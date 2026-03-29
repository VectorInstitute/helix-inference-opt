"""Fixed infrastructure for the inference-opt helix.

Handles model loading, WikiText-2 dataset preparation, and the generation
throughput evaluation harness. Do not modify — this is the fixed ground truth.

Requires a single NVIDIA GPU (CUDA). CPU and MPS are not supported.

Usage (one-time setup):
    uv run prepare.py
"""

import itertools
import math
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging


hf_logging.set_verbosity_error()
console = Console()

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU required. This harness does not support CPU or MPS.\n"
        "Set CUDA_VISIBLE_DEVICES or run on a machine with an NVIDIA GPU."
    )

MODEL_ID = os.environ.get("HELIX_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TIME_BUDGET = int(os.environ.get("HELIX_TIME_BUDGET", "300"))
PROMPT_TOKENS = int(os.environ.get("HELIX_PROMPT_TOKENS", "128"))     # context length per prompt
MAX_NEW_TOKENS = int(os.environ.get("HELIX_MAX_NEW_TOKENS", "256"))   # tokens to generate per prompt
BPB_CHUNKS = int(os.environ.get("HELIX_BPB_CHUNKS", "100"))           # chunks for BPB quality guard
BPB_CHUNK_TOKENS = 512                                                  # fixed chunk size for BPB eval

DEVICE = "cuda"
DTYPE = torch.bfloat16


def load_model_and_tokenizer():  # type: ignore[no-untyped-def]
    """Load the configured model from cache onto CUDA."""
    with console.status(f"[bold cyan]Loading {MODEL_ID}...[/]", spinner="dots"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="cuda:0",
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        model.eval()
    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    console.print(f"[green]✓[/green] Model loaded  [dim]VRAM: {mem_gb:.1f} GB[/dim]")
    return model, tokenizer


def load_generation_prompts(tokenizer) -> list[list[int]]:  # type: ignore[no-untyped-def]
    """Load WikiText-2 validation set as fixed-length generation prompts.

    Returns a list of token-id lists, each exactly PROMPT_TOKENS long.
    All prompts are the same length so batching requires no padding.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", cache_dir=CACHE_DIR)
    full_text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompts = []
    for start in range(0, len(all_ids) - PROMPT_TOKENS + 1, PROMPT_TOKENS):
        prompts.append(all_ids[start : start + PROMPT_TOKENS])
    return prompts


def load_bpb_chunks(tokenizer) -> list[dict]:  # type: ignore[no-untyped-def]
    """Load the first BPB_CHUNKS chunks of WikiText-2 test set for quality guard.

    Returns a list of dicts with keys ``ids`` (list[int]) and ``utf8_bytes`` (int).
    This set is fixed and never changes so BPB comparisons across experiments are valid.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=CACHE_DIR)
    full_text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(all_ids) - BPB_CHUNK_TOKENS + 1, BPB_CHUNK_TOKENS):
        ids = all_ids[start : start + BPB_CHUNK_TOKENS]
        utf8_bytes = len(tokenizer.decode(ids).encode("utf-8"))
        chunks.append({"ids": ids, "utf8_bytes": utf8_bytes})
        if len(chunks) >= BPB_CHUNKS:
            break
    return chunks


def compute_bpb(model, chunks: list[dict]) -> float:  # type: ignore[no-untyped-def]
    """Compute bits-per-byte via teacher-forced eval on the fixed BPB chunks.

    This is the quality guard metric. It runs on the model in its current state
    (after any modifications applied during warmup) to catch quantization or
    kernel changes that degrade model quality.
    """
    total_nll_nats = 0.0
    total_utf8_bytes = 0
    with torch.inference_mode():
        for chunk in chunks:
            ids = torch.tensor([chunk["ids"]], dtype=torch.long, device=DEVICE)
            logits = model(ids).logits
            nll = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),
                ids[:, 1:],
                reduction="sum",
            )
            total_nll_nats += nll.item()
            total_utf8_bytes += chunk["utf8_bytes"]
    return total_nll_nats / math.log(2) / total_utf8_bytes


def evaluate(infer_fn, batch_size: int = 1) -> dict:  # type: ignore[no-untyped-def]
    """Run the fixed evaluation harness.

    Calls ``infer_fn(model, tokenizer, prompts: list[list[int]], max_new_tokens: int)
    -> list[list[int]]`` where each inner list is the *newly generated* token ids
    (prompt tokens excluded). The harness passes ``batch_size`` prompts per call.

    All prompts are the same length (PROMPT_TOKENS tokens) so no padding is needed.

    Evaluation proceeds in three phases:

    1. **Warmup** — one call to ``infer_fn`` (not timed, not counted). Triggers any
       lazy setup such as ``torch.compile`` JIT or in-place quantization.
    2. **BPB quality guard** — teacher-forced eval on the first BPB_CHUNKS test chunks.
       Runs on the model in its post-warmup state, so quantization effects are captured.
    3. **Generation benchmark** — autoregressive generation for TIME_BUDGET seconds.
       ``tokens_per_sec`` counts only the newly generated output tokens.

    Args:
        infer_fn: Callable conforming to the signature above.
        batch_size: Number of prompts per ``infer_fn`` call.

    Returns
    -------
        Dict with keys ``tokens_per_sec``, ``bpb``, ``prompts_processed``, ``time_elapsed``.
    """
    model, tokenizer = load_model_and_tokenizer()
    prompts = load_generation_prompts(tokenizer)
    bpb_chunks = load_bpb_chunks(tokenizer)
    n_prompts = len(prompts)

    # --- Phase 1: Warmup ---
    console.print("[bold]Warmup:[/bold] one forward pass to trigger lazy init...")
    with console.status("[cyan]Warming up...[/]", spinner="dots"):
        warmup_batch = prompts[:batch_size]
        try:
            infer_fn(model, tokenizer, warmup_batch, MAX_NEW_TOKENS)
        except Exception as exc:
            console.print(f"[red]✗[/red] Warmup failed: {exc}")
            raise
    console.print("[green]✓[/green] Warmup complete")

    # --- Phase 2: BPB quality guard ---
    console.print("[bold]BPB guard:[/bold] teacher-forced eval on fixed test chunks...")
    with console.status("[cyan]Computing BPB...[/]", spinner="dots"):
        bpb = compute_bpb(model, bpb_chunks)
    console.print(f"[green]✓[/green] BPB quality guard  [dim]bpb={bpb:.4f}  ({len(bpb_chunks)} chunks × {BPB_CHUNK_TOKENS} tokens)[/dim]")

    # --- Phase 3: Generation benchmark ---
    total_output_tokens = 0
    prompts_processed = 0

    console.print(
        f"\n[bold]Generation benchmark:[/bold]  "
        f"[dim]budget: {TIME_BUDGET}s · {n_prompts} prompts · "
        f"{PROMPT_TOKENS} ctx · {MAX_NEW_TOKENS} max new tokens · batch={batch_size}[/dim]"
    )

    t_start = time.time()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(bar_width=28),
        TextColumn("[dim]{task.completed:.0f}s / {task.total}s[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    prompt_cycle = itertools.cycle(range(n_prompts))

    with progress:
        task = progress.add_task("generating...", total=TIME_BUDGET)
        while True:
            elapsed = time.time() - t_start
            if elapsed >= TIME_BUDGET:
                break

            batch_indices = [next(prompt_cycle) for _ in range(batch_size)]
            batch_prompts = [prompts[j] for j in batch_indices]

            try:
                generated = infer_fn(model, tokenizer, batch_prompts, MAX_NEW_TOKENS)
                total_output_tokens += sum(len(g) for g in generated)
                prompts_processed += len(batch_prompts)
            except Exception as exc:
                console.print(f"[yellow]⚠[/yellow] infer_fn raised [red]{type(exc).__name__}[/red]: {exc}")
                break

            elapsed = time.time() - t_start
            tps = total_output_tokens / elapsed if elapsed > 0 else 0.0
            desc = (
                f"[cyan]{prompts_processed}[/cyan] prompts  "
                f"[cyan]{tps:.0f}[/cyan] output tok/s"
            )
            progress.update(task, completed=min(elapsed, TIME_BUDGET), description=desc)
            if not console.is_terminal:
                remaining = max(0, TIME_BUDGET - elapsed)
                print(
                    f"  [{int(elapsed):3d}s/{TIME_BUDGET}s | {int(remaining):3d}s left]"
                    f"  {prompts_processed} prompts  {tps:.0f} tok/s",
                    flush=True,
                )

    time_elapsed = time.time() - t_start
    tokens_per_sec = total_output_tokens / time_elapsed if time_elapsed > 0 else 0.0

    return {
        "tokens_per_sec": tokens_per_sec,
        "bpb": bpb,
        "prompts_processed": prompts_processed,
        "time_elapsed": time_elapsed,
    }


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    console.rule("[bold cyan]inference-opt setup[/bold cyan]")
    console.print(f"[dim]Cache directory:[/dim] {CACHE_DIR}\n")

    console.print("[bold]Step 1:[/bold] Downloading model weights...")
    _, tokenizer = load_model_and_tokenizer()
    console.print()

    console.print("[bold]Step 2:[/bold] Caching WikiText-2 datasets...")
    with console.status("[cyan]Fetching WikiText-2...[/]", spinner="dots"):
        prompts = load_generation_prompts(tokenizer)
        bpb_chunks = load_bpb_chunks(tokenizer)
    console.print(f"[green]✓[/green] Generation prompts  [dim]{len(prompts)} × {PROMPT_TOKENS} tokens[/dim]")
    console.print(f"[green]✓[/green] BPB quality chunks  [dim]{len(bpb_chunks)} × {BPB_CHUNK_TOKENS} tokens[/dim]\n")

    console.print(
        Panel(
            "[green]Setup complete![/green]\n\nRun a session with:  [bold cyan]helix run[/bold cyan]",
            title="[bold]inference-opt[/bold]",
            expand=False,
        )
    )
