"""Inference strategy for the inference-opt helix.

Agent instructions:
- Modify the INFERENCE STRATEGY section freely.
- Goal: maximize ``tokens_per_sec`` (output tokens generated per second).
- Baseline: greedy decoding with model.generate(), batch_size=1.

Usage:
    uv run infer.py
"""

import os
# Optimize CUDA memory allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import StaticCache
from prepare import evaluate
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()

# Hyperparameters (do not edit)
BATCH_SIZE = 1


# INFERENCE STRATEGY — modify this section

_decode_compiled = None
_setup_done = False
_device = None


def _setup(model):
    """One-time setup: eager attention + compile decode."""
    global _decode_compiled, _setup_done, _device
    if _setup_done:
        return
    _setup_done = True

    _device = next(model.parameters()).device

    # Use eager attention (faster for batch=1 single-token decode)
    model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"

    @torch.compile(mode="max-autotune", fullgraph=True)
    def decode(token, cache_position, past_kv):
        out = model(
            token,
            cache_position=cache_position,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=False,
        )
        return out[0][:, -1, :].argmax(dim=-1), out[1]

    _decode_compiled = decode


def infer(model, tokenizer, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """Greedy decode: uncompiled prefill + CUDA graph decode, eager attention."""
    _setup(model)

    input_ids = torch.tensor(prompts, dtype=torch.long, device=_device)
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    cache = StaticCache(
        model.config,
        max_batch_size=batch_size,
        max_cache_len=prompt_len + max_new_tokens,
        dtype=model.dtype,
        device=_device,
    )

    generated = torch.zeros(batch_size, max_new_tokens, dtype=torch.long, device=_device)

    with torch.inference_mode():
        # Prefill (uncompiled — needed for mark_static_address)
        cache_position = torch.arange(prompt_len, device=_device, dtype=torch.long)
        out = model(input_ids, cache_position=cache_position, past_key_values=cache,
                    use_cache=True, return_dict=False)
        next_token = out[0][:, -1, :].argmax(dim=-1)
        past_kv = out[1]
        generated[:, 0] = next_token

        # Decode with CUDA graphs
        cache_pos = torch.zeros(1, device=_device, dtype=torch.long)
        base = prompt_len
        for i in range(1, max_new_tokens):
            cache_pos.fill_(base + i - 1)
            torch.compiler.cudagraph_mark_step_begin()
            next_token, past_kv = _decode_compiled(
                next_token.unsqueeze(1), cache_pos, past_kv
            )
            next_token = next_token.clone()
            generated[:, i] = next_token

    return [generated[b].tolist() for b in range(batch_size)]


# Entry point (do not modify)
if __name__ == "__main__":
    results = evaluate(infer, batch_size=BATCH_SIZE)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("tokens_per_sec", f"[cyan]{results['tokens_per_sec']:.1f}[/cyan]")
    table.add_row("bpb", f"{results['bpb']:.4f}")
    table.add_row("prompts_processed", str(results["prompts_processed"]))
    table.add_row("time_elapsed", f"{results['time_elapsed']:.1f}s")

    console.print()
    console.print(Panel(table, title="[bold cyan]results[/bold cyan]", expand=False))

    # Machine-parseable summary for grep in the experiment loop
    print("---")
    print(f"tokens_per_sec:    {results['tokens_per_sec']:.1f}")
    print(f"bpb:               {results['bpb']:.4f}")
    print(f"prompts_processed: {results['prompts_processed']}")
    print(f"time_elapsed:      {results['time_elapsed']:.1f}")