"""Inference strategy for the inference-opt helix.

Agent instructions:
- Modify the INFERENCE STRATEGY section freely.
- Goal: maximize ``tokens_per_sec`` (output tokens generated per second).
- Baseline: greedy decoding with model.generate(), batch_size=1.

Usage:
    uv run infer.py
"""

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


def _make_decode_fn(model):
    """Create and compile the decode step function with reduce-overhead (CUDA graphs)."""
    global _decode_compiled
    if _decode_compiled is not None:
        return

    @torch.compile(mode="max-autotune", fullgraph=True)
    def decode(token, cache_position, past_kv):
        outputs = model(
            token,
            cache_position=cache_position,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=False,
        )
        logits = outputs[0]
        new_past_kv = outputs[1]
        next_tok = logits[:, -1, :].argmax(dim=-1)
        return next_tok, new_past_kv

    _decode_compiled = decode


def infer(model, tokenizer, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """Greedy decode: uncompiled prefill + compiled decode with CUDA graphs."""
    device = next(model.parameters()).device
    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    # Create a fresh static cache for this generation
    cache = StaticCache(
        model.config,
        max_batch_size=batch_size,
        max_cache_len=prompt_len + max_new_tokens,
        dtype=model.dtype,
        device=device,
    )

    generated = torch.zeros(batch_size, max_new_tokens, dtype=torch.long, device=device)

    with torch.inference_mode():
        # Prefill: uncompiled, triggers lazy cache initialization + mark_static_address
        cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)
        outputs = model(
            input_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=False,
        )
        logits = outputs[0]
        past_kv = outputs[1]
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated[:, 0] = next_token

        # Set up compiled decode after prefill has initialized the cache
        _make_decode_fn(model)

        # Decode with CUDA graphs — clone outputs to avoid overwrite
        cache_pos = torch.zeros(1, device=device, dtype=torch.long)
        for i in range(1, max_new_tokens):
            cache_pos.fill_(prompt_len + i - 1)
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
