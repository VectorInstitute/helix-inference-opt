"""Inference strategy for the inference-opt helix.

Agent instructions:
- Modify the INFERENCE STRATEGY section freely.
- Goal: maximize ``tokens_per_sec`` (output tokens generated per second).
- Baseline: greedy decoding with model.generate(), batch_size=1.

Usage:
    uv run infer.py
"""

import torch
from prepare import evaluate
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()

# Hyperparameters (do not edit)
BATCH_SIZE = 1


# INFERENCE STRATEGY — modify this section

_decode_step = None
_prefill_step = None


def _setup(model):
    """One-time: compile prefill and decode step functions separately."""
    global _decode_step, _prefill_step

    if _decode_step is not None:
        return

    @torch.compile(mode="default", fullgraph=True)
    def prefill(input_ids):
        outputs = model(input_ids, use_cache=True, return_dict=False)
        logits = outputs[0]
        past_kv = outputs[1]
        next_tok = logits[:, -1, :].argmax(dim=-1)
        return next_tok, past_kv

    @torch.compile(mode="default", fullgraph=True)
    def decode(token, past_kv):
        outputs = model(token.unsqueeze(1), past_key_values=past_kv, use_cache=True, return_dict=False)
        logits = outputs[0]
        new_past_kv = outputs[1]
        next_tok = logits[:, -1, :].argmax(dim=-1)
        return next_tok, new_past_kv

    _prefill_step = prefill
    _decode_step = decode


def infer(model, tokenizer, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """Manual greedy decode with compiled prefill and decode steps."""
    _setup(model)

    device = next(model.parameters()).device
    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    batch_size = input_ids.shape[0]

    generated = torch.zeros(batch_size, max_new_tokens, dtype=torch.long, device=device)

    with torch.inference_mode():
        # Prefill
        next_token, past_kv = _prefill_step(input_ids)
        generated[:, 0] = next_token

        # Decode
        for i in range(1, max_new_tokens):
            next_token, past_kv = _decode_step(next_token, past_kv)
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
