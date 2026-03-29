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

_compiled = False


def _compile_model(model):
    """Apply torch.compile to the model forward pass."""
    global _compiled
    if not _compiled:
        model.forward = torch.compile(model.forward, mode="default", fullgraph=True)
        _compiled = True


def infer(model, tokenizer, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
    """Manual greedy decode loop with torch.compile."""
    _compile_model(model)

    device = next(model.parameters()).device
    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    generated = torch.zeros(batch_size, max_new_tokens, dtype=torch.long, device=device)

    with torch.inference_mode():
        # Prefill: process the full prompt
        outputs = model(input_ids, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated[:, 0] = next_token
        past_key_values = outputs.past_key_values

        # Decode: one token at a time
        for i in range(1, max_new_tokens):
            outputs = model(next_token.unsqueeze(1), past_key_values=past_key_values, use_cache=True)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            generated[:, i] = next_token
            past_key_values = outputs.past_key_values

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
