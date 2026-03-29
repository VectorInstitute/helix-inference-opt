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
def infer(model, tokenizer, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:  # type: ignore[no-untyped-def]
    """Autoregressive generation. Returns only the newly generated token ids.

    Parameters
    ----------
    model :
        The causal LM (already on CUDA, bfloat16).
    tokenizer :
        The model's tokenizer.
    prompts : list[list[int]]
        List of prompt token-id sequences, all the same length.
    max_new_tokens : int
        Number of new tokens to generate per prompt.

    Returns
    -------
    list[list[int]]
        List of lists, each containing only the generated (non-prompt) token ids.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    return [out[prompt_len:].tolist() for out in output_ids]


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
