# inference-opt

Autonomous research into autoregressive generation throughput for a causal language model.

## Setup

Before starting a session:

1. Verify the model is cached: check `~/.cache/autoresearch/`. If not, run `uv run prepare.py` once.
2. Read `prepare.py` to understand the fixed evaluation harness.
3. Read `infer.py` to understand the current inference strategy.

Requires a single NVIDIA GPU (CUDA). Does not run on CPU or MPS.

## The task

Maximize how fast the model generates text. The harness measures **output tokens per second**
during autoregressive decoding from WikiText-2 validation prompts (128-token context,
256 max new tokens). This is real inference — token-by-token generation using the KV cache —
not teacher-forced perplexity evaluation.

## Experimentation

Each experiment runs for a **fixed time budget of 1 minute** (wall clock, excluding model loading ~30s).

**What you CAN do:**

- Modify `infer.py` freely. There are no prescribed approaches. Reason from first principles.

**What you CANNOT do:**

- Modify `prepare.py`. It owns the evaluation harness and BPB quality guard. It is read-only.
- Fine-tune or modify the model weights.
- Install new packages beyond those in `pyproject.toml`.
- Tune hyperparameters such as batch size, temperature, or repetition penalty.

## Harness phases

Each run of `uv run infer.py` proceeds in three phases:

1. **Warmup** — one call to `infer_fn` (not timed). Triggers `torch.compile` JIT,
   quantization initialization, or any other lazy setup.
2. **BPB quality guard** — teacher-forced eval on 100 fixed WikiText-2 test chunks.
   Runs *after* warmup so quantization effects are reflected. Guards against quality regression.
3. **Generation benchmark** — autoregressive generation for 1 minute.
   `tokens_per_sec` counts only output (non-prompt) tokens.

## infer_fn signature

```python
def infer(
    model,                        # CausalLM, CUDA, bfloat16
    tokenizer,
    prompts: list[list[int]],     # batch_size prompts, all length PROMPT_TOKENS (128)
    max_new_tokens: int,          # tokens to generate (256)
) -> list[list[int]]:             # newly generated token ids only (no prompt)
```

All prompts in a batch are the same length, so **no padding is needed**.

## Metrics

**Primary: `tokens_per_sec` (maximize)**

Output tokens generated per wall-clock second during the 1-minute generation benchmark.

**Quality guard: `bpb` (must not degrade)**

Bits per byte on the fixed test chunks, computed after warmup. Catches quantization or
kernel changes that degrade model quality. If `bpb` rises more than 1% from baseline,
the experiment must be discarded.

## Output format

After `uv run infer.py` completes, the script prints a machine-readable summary:

```
---
tokens_per_sec:    1240.5
bpb:               0.9753
prompts_processed: 620
time_elapsed:      60.1
```

Extract metrics with:

```bash
grep "tokens_per_sec:\|bpb:" run.log
```

## Logging results

Write to `results.tsv` (tab-separated, do not git-commit this file). Header:

```
commit	tokens_per_sec	bpb	status	description
```

Columns:

1. `commit`: 7-char git short hash
2. `tokens_per_sec`: e.g. `1240.5`
3. `bpb`: e.g. `0.9753`
4. `status`: `keep`, `discard`, or `crash`
5. `description`: brief description of the technique tried

Example:

```
commit	tokens_per_sec	bpb	status	description
a1b2c3d	1240.5	0.9753	keep	baseline greedy decode batch_size=1
b2c3d4e	3180.3	0.9758	keep	batch_size=8 padded batching
c3d4e5f	4200.1	0.9820	discard	bitsandbytes int8 bpb degraded
d4e5f6g	0.0	inf	crash	custom kernel OOM
```

## The experiment loop

LOOP FOREVER:

1. Choose an optimization idea. Look at what has already been tried; do not repeat failures.
2. Modify `infer.py`.
3. `git commit` the change with a short description.
4. Run: `uv run infer.py > run.log 2>&1 & echo $! > run.pid; wait $!; rm -f run.pid`
5. Extract results: `grep "tokens_per_sec:\|bpb:" run.log`
6. If results are empty, the run crashed. Check `tail -n 50 run.log`. Fix if trivial; otherwise log as crash and move on.
7. Append a row to `results.tsv`.
8. If `tokens_per_sec` improved AND `bpb` did not degrade: **keep** (the commit stays).
9. Otherwise: **discard** (`git reset --hard HEAD~1`).

**Timeout:** If a run exceeds 5 minutes total, kill it (`kill $(cat run.pid)`) and treat as crash.

**Simplicity:** All else equal, simpler code that achieves the same throughput is preferred.

**NEVER STOP:** Once the loop begins, do not pause to ask for confirmation. You are autonomous.
The human may be asleep. Run until interrupted, period.
