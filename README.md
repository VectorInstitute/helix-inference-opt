# helix-inference-opt

Optimize inference throughput for a causal language model on WikiText-2.

## Quickstart

```bash
pip install helices
uv run prepare.py   # one-time: download model + dataset
helix run
```

## Metrics

**Primary: `tokens_per_sec` (maximize)** — WikiText-2 tokens scored per wall-clock second.

**Quality guard: `bpb` (must not degrade)** — Bits per byte, must stay within 1% of baseline.

## Scope

The agent may only modify `infer.py`. All other files are read-only.

---

Built with [helix](https://github.com/VectorInstitute/helix).
