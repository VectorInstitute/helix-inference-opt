# helix-inference-opt

Run `Qwen/Qwen2.5-0.5B-Instruct` on WikiText-2 prompts and let an autonomous agent maximize how fast it generates text. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The agent has a fixed **1-minute generation benchmark** per experiment. It modifies `infer.py` — batching strategy, attention kernels, quantization, speculative decoding — and tries to maximize **output tokens per second (`tokens_per_sec`)**, subject to a bits-per-byte quality guard.

## Quickstart

```bash
pip install 'helices[claude]'

git clone https://github.com/VectorInstitute/helix-inference-opt.git
cd helix-inference-opt

uv run prepare.py   # one-time: download model + dataset (~1 GB)
helix run
```

## Metric

**Primary: `tokens_per_sec` (maximize)** — Output tokens generated per wall-clock second during autoregressive decoding. Measured over a 1-minute window with 128-token WikiText-2 prompts and 256 max new tokens per call.

**Quality guard: `bpb` (must not degrade)** — Bits per byte on a fixed WikiText-2 test set, evaluated after warmup. Must stay within 1% of baseline. Guards against optimizations (e.g. aggressive quantization) that trade quality for speed.

## Scope

The agent may only modify `infer.py`. All other files are read-only.

What's in scope: batch size, `torch.compile`, flash attention, speculative decoding, static KV cache, quantization, custom CUDA kernels — anything that speeds up `model.generate()` without touching the weights.

## Hardware

Requires a single NVIDIA GPU (CUDA). CPU and MPS are not supported.

---

Built with [helix](https://github.com/VectorInstitute/helix).
