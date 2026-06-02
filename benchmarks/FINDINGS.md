# Multi-shape benchmark campaign — findings log

MI350X / gfx950, ROCm 7.2, torch 2.9.1+rocm. Headline metric = **kernel-only**
(CUDA-graph) median speedup vs the best available baseline; eager/host-overhead
reported separately. Per-kernel detail in `examples/<kernel>/benchmark_summary.md`.

| kernel | shapes | geomean vs best (kernel-only) | decision | headline |
|---|---:|---:|---|---|
| rmsnorm | 117 | 0.76 (weighted 0.70) | tune_needed | wins 2048-aligned; loses non-aligned generic path; **crash bug fixed → PR #615** |
| layernorm | 73 | 0.84 | tune_needed | wins aligned/large (1.08–1.10×); loses small-M/non-aligned (0.71×) |
| softmax | 73 | **1.13** | promote | **wins overall** despite vectorized path being dead-coded off → headroom |

## Per-kernel notes

### rmsnorm — tune_needed
- Wins on 2048-aligned shapes (synthetic 0.94×, diagnostic 1.04×; beats PyTorch ~1.4×). Fastest kernel on the 32768×8192 diagnostic.
- `implementation_gap`: non-2048-aligned N (2560/5120/7168 — the real model hidden sizes) drop to the generic scalar path; profiler shows same block count as triton, slower per block.
- Eager launcher host-overhead ~46 µs (decode) — separate `launch_or_roofline_limited` (eager only; CUDA-graph decode hides it).
- **Hard crash** M>8192 & N≤2048 (block size > AMDGPU max_flat_workgroup_size). **Fixed**: ROCm/FlyDSL issue #614, PR #615 (`known_block_size` annotation). Reproduced on a real Qwen3-4B prefill q/k-norm shape.

### layernorm — tune_needed
- Same profile as rmsnorm: wins large-aligned (diagnostic 1.10×, synthetic 1.08×, vs aiter_triton 1.05×, vs pytorch 1.49×), loses small-M/non-aligned (model_config 0.71×, worst M=1,N=16384 → 0.38×).
- **Candidate gap**: the layernorm fast-vectorized path triggers ONLY at N==8192 (16-bit) — far narrower than rmsnorm's N≥2048 & N%2048==0. Most production hidden sizes miss it → generic scalar path. Worth widening the fast-path condition. (Needs a profiler pass before filing.)

### softmax — promote (with headroom)
- FlyDSL softmax **wins overall** (geomean 1.13×; vs aiter_triton 1.96×, vs triton 1.17×, vs pytorch 1.48×), even on the generic scalar path.
- **Candidate FlyDSL issue/PR (high value)**: the vectorized path is **dead-coded off** — `softmax_kernel.py:104` guards it with `if const_expr(False and N >= tile_cols and N % tile_cols == 0):`. The leading `False and` makes the BufferCopy128b VEC_WIDTH=8 path unreachable for every shape. Re-enabling it should improve the bandwidth-bound large-N shapes further. Already wins despite this → clear headroom. (Matches the prior softmax dead-code finding.)
- Worst: M=16384,N=128 → 0.39× vs pytorch (generic path inefficiency at tiny N / large M).

## Candidate FlyDSL issues/PRs

| finding | kernel | status |
|---|---|---|
| large-M small-N block size > max_flat_workgroup_size (crash) | rmsnorm | **filed: #614 / PR #615** |
| softmax vectorized path dead-coded off (`False and ...`) | softmax | candidate — re-enable + verify (high value, low risk) |
| narrow fast-path (N==8192 only) + small-M one-block-per-row | layernorm, rmsnorm | candidate — needs profiler pass; parallelize across N for small M |

## Pending kernels
rope/fused_rope_cache, hgemm_splitk (+ preshuffle), moe_gemm, pa_decode, mla_decode, flash_attn — adapters in progress.
