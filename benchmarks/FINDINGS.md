# Multi-shape benchmark campaign — findings log

Result rows are MI350X / gfx950, ROCm 7.2, torch 2.9.1+rocm (provenance now
recorded per row). Headline metric = **kernel-only, cold-cache**: CUDA/HIP graph
captured + replayed on a side stream, L2 flushed before each replay
(`cache_state=l2_flushed_graph`) → steady-state HBM-bound; eager/host-overhead
reported separately. Per-kernel detail in `examples/<kernel>/benchmark_summary.md`.

> **PR #1 review round (addressed):** (1) capture+replay the graph on a side
> stream, not the default stream; (2) switch the graph metric to single-launch +
> L2-flush cold-cache; (3) `_elem_bytes` now maps int8/fp4/mixed for correct
> effective-GB/s; (4) enable the stage2 CShuffle epilogue → **FlyDSL moe now
> runs**. Cold-cache removes the baselines' warm-L2 edge on the small norm
> tensors, so FlyDSL is materially more competitive than the earlier warm numbers
> (rmsnorm 0.71→0.88) — both valid, distinguished by `cache_state`.

| kernel | shapes | geomean vs best (kernel-only, cold) | decision | headline |
|---|---:|---:|---|---|
| rmsnorm | 159 | 0.88 (weighted **0.82**, 2 models) | tune_needed | wins 2048-aligned; loses non-aligned generic path; **crash fixed → PR #615** |
| layernorm | 73 | 0.88 | tune_needed | wins aligned/large; loses small-M/non-aligned |
| softmax | 73 | **1.08** | promote | **wins overall** despite vectorized path dead-coded off → **issue #627** |
| gemm (hgemm_splitk) | 30/265 | 0.50 | tune/rewrite | ≈parity vs aiter; slower vs hipBLASLt/aiter_triton on small-N model shapes |
| fused_rope_cache | 30 | ≈parity vs aiter_triton | promote_with_guardrails | beats PyTorch eager strongly; ≈parity with aiter_triton where it exists |
| moe_gemm (2-stage) | 3 FlyDSL ok | unblocked (no fair baseline yet) | fix aiter baseline | **CShuffle fix → FlyDSL fp8 moe runs + correct (~44µs)**; aiter baseline errors on fp8 out dtype → only pytorch-eager left, so vs-best is meaningless until that's fixed |

### fused_rope_cache — promote_with_guardrails
- Overall vs-best is inflated because PyTorch is the only correct baseline on several rows; on rows where aiter_triton runs, FlyDSL is ≈ parity. Eager-vs-graph gap is large (single 64-lane wave/block → launch-overhead-bound), so kernel-only is the right metric.
- Worst ~0.67× at head_dim=64, T=2048 vs aiter_triton. (Discovery noted a 0.17× rocprof point at a GPT-OSS-TP8 trivial-T shape — serialized `lgkmcnt(0)` fences; worth a profiler pass.)

### moe_gemm — unblocked by the CShuffle fix; aiter baseline still broken
- The stage2 CShuffle epilogue fix (PR #1 review) makes **FlyDSL's fp8 2-stage MoE run and pass correctness** (3 shapes, ~44µs). int8/fp4/mixed remain out of this adapter's scope (route to mixed_moe_gemm_2stage).
- **Open follow-up**: the aiter `fused_moe` baseline raises `AssertionError: unsupported out dtype torch.float8_e4m3fn` on these shapes, leaving only slow pytorch-eager → the "vs best" geomean is meaningless (looks ~157× but that's vs eager torch). Fix the aiter-moe baseline out-dtype before drawing a FlyDSL moe perf verdict.
- 6 fp8 shapes still flagged `incorrect`: fp8 round-trips through two quantizations (inputs + inter-stage) so the gap exceeds even tol(fp8)=0.15 on some shapes — needs a per-shape look (scale handling) or a looser fp8-MoE tol.

### gemm (hgemm_splitk) — rewrite_needed
- Only bf16 in scope (30/265 shapes; int8/fp4/fp8 are other kernels → correctly unsupported). Tolerance is op-aware now (GEMM accumulates over K → (0.1,0.1), matching FlyDSL's own hgemm_splitk test).
- FlyDSL split-K hgemm ≈ parity with aiter compiled (0.98×) but **0.42–0.44× vs hipBLASLt / aiter_triton** on the AITER model GEMM shapes (small N=128/256/640, large K) — these are GEMV-ish; hgemm_splitk isn't tuned for them. Worst M=1,N=256,K=7168 → 0.10×.
- NOTE: aiter `gemm_a16w16_asm` returns structurally-wrong output on some small-M N=128 shapes (flagged `incorrect`, excluded from best) — likely an adapter orientation issue for those shapes, to refine.

### Adapter bugs found + fixed during the campaign (harness, not FlyDSL)
- fused_rope_cache adapters cached T-sized output/KV-cache buffers under a key that omitted T → cross-shape buffer reuse → **GPU memory access fault**. Fixed (T in cache key).
- moe_gemm Op.make_inputs KeyError on quant dtypes (fp4/int8/mixed) → guarded (`_safe_torch_dtype`), so those become per-provider `unsupported` not a shared crash.
- Added an op-aware correctness tolerance hook (`Op.tolerance`): GEMM (0.1,0.1) accumulation; RoPE atol-dominated (near-zero rotation outputs) so a correct f16 baseline isn't false-failed and doesn't inflate FlyDSL's vs-best.

## Per-kernel notes

### rmsnorm — tune_needed
- **Two live serving traces** (Qwen3-4B hidden=2560 + DeepSeek-R1-MXFP4 hidden=7168, both captured via SGLang `bench_serving` ISL/OSL/concurrency sweeps; DeepSeek-R1 ran fine on TP=8, no MXFP4 blocker). Production-weighted geomean vs best = **0.65×** (unweighted 0.71×). Serving-stage by model: DeepSeek-R1 0.61×, Qwen3-4B 0.69× — the heavily-weighted non-2048-aligned hidden sizes (2560/7168) are exactly where FlyDSL drops to the generic scalar path. The more real traffic you weight by, the worse it looks.
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
| softmax vectorized path dead-coded off + won't compile (fastmath bool + LLVM cast) | softmax | **filed: #627** |
| narrow fast-path (N==8192 only) + small-M one-block-per-row | layernorm, rmsnorm | candidate — needs profiler pass; parallelize across N for small M |

## Pending kernels
rope/fused_rope_cache, hgemm_splitk (+ preshuffle), moe_gemm, pa_decode, mla_decode, flash_attn — adapters in progress.
