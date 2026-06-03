# Multi-shape benchmark campaign — findings log

Result rows are MI350X / gfx950, ROCm 7.2, torch 2.9.1+rocm (provenance now
recorded per row). Headline metric = **kernel-only, cold-cache**: CUDA/HIP graph
captured + replayed on a side stream, L2 flushed before each replay
(`cache_state=l2_flushed_graph`) → steady-state HBM-bound; eager/host-overhead
reported separately. Per-kernel detail in `examples/<kernel>/benchmark_summary.md`.

> **Harness changes in this rerun:** (1) capture+replay the graph on a side
> stream; (2) switch the graph metric to single-launch + L2-flush cold-cache;
> (3) `_elem_bytes` maps int8/fp4/mixed for correct effective GB/s; (4) enable
> the stage2 CShuffle epilogue, so FlyDSL MoE now runs on fp8 rows. Cold-cache
> removes the baselines' warm-L2 edge on small norm tensors, so FlyDSL is more
> competitive than the earlier warm-cache rows (rmsnorm 0.71→0.88); both are
> valid only when distinguished by `cache_state`.

| kernel | shapes | geomean vs best (kernel-only, cold) | decision | headline |
|---|---:|---:|---|---|
| rmsnorm | 159 | 0.88 (weighted **0.82**, 2 models) | tune_needed | wins 2048-aligned; loses non-aligned generic path; **crash fixed → PR #615** |
| layernorm | 73 | 0.88 | tune_needed | wins aligned/large; loses small-M/non-aligned |
| softmax | 73 | **1.08** | promote | **wins overall** despite vectorized path dead-coded off → **issue #627** |
| gemm (hgemm_splitk) | 30/265 | 0.50 | rewrite_needed | 0.89× vs aiter; 0.55–0.59× vs hipBLASLt/aiter_triton/PyTorch on small-N model shapes |
| fused_rope_cache | 30 | 1.72 vs best; 0.97 vs aiter_triton | promote | beats PyTorch reference strongly; near parity with aiter_triton where it exists |
| moe_gemm (2-stage) | 3/29 correct+timed | 157.59 vs PyTorch-only | tune_needed | CShuffle fix unblocks fp8, but 6 fp8 rows are incorrect and 20 quantized rows unsupported |

### fused_rope_cache — promote
- Overall vs-best is 1.72× because PyTorch is the only correct baseline on several rows; on rows where aiter_triton runs, FlyDSL is near parity (0.97×). Eager-vs-graph gap is large (single 64-lane wave/block → launch-overhead-bound), so kernel-only is the right metric.
- Worst 0.84× at head_dim=64, T=2048 vs aiter_triton. (Discovery noted a 0.17× rocprof point at a GPT-OSS-TP8 trivial-T shape — serialized `lgkmcnt(0)` fences; worth a profiler pass.)

### moe_gemm — unblocked by CShuffle, still not promotable
- The stage2 CShuffle epilogue fix makes FlyDSL's fp8 2-stage MoE run and pass correctness on 3 rows (~43–45 us). int8/fp4/mixed remain out of this adapter's scope (route to mixed_moe_gemm_2stage).
- **Open follow-up**: the aiter `fused_moe` baseline fails on the fp8 rows, leaving only slow PyTorch eager as the correct baseline for the 3 passing FlyDSL rows. The 157.59× vs-best geomean is therefore not a fair FlyDSL-vs-optimized-MoE verdict.
- 6 fp8 rows still fail correctness: fp8 round-trips through two quantizations (inputs + inter-stage), and the gap exceeds tol(fp8)=0.15 on some shapes. Inspect scale handling or define a MoE-specific fp8 tolerance before promotion.

### gemm (hgemm_splitk) — rewrite_needed
- Only bf16 in scope (30/265 shapes; int8/fp4/fp8 are other kernels → correctly unsupported). Tolerance is op-aware now (GEMM accumulates over K → (0.1,0.1), matching FlyDSL's own hgemm_splitk test).
- FlyDSL split-K hgemm is 0.89× vs aiter compiled and **0.55–0.59× vs hipBLASLt / aiter_triton / PyTorch** on the AITER model GEMM shapes (small N=128/256/640, large K) — these are GEMV-ish; hgemm_splitk isn't tuned for them. Worst M=1,N=256,K=7168 → 0.26×.
- NOTE: aiter `gemm_a16w16_asm` returns structurally-wrong output on some small-M N=128 shapes (flagged `incorrect`, excluded from best) — likely an adapter orientation issue for those shapes, to refine.

### Adapter bugs found + fixed during the campaign (harness, not FlyDSL)
- fused_rope_cache adapters cached T-sized output/KV-cache buffers under a key that omitted T → cross-shape buffer reuse → **GPU memory access fault**. Fixed (T in cache key).
- moe_gemm Op.make_inputs KeyError on quant dtypes (fp4/int8/mixed) → guarded (`_safe_torch_dtype`), so those become per-provider `unsupported` not a shared crash.
- moe_gemm stage2 f16/bf16 output requires the CShuffle epilogue → enabled, changing old fp8 failures into 3 correct rows + 6 correctness gaps.
- Added an op-aware correctness tolerance hook (`Op.tolerance`): GEMM (0.1,0.1) accumulation; RoPE atol-dominated (near-zero rotation outputs) so a correct f16 baseline isn't false-failed and doesn't inflate FlyDSL's vs-best.

## Per-kernel notes

### rmsnorm — tune_needed
- **Two live serving traces** (Qwen3-4B hidden=2560 + DeepSeek-R1-MXFP4 hidden=7168, both captured via SGLang `bench_serving` ISL/OSL/concurrency sweeps; DeepSeek-R1 ran fine on TP=8, no MXFP4 blocker). Production-weighted geomean vs best = **0.82×** (unweighted 0.88×). Serving-stage by model: DeepSeek-R1 0.83×, Qwen3-4B 0.89×.
- Near parity on synthetic/diagnostic rows (0.97× / 0.90×) and beats PyTorch overall (1.10×), but still trails tuned aiter/aiter_triton/triton on enough shapes to block promotion.
- `implementation_gap`: non-2048-aligned N (2560/5120/7168 — the real model hidden sizes) drop to the generic scalar path; profiler shows same block count as triton, slower per block.
- Eager launcher host-overhead ~46 µs (decode) — separate `launch_or_roofline_limited` (eager only; CUDA-graph decode hides it).
- **Hard crash** M>8192 & N≤2048 (block size > AMDGPU max_flat_workgroup_size). **Fixed**: ROCm/FlyDSL issue #614, PR #615 (`known_block_size` annotation). Reproduced on a real Qwen3-4B prefill q/k-norm shape.

### layernorm — tune_needed
- Same profile as rmsnorm: near parity on synthetic/diagnostic (1.00× / 1.02×), but model_config rows are 0.81× overall and worst M=32,N=16384 is 0.52× vs aiter_triton.
- **Candidate gap**: the layernorm fast-vectorized path triggers ONLY at N==8192 (16-bit) — far narrower than rmsnorm's N≥2048 & N%2048==0. Most production hidden sizes miss it → generic scalar path. Worth widening the fast-path condition. (Needs a profiler pass before filing.)

### softmax — promote (with headroom)
- FlyDSL softmax **wins overall** (geomean 1.08×; vs aiter_triton 1.53×, vs triton 1.09×, vs pytorch 1.27×), even on the generic scalar path.
- **Candidate FlyDSL issue/PR (high value)**: the vectorized path is **dead-coded off** — `softmax_kernel.py:104` guards it with `if const_expr(False and N >= tile_cols and N % tile_cols == 0):`. The leading `False and` makes the BufferCopy128b VEC_WIDTH=8 path unreachable for every shape. Re-enabling it should improve the bandwidth-bound large-N shapes further. Already wins despite this → clear headroom. (Matches the prior softmax dead-code finding.)
- Worst: M=16384,N=128 → 0.65× vs pytorch (generic path inefficiency at tiny N / large M).

## Candidate FlyDSL issues/PRs

| finding | kernel | status |
|---|---|---|
| large-M small-N block size > max_flat_workgroup_size (crash) | rmsnorm | **filed: #614 / PR #615** |
| softmax vectorized path dead-coded off + won't compile (fastmath bool + LLVM cast) | softmax | **filed: #627** |
| narrow fast-path (N==8192 only) + small-M one-block-per-row | layernorm, rmsnorm | candidate — needs profiler pass; parallelize across N for small M |

## Campaign expansion (2026-06-02) — all-kernel coverage + 3-model pass

**Goal**: cover every remaining FlyDSL kernel and weight by 3 models (Qwen, DeepSeek, Kimi).

**3-model pass (honest framing).** The two live SGLang traces stand (Qwen3-4B dense hidden=2560;
DeepSeek-R1 MLA hidden=7168 with q_lora=1536 / kv_lora=512 norms). **Kimi-K2.5 shares DeepSeek-R1's
exact MLA norm dims (7168 / 1536 / 512)**, so its norm profile is already represented by the DeepSeek
rows — fabricating config-derived Kimi norm rows would only double-count. Where Kimi genuinely differs
is the **MoE/gating** path: **384 routed experts, top-8** (vs DeepSeek 256/top-8), model_dim 7168,
inter_dim 2048. So Kimi enters the 3-model pass through the MoE-family shape ledgers
(moe_gemm / moe_blockscale / topk_gating_softmax / moe_reduce), seeded per-model.

**Kimi live-serving blocker (recorded).** `amd/Kimi-K2.6-MXFP4` fails to load under this sglang
(`AssertionError: param_data.shape == loaded_weight.shape` in deepseek_v2 weight_loader, even with
`--disable-shared-experts-fusion`). sglang's 384-expert path is validated only for `amd/Kimi-K2.5-MXFP4`
(not downloaded here; only an empty 0-safetensors shell). The complete `moonshotai/Kimi-K2.5` is the
same `kimi_k25` arch (compressed-tensors INT4). Rather than burn the 8-GPU pool on a checkpoint-compat
gamble whose live token distribution would ≈ DeepSeek-R1's anyway, Kimi is brought in via
config-derived MoE architecture shapes. (Env gotcha: must `source benchmarks/env.sh` so `import aiter`
resolves the FlyDSL build tree, else sglang dies on `SGLANG_USE_AITER=1`.)

## Campaign results — 11 new kernels (kernel-only, cold-cache, vs best correct baseline)

Adapters generated one-agent-per-kernel, merged into ops.py, run across the 8-GPU pool.
**17 kernels / 781 shapes total now benchmarked.** Headline geomean = best-correct-baseline / flydsl
(<1 = FlyDSL slower). Dashboard data: `tools/export_multishape_dashboard.py` → `docs/data/`.

| kernel | shapes | fly ok | geomean vs best | verdict | note |
|---|---:|---:|---:|---|---|
| topk_gating_softmax | 16 | 13 | **1.61** | promote | wins; **3 shapes unsupported = Kimi 384 experts** (FlyDSL finding) |
| moe_reduce | 16 | 16 | 1.03 | parity | bandwidth-bound sum-reduce, ≈ best |
| vec_add | 12 | 12 | 0.95 | tune | pure-BW sanity; ≈ triton/aiter_triton |
| mla_decode | 14 | 14 | 0.94 | tune | strong for a complex MLA fp8 decode kernel |
| blockscale_preshuffle_gemm | 14 | 14 | 0.94 | tune | fp8 block-scale GEMM, close to aiter |
| quant | 12 | 12 | 0.89 | tune | per-token int8; ≈ aiter_triton (after graph-stream fix) |
| flash_attn | 16 | 16 | 0.87 | tune | ≈ aiter_triton at large seq (after graph-stream fix) |
| preshuffle_gemm | 14 | 14 | 0.72 | tune | fp8 preshuffle GEMM, trails aiter |
| fp8_gemm_rowscale | 14 | 13 | 0.69 | tune | 1 unsupported (N%BLOCK_N≠0, M=512/N=2112) |
| pa_decode | 12 | 12 | **0.49** | rewrite | 2× slower than aiter gluon paged-attn decode |
| moe_blockscale | 12 | 0 | — | **blocked** | **FlyDSL kernel emits inf** (see findings) |

### Harness/methodology bugs found + fixed (not FlyDSL)
- **CUDA-graph stream omission** → bogus speedups. flash_attn (85×) and quant (5.4×) were physically
  impossible: the FlyDSL `@flyc.jit` launcher defaults to `stream=fx.Stream(None)` (stream 0), so
  graph capture on a side stream missed the kernel → empty-graph replay (~4µs). Fixed by passing
  `stream=torch.cuda.current_stream()`. Real numbers: flash_attn 0.87, quant 0.89. Caught by a
  physical-sanity gate (effective GB/s vs MI350X ~8 TB/s; effective TFLOP/s vs peak). All other
  kernels' cold-cache timings verified physical.
- pa_decode Op built the KV cache as `uint8` then `.uniform_()` (not implemented for Byte) → must be a
  bf16 cache quantized to fp8 (mirroring test_pa create_kv_cache). Fixed.
- moe_blockscale adapter flattened the 384-expert weight to >2³¹ elements with `.view(-1)` → fixed by a
  multi-dim reshape (bitwise-identical kernel output; see int32 finding).

### New FlyDSL findings from the expansion (candidate issues)
| finding | kernel | severity | repro |
|---|---|---|---|
| **block-scale stage1 emits `inf` (~47% of outputs) → all-NaN** | moe_blockscale | HIGH | `E=8,M=16,model_dim=7168,inter_dim=256,topk=2,scale_block=128`. Independent aiter CK stage1 reproduces bit-identical inf; fp32 ref finite. **Upstream test masks it** (inf-filtered `checkAllclose` + non-asserting body). Strict elementwise check flags it. |
| `topk_gating_softmax` has no valid multi-token layout for **384 experts** (Kimi) | topk_gating_softmax | MED | num_experts=384, multi-token; "needs num_experts//VPT…". Works ≤256 (DeepSeek). |
| int32 buffer-shape codec | (host-side) | LOW | `pack_layout_buffer` packs each dim as int32; a flat 1-D weight ≥2³¹ elems overflows `struct`. Workaround: multi-dim view. Not a kernel limit. |
| `fp8_gemm_rowscale` 8-wave path rejects N not divisible by BLOCK_N | fp8_gemm_rowscale | LOW | M=512,N=2112 → unsupported. |

### Baseline-blocked (vs-best not yet a fair FlyDSL verdict)
- **moe_gemm** (157×) and the high-multiplier reads come from the only *correct* baseline being slow
  PyTorch eager (optimized aiter baseline fails on fp8-out dtype). Flagged `baseline_blocked` in the
  dashboard. Fix the aiter moe baseline before drawing a moe_gemm perf conclusion.
