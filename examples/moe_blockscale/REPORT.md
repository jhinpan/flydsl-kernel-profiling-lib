# MoE Block-Scale GEMM Stage-1 (FP8->f16) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X (CDNA4) · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `mfma_moe1_bs_fp8_f16_cshuffle_t16x256x128_wpe2_abi8`
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_moe_blockscale/`

## Workload & headline

Shape captured: `tokens=256, model_dim=7168, inter_dim=256, E=8, topk=2`, tile `16x256x128`, 2 waves per expert block.
This is MoE **stage-1**: `X @ W1[expert].T -> SiLU(gate)*up -> f16` with per-token (X) and per-block (W) FP8 scales, fused MFMA-scale, cshuffle epilogue.

| Metric | Value |
|---|---|
| FlyDSL latency | **53.8 us** |
| Baseline (CK: `ck_moe_stage1_fwd`, blockscale) | **44.0 us** |
| Speedup vs baseline | **0.818x -> FlyDSL is ~22% slower** |
| TFLOPS / bandwidth | not recorded in sweep |

Verdict: **slower than CK, and memory-stall-bound, not compute-bound.** 82.8% of cycles on the sampled CU are stalls; ~78% of that is VMEM (load + wait). The matrix cores are idle most of the time waiting on FP8 operand and scale loads — the kernel is starved, not saturated. There is real headroom here: closing the 9.8 us gap to CK is a latency-hiding / occupancy problem, not an algebra problem.

> Note on arch label: the analyzer correctly detected **gfx950 (CDNA4)** for this kernel (its header prints `gfx950`). For most other kernels in this bundle it defaults to `gfx942` — that is an analyzer arch-detection default, not the real target. CDNA4 has a **combined 512-VGPR pool** (no split arch/accum file), which is what the occupancy math below uses.

## 1. Wave-state / stall breakdown

Sampled one CU (`att_target_cu`), iteration range `[3,[4-4]]`, 885/886 instructions mapped (99.9%). 552.2K total cycles, **457.4K stalled (82.8%)**.

| Stall type | Cycles | % | |
|---|---|---|---|
| **VMEM-wait** | 189.1K | **41.3%** | `s_waitcnt vmcnt(N)` — waiting on outstanding loads |
| **VMEM-load** | 166.0K | **36.3%** | issue/in-flight latency of `buffer_load` |
| barrier | 54.2K | 11.9% | `s_barrier` (LDS producer/consumer sync) |
| other | 21.2K | 4.6% | |
| LDS/SMEM-wait | 13.4K | 2.9% | `lgkmcnt` on `ds_read` |
| MFMA/FMA | 13.3K | 2.9% | matrix-core issue |
| VMEM-store / LDS | ~100 | 0.0% | negligible |

**Bound class: bandwidth/latency-bound (memory-bound).** VMEM-wait + VMEM-load = **77.6%** of stalls. MFMA stall is only 2.9% — the matrix cores are not the ceiling; they sit idle behind the load chain. The 11.9% barrier bucket is secondary but real: it's the cost of serializing the LDS staging of the A-tile against the MFMA consumers, amplified by low occupancy.

**Register pressure & occupancy:**
- arch_vgpr **~203 (alloc 208)** of 512 combined pool; accum_vgpr 0 (CDNA4 uses the unified file).
- occupancy **2 waves/SIMD**.
- Analyzer: **3 waves/SIMD requires total_vgpr <= 170** — i.e. shaving ~38 VGPR off the live set unlocks a 50% occupancy bump.

This is the crux. At 2 waves/SIMD there is almost no other wave to swap in while one wave waits on `vmcnt`, so the VMEM latency that dominates this section cannot be hidden by thread-level parallelism — it has to be hidden by ILP (deeper prefetch) instead, and right now it isn't. Instruction mix: 104 `buffer_load`, 32 MFMA, 20 `ds_write`, 12 `ds_read`, 4 `buffer_store` — a load-heavy GEMM tile with a small MFMA count per trip, exactly the profile that punishes you for not prefetching far enough ahead.

## 2. Top instruction-level hotspots

Source paths are truncated in the trace (`b/kernels/...`, `lyDSL-lab/...`); resolved against `/sgl-workspace/FlyDSL-lab/kernels/`.

**#1 — `moe_blockscale_2stage.py:740` · 140.6K (30.7%) · VMEM-wait.**
Line 740 is the second `rocdl.mfma_scale_f32_16x16x128_f8f6f4(...)` (the `up` projection). The per-instruction view shows the cost is not the MFMA itself but the `s_waitcnt vmcnt(4/6/12/0)` instructions attributed to this line (48.0K + 37.4K + 32.4K + 10.5K). These are the gates the MFMA waits on before it can consume `a128`/`bu128` operands — i.e. matrix issue is blocked on the FP8 operand loads landing. **This is the real bottleneck: the load->MFMA dependency chain, not enough loads in flight to cover the wait.**

**#2 — `mfma_preshuffle_pipeline.py:83` · 112.2K (24.5%) · VMEM-load.**
This is `buffer_ops.buffer_load(...)` inside `_buffer_load_vec` — the vectorized weight-tile load helper that issues `buffer_load_dwordx4` for the W1 (gate/up) FP8 operands (per-instr: four `buffer_load_dwordx4 v[...:...]` at 13–17K each). This is the W-operand HBM traffic feeding the MFMA. High in-flight latency cost; genuine bandwidth/latency, not an artifact.

**#3 — `moe_blockscale_2stage.py:196` · 91.1K (19.9%) · barrier.**
Line 196 is the `@flyc.kernel def moe_blockscale_gemm1(...)` decorator/signature — the trace attributes the kernel's `s_barrier` instructions (31.2K + 20.4K) here. This is the **source-loc-granularity collapse** (FlyDSL #587 / PR #593): the barriers belong to the LDS A-tile staging loop body, but the debug line-table folds them onto the kernel entry. Treat this as "barrier cost in the main loop," not as a problem at the function signature. Real signal: ~12% of stalls is LDS sync, consistent with the double-buffered A-tile staging contending with too few waves to hide it.

**#4 — `moe_blockscale_2stage.py:674` · 44.4K (9.7%) · VMEM-load.**
`s_a_val = buffer_ops.buffer_load(sx_rsrc, sa_idx, vec_width=1, dtype=f32)` — the **per-token activation scale** load (the X block-scale). Note `vec_width=1`: scalar f32 loads, one per (mi, ii). The per-instruction view confirms a hot `buffer_load_dword v42` here at 36.4K. Many tiny uncoalesced scalar loads on the scale path — a classic small-load inefficiency.

**#5 — `moe_blockscale_2stage.py:737` · 40.2K (8.8%) · VMEM-wait.**
The first `mfma_scale` (the `gate` projection); same story as #1 — `s_waitcnt vmcnt(10)`/`vmcnt(21) lgkmcnt(0)` gating the gate MFMA on its operand loads.

**#6 — `moe_blockscale_2stage.py:688` · 14.4K (3.2%) · VMEM-load.**
`s_w_up = buffer_ops.buffer_load(sw_rsrc, sw_up_idx, vec_width=1, dtype=f32)` — the **per-block weight scale** load (up). Again `vec_width=1` scalar f32. Lines 686/688 are the gate/up weight-scale loads; small, scattered, on the critical path into the scale-FMA.

The rest (306/341 `ds_read` LDS-waits, 273, 450) are <1% each — noise.

**Summary:** every hot line is on the **operand-supply path** into the MFMA-scale calls. The MFMA-scale lines (737/740) top the list only because they are where the wave *stalls waiting* for those operands. The fix is upstream: get the FP8 weight/activation tiles and the f32 scales into registers/LDS earlier and in bigger chunks.

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Deepen the global->LDS software pipeline (prefetch the A/W tiles 1–2 stages ahead)
**Root cause:** the dominant bucket is VMEM-wait (41.3%) + VMEM-load (36.3%); section 2 shows the MFMA-scale issue (lines 737/740) blocked on `s_waitcnt vmcnt(N)` for operands issued too close to their use. With only 2 waves/SIMD there is no other wave to cover the latency, so it must be covered by **more outstanding loads per wave**.
**Change:** widen the prefetch distance in the K-loop so that iteration *k+1*'s (and ideally *k+2*'s) `buffer_load_dwordx4` are in flight before the iteration-*k* MFMA-scale consumes its operands — classic multi-stage `vmcnt`-gated direct-to-LDS double/triple buffering, draining `vmcnt(N)` only at the point of use. The kernel already prefetches `a0_prefetch` for the first MFMA (line 725); extend that pattern to all stages and to the W tiles.
**Grounding:** ROCmKernelWiki `technique-lds-double-buffering` (direct-to-LDS + `s_waitcnt vmcnt(N)` counter gating) and `technique-mfma-pipelining` (interleave loads and matrix issue so `vmcnt`/`lgkmcnt` rarely block the MFMA). Both list gfx950 and prior FlyDSL PRs (pr-FlyDSL-346, -579, -278).
**Expected gain:** large — directly attacks ~78% of stalls. Realistically closes most of the 9.8 us gap to CK and could overtake it, since CK's stage-1 uses exactly this pipelined structure.
**Effort:** medium-high (touches the K-loop staging in `moe_blockscale_2stage.py` around lines 668–760).

### #2 — Cut VGPR pressure from ~203 -> <=170 to reach 3 waves/SIMD
**Root cause:** occupancy is 2 waves/SIMD; the analyzer states 3 waves needs total_vgpr <= 170. A third wave gives the scheduler another wave to hide the very VMEM latency that #1 attacks — the two fixes compound.
**Change:** reduce the live VGPR set in the inner loop: the `pending_gate_up` one-deep software-pipelined FMA (lines 744–756) and the `_pack128` accumulator vectors hold a lot of f32x4 state. Consider narrowing the `num_acc_n` register-blocking factor, reusing accumulator temporaries, or recomputing some scale broadcasts rather than holding them live. Confirm spills with `--save-temps`/ISA dump before/after.
**Grounding:** ROCmKernelWiki `technique-vgpr-budgeting` and `technique-occupancy-tuning` (on CDNA the occupancy limiter is `floor(ArchVGPR_file / vgprs_per_wave)`; combined pool on gfx950).
**Expected gain:** moderate-high; +50% occupancy headroom that amplifies #1. Watch the ILP trade-off — too-aggressive blocking reduction can hurt the very pipelining #1 wants; tune together.
**Effort:** medium (register-blocking knobs `tile_n`/`num_acc_n`/`m_repeat` already exist).

### #3 — Vectorize / batch the scale loads (lines 674, 686, 688)
**Root cause:** the per-token X-scale (674) and per-block W-scale (686/688) loads are `vec_width=1` scalar f32 `buffer_load_dword` — 4 separate loads per (mi) for X, plus per-`ni` W scales. ~13% of stalls combined; each consumes a full VMCNT queue slot for 4 bytes.
**Change:** the four `s_a_val` loads per `mi` (`for ii in range_constexpr(4)` at 670–675) can be a single `vec_width=4` f32x4 load when the four scale indices are contiguous (they feed `vector.from_elements(f32x4, ...)` at 694 anyway). Likewise coalesce the gate/up weight-scale loads where the indices permit.
**Grounding:** ROCmKernelWiki `technique-vectorized-loads` (one wide `buffer_load` = one issue slot + one VMCNT entry instead of four).
**Expected gain:** small-moderate (~13% of stalls, but partly latency that #1 already hides). Cheap and orthogonal.
**Effort:** low-medium.

### #4 — Reduce LDS-barrier serialization in the A-tile stage (the line-196 barrier bucket)
**Root cause:** 11.9% barrier stalls; with double-buffered LDS staging, a single `s_barrier` between producer (ds_write) and consumer (MFMA reading the prior buffer) serializes when there aren't enough waves to overlap.
**Change:** ensure the A-tile uses ping-pong LDS buffers so the barrier guards only the buffer being filled, letting MFMA proceed on the other; this is the LDS half of the #1 pipeline.
**Grounding:** same `technique-lds-double-buffering`.
**Expected gain:** small-moderate; folds into #1.
**Effort:** medium.

**Do #1 first** (biggest bucket), co-tune with #2; #3 and #4 are cheap follow-ups.

## 4. Re-run

Capture command (cwd `/sgl-workspace/FlyDSL-lab`, target GPU 3, single-CU ATT):

```bash
/opt/venv/bin/python /sgl-workspace/flydsl-prof/drivers/att_capture.py \
  --test test_moe_blockscale --gpu 3 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_moe_blockscale \
  --tag big --iter-range "[3,[4-4]]" --buffer 0x6000000 --target-cu 1 \
  --cmd-override "python tests/kernels/test_moe_blockscale.py -m 256 -dim 7168 -idim 256 -e 8 -k 2 --tile_m 16 --tile_n 256 --tile_k 128 --wpe 2 --num_iters 5 --num_warmup 2"
```

Underlying workload invocation:
```bash
python tests/kernels/test_moe_blockscale.py -m 256 -dim 7168 -idim 256 -e 8 -k 2 \
  --tile_m 16 --tile_n 256 --tile_k 128 --wpe 2 --num_iters 5 --num_warmup 2
```

Hotspot re-analysis: `hotspot_analyzer.py` over `att/ui_output_agent_39345_dispatch_668` -> `hotspot_big.txt`.
