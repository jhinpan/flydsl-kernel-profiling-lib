# Per-Token Quantization (fp16 → int8) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X (CDNA4) · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `quant_kernel_0` (ATT dispatch `ui_output_agent_34678_dispatch_8`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_quant/`

> Arch caveat: the analyzer prints `gfx942 (CDNA3)` — that is its arch-detection default, not the real target. This capture ran on **gfx950 / MI350X / CDNA4**, which has a **combined VGPR pool** (no separate accum bank). All occupancy reasoning below uses the combined pool.

## Workload & headline

- **Shape:** M=4096 tokens × N=8192 hidden. fp16 in → int8 out, one f32 scale per row. Traffic model = `M*N*2 (fp16 read) + M*N*1 (int8 write) + M*4 (scales)` = **100,679,680 B/call**.
- **Latency / bandwidth (HIP-event sweep, GPU 5):** FlyDSL **16.74 µs** → **6013 GB/s**; the manifest sweep records 5910 GB/s on a separate run. This op is pure-streaming, so **GB/s is the metric — TFLOPS is N/A** (zero MFMA, `inst_mix.mfma == 0`).
- **Baseline head-to-head (authoritative, `baselines.json`):** aiter `per_token_quant_hip` = **16.045 µs / 6275 GB/s**. **FlyDSL is 0.96× — i.e. ~4% slower.** Correctness matches (max output diff 1.0 = int8 rounding boundary; scale diff ~0).
- **Verdict — bandwidth-bound, FlyDSL does NOT win, but the gap is structural, not algorithmic.** Both kernels sit at ~6.0–6.3 TB/s, near the MI350X HBM3E roofline. There is no CK kernel for this op; aiter is the strongest comparable. The 4% is the cost of a leftover LDS/barrier reduction path that aiter avoids — see §3.

## 1. Wave-state / stall breakdown

Sampled on one CU: 301 instructions, **2.62M total cycles, 2.03M stalled (77.5%)**. A 77.5%-stall, zero-MFMA, all-VMEM kernel is the textbook signature of **bandwidth/latency-bound streaming** — the SIMDs are parked waiting on HBM, exactly as a roofline-saturated copy should look.

| Stall type | Cycles | % | Reading |
|---|---|---|---|
| **VMEM-wait** | 730.0K | **35.9%** | `s_waitcnt vmcnt(*)` — waiting for in-flight loads to land. **#1.** |
| VMEM-load | 425.0K | 20.9% | the `buffer_load_dwordx4` issue/queue pressure |
| other | 352.1K | 17.3% | ALU: f16→f32 cvt, abs-mask, scale mul, fptosi |
| barrier | 265.2K | 13.1% | `s_barrier` around the block reduction |
| LDS/SMEM-wait | 208.1K | 10.2% | `lgkmcnt(0)` on the LDS reduction scratch |
| VMEM-store | 41.3K | 2.0% | int8 + scale stores |
| MFMA/FMA | 8.5K | 0.4% | — |
| LDS | 644 | 0.0% | — |

VMEM-wait + VMEM-load together = **56.8%** of all stalls. Add the store path and it's ~59% pure memory traffic; that is the dominant axis. The remaining ~23% (barrier + LDS-wait) is overhead from the in-kernel block-max reduction, and **that is the only part worth attacking** — the memory traffic itself is irreducible (you must read every fp16 and write every int8 once).

**Register pressure & occupancy:** arch_vgpr ~44 (alloc 48), accum_vgpr 0, limiting pool 48/256 → **5 waves/SIMD**. The analyzer notes the next step (**6 waves/SIMD requires VGPR ≤ 42**), i.e. shaving ~6 VGPRs. With a 256-VGPR combined pool on gfx950, 48 VGPRs is *not* the occupancy ceiling at the per-SIMD level the way it would be under a split gfx942 budget — but for a latency-hiding streaming kernel more waves = more outstanding loads, so the 5→6 step still matters (see §3 #2). Note this kernel is *also* limited by `BLOCK_THREADS=256` → 4 waves/WG, so per-CU occupancy is governed by how many 256-thread WGs co-reside, not purely by VGPR.

## 2. Top instruction-level hotspots

| # | %total | DomType | Source | What runs there |
|---|---|---|---|---|
| 1 | **66.5%** | VMEM-wait | `test_quant.py:99` | `@flyc.kernel def quant_kernel` — the kernel-body anchor line |
| 2 | 23.8% | VMEM-load | `test_quant.py:178` | `fx.copy_atom_call(copy_atom_in, ...)` — the `buffer_load_dwordx4` vec-f16 load |
| 3 | 6.4% | LDS/SMEM-wait | `flydsl/expr/rocdl/universal.py:144` | buffer-descriptor / `make_ptr` intrinsic helper |
| 4 | 1.8% | VMEM-store | `test_quant.py:242` | `fx.copy_atom_call(copy_atom_out, ...)` — int8 `buffer_store_dwordx2` |
| 5 | 0.6% | other | `test_quant.py:128` | `lane == 0` guard in `block_reduce_max` |

**Line 99 (66.5%) is the real bottleneck, correctly attributed.** The per-instruction view confirms it: the top three stalls there are `s_waitcnt vmcnt(2)` (18.2%), `s_waitcnt vmcnt(0)` (13.3%), and two `s_barrier`s (7.0% + 6.1%) — plus the `v_cvt_f32_f16` conversions. Line 99 is the `@flyc.kernel` decorator line, so the compiler attributes the kernel's prologue/reduction `s_waitcnt`/`s_barrier` to it. The signal is unambiguous: **the kernel spends two-thirds of its stall budget waiting on loads to drain (`vmcnt`) and on the reduction barrier.** This is Pass-1: load fp16, cvt to f32, abs-max-reduce. The `vmcnt(2)/(1)/(0)` ladder shows the compiler *is* keeping ~3 loads in flight before draining — decent latency hiding, but it then hits the barrier.

**Line 178 (23.8%, VMEM-load)** is the actual `buffer_load_dwordx4 v[2:5], ...` — the 128-bit (8×f16) vectorized load in `_load_vec_f16`. This is already optimal width per the wiki (`technique-vectorized-loads`): 16 B/lane, one VMCNT slot. The stall here is HBM latency/queue depth, not a width problem.

**Line 144 in `universal.py` (6.4%) is a source-loc-collapse artifact** (FlyDSL issue #587 / PR #593): the hot line maps into the `make_buffer_tensor` / buffer-descriptor intrinsic helper (`make_ptr`, buffer flags), not a distinct bottleneck. Its DomType is `LDS/SMEM-wait` (`lgkmcnt(0)`), so it's really the reduction-scratch LDS traffic being attributed to the helper. **Do not treat universal.py:144 as the bottleneck** — fold its cost into the reduction-overhead bucket (barrier + LDS-wait) discussed below.

**Line 242 (1.8%)** is the int8 store — `buffer_store_dwordx2` (8×i8 = 64 b). Cheap, as expected for a 1-byte-output streaming write.

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Cut the in-kernel block reduction; trim the barrier + LDS-wait overhead (the only non-traffic stalls)
**Root cause (ties to §1):** the memory stalls (VMEM-wait + VMEM-load = 56.8%) are *irreducible* at the roofline — you must touch every byte once. The only recoverable stall is the **barrier (13.1%) + LDS/SMEM-wait (10.2%) = ~23%** spent in `block_reduce_max` (`test_quant.py:118–147`): a wave shuffle-xor reduce, an LDS scratch store/load across waves, and **two `gpu.barrier()` calls** (lines 131, 144). The per-instruction view shows those two `s_barrier`s alone are 13.1% of all stalls. aiter's HIP kernel almost certainly fuses the cross-wave max without two full block barriers — that is plausibly the entire 4% gap.

**Concrete change:** the block has 256 threads = **4 waves** (`RED_SLOTS = 256//64 = 4`). After `wave_reduce_max`, you only need to combine 4 partial maxes. Options, cheapest first:
- Replace the LDS round-trip + double barrier with a **single barrier + one LDS slot** broadcast, or
- Better: drop `BLOCK_THREADS` to **64 (one wave per row)** so `RED_SLOTS == 1` and `block_reduce_max` returns `wave_reduce_max(val)` directly — **zero barriers, zero LDS scratch**. With N=8192 and VEC_WIDTH=8, one 64-thread wave covers 512 cols/iter → 16 iters/row. That trades a slightly longer per-wave loop for the elimination of the entire barrier+LDS-wait bucket, and lets far more WGs co-reside per CU (more loads in flight → better latency hiding on the dominant VMEM-wait).

**Grounding:** `technique-occupancy-tuning` (ROCmKernelWiki) — on CDNA, for a memory-bound kernel the lever is "enough waves to hide VMEM latency" rather than ILP within one fat WG; a 64-thread WG maximizes the number of independent reductions in flight. Also `technique-vectorized-loads` confirms the load width is already correct, so the win is in removing synchronization, not in the loads.
**Expected gain:** recover a meaningful fraction of the ~23% sync stall → close most or all of the 4% gap to aiter; could nudge FlyDSL to ≈ parity (≥1.0×).
**Effort:** Medium — change `BLOCK_THREADS`/`WARP_SIZE` tiling and re-verify correctness across N that isn't a multiple of `64*8`.

### #2 — Shave ~6 VGPRs to reach 6 waves/SIMD, deepening the load pipeline
**Root cause:** VMEM-wait is #1 (35.9%); the cure for load-latency stalls is more outstanding loads, which more resident waves provide. Analyzer: **6 waves/SIMD needs VGPR ≤ 42**, currently ~44 (alloc 48).
**Concrete change:** the kernel caches *all* `vec_f32` tiles in registers across Pass 1 → Pass 2 (`cached_vecs`), which inflates live VGPRs (8 f32 × num_tiles held live). For N=8192, num_tiles is small, but the cache is the obvious VGPR sink. Consider recomputing/re-loading in Pass 2 instead of caching (this *adds* read traffic, so only worth it if it frees enough VGPRs to gain a wave **and** the kernel is latency- not bandwidth-limited), or narrowing intermediates. Even a 6-VGPR trim flips 5→6 waves on gfx950's combined 256-VGPR pool.
**Grounding:** `technique-vgpr-budgeting` + `technique-occupancy-tuning`.
**Expected gain:** Small–Medium. Since the kernel is already near roofline, extra waves mostly help if it's latency- rather than pure-bandwidth-bound; treat as a measure-then-keep experiment. Likely <=2%.
**Effort:** Low-Medium — one tiling/caching change, re-profile occupancy.

### #3 — Non-temporal / streaming hint on loads and stores
**Root cause:** every byte is read once and written once — zero reuse. Default loads still consult L2 residency policy.
**Concrete change:** if FlyDSL's copy-atom / `BufferCopy128b` path exposes a non-temporal (glc/slc/nt) flag, set it on both the fp16 load (line 178) and int8 store (line 242) so streaming traffic bypasses L2 retention and doesn't evict useful lines.
**Grounding:** `technique-vectorized-loads` (non-temporal section — "read once, never reused" is exactly this kernel).
**Expected gain:** Small (1–2%) at this scale; both kernels are already near the HBM ceiling so L2 policy has limited headroom. Worth it only if the flag is a one-line change.
**Effort:** Low (if exposed) / N/A (if not exposed by the copy-atom API).

**Bottom line:** the load width and traffic are already optimal — this is a near-roofline streaming kernel. The recoverable budget is the ~23% barrier + LDS-wait from the two-barrier block reduction (#1). Everything else is roofline.

## 4. Re-run

ATT capture (one CU, representative wave-state sample — latency/GB/s come from the HIP-event sweep below, **not** from ATT):

```bash
/opt/venv/bin/python /sgl-workspace/flydsl-prof/att_capture.py \
  --stem test_quant \
  --test /sgl-workspace/FlyDSL-lab/tests/kernels/test_quant.py \
  --kernel quant_kernel_0 \
  -m 4096 -n 8192
```

HIP-event timing / baseline head-to-head (from `/sgl-workspace/FlyDSL-lab`):

```bash
HIP_VISIBLE_DEVICES=5 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
FLYDSL_RUN_QUANT=1 \
/opt/venv/bin/python tests/kernels/test_quant.py -m 4096 -n 8192
```
