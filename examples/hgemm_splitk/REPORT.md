# HGEMM Split-K (bf16) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X / CDNA4 · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `hgemm_bf16_32x64x256_W1x2x2_S2_BT_BLDS1_AS1_SPK14_0`
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_hgemm_splitk/`

> Arch caveat: the analyzer prints `gfx942 (CDNA3)`. That is the known arch-detection default — this capture is gfx950 / MI350X / CDNA4. CDNA4 has a **combined** VGPR pool (no split arch/accum file), so `accum_vgpr=0` here is expected and occupancy is governed by the single ~117-VGPR allocation. All occupancy reasoning below uses the combined-pool model.

---

## Workload & headline

- **Shape:** M=32, N=384, K=7168, bf16, **SPLIT_K=14**. Block tile 32×64×256, warps W1×2×2, STAGES=2, B staged through LDS, direct-to-LDS async copy (`AS1`).
- **Latency:** FlyDSL **7.0 µs** → **25.1 TFLOPS**, **0.85 TB/s** effective (HIP-event sweep timing; not ATT).
- **Head-to-head:** sweep baseline is **PyTorch `torch.mm` 11.6 µs** → FlyDSL is **1.66× faster**. There is **no matched-shape entry for `test_hgemm_splitk` in `baselines.json`** (the authoritative file holds only 8 kernels, none splitk), so the PyTorch sweep number is the only baseline available — treat it as indicative, not a tuned-library head-to-head (rocBLAS/CK bf16 GEMM would be the right rival and is unmeasured here).

**Verdict: latency-bound, not throughput-bound.** AI is ~29 FLOP/byte (compute-leaning roofline), yet the kernel only achieves ~1% of MI350X bf16 peak and ~11% of HBM3e BW. The cause is structural: M=32 × N=384 produces just **6 MN-tiles**; SPLIT_K=14 fans that to **84 workgroups** on a 256-CU part. The device is nearly empty, every block runs a short K-slice, and the runtime is dominated by the serial latency of a few global-load → MFMA → split-K reduction chains. The kernel beats PyTorch because PyTorch has no split-K for this skinny shape — but the absolute number is set by launch latency, not by the MFMA or HBM ceiling.

---

## 1. Wave-state / stall breakdown

ATT sampled one CU (`att_target_cu`); 521/522 instructions mapped (99.8%). 71.3% of all cycles are stalls (58.2K / 81.7K).

| Stall type      | Cycles | %    |
|-----------------|-------:|-----:|
| **VMEM-wait**   | 32.5K  | **55.8** |
| barrier         |  8.3K  | 14.2 |
| LDS/SMEM-wait   |  7.7K  | 13.3 |
| VMEM-load       |  4.3K  |  7.4 |
| other           |  3.3K  |  5.7 |
| MFMA/FMA        |  1.2K  |  2.0 |
| LDS             |   868  |  1.5 |

**#1 is VMEM-wait at 55.8%** — `s_waitcnt vmcnt(0)`, the wave sitting on global-load completion. Add the 7.4% VMEM-load and **~63% of all stall cycles are global-memory latency**. The next-largest bucket (14.2% barrier) is the split-K / staging `s_barrier` traffic, and 13.3% LDS/SMEM-wait is `lgkmcnt(0)` on the LDS read of staged tiles. **MFMA is 2.0%** — the matrix core is essentially idle, waiting. This is a **wait-bound / latency-bound kernel** dominated by global-load latency that the pipeline fails to hide.

**Register pressure & occupancy:**
- arch_vgpr ≈ 117 (alloc 120), accum_vgpr 0, limiting pool 120/256.
- **occupancy = 2 waves/SIMD.** Analyzer: *3 waves/SIMD requires max(arch,accum) ≤ 85* (manifest `next_occ_step`: 3 waves @ VGPR budget 85).
- With only 84 workgroups across 256 CUs, occupancy-per-CU is almost irrelevant for *throughput* — there aren't enough waves resident anywhere to hide latency regardless of the 2-vs-3 SIMD budget. Raising occupancy helps only if there were more blocks to fill the holes; here the latency comes from the *critical path of one block*, not from too few co-resident waves on a busy CU.

---

## 2. Top instruction-level hotspots

| # | %tot | DomType | Source | What runs there |
|---|-----:|---------|--------|-----------------|
| 1 | **83.5%** | VMEM-wait | `hgemm_splitk.py:217` | **Source-loc collapse** — line 217 is the `@flyc.kernel def hgemm_kernel(...)` *signature*. Nearly the entire VMEM-wait + barrier + VMEM-load mass folds onto the kernel def line, so this is not one statement. See note below. |
| 2 | 11.2% | LDS/SMEM-wait | `flydsl/expr/numeric.py:872` | FlyDSL intrinsic helper, **not** the real bottleneck — `Index.__init__` → `index_cast`. The `s_waitcnt lgkmcnt(0)` that gates the LDS-tile read gets debug-attributed to the index-cast helper instead of the `lds_matrix_a/b` consumer (issue #587 / PR #593 source-loc granularity collapse). Read it as "the LDS-read wait", located in the main loop. |
| 3 | 2.5% | MFMA/FMA | `hgemm_splitk.py:80` | `rocdl.mfma_f32_16x16x32_bf16` — the actual matrix-core op (`WmmaHalf_m16n16k32.__call__`). Only 2.5% of stall: the MFMA pipe is not the limiter. |
| 4 | 2.2% | LDS | `tensor_shim.py:242` | `ds_write`/`ds_read` shim for the staged A/B tiles. |
| 5+ | <0.4% each | — | `hgemm_splitk.py:448/449/456/487/495`, `tensor_shim.py:225/255` | `arith.select` boundary guards, `linear_offset` index math, and async-copy address setup in `ldg_sts_a/b_async`. Negligible. |

**Reading the real bottleneck.** Because lines 217 and numeric.py:872 are artifacts, go to the instruction-level table in the hotspot text, which keeps the asm:

- `s_waitcnt vmcnt(0)` appears **three times in the top-6 instructions** (28.4% + 24.5% + 2.3% = **~55% of all stalls on a single waitcnt pattern**), each tagged `hgemm_splitk.py:217`.
- The loads it waits on are `buffer_load_dwordx4 ... sc0` (128-bit, system-coherent) — the direct-to-LDS async loads issued by `ldg_sts_a_async` / `ldg_sts_b_async` (lines 440–514, `rocdl.raw_ptr_buffer_load_lds`).
- `s_barrier` is 8.8% + smaller (the staging/split-K barriers at `gpu.barrier()`, lines 600/635 and the split-K reduction barrier line 731).

So the true critical path is: **issue 128-bit direct-to-LDS loads → `s_waitcnt vmcnt(0)` → `s_barrier` → `lds_matrix` read (`lgkmcnt(0)`) → MFMA.** The kernel *already* implements the right structure — STAGES=2 double-buffering, `raw_ptr_buffer_load_lds`, and an explicit `hot_loop_scheduler()` with `sched_vmem`/`sched_mfma`/`sched_dsrd` interleave hints (lines 604–639). The problem is that with K=7168 / BLOCK_K=256 / SPLIT_K=14, each block only runs **~2 K-loop iterations** — far too few to amortize the prologue load latency or fill the 2-stage pipeline. The wave spends most of its life in the *prologue* `vmcnt(0)` waiting for the first tiles, with almost no steady-state to hide behind.

---

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Reduce SPLIT_K / lengthen the per-block K-loop so each block has enough iterations to hide load latency
**Root cause:** the 55.8% VMEM-wait is prologue-dominated global-load latency that the 2-stage pipeline can't hide because each block runs only ~2 K-loop iterations (K=7168, BLOCK_K=256, SPLIT_K=14 → ~511 K per block ÷ 256 ≈ 2 iters). A double-buffer needs ≥ (load-latency / MFMA-time-per-iter) iterations of steady state to pay back; here there is essentially none.
**Change:** sweep SPLIT_K **down** (the autotune space is `range(1,17)`, line 870). Fewer splits → more K per block → more pipeline iterations to overlap loads with MFMA. The autotuner picked SPLIT_K=14 to maximize the 84-block grid, but that trades latency-hiding for occupancy that 84 blocks can't even use on 256 CUs. Try SPLIT_K ∈ {4, 7} and re-measure; the optimum balances "enough blocks to populate CUs" against "enough K-iters per block to hide latency."
**Expected gain:** this directly attacks the 63% memory-latency stall; plausibly the largest single lever for this skinny shape. Effort: low (autotune sweep, no code change).
**Grounding:** ROCmKernelWiki `technique-mfma-pipelining` — software pipelining hides global-load latency behind MFMA via `vmcnt` gating, but only works when the steady-state loop is long enough; and `technique-lds-double-buffering` — direct-to-LDS + `s_waitcnt vmcnt(N)` overlap requires sufficient in-flight iterations.

### #2 — Deepen the pipeline to STAGES=3 (only if VGPR budget allows)
**Root cause:** same VMEM-wait. The kernel is `S2` (2-stage). A 3rd stage keeps more loads in flight so a `vmcnt` wait resolves against an already-landed tile rather than a fresh HBM round-trip.
**Change:** STAGES=3 in the config. **Caveat:** arch_vgpr is already ~117 and occupancy is 2 waves/SIMD; a 3rd LDS stage costs more VGPRs for the extra fragment buffers and LDS for the extra tile. On CDNA4's combined pool, pushing arch_vgpr above the 2-wave ceiling won't drop occupancy further (already 2), but spilling would be catastrophic — verify no spill in the post-compile VGPR count. Pairs best with #1.
**Expected gain:** moderate; secondary to #1 because the iteration count is the binding constraint. Effort: medium (config + spill check).
**Grounding:** `technique-lds-double-buffering`, `technique-vgpr-budgeting` (ArchVGPR pressure vs occupancy on the combined CDNA4 pool).

### #3 — Drop `sc0` system-coherence on the input loads if semantics permit
**Root cause:** the hot loads are `buffer_load_dwordx4 ... sc0` (lines 469/508, the `sc0/sc1` flags into `raw_ptr_buffer_load_lds`). `sc0` forces system-scope coherence, lengthening each load's latency and feeding directly into the dominant `vmcnt(0)` wait.
**Change:** A and B are read-only inputs within a split-K group; the coherence traffic is needed for the *signal/semaphore* path (the `global_store ... sc0 sc1` + atomic in `zero_c`/`split_k_barrier`), not necessarily for the A/B tile loads. If the A/B `raw_ptr_buffer_load_lds` can use agent/device scope instead of system scope, each load's latency shrinks. **Verify correctness carefully** — split-K reduction relies on signal ordering; only relax scope on the *data* loads, never on the semaphore.
**Expected gain:** small-to-moderate per-load latency cut, multiplied across the prologue. Effort: medium (needs a correctness pass on the coherence model).
**Grounding:** `technique-vectorized-loads` (128-bit loads + coherence/`vmcnt` queue behavior).

### #4 — Fix source-loc granularity so future captures are actionable (tooling, not perf)
83.5% of stall folds onto the kernel-def line and 11.2% onto `numeric.py:872` (`index_cast`). This is issue #587 / PR #593. Landing finer debug-loc propagation would let the analyzer attribute the `vmcnt`/`lgkmcnt` waits to the actual `ldg_sts_*`/`lds_matrix_*` statements, removing the manual asm-cross-reference step done in §2. No runtime gain; pure observability.

---

## 4. Re-run

```bash
# Capture (gfx950 / MI350X), shape M=32 N=384 K=7168, bf16, SPLIT_K=14, tile 32x64x256, W1x2x2, S2
/opt/venv/bin/python /sgl-workspace/flydsl-prof/att_capture.py \
  --test /sgl-workspace/FlyDSL-lab/tests/kernels/test_hgemm_splitk.py \
  --kernel hgemm_bf16_32x64x256_W1x2x2_S2_BT_BLDS1_AS1_SPK14_0 \
  --stem test_hgemm_splitk \
  --out /sgl-workspace/flydsl-prof/results/att/test_hgemm_splitk

# Underlying test param (first gfx950 row):
#   m=32 n=384 k=7168 TILE_M=32 TILE_N=64 TILE_K=256 SPLIT_K=14 BLOCK_M_WARPS=1 BLOCK_N_WARPS=2 BLOCK_K_WARPS=2  dtype=bf16
ARCH=gfx950 /opt/venv/bin/python -m pytest \
  /sgl-workspace/FlyDSL-lab/tests/kernels/test_hgemm_splitk.py \
  -k "bf16 and splitk" -v
```
