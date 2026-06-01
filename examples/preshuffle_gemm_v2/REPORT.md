# Preshuffle GEMM v2 (fp8 a8w8 B-preshuffle) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X (CDNA4) · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `kernel_gemm_0` (ATT dispatch `ui_output_agent_63371_dispatch_15`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/bench_preshuffle_gemm_v2/`

> Arch caveat: the hotspot analyzer printed `gfx942 (CDNA3)`. That is the analyzer's arch-detection default, not the real target. This capture ran on **gfx950 / MI350X / CDNA4**, which has a **single combined VGPR pool** (no split arch/accum file like gfx942) — occupancy reasoning below uses the combined-pool model, and `accum_vgpr=0` is expected, not a missing-AGPR signal.

## Workload & headline

- **Shape:** M=16, N=1024, K=2048; in=fp8 e4m3, out=bf16; tile 16x64x512. Total work ~0.069 GFLOP — a **tiny, K-skinny, memory-bound** GEMM.
- **Latency (HIP-event, sweep/baseline):** FlyDSL **5.0 us** ~ **13.3 TFLOPS**.
- **Head-to-head (authoritative, baselines.json):** AIter `gemm_a8w8_bpreshuffle` (CK fp8 a8w8 B-preshuffle, untuned default config — AIter found **no tuned entry** for M16/N1024/K2048) = **4.4 us / 15.23 TFLOPS / 491.9 GB/s**. **speedup = 0.88x** → FlyDSL ~12% slower on median.
- **Verdict:** FlyDSL does **not** beat the strongest comparable at this shape — it trades **within noise** and lands slightly behind. Across 4 AIter runs the baseline ranged 4.2–4.8 us and FlyDSL 4.5–5.1 us; in one run FlyDSL was faster. At 0.069 GFLOP both kernels are launch+memory bound, so the 0.6 us gap is ~jitter-sized. The 13.3 TFLOPS figure is far below MI350X peak purely because the shape can't fill the machine — a fair "does FlyDSL win" test needs a large compute-bound shape (e.g. M=5120 N=2048 K=8320), not this one.

Latency/TFLOPS are HIP-event timings. The ATT numbers below sample **one CU** — a representative wave-state snapshot, not device-wide timing.

## 1. Wave-state / stall breakdown

428 instructions, 1.72M total cycles, **1.40M stalled = 81.1%**. This kernel spends 4 of every 5 cycles waiting, not issuing.

| Type | Stall | % |
|---|---:|---:|
| **LDS/SMEM-wait** | 567.6K | **40.6%** |
| VMEM-wait | 461.5K | 33.0% |
| barrier | 211.5K | 15.1% |
| MFMA/FMA | 87.1K | 6.2% |
| LDS | 35.0K | 2.5% |
| VMEM-load | 22.4K | 1.6% |
| other | 12.2K | 0.9% |
| VMEM-store | 220 | 0.0% |

**Bound class: stall-bound, memory/sync-latency dominated.** The top three buckets (LDS-wait 40.6% + VMEM-wait 33.0% + barrier 15.1% = **88.7%**) are all "wait for data or peers," and MFMA-issue stall is only 6.2%. The matrix core is essentially idle waiting on the operand-staging pipeline — A goes HBM→LDS→register, B goes HBM→register, and the MFMA front-end blocks on the `s_waitcnt` that gates them.

**Register pressure & occupancy:**
- arch_vgpr ~97 (alloc 104), accum_vgpr 0, limiting pool **104 / 256** (combined pool on gfx950).
- **occupancy = 2 waves/SIMD.**
- Next step: **3 waves/SIMD requires VGPR <= 85** (analyzer `next_occ_step`).

At only 2 waves/SIMD the scheduler has almost no other wave to hide the long LDS/VMEM round-trips behind — which is exactly why ~89% of cycles are wait. The fp8 16x64x512 tile is VGPR-heavy (acc + A retile-frag + double-buffered B frags), and the inst-mix confirms a tight inner loop: **MFMA 64, buffer_load 24, buffer_store 32, ds_read 32, ds_write 10**.

## 2. Top instruction-level hotspots

| # | %Total | DomType | Source | What runs there |
|---|---:|---|---|---|
| 1 | **64.5%** | LDS/SMEM-wait | `preshuffle_gemm_v2.py:309` | `fx.gemm(tiled_mma, frag_C, frag_A, cur_frag_B, frag_C)` — the MFMA |
| 2 | 20.4% | barrier | `preshuffle_gemm_v2.py:100` | `@flyc.kernel def kernel_gemm(...)` — collapsed `s_barrier` |
| 3 | 9.4% | VMEM-wait | `preshuffle_gemm_v2.py:311` | `fx.copy(uni_copy_g2s, frag_copy_A, pA_s[...,write_stage])` — write A→LDS |
| 4 | 2.2% | LDS | `preshuffle_gemm_v2.py:305` | `fx.copy(mma_uni, pA_s2r[...], frag_A_retile[...])` — A LDS→reg |
| 5 | 1.4% | VMEM-load | `preshuffle_gemm_v2.py:302` | `fx.copy(mma_copy, pB_g[...], frag_B_retile_stages[...])` — load next B |
| 6 | 0.8% | VMEM-load | `preshuffle_gemm_v2.py:298` | `fx.copy(buf_copy_g2s, pA_g[...], frag_copy_A)` — prefetch next A |

**Line 309 — the real bottleneck (64.5%).** This is the `fx.gemm` MFMA call inside the unrolled `k_iters` loop. The per-instruction breakdown shows the stall is not the matrix multiply itself but the `s_waitcnt vmcnt(0) lgkmcnt(0)` and repeated `s_waitcnt lgkmcnt(0)` fences immediately ahead of the MFMA issue (entries #1, #3, #8–#15 in the per-instruction table). The MFMA front-end blocks until (a) the A operand has landed from LDS (`lgkmcnt`) and (b) outstanding global loads of B have retired (`vmcnt`). With occupancy 2, there is no spare wave to cover that latency, so the wait is exposed. This is a genuine pipeline-coverage problem, not a source-loc artifact.

**Line 311 — write A to LDS (9.4%, VMEM-wait).** `fx.copy(uni_copy_g2s, frag_copy_A, pA_s[..., write_stage])` stores the prefetched A tile into the LDS write-stage of the double buffer. It carries VMEM-wait because the data being written depends on the global A load (line 298) completing — the store-to-LDS is gated on the load it consumes. It feeds the next iteration's line-309 reads, so it is on the critical staging path.

**Line 100 — barrier collapse (20.4%), an attribution artifact to read carefully.** Line 100 is the `@flyc.kernel def kernel_gemm(` signature line — code does not "execute" there. The per-instruction table shows the cost is `s_barrier` (entries #2, #4) plus `s_waitcnt lgkmcnt(0)` (#7). These are the `gpu.barrier()` calls at lines 314 (end of `pipeline_stage`) and 321 (prologue) being attributed back to the function-definition line — a **source-location-granularity collapse** (the same class of issue as FlyDSL #587 / PR #593, where hot lines fold onto a helper/def rather than the emitting statement). Do **not** read line 100 as "the def is slow." The real meaning: **15% of stall is the ping-pong barrier between the LDS write-stage flip and the next read** — i.e. the cost of synchronizing the double buffer. That is a consequence of the same shallow pipeline as line 309, not an independent bug.

Net: lines 309 + 311 + the line-100 barriers are **one story** — a 2-stage double-buffered pipeline (`frag_B_stages`, `pA_s` write/read stages) that, at occupancy 2 and this skinny K, cannot hide its own HBM→LDS→register latency. ~84% of all stall sits on that staging path.

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Raise occupancy past 2 waves/SIMD by cutting VGPR <= 85 (addresses the 40.6% LDS-wait + most of VMEM-wait)
**Root cause:** Sec 1 — 89% of cycles are wait, and the dominant LDS/SMEM-wait + VMEM-wait on line 309 are *exposed* because occupancy is only 2 waves/SIMD. A third resident wave gives the scheduler something to issue while one wave's `s_waitcnt` drains, directly converting wait-cycles into useful issue. The analyzer states the lever explicitly: **3 waves/SIMD needs max VGPR <= 85**, and we are at ~97 (alloc 104). Closing that ~12–19 VGPR gap is the single highest-leverage change.
**Concrete change:** trim live VGPRs in the inner loop — shrink the C accumulator footprint (smaller register-blocking / narrower `acc_size`), reuse the A retile fragment instead of holding a distinct prefetch frag live across the MFMA, and check whether the double-buffered B fragments can share storage. Validate with `FLYDSL_DUMP_IR=1` + the kernel's own VGPR report after each trim; the target is `arch_vgpr <= 85`.
**Expected gain:** 2→3 waves is a 50% occupancy bump; on a latency-exposed GEMM this can recover a meaningful slice of the 73.6% LDS+VMEM-wait. Likely closes the 0.88x gap and could push past AIter at this shape. **Risk:** too-aggressive trimming can shrink the tile and lose MFMA efficiency — measure TFLOPS, not just occupancy.
**Effort:** medium (register-budget tuning + re-verify correctness/perf).
**Grounding:** ROCmKernelWiki `technique-occupancy-tuning` (waves/SIMD vs ILP on CDNA) and `technique-vgpr-budgeting` (ArchVGPR pressure vs occupancy). The wiki's occupancy formula `floor(ArchVGPR_file_per_simd / vgprs_per_wave)` is exactly the VGPR<=85 → 3-waves relation the analyzer reports; relevant PR `pr-FlyDSL-591`.

### #2 — Deepen the software pipeline / reorder waits so MFMA issue isn't gated on the immediate next load (addresses line 309 + line 100 barrier)
**Root cause:** Sec 2 — line 309's MFMA blocks on `s_waitcnt vmcnt(0) lgkmcnt(0)`, and the line-100 barriers (15%) synchronize a 2-stage double buffer. A 2-stage buffer at low occupancy cannot hide the HBM→LDS→reg latency by itself.
**Concrete change:** the kernel already has a `hot_loop_scheduler()` with `rocdl.sched_vmem/dsrd/dswr/mfma` knobs (lines 248–290) and an fp8/gfx950 preload path (`_get_preload`, line 95). Tune the schedule so more B-load issue and A LDS-reads are launched *earlier* relative to the MFMA cluster, and relax the `vmcnt`/`lgkmcnt` targets so the MFMA only waits on the operands it truly needs that iteration (issue #N's MFMA shouldn't block on issue #N+1's prefetch). Where VGPR budget permits after #1, consider a 3-stage B buffer to add one more iteration of slack.
**Expected gain:** secondary to #1 but compounds — better hiding of the same 73.6% wait without needing extra waves. **Effort:** medium-high (scheduler tuning is finicky and shape-sensitive; verify per shape).
**Grounding:** `technique-mfma-pipelining` (interleaving loads and matrix issue; "front stalls on `s_waitcnt` waiting for the next K-slice before it can issue the next MFMA") and `technique-lds-double-buffering`; relevant PRs `pr-FlyDSL-346`, `pr-FlyDSL-579`.

### #3 — Confirm LDS staging is conflict-free (smaller, but cheap to rule out)
**Root cause:** part of the LDS/SMEM-wait could be `ds_read`/`ds_write` (lines 305/311; ds_read 32, ds_write 10) hitting bank conflicts on the A tile, which inflate the `lgkmcnt` the MFMA waits on.
**Concrete change:** verify the `pA_s` LDS layout is padded/swizzled so the 32 ds_reads per wave are conflict-free for the MFMA operand pattern; add padding or a swizzled layout if profiling shows conflicts.
**Expected gain:** modest at this shape (LDS line 305 is only 2.2%), but it's a low-cost check that de-risks #1/#2. **Effort:** low-medium.
**Grounding:** `technique-bank-conflict-avoidance`, `technique-lds-swizzling`.

### #4 — Recognize this shape is launch/memory-bound; benchmark the win on a compute-bound shape
**Root cause:** Sec Workload — 0.069 GFLOP at M=16 leaves the device starved; ~5 us is dominated by launch + HBM, so kernel-internal optimization has a low ceiling here regardless of #1–#3.
**Concrete change:** for the "does FlyDSL beat AIter" verdict, re-run on a large compute-bound shape (large_shape M=5120 N=2048 K=8320). For deployment in a many-small-call decode loop, amortize launch with HIP graphs.
**Expected gain:** changes the *evaluation*, not the kernel; clarifies whether #1–#3 actually matter in production. **Effort:** low.
**Grounding:** `technique-hip-graphs`, `technique-persistent-kernel`.

## 4. Re-run

```bash
# ATT capture (from /sgl-workspace/flydsl-prof)
/opt/venv/bin/python att_capture.py \
  --test /sgl-workspace/FlyDSL-lab/tests/kernels/test_preshuffle_gemm.py \
  --kernel kernel_gemm_0 \
  --args "-M 16 -N 1024 -K 2048 --tile_m 16 --tile_n 64 --tile_k 512 --num_iters 50 --num_warmup 15" \
  --iter-range "[6,[8-8]]" \
  --out /sgl-workspace/flydsl-prof/results/att/bench_preshuffle_gemm_v2

# Same-shape baseline (AIter compare is ON by default — do NOT pass --no_aiter_bench)
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
/opt/venv/bin/python tests/kernels/test_preshuffle_gemm.py \
  -M 16 -N 1024 -K 2048 --tile_m 16 --tile_n 64 --tile_k 512 \
  --num_iters 50 --num_warmup 15   # run from /sgl-workspace/FlyDSL-lab
```

Shape: M=16, N=1024, K=2048, fp8 e4m3 → bf16, tile 16x64x512. Hotspot re-analysis: `hotspot_big.txt` in the bundle dir.

## Addendum — compute-bound re-test (2026-06-01)

This bench compares the **v2 layout-API** kernel against the **v1 manual** kernel (internal, not an external lib). At a saturating **M=4096, N=5120, K=8192 bf16** shape:

| impl | µs | TFLOPS |
|---|---:|---:|
| v2 (layout API) | 447.9 | 767.2 |
| v1 (manual) | 538.2 | 638.4 |

→ **v2 is 1.20× over v1** — the layout-API refactor is a real FlyDSL-internal win. The external CK gap is tracked under [`preshuffle_gemm`](../preshuffle_gemm/REPORT.md).
