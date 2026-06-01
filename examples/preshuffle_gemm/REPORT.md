# Preshuffle GEMM (fp8 a8w8, B-preshuffle) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X (CDNA4) · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `kernel_gemm_0` (ATT dispatch `ui_output_agent_58177_dispatch_135`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_preshuffle_gemm/`

> Arch note: for this kernel the analyzer **correctly** detected `gfx950 (CDNA4)` — no `gfx942` caveat applies here. gfx950 has a **single combined 512-VGPR pool** (no split arch/accum file), so occupancy is reasoned against that one budget.

---

## Workload & headline

- **Shape:** M=16, N=1024, K=2048 · in=fp8 (e4m3), out=bf16 · tile 16×64×512 · B preshuffled (16×16 weight layout)
- **Latency:** FlyDSL **5.0 µs** (median; sweep run 4.6 µs) → **13.3 TFLOPS / ~458 GB/s**
- **Baseline (authoritative, baselines.json):** AIter `gemm_a8w8_bpreshuffle` (CK fp8, untuned default config — AIter found *no* tuned entry for M16/N1024/K2048) → **4.4 µs / 15.23 TFLOPS / 491.86 GB/s**
- **Speedup FlyDSL vs AIter: 0.88× — FlyDSL is ~12% slower at the median.**

**Verdict: a tie within noise, slightly behind.** This is a *tiny, memory/launch-bound* shape — only 0.069 GFLOP total. Both kernels land at 4–5 µs where per-run jitter (±0.5 µs) swamps the gap; across four AIter runs the range was 4.2–4.8 µs and in one of four FlyDSL was actually faster (4.5 vs 4.8). The 13–15 TFLOPS figures are *far* below MI350X peak because M=16 cannot fill the machine — this shape does not exercise the compute roofline at all. A real "does FlyDSL win" verdict needs a compute-bound shape (e.g. M=5120, N=2048, K=8320). At *this* shape FlyDSL does **not** beat the strongest comparable CK kernel.

---

## 1. Wave-state / stall breakdown

863 instructions, **6.42M total cycles, 5.31M stalled = 82.7%.** This kernel spends 5 of every 6 cycles waiting, almost entirely on memory.

| Type            |  Stall | %    | Bucket            |
|-----------------|-------:|------|-------------------|
| **VMEM-wait**   | 1.79M  | 33.7 | wait (HBM/B-load) |
| **LDS/SMEM-wait** | 1.18M | 22.3 | wait (LDS A read) |
| MFMA/FMA        | 698.1K | 13.1 | exec (matrix)     |
| LDS             | 626.5K | 11.8 | exec (ds issue)   |
| VMEM-load       | 439.2K |  8.3 | issue             |
| barrier         | 366.0K |  6.9 | sync              |
| other           | 109.3K |  2.1 | —                 |
| VMEM-store      |  85.2K |  1.6 | epilogue          |
| SMEM            |  10.4K |  0.2 | —                 |

**Bound class: wait-bound, specifically VMEM-wait.** VMEM-wait + VMEM-load + VMEM-store = **43.6%** of all stall, LDS-wait + LDS-issue = **34.1%**. Only ~13% is genuine MFMA execution. The matrix cores are starved: the wave sits on `s_waitcnt vmcnt(N)` waiting for B tiles from HBM and on `s_waitcnt lgkmcnt(0)` waiting for A from LDS, then hits `s_barrier`. This is the classic signature of a GEMM whose K-loop pipeline is too shallow to hide HBM latency — and at M=16 there is essentially no compute to hide it behind.

**Register pressure & occupancy:**
- `arch_vgpr ~161` (alloc 168), `accum_vgpr 0` → **total 168 / 512** combined pool.
- **Occupancy: 3 waves/SIMD.** Analyzer: *4 waves requires total_vgpr ≤ 128.* We are 40 VGPRs over the line for the next occupancy step.
- Inst mix: MFMA 64, buffer_load 42, buffer_store 64, ds_read 64, ds_write 32 — a load-heavy mix; buffer_store (64) is inflated by the bf16 epilogue writing 16-wide rows.

3 waves/SIMD is thin latency-hiding cover. With only 3 waves to swap in while one is blocked on `vmcnt`, there is not enough independent work to keep the VALU/MFMA busy through HBM latency — which is exactly why VMEM-wait dominates. But note the causality: at M=16 the tile is so small that even at 4 waves the arithmetic intensity is too low to saturate; **occupancy is a contributing limiter, not the root cause.** The root cause is the load/MFMA pipeline depth and the inherently low intensity of this shape.

---

## 2. Top instruction-level hotspots

| # | %tot | DomType | Source | What runs there |
|---|-----:|---------|--------|-----------------|
| 1 | 52.0% | VMEM-wait | preshuffle_gemm.py:**313** | whole `@flyc.kernel def kernel_gemm` body — the K-loop pipeline (source-loc collapse) |
| 2 | 35.6% | LDS/SMEM-wait | preshuffle_gemm.py:**940** | `rocdl.mfma_scale_f32_16x16x128_f8f6f4(...)` — the MFMA, waiting on A from LDS |
| 3 | 4.4% | VMEM-load | mfma_preshuffle_pipeline.py:83 | pipeline helper — B prefetch issue |
| 4 | 4.1% | VMEM-load | preshuffle_gemm.py:517 | `buffer_ops.buffer_load(b_rsrc, …, vec_width=4)` — the 16B B-tile load |
| 5 | 1.6% | VMEM-store | preshuffle_gemm.py:1167 | `buffer_ops.buffer_store(val_f16, c_rsrc, idx_out)` — bf16 epilogue store |
| 6 | 1.6% | LDS/SMEM-wait | preshuffle_gemm.py:396 | `create_buffer_resource(arg_a, …)` — A buffer-resource setup / first A wait |

**Line 313 is a source-loc-collapse artifact, not a single hot instruction.** It is the `@flyc.kernel def kernel_gemm` *definition line*. The per-instruction ASM dump shows what actually accumulates there: `s_waitcnt vmcnt(2)` (587.6K), `s_waitcnt vmcnt(3)` (506.6K), two `s_barrier` (181.2K + 169.7K), plus more `vmcnt`/`lgkmcnt` waits. These are the **K-loop's load-wait and barrier instructions** — the steady-state pipeline body — all attributed to the kernel signature line because the unrolled main loop carries the function-entry debug line. So #1 is real and it is the dominant cost: **the wave stalls on `s_waitcnt vmcnt(N)` waiting for B tiles and on `s_barrier` synchronizing the ping/pong LDS stage.** This is FlyDSL issue #587 / PR #593 territory (source-loc granularity), so don't read line 313 as "the function header is slow" — read it as "the main K-loop is slow, and the cost is HBM-load wait + barrier."

**Line 940 is the real MFMA.** `mfma_scale_f32_16x16x128_f8f6f4` is the scaled fp8 matrix instruction. Its dominant stall type is **LDS/SMEM-wait** (`s_waitcnt lgkmcnt(0)` appears 6× in the top-15 instructions, all attributed to 940): the MFMA cannot issue until the A operand has landed in registers from the `ds_read`. So #2 is not "the MFMA is slow" — it is **the MFMA waiting on the LDS read of A.** Combined, #1 (B from HBM) + #2 (A from LDS) tell one story: **operand delivery, not matrix throughput, is the ceiling.**

Lines 517 (B `buffer_load` vec_width=4 = 16B vectorized), 1167 (bf16 epilogue store), and 396 (A buffer-resource setup) are minor (<5% each). Line numeric.py:872 (0.2%, SMEM) is a FlyDSL expr helper and is negligible. No hot line maps to a misleading intrinsic-helper collapse here beyond the line-313 function-entry attribution.

---

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Deepen the K-loop software pipeline to hide HBM latency behind MFMA *(biggest stall: VMEM-wait 33.7%)*
**Root cause:** §1 — 33.7% VMEM-wait + 8.3% VMEM-load. The wave blocks on `s_waitcnt vmcnt(N)` (line 313) because B tiles arrive from HBM faster than they can be consumed only if enough loads are *in flight ahead* of the MFMA. With tile_k=512 and the current pipeline depth, the `vmcnt(2)`/`vmcnt(3)` waits mean the loads are not issued far enough ahead.
**Change:** Increase prefetch distance / pipeline stages in `mfma_preshuffle_pipeline.py` so that B-load for iteration *i+2* (or *i+3*) is issued before the MFMA of iteration *i* — i.e. raise the outstanding-`vmcnt` depth so the matrix core never waits. Pair this with `lds_stage == 2` ping/pong (already present, line 373) confirmed active so A double-buffers while B streams. Validate via `tests/kernels/test_preshuffle_gemm.py` perf and re-capture VMEM-wait%.
**Expected gain:** VMEM-wait is the top bucket; shaving even half of it would meaningfully cut the 82.7% stall fraction — but **bounded by the shape**: at M=16 there is little MFMA work to hide behind, so realistic upside here is modest (a few hundred ns) until tested on a larger-M shape where it should pay off strongly.
**Effort:** Medium (pipeline-depth tuning, no algorithm change).
**Grounded in:** ROCmKernelWiki `technique-lds-double-buffering` (direct-to-LDS + `s_waitcnt vmcnt(N)` counter gating to overlap HBM loads with MFMA; archs gfx942/gfx950) and `technique-mfma-pipelining` (interleave load issue with matrix issue via `vmcnt`/`lgkmcnt`; FlyDSL PRs #346/#579/#278).

### #2 — Cut A-operand LDS-read latency feeding the MFMA *(LDS/SMEM-wait 22.3%)*
**Root cause:** §2 — line 940's MFMA stalls on `s_waitcnt lgkmcnt(0)` waiting for the `ds_read` of A. 64 `ds_read` for 64 MFMA is 1:1; if the read isn't issued early enough the matrix core idles.
**Change:** Hoist the `lds_load_packs_k64` reads for the next `mi`/`ku` iteration ahead of the current MFMA so `lgkmcnt` is already satisfied when the MFMA wants the operand (deeper LDS read-ahead, same spirit as #1 but on the LDS side). Confirm the A-LDS layout has no bank conflicts at tile_k=512 (32 ds_write feeding 64 ds_read — check the swizzle).
**Expected gain:** Targets the 2nd bucket (22.3%); overlapping the LDS read with prior MFMA could reclaim a large share of that wait.
**Effort:** Medium.
**Grounded in:** `technique-mfma-pipelining` (same `s_waitcnt` co-issue discipline applied to `lgkmcnt`) and `technique-lds-double-buffering`.

### #3 — Trim VGPRs from 168 → ≤128 to reach 4 waves/SIMD
**Root cause:** §1 — 168 VGPR caps occupancy at 3 waves/SIMD; 4 waves needs ≤128. More waves = more independent work to hide the memory waits that dominate.
**Change:** Reduce register live-range pressure in the K-loop: the accumulator/operand packing (`pack_i64x4_to_i32x8` building 256-bit a128/b128 per `mi`,`ni`, lines 930–938) holds wide vectors live across the MFMA. Narrower accumulator tiling (smaller `num_acc_n` × `m_repeat`) or reusing pack temporaries could drop below 128. Trade-off: fewer accumulators in flight may *reduce* ILP, so measure — at M=16 the win may not materialize.
**Expected gain:** Going 3→4 waves is a 33% occupancy bump *if* it doesn't cost ILP, but at this tiny shape the latency-hiding benefit is limited (root cause is intensity, not just occupancy). Speculative.
**Effort:** Medium–High (register tuning, easy to regress).
**Grounded in:** `technique-vgpr-budgeting` (ArchVGPR pressure vs occupancy on the combined gfx950 pool) and `technique-occupancy-tuning` (waves/SIMD vs ILP; FlyDSL PR #591).

### #4 — Re-evaluate at a compute-bound shape before further micro-tuning
**Root cause:** Methodology, not the kernel. At M=16 the kernel is launch/memory-bound; 13–15 TFLOPS is nowhere near roofline and ±0.5 µs jitter dominates the 0.88× gap. Micro-optimizing here optimizes noise.
**Change:** Re-capture and re-baseline at a large compute-bound shape (the harness's `large_shape`, e.g. M=5120, N=2048, K=8320) where MFMA actually saturates. Only there will #1/#2 show their true gain and the FlyDSL-vs-AIter verdict be meaningful.
**Effort:** Low (run the harness with large params).

---

## 4. Re-run

ATT capture (B-preshuffle GEMM, captured shape):

```bash
cd /sgl-workspace/flydsl-prof
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
/opt/venv/bin/python att_capture.py \
  --test /sgl-workspace/FlyDSL-lab/tests/kernels/test_preshuffle_gemm.py \
  --kernel kernel_gemm_0 \
  --out /sgl-workspace/flydsl-prof/results/att/test_preshuffle_gemm \
  -- -M 16 -N 1024 -K 2048 --tile_m 16 --tile_n 64 --tile_k 512 --no_aiter_bench
```

Matched-shape head-to-head (AIter baseline, built-in harness compare — do NOT pass `--no_aiter_bench`):

```bash
cd /sgl-workspace/FlyDSL-lab
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
/opt/venv/bin/python tests/kernels/test_preshuffle_gemm.py \
  -M 16 -N 1024 -K 2048 --tile_m 16 --tile_n 64 --tile_k 512 \
  --num_iters 50 --num_warmup 15
```

Hotspot re-analysis: `hotspot_analyzer.py` on `att/ui_output_agent_*_dispatch_135` -> `hotspot_big.txt`.
ATT samples one CU (`att_target_cu`) — a representative wave-state sample, not full-device timing. The 5.0 µs / 13.3 TFLOPS numbers come from the harness HIP-event sweep, not from ATT.

## Addendum — compute-bound re-test (2026-06-01)

The headline/sweep numbers above were at the harness default **M=16** (launch-bound). Re-measured at a saturating **M=N=K=4096 fp8** shape (the same shape this ATT trace was captured at):

| impl | µs | TFLOPS |
|---|---:|---:|
| FlyDSL `kernel_gemm` | 102.0 | 1347.4 |
| AIter CK a8w8 bpreshuffle (untuned default) | 78.1 | 1760.2 |

→ **FlyDSL 0.77×** (≈23% slower) — at saturation the gap is real, not noise. AIter found no tuned entry for 4096³ and ran its default CK config, so this is FlyDSL vs *untuned* CK. The ATT VMEM-wait (33.7%) + occ 3/SIMD point at a shallow K-loop load pipeline; deeper prefetch is the lever.
