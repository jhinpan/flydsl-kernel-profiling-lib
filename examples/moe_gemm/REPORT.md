# MoE GEMM (2-stage, stage-1) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X / CDNA4 · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `moe_gemm1_0` (ATT dispatch label `ui_output_agent_56573_dispatch_87`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_moe_gemm/`

> Arch caveat: the analyzer prints `gfx942 (CDNA3)` — that is its arch-detection default, not the real target. This was captured on **gfx950 / MI350X / CDNA4**, which has a **combined VGPR pool** (no split arch/accum file like gfx942). Occupancy reasoning below uses the combined-pool model. accum_vgpr reads 0, consistent with gfx950 lowering all accumulators into the unified VGPR file.

---

## Workload & headline

- **Shape:** model_dim=6144, inter_dim=4096, experts=8, topk=2, tokens=32 (M_eff=64 after topk fan-out), in_dtype=fp8, tile_m=32 / tile_n=128 / tile_k=256. This is **MoE stage-1** (gate+up GEMM, the `moe_gemm1` kernel).
- **Latency / throughput (HIP-event timing, not ATT):** FlyDSL stage-1 **70.8 us @ 91.06 TFLOPS**. Full 2-stage path 111.1 us.
- **Head-to-head (authoritative, `baselines.json`):** vs AIter CK MoE 2-stage (`ck_moe_stage1_fwd` + `ck_moe_stage2_fwd`, per-Token fp8, Silu), same shape, same `run_perftest` harness, output-asserted at rtol/atol=0.25:
  - **Stage-1: FlyDSL 70.8 us vs CK 71.1 us -> ~1.00x (parity).**
  - Stage-2 atomic (FlyDSL default): FlyDSL 40.3 us vs CK 52.3 us -> 1.30x.
  - **Total: FlyDSL 111.1 us vs CK 123.4 us -> 1.11x faster.**

**Verdict: FlyDSL wins overall (1.11x), driven entirely by stage-2; stage-1 — the kernel captured here — is dead even with CK.** The 142.9 us / 93.01 TFLOPS in the sweep is stale (it timed a slower stage-2 variant, s2=73.6 us); the stage-1 figure (~70 us) matches. So this ATT capture is the *parity* kernel: any stage-1 win has to come out of the stall profile below.

This is a tiny problem (32 tokens, M_eff=64 -> 24 waves, occupancy 1 wave/SIMD). It is **latency-bound on the weight-load pipeline**, not compute-bound — 91 TFLOPS is far under the gfx950 fp8 peak, and 91% of all cycles are stalls.

---

## 1. Wave-state / stall breakdown

996 instructions mapped (99.9%), 24 waves, **2.23M total cycles, 2.02M stalled = 91.0%**. This kernel barely computes; it waits.

| Stall type      |    Stall | %    | Meaning |
|-----------------|---------:|------|---------|
| **VMEM-wait**   | **1.11M**| **55.0%** | `s_waitcnt vmcnt(N)` — waiting on outstanding global loads (the A/B tile feeds) |
| VMEM-load       |  304.7K  | 15.0% | the `buffer_load_dwordx4` issues themselves stalling for issue slots / queue |
| barrier         |  263.9K  | 13.0% | `s_barrier` — LDS producer/consumer sync between the load stage and MFMA stage |
| MFMA/FMA        |  227.5K  | 11.2% | matrix-core issue + accumulate |
| LDS/SMEM-wait   |  101.5K  |  5.0% | `s_waitcnt lgkmcnt(0)` on LDS reads/scale loads |
| other / LDS / VMEM-store | <15K | <0.6% | negligible |

**Bound type: stall-bound, specifically VMEM latency-bound.** VMEM-wait + VMEM-load together are **70%** of all stall cycles. The barrier slice (13%) is downstream of the same problem — waves arrive at `s_barrier` and idle because the *other* waves are still draining their `vmcnt` queue. MFMA at 11% is the only "useful-work" stall and it is small; the matrix cores are starved.

**Register pressure & occupancy:**
- arch_vgpr ~155 (alloc 160), accum_vgpr 0, limiting pool 160/256.
- **Occupancy = 1 wave/SIMD.** This is the core disease: with only one wave resident, there is **zero inter-wave latency hiding** — when that wave issues a load and hits `s_waitcnt`, the SIMD has nothing else to run, so the load latency lands directly on the critical path. That is exactly the 55% VMEM-wait.
- Analyzer: **"2 waves requires max(arch,accum) <= 128."** We are at 160. On gfx950's combined pool, getting to 2 waves/SIMD means shedding **~32 VGPRs** (160 -> 128) per wave. That doubles the latency-hiding headroom.
- Inst mix: MFMA 256, buffer_load 96, buffer_store 16, ds_read 32, ds_write 8 — a load-heavy GEMM inner loop. 96 buffer_loads feeding 256 MFMAs through LDS.

---

## 2. Top instruction-level hotspots

Source lines below are aggregated stall cycles; the analyzer truncates the path prefix (`yDSL-lab/...`, `lyDSL-lab/...` are `.../FlyDSL-lab/...`). Two source files dominate: the kernel body `moe_gemm_2stage.py` and the shared MFMA-preshuffle pipeline helper `mfma_preshuffle_pipeline.py`.

**#1 — `moe_gemm_2stage.py:880` · 639.6K (31.6%) · VMEM-wait**
Line 880 is `acc_mid = mfma_fn(mfma_res_ty, [a0, b0, acc_in, 0, 0, 0])` — the first of the two K=32 MFMA issues inside `mfma_k64()`. The stall is attributed here because the `s_waitcnt vmcnt(N)` that gates the MFMA's input operands is scheduled immediately ahead of the matrix issue. The instruction-level view confirms it: the hot ASM at :880 is `s_waitcnt vmcnt(7)`, `vmcnt(9)`, `vmcnt(5)`, `vmcnt(15) lgkmcnt(0)` — **the MFMA is blocked waiting for its A/B tiles to arrive from HBM.** This is the single biggest bucket and it is pure load latency, not compute.

**#2 — `mfma_preshuffle_pipeline.py:585` · 570.7K (28.2%) · VMEM-wait**
Line 585 is `vector.store(v16, lds_memref, [idx0])` inside `lds_store_16b_xor16()` — the direct-to-LDS store of a 16-byte tile chunk with CK-style XOR16 swizzle. The dominant ASM here is **`s_waitcnt vmcnt(1)`** (the top-2 single instructions in the whole kernel, 312.6K + 237.1K). Reading: the global->LDS staging path waits on `vmcnt` for the buffer_load to complete *before* the LDS write can fire. This is the **load->LDS leg of the software pipeline draining synchronously** — the loads are not running far enough ahead of the stores.

**#3 — `moe_gemm_2stage.py:280` · 306.9K (15.2%) · barrier**
Line 280 is the `@flyc.kernel def moe_gemm1(...)` decorator/entry — the barrier stall (`s_barrier`, 158K+72K+20K) gets attributed to the kernel-scope frame because the LDS double-buffer barrier lives at top-level scope. This is the producer/consumer fence between the LDS-store stage (#2) and the LDS-read->MFMA stage (#1). It is **a symptom of #1/#2**, not an independent cost: waves stall at the barrier because peers are still draining `vmcnt`. Fix the load pipeline and this shrinks with it.

**#4 — `mfma_preshuffle_pipeline.py:83` · 306.4K (15.1%) · VMEM-load**
Line 83 is the `buffer_ops.buffer_load(rsrc, idx_i32, vec_width=..., dtype=T.i32)` in the preshuffle loader. ASM: `buffer_load_dwordx4 v[74:77], v62, s[16:19] ...` — the actual 128-bit A/B-tile loads. This is the issue-side cost of the same loads that #1/#2 wait on. Already 128-bit (`dwordx4`) and vectorized, so the *width* is right; the problem is depth (how many are in flight), not width.

**Minor (each <2%):** :363 (`create_buffer_resource` for max_token_ids, LDS/SMEM-wait on the scalar guard load), :370 (block-validity `cmpi`), :405/:442/:415 (more `create_buffer_resource` scale/expert setup). These are one-time prologue scalar loads, not loop hot spots — ignore.

No source-loc-collapse artifact here (no `rocdl/universal.py` frames); :280 is the closest to one, and it is correctly read as the barrier-at-kernel-scope, a real fence.

---

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Deepen the global->LDS software pipeline (more loads in flight before the wait)
**Root cause:** §1 dominant stall — 55% VMEM-wait + the :585 `vmcnt(1)` stores. With occupancy 1, the *only* latency hiding available is **intra-wave**: issue more `buffer_load`s ahead of the consuming `s_waitcnt`, so the `vmcnt` queue has many outstanding loads instead of stalling at `vmcnt(1)`/`vmcnt(7)`. Today the load->store leg drains at `vmcnt(1)` (store waits for nearly every load), which means the prefetch depth is ~1 — effectively unpipelined.
**Change:** increase the prefetch/pipeline stage count in the MFMA-preshuffle loop (the `num_stages` / double->triple-buffer parameter feeding `lds_store_16b_xor16`), and relax the `s_waitcnt vmcnt(N)` target so the LDS store only waits on its *own* tile, not the whole queue. Concretely: split-K / K-stage the inner loop so the next K-tile's `buffer_load_dwordx4` is issued before the current tile's MFMA, and let `vmcnt` ride at a higher N.
**Grounded in:** ROCmKernelWiki `technique-mfma-pipelining` (MFMA Software Pipelining — interleaving loads and matrix issue; gfx950) and `technique-lds-double-buffering` (direct-to-LDS + `s_waitcnt vmcnt(N)` counter gating). Both are already partially present in this kernel (it uses XOR16 direct-to-LDS); the depth is the gap. Both cite FlyDSL PR-346 / PR-579 as in-tree precedent.
**Expected gain:** this is the 70% stall bucket. Even halving VMEM-wait would cut a large fraction of the 91% stall time. Since stage-1 is at parity with CK, this is the lever that turns parity into a win.
**Effort:** medium — pipeline-depth and waitcnt tuning inside the existing preshuffle loop, no algorithmic change.

### #2 — Cut VGPR below 128 to reach 2 waves/SIMD (restore inter-wave latency hiding)
**Root cause:** occupancy 1 (§1). One resident wave = no other wave to cover the load latency, so every `vmcnt` stall is naked on the critical path. The analyzer states the budget exactly: **2 waves needs VGPR <= 128; we are at 160.**
**Change:** shed ~32 VGPRs. Candidates: shorten the lifetime of the dequantized/converted vectors (`_i64x2_to_v8f16` etc. at :848-865 hold wide vector temporaries), reuse accumulator/operand registers across the two `mfma_k64` issues at :880-881, and reduce the per-`num_acc_n` prefetch fan-out at :824-838 (`sw_gate_pf`/`sw_up_pf` lists keep many scale VGPRs live). On gfx950's **combined pool**, every VGPR saved counts directly against the single 256-wide file — there is no separate accum file to spill into.
**Grounded in:** `technique-occupancy-tuning` and `technique-vgpr-budgeting` (both gfx950; cite FlyDSL PR-591). On the combined pool, occ = floor(VGPR_file_per_simd / vgprs_per_wave) is the binding constraint here.
**Expected gain:** 2 waves/SIMD doubles the latency-hiding budget and is synergistic with #1 — a deeper pipeline plus a second wave to overlap it. Risk: too-aggressive VGPR reduction can re-serialize MFMA; tune against the 11% MFMA bucket.
**Effort:** medium-high — register-lifetime surgery, easy to regress correctness; verify with the harness's built-in output assert.

### #3 — Confirm tile_n=128 / tile_k=256 is right for M_eff=64 (avoid over-fetching weights)
**Root cause:** this is a fat-weight, thin-token problem (M_eff=64, N=inter, K=model_dim). The 96 buffer_loads per 256 MFMAs and the VMEM-load (#4) cost suggest the kernel is moving a lot of W per unit of useful M work. With only 24 waves the launch is small; a tile that reduces redundant W re-fetch (or a split-K that overlaps better) may shrink #4 directly.
**Change:** sweep tile_m/tile_n/tile_k around the captured 32/128/256 for this exact shape; check whether a larger tile_m (more token reuse of each loaded W tile) lowers VMEM-load without blowing VGPR past the #2 budget.
**Grounded in:** `technique-vectorized-loads` — loads are already 128-bit `dwordx4` (width is correct), so the remaining lever is *fewer* fetches via better reuse, not wider ones.
**Expected gain:** secondary; trims the 15% VMEM-load bucket. Bounded by #1/#2.
**Effort:** low — pure autotune sweep, no code change; the harness exposes `--tile_m/--tile_n/--tile_k`.

> **Why #1 first:** it directly attacks the 55% (+15%) VMEM-wait bucket — the single largest stall — without needing the occupancy win. #2 amplifies #1 but is riskier and bounded by the same load latency. Order: pipeline depth (#1) -> occupancy (#2) -> tile reuse (#3).

---

## 4. Re-run

ATT capture (this bundle):
```bash
/opt/venv/bin/python /sgl-workspace/flydsl-prof/att_capture.py \
  --test test_moe_gemm \
  --out /sgl-workspace/flydsl-prof/results/att/test_moe_gemm
```

Underlying kernel + baseline (the shape these numbers came from):
```bash
cd /sgl-workspace/FlyDSL-lab && \
HIP_VISIBLE_DEVICES=2 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
/opt/venv/bin/python tests/kernels/test_moe_gemm.py \
  -t 32 -e 8 -k 2 -dim 6144,4096 \
  --tile_m 32 --tile_n 128 --tile_k 256 \
  --in_dtype fp8 --num_iters 20 --num_warmup 10 \
  --compare_aiter_ck true
```

Hotspot re-derive: `hotspot_analyzer.py` on `att/ui_output_agent_56573_dispatch_87` -> `hotspot_big.txt`.
ATT samples one CU (`att_target_cu`); it is a representative wave-state sample, not full-device timing. All us/TFLOPS are HIP-event timing from the harness, not ATT.
