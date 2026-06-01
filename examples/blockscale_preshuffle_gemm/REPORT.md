# Blockscale Preshuffle GEMM (fp8 a8w8, bf16 out) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X / CDNA4 · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `bs_gemm_bf16_direct_t16x64x256` (ATT dispatch `ui_output_agent_14065_dispatch_120`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_blockscale_preshuffle_gemm/`

This is the one kernel where the analyzer correctly detected **gfx950 (CDNA4)** — no arch-label caveat needed here, unlike the gfx942 default it prints elsewhere. gfx950 has a single **combined VGPR pool** (no split arch/accum file), and `accum_vgpr=0` confirms the MFMA accumulators live in that combined pool.

## Workload & headline

- **Shape:** M=16, N=7168, K=2304 — fp8 A8W8 blockscale (128x128 scale blocks), bf16 output. Tile **16x64x256**.
- **FlyDSL:** 11.0 us / 48.1 TFLOPS (sweep counter run: 43.72 TFLOPS, 1.389 TB/s on the captured CU).
- **Head-to-head (authoritative, same process, same shape):** AIter CK `gemm_a8w8_blockscale_bpreshuffle` (auto-tuned, weight layout 16x16, scale_block 128x128) = **9.7 us / 54.54 TFLOPS**. Second run: 10.1 us.
- **Verdict: FlyDSL loses, 0.88x (AIter is ~13% faster).** This is a tiny-M=16 decode-style shape — exactly the regime where a single output-tile row across N starves the machine. FlyDSL is not slow because of bad inner-loop codegen; it is slow because the launch under-fills the GPU and each surviving workgroup serially streams all of K=2304 through VMEM.

ATT samples one CU (`att_target_cu=1`) — it is a representative wave-state sample, not full-device timing. The us/TFLOPS above come from the HIP-event sweep/baseline, **not** from ATT.

## 1. Wave-state / stall breakdown

390 of 391 instructions mapped (99.7%). Of 90.1K total cycles, **82.7K (91.8%) are stalls** — the kernel spends almost all its time waiting, not issuing.

| Stall type      |  Cycles | %    |
|-----------------|--------:|------|
| **VMEM-wait**   |   63.4K | **76.6%** |
| barrier         |    6.0K | 7.3% |
| LDS/SMEM-wait   |    6.0K | 7.2% |
| VMEM-load       |    4.7K | 5.6% |
| other           |    2.4K | 2.9% |
| LDS             |     192 | 0.2% |
| MFMA/FMA        |      84 | 0.1% |
| VMEM-store      |      28 | 0.0% |

**Bound class: latency-bound (VMEM-wait dominated).** 76.6% of stall cycles are `s_waitcnt vmcnt(N)` — the wave is parked waiting for global (HBM) loads to land. MFMA/FMA stall is **0.1%** and ds (LDS read/write) stall is ~0.2%: the matrix core and the LDS staging path are essentially never the bottleneck. The instruction mix backs this up — 18 MFMA vs **72 buffer_load** — this is a load-issue / load-wait machine with a thin compute tail.

**Register pressure & occupancy.** arch_vgpr ~73 (alloc 80), accum_vgpr 0, total 80/512 in the combined pool. Occupancy is **6 waves/SIMD**. The analyzer notes the next step (**7 waves/SIMD**) requires `total_vgpr <= 73` — i.e. shaving the 80-VGPR allocation by 7 registers buys one more wave. But note: at 6 waves/SIMD this kernel is **not** occupancy-starved *per CU*. The starvation is at the **grid** level (see below), so chasing the 7th wave is a second-order lever, not the fix.

**Why VMEM-wait dominates at M=16.** tile_m=16 means ceil(M/16)=1 M-block; N=7168/tile_n=64 -> 112 N-blocks. The grid is ~112 workgroups against MI350X's ~256+ CUs — under half the machine is lit, and every active workgroup serially walks all K=2304 (9 tiles of tile_k=256). The few busy CUs run long on a single deep K-stream while the rest sit idle, so the kernel's wall-clock is gated by HBM load latency on a serial K-walk, exactly the VMEM-wait signature. This is the textbook small-M.N / large-K occupancy-starved GEMM.

## 2. Top instruction-level hotspots

| # | Stall | %tot | DomType | Source |
|---|------:|-----:|---------|--------|
| 1 | 44.2K | 53.5% | VMEM-wait | `blockscale_preshuffle_gemm.py:138` |
| 2 | 30.4K | 36.7% | VMEM-wait | `mfma_preshuffle_pipeline.py:585` |
| 3 |  2.9K |  3.5% | LDS/SMEM-wait | `blockscale_preshuffle_gemm.py:541` |
| 4 |  2.6K |  3.1% | VMEM-load | `blockscale_preshuffle_gemm.py:481` |
| 5 |  1.8K |  2.2% | VMEM-load | `mfma_preshuffle_pipeline.py:83` |
| 6 |  272  |  0.3% | VMEM-load | `blockscale_preshuffle_gemm.py:489` |

**#1 — `blockscale_preshuffle_gemm.py:138` (44.2K, 53.5%, VMEM-wait).** Line 138 is `@flyc.kernel def kernel_gemm(...)` — the kernel function header. This is a **source-loc collapse** (granularity loss a la FlyDSL #587/#593): stall cycles for instructions that don't carry a finer line attribution roll up to the function-def line. The per-instruction view confirms what's actually there: a fan of `s_waitcnt vmcnt(0/1/2)` plus one `s_waitcnt lgkmcnt(0)` — these are the **A-matrix HBM-load waits and the LDS-staging drain** for the A operand. So #1 is real VMEM-wait traffic (the A/scale global loads + the wait before MFMA can consume them), just attributed to the function header rather than the exact load site. Treat it as "the A-load + drain waits," not "line 138 is slow."

**#2 — `mfma_preshuffle_pipeline.py:585` (30.4K, 36.7%, VMEM-wait).** Line 585 is `vector.store(v16, lds_memref, [idx0])` inside `lds_store_16b_xor16` — the CK-style XOR16-swizzled store of a 16B B-chunk into LDS. The DomType is **VMEM-wait**, not LDS: the store can't issue until the preceding `buffer_load` of that B-pack has landed, so the `s_waitcnt vmcnt(5/7)` gating the global->LDS staging is charged here. The per-instruction list shows a ladder of `vmcnt(7)`/`vmcnt(0)` at this line — the wave is draining the B-load queue before swizzling into LDS. This is the **B-operand global-load latency**, the other half of the same VMEM-wait wall as #1.

Together #1 + #2 are **90.2%** of all stalls and both are global-load waits (A side + B side). The kernel is one big load-latency stall.

**#3 — `:541` (2.9K, LDS/SMEM-wait).** This is `rocdl.mfma_scale_f32_16x16x128_f8f6f4(...)` — the actual scaled-MFMA. The LDS/SMEM-wait here is the `lgkmcnt` wait for the A packs to arrive from LDS (`lds_load_packs_k64`) just before the matrix issue. Tiny (3.5%) — the MFMA feed from LDS is well hidden.

**#4 / #6 — `:481` / `:489` (scale loads).** `:481` is `buffer_load(scale_a_rsrc, ..., vec_width=4)` and `:489` is `buffer_load(scale_b_rsrc, ..., vec_width=1)` in `load_scales_for_tile`. These are the per-K-tile A/B scale fetches. `:489` loads scale_b **one f32 at a time** (`vec_width=1`) — a narrow load — but at 0.3% it's not material here.

**#5 — `mfma_preshuffle_pipeline.py:83` (1.8K, VMEM-load).** `_buffer_load_vec`'s `buffer_load(...)` — the generic vectorized B/scale load helper. Real load-issue cost, small.

Net: the only stalls that matter are #1 and #2, and they are the **same problem** — waiting on HBM for A and B while a half-empty grid serially streams K.

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Split-K to fill the grid and shorten each K-walk  ·  root cause: §1 VMEM-wait (76.6%) / occupancy-starved grid  ·  effort: medium-high

The dominant 76.6% VMEM-wait is a *serial-K-stream-on-an-underfilled-grid* symptom, not an inner-loop codegen defect (MFMA stall 0.1%, LDS stall 0.2%). At M=16 the launch makes only ~112 workgroups for ~256+ CUs and each one streams all K=2304. **Split the K reduction**: partition K into `SplitK` slices so the grid becomes `112 . SplitK` workgroups — enough to cover every CU — and each workgroup now walks only K/SplitK, so its VMEM-wait critical path shrinks proportionally. Reduce the partials with **atomic split-K** (`global_atomic_add_f32` into a pre-zeroed f32 accumulator, no second launch) for the low-overhead path, or a **workspace reduction** kernel for determinism. For K=2304 / tile_k=256 (9 K-tiles), SplitK=2-4 is the natural sweep — start at 3 (3 K-tiles each). Expected gain: this is the lever that closes the 0.88x gap and can overtake AIter at small M, since CK's own win here comes from its auto-tuned GSU/k_batch picking exactly this. **ROCmKernelWiki: `technique-split-k`** ("Split-K / GlobalSplitU — Partial-Sum Reduction GEMM for Small M.N, Large K"; implemented in FlyDSL by pr-FlyDSL-370, pr-FlyDSL-346, and in CK by composable_kernel-933/2152). Related: `technique-occupancy-tuning`, the low-occupancy pattern.

### #2 — Deepen the global->LDS prefetch pipeline to hide load latency  ·  root cause: §1 VMEM-wait, hotspots #1/#2  ·  effort: medium

Even after split-K each slice still serially waits on `vmcnt` (the `vmcnt(7)` ladder at pipeline:585 and the `vmcnt(0/1/2)` fan at :138). The kernel already double-buffers (separate ping/pong LDS at lines 119-123), but the VMEM-wait wall says the prefetch depth isn't covering HBM latency for this load-heavy mix (72 buffer_load : 18 MFMA). Increase the **software-pipeline prefetch distance** — issue the A and B `buffer_load`s for K-tile *t+2* (not just *t+1*) before the `s_waitcnt` for tile *t*, so more loads are in flight per `vmcnt` wait and the matrix core never stalls on a single outstanding load. Expected gain: directly eats into the 76.6% VMEM-wait by overlapping more loads with compute; secondary to #1 because at M=16 the grid starvation dominates, but it compounds with split-K. **ROCmKernelWiki: `technique-mfma-pipelining`** (interleave loads and matrix issue under `vmcnt`/`lgkmcnt`; FlyDSL pr-FlyDSL-579/346) and **`technique-lds-double-buffering`** (direct-to-LDS + `vmcnt(N)` gating).

### #3 — Widen / coalesce the scale_b load  ·  root cause: hotspot #6 (`:489`, vec_width=1)  ·  effort: low

`load_scales_for_tile` fetches `scale_b` with `buffer_load(..., vec_width=1)` (one f32 per `num_acc_n` step, line 489) while `scale_a` uses `vec_width=4`. Each scalar load is its own VMCNT entry and address calc. Coalesce the per-`ni` scale_b fetches into a single vectorized load where the N-block layout permits (the 128x128 scale blocks mean adjacent `ni` often share or stride a scale row predictably). Expected gain: small (#6 is 0.3% of stalls) — do this only as cleanup alongside #1/#2, not on its own. **ROCmKernelWiki: `technique-vectorized-loads`** (128-bit loads to saturate HBM; one issue slot, one VMCNT entry per wide load).

### #4 — Trim VGPRs to reach 7 waves/SIMD  ·  root cause: §1 occupancy  ·  effort: low-medium, speculative

The analyzer says 7 waves/SIMD needs `total_vgpr <= 73` (currently alloc 80, live ~73). Shaving 7 VGPRs — e.g. shorter live ranges on the prefetch buffers, reusing the `a0/a1` prefetch temporaries — buys one more wave per SIMD to hide latency. But this is a **per-CU** lever and the real deficit is **grid-level** (too few workgroups), so expected gain is marginal until #1 is in place. Verify any change against scratch spill (`ScratchSize` must stay 0). **ROCmKernelWiki: `technique-vgpr-budgeting`**, `technique-occupancy-tuning`.

## 4. Re-run

ATT capture (one CU, deterministic debug cache):

```bash
/opt/venv/bin/python /sgl-workspace/flydsl-prof/drivers/att_capture.py \
  --test test_blockscale_preshuffle_gemm.py \
  --gpu 1 --target-cu 1 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_blockscale_preshuffle_gemm \
  --tag big --iter-range "[6,[8-8]]" \
  --cmd-override "python tests/kernels/test_blockscale_preshuffle_gemm.py -M 16 -N 7168 -K 2304 --out_dtype bf16 --num_iters 20 --num_warmup 3"
```

Underlying harness shape (also the matched-baseline timing run, with AIter compare enabled):

```bash
cd /sgl-workspace/FlyDSL-lab && HIP_VISIBLE_DEVICES=1 \
  PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
  /opt/venv/bin/python tests/kernels/test_blockscale_preshuffle_gemm.py \
  -M 16 -N 7168 -K 2304 --out_dtype bf16 --num_iters 50 --num_warmup 10
```

Tile 16x64x256 is auto-selected by `select_tile_config(M=16,N=7168,K=2304)`; pass `--tile_m/--tile_n/--tile_k` to override.

## Addendum — compute-bound re-test (2026-06-01)

Re-measured at a saturating **M=4096, N=7168, K=2304 fp8 blockscale** shape:

| impl | µs | TFLOPS |
|---|---:|---:|
| FlyDSL `bs_gemm` | 155.7 | 869.1 |
| AIter CK a8w8_blockscale_bpreshuffle (**tuned**) | 102.3 | 1322.0 |

→ **FlyDSL 0.66×** — the gap *widens* vs the M=16 result (0.88×) because AIter loaded a **per-shape tuned** blockscale config (dsv3/qwen tables) while FlyDSL runs a fixed untuned schedule. Part of this is a tuning-parity gap rather than pure codegen: add split-K + per-shape tile selection to close it.
