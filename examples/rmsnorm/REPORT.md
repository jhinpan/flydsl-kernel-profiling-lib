# RMSNorm (bf16) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X / CDNA4 · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `rmsnorm_kernel_0` (dispatch `ui_output_agent_19276_dispatch_18`, one CU sampled)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_rmsnorm/`

> Arch caveat: the analyzer header prints `gfx942 (CDNA3)` — that is its arch-detection default, not the real target. This was captured on **gfx950 / MI350X / CDNA4**. All occupancy reasoning below uses the gfx950 **combined VGPR pool** (256 regs/SIMD, no split arch/accum), not the gfx942 split model.

## Workload & headline

- Shape: **M=32768 rows × N=8192**, bf16, fast path (`N % (256·8)=0`, so `num_tiles=4`, one row per block, 256 threads).
- Latency (HIP-event sweep, not ATT): FlyDSL **25.1 µs**; baseline **AIter `rms_norm` 22.4 µs**.
- Speedup vs baseline: **0.892× — FlyDSL is slower** (AIter wins by ~2.7 µs / ~12%).
- Effective bandwidth: **4729.73 GB/s** (~59% of MI350X HBM3E peak ~8 TB/s). No TFLOPS — RMSNorm is a memory-bound reduction, not a GEMM; no rocBLAS/CK GEMM baseline exists.

Verdict: this is a **bandwidth-bound** kernel that is *not yet bandwidth-saturated*. The ATT wave-state says the bottleneck is **VMEM-wait** (waiting on `buffer_load` data to land), amplified by a **two-pass load** structure and a **barrier-gated block reduction**. AIter beats us because it hides load latency better at this occupancy; closing ~12% is realistic.

## 1. Wave-state / stall breakdown

337 instructions, 331.9K total cycles, **239.4K (72.1%) spent stalled**. The kernel does essentially zero compute (MFMA: 0); every cycle is memory or sync.

| Stall type      | Cycles | % of stalls | Meaning |
|-----------------|--------|-------------|---------|
| **VMEM-wait**   | 98.4K  | **41.1%**   | `s_waitcnt vmcnt(N)` — wave parked waiting for `buffer_load` data |
| LDS/SMEM-wait   | 48.1K  | 20.1%       | `s_waitcnt lgkmcnt(N)` — waiting on LDS reads (cross-wave reduction) |
| other           | 37.2K  | 15.6%       | mostly `v_permlane*_swap` (the in-wave shuffle reduction) |
| barrier         | 32.5K  | 13.6%       | `s_barrier` between reduction phases |
| VMEM-load       | 22.2K  | 9.3%        | the `buffer_load_dwordx4` issue itself |
| VMEM-store      | 796    | 0.3%        | output writes (cheap) |
| LDS / SMEM      | 88     | ~0%         | the actual `ds_read`/`ds_write` ops |

**#1 stall is VMEM-wait at 41.1%.** Combined with VMEM-load (9.3%) and the store, raw memory traffic is ~51% of all stalls; the LDS-wait + barrier reduction machinery is another ~34%. So the kernel is **bandwidth-/latency-bound on the input loads, with a non-trivial reduction-sync tax on top.**

Instruction mix per wave: `buffer_load: 8, buffer_store: 4, ds_read: 2, ds_write: 2, MFMA: 0`. The 8 loads = 4 tiles × (input pass-1 + gamma pass-2); 4 stores = 4 output tiles. The 2+2 LDS ops are the block-reduce scratch.

Register / occupancy:
- `arch_vgpr ≈ 60` (alloc 64), `accum_vgpr 0`. On gfx950's combined pool the 64-VGPR allocation is what gates occupancy.
- Occupancy: **4 waves/SIMD** (limiting pool 64/256).
- Next step: **5 waves/SIMD requires VGPR ≤ 51**. We are 13 VGPRs over the threshold for the next occupancy tier.

The VGPR pressure comes directly from **caching all 4 input tiles in registers across the reduction** (`in_local[]`, lines 201–224): 4 tiles × 8 bf16 lanes held live through the barrier. That is what pins us at 60 VGPRs / 4 waves and limits the GPU's ability to hide the very VMEM-wait that dominates §1.

## 2. Top instruction-level hotspots

| # | Stall | %tot | DomType | Source |
|---|-------|------|---------|--------|
| 1 | 179.6K | 75.0% | VMEM-wait | `rmsnorm_kernel.py:115` |
| 2 | 32.8K | 13.7% | LDS/SMEM-wait | `flydsl/expr/rocdl/universal.py:144` |
| 3 | 26.1K | 10.9% | VMEM-load | `rmsnorm_kernel.py:59` |
| 4 | 892 | 0.4% | VMEM-store | `rmsnorm_kernel.py:66` |

**Line 115 — `@flyc.kernel def rmsnorm_kernel(...)` (75%, VMEM-wait).** This is a **source-loc collapse**: the decorated function-definition line absorbs every instruction whose debug-loc points at the kernel body broadly — the `s_waitcnt vmcnt(0)` barriers between the two load passes, the `s_barrier` of the block reduction, and the `v_permlane32/16_swap_b32` in-wave shuffles all map here (see the per-instruction table: rows 1–6, 9, 11, 12, 14, 15 are all `:115`). The real story line 115 tells: the wave repeatedly issues `s_waitcnt vmcnt(0)` and stalls because **the reduction can't proceed until all 4 input tiles have landed** — pass 1 loads all four tiles, then `block_reduce_add2` forces a full drain (`vmcnt(0)`) before the sum-of-squares is correct. That serial drain is the dominant cost.

**Line 59 — `fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)` inside `_load_vec` (10.9%, VMEM-load).** This is the actual `buffer_load_dwordx4 v[..], v.., s[4:7], 0 offen` — the 128-bit vectorized input/gamma loads. This is the genuine memory-issue hotspot and is exactly where the bytes come from. The good news: the loads are already 128-bit wide (`BufferCopy128b`), so per-instruction issue is efficient; the cost is *latency*, not narrow access.

**Line 66 — `fx.copy_atom_call(... r, fx.slice(div_tensor, (None, idx)))` inside `_store_vec` (0.4%, VMEM-store).** Output writes, cheaply absorbed into the tail — stores are fire-and-forget here.

**Line 144 — `make_ptr(...)` in `flydsl/expr/rocdl/universal.py` (13.7%, LDS/SMEM-wait).** This is **not** the buffer-resource construction it textually appears to be — it is the same **source-loc-granularity collapse** seen in rope/layernorm (FlyDSL issue #587 / PR #593). The `s_waitcnt lgkmcnt(0)` of the cross-wave LDS reduction read (`block_reduce_add2`, lines 165–177: `memref_load(s_red, ...)`) folds back onto the intrinsic helper's loc. Read it as **"LDS reduction-read wait,"** attributable to `block_reduce_add2`, not to buffer setup.

## 3. Optimisation recommendations (RANKED by expected impact)

### #1 — Software-pipeline pass-1 loads to hide VMEM-wait (root cause: 41.1% VMEM-wait + the `vmcnt(0)` drain at :115)
The kernel loads all 4 tiles, then does **one** `s_waitcnt vmcnt(0)` drain before the reduction (`block_reduce_add2`, line 214). With only 4 waves/SIMD the scheduler can't hide that full-drain latency. Instead, **interleave the partial sum-of-squares with the loads using staged `vmcnt(N)` gates** so each tile's reduction starts as soon as *that* tile lands, rather than waiting for all four:
- Issue all 4 `buffer_load`s up front (already done), then consume tile *i* gated on `vmcnt(num_tiles-1-i)` (descending counter), accumulating `thread_sumsq` per tile. The per-instruction table already shows a `vmcnt(3)` partial wait (row 11) — push that pattern to cover all tiles instead of collapsing to `vmcnt(0)`.
- This is precisely **`technique-vectorized-loads`** (id `technique-vectorized-loads`, archs incl. gfx950): the wiki notes each 128-bit `buffer_load` is "one entry in the VMCNT queue … 1 KiB per instruction," and that saturating HBM requires keeping multiple loads in flight and gating consumers on partial `vmcnt` rather than a single full drain. The same `s_waitcnt vmcnt(N)` counter-gating idea underpins **`technique-lds-double-buffering`** (id `technique-lds-double-buffering`).
- Expected gain: this is the 41.1% bucket; even halving the drain stall recovers a few µs and should erase the 0.89× gap to AIter. **Medium effort** (reorder pass-1 to per-tile gated accumulation; FlyDSL exposes the copy/`vmcnt` ordering, so it's a kernel-body change, not a compiler change).

### #2 — Cut VGPR pressure to reach 5 waves/SIMD (root cause: occupancy 4 → can't hide latency)
We're at 60 VGPR / 4 waves; **5 waves needs VGPR ≤ 51** (13 over). The pressure is `in_local[]` caching all 4 input tiles in registers across the barrier (lines 201–224). Options:
- Don't cache: re-load the input in pass 2 instead of holding 4×8 bf16 lanes live across the reduction. Trades 4 extra `buffer_load`s for ~quartered live VGPR footprint. Because the kernel is bandwidth-bound *and* occupancy-starved, the extra loads may pay for themselves via better latency hiding at 5 waves — **measure both ways**.
- Or cache in **packed bf16** (keep `in_local` as the 16-bit `elem_dtype`, convert to f32 only at use in pass 2) instead of widening to f32 early — halves the cached footprint without re-loading.
- Expected gain: +1 wave/SIMD ≈ 25% more in-flight latency hiding; compounds with #1. **Low–medium effort** (the packed-cache variant is a few lines: defer the `.to(fx.Float32)` in lines 207/224).

### #3 — Confirm the reduction barrier path; reduce LDS-wait + barrier (root cause: 20.1% LDS-wait + 13.6% barrier = 33.7%)
`block_reduce_add2` does wave-shuffle → LDS store → `s_barrier` → wave-0 re-reduce → `s_barrier` → broadcast-load. With `BLOCK_THREADS=256` and WARP_SIZE 64, `RED_SLOTS=4`, so the LDS round-trip and two barriers are real. The `v_permlane32/16_swap_b32` shuffles (rows 12, 14 at :115) are the in-wave tree. This is a single reduction per block; the cost is the **serialization point**, which mostly disappears once #1 overlaps it with loads. If after #1 the barrier bucket persists, consider one fewer barrier (the final broadcast can use a shuffle from lane 0 instead of an LDS load+barrier). **Low effort, lower priority** — overlap (#1) is the higher-leverage fix.

### #4 — Reporting/attribution: fix source-loc collapse (not a perf fix)
75% of stalls collapse onto the `@flyc.kernel` decorator line (:115) and 13.7% onto `universal.py:144`. This is **issue #587 / PR #593** — the debug-loc granularity hides which *body* line stalls. Landing #593 would let the next capture attribute the `vmcnt(0)` drain vs. the reduction barrier separately, making #1 vs #3 trivially decidable from the trace instead of from source reading. **No runtime gain**, but it's the difference between guessing and measuring on the next pass.

## 4. Re-run

```bash
# Shape is baked into the harness active config (M=32768, N=8192, bf16).
python /sgl-workspace/flydsl-prof/drivers/att_capture.py \
  --test test_rmsnorm.py \
  --gpu 1 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_rmsnorm \
  --tag big \
  --iter-range "[6, [8-8]]" \
  --target-cu 1

# hotspot view:
#   cat /sgl-workspace/flydsl-prof/results/att/test_rmsnorm/hotspot_big.txt
# to change shape: export ROCDSL_RMSNORM_SHAPES="32768x8192:bf16" before the capture
```
