# Softmax (row-wise) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed (branch `docs/update-compile-pipeline`), gfx950 / AMD Instinct MI350X (CDNA4), ROCm 7.2.0, rocprofv3 1.1.0, captured 2026-06-01.
JIT kernel: `softmax_kernel_0` (ATT dispatch `ui_output_agent_47920_dispatch_13`, iter_range `[6, [8-8]]`).
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_softmax/`.

> Arch caveat: the analyzer prints "gfx942 (CDNA3)" — that is its arch-detection default, not the real target. This capture is **gfx950 / MI350X / CDNA4**; occupancy/VGPR reasoning below uses the gfx950 **combined** VGPR pool (256/SIMD), not the gfx942 split arch/accum model.

---

## Workload & headline

- **Shape captured:** 32768 rows × 8192 cols, **bf16** (default sweep config; `buffer_load_ushort` in the trace = 16-bit loads confirm bf16). One block per row (grid = M, block = 256 threads).
- **Latency (HIP-event sweep, not ATT):** FlyDSL **271.8 µs**, achieved **bandwidth 5246.7 GB/s** (read 2·M·N·2 B ≈ 1.07 GB moved / 271.8 µs). No TFLOPS — softmax is a bandwidth-bound reduction, not a FLOP kernel.
- **Head-to-head:** baseline **AIter** (`aiter.ops.triton.softmax`) = **558.0 µs**. FlyDSL is **2.05× faster**. Verdict: **FlyDSL wins.**

> Baseline note: `baselines.json` has **no `test_softmax.py` entry** (its only softmax row is `test_topk_gating_softmax.py`, a different fused MoE-gating kernel). The authoritative head-to-head for this kernel is therefore the manifest `sweep` block — AIter triton softmax at the matched 32768×8192 bf16 shape. The 5.25 TB/s effective bandwidth is a strong number; MI350X HBM3E peaks well above this, so there is still headroom, but FlyDSL is comfortably ahead of the triton reference here.

This kernel is a **win that is leaving performance on the table** — see §3.

---

## 1. Wave-state / stall breakdown

ATT samples one CU (representative wave-state, not device timing): 731 instructions, **41.65M total cycles, 31.80M stalled = 76.4%**. This is a heavily stall-bound kernel; the EUs spend three of every four cycles waiting.

| Stall type      |   Cycles | %    |
|-----------------|---------:|------|
| **VMEM-load**   |  14.60M  | 45.9 |
| VMEM-store      |   4.65M  | 14.6 |
| barrier         |   4.54M  | 14.3 |
| VMEM-wait       |   3.86M  | 12.1 |
| other           |   2.73M  |  8.6 |
| LDS/SMEM-wait   |   1.36M  |  4.3 |
| SMEM            |  29.0K   |  0.1 |
| MFMA/FMA        |  23.0K   |  0.1 |

**Bound type: stall-bound, specifically VMEM-load-bound.** VMEM-load alone is 45.9%; adding VMEM-store (14.6%) and VMEM-wait (12.1%) puts **~73% of all stall in the global-memory path**. Barrier (14.3%) is the secondary cost, from the block-reduce. MFMA is zero (no matmul) — this is a pure load → reduce → exp → reduce → store pipeline.

**Register pressure & occupancy (gfx950 combined pool):**
- `arch_vgpr ≈ 47` (alloc 48), `accum_vgpr = 0`. Limiting pool **48 / 256**.
- **Occupancy = 5 waves/SIMD.** Not VGPR-starved in absolute terms (48 of 256), but the allocation granularity caps it at 5.
- Analyzer's next step: **6 waves/SIMD requires VGPR ≤ 42**. Going from 5→6 waves is +20% latency-hiding slots — relevant because the kernel is latency-bound on loads (more waves = more loads in flight to cover HBM latency).

**Instruction mix:** `buffer_load: 32, buffer_store: 32, ds_read: 4, ds_write: 4, MFMA: 0`. The 4 ds_read/ds_write are the LDS hops of the two-level block-reduce; everything else is the global load/store stream.

---

## 2. Top instruction-level hotspots

All three real hotspots live in the kernel body's **generic scalar path** (`softmax_kernel.py`). Critical structural finding: the **fast vectorized path is dead code** — line 104 guards it with `if const_expr(False and N >= tile_cols ...)`, and `False and ...` is unconditionally false. So every shape, including this aligned 8192-wide bf16 row, falls through to the **scalar** generic branch.

**#1 — `softmax_kernel.py:186` — 14.75M / 46.4% / VMEM-load (the real bottleneck).**
```python
def _load_scalar(divided, index):
    view = fx.slice(divided, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom_s, view, r)   # <- line 186
    return fx.memref_load_vec(r)[0]
```
This is a **scalar `BufferCopy16b` load — one bf16 element (2 bytes) per lane per instruction.** The trace bears this out: the #1 instruction is `buffer_load_ushort v2, v1, s[12:15], 0 offen` at 8.47M cycles (26.6% alone), with a long tail of more `buffer_load_ushort`s. Each row of 8192 cols is loaded 256 threads × scalar steps, **2 B/lane/instruction** — 1/8 of the 16 B/lane a `dwordx4` would move. The kernel pays full HBM latency per element and never fills the VMCNT queue with wide transactions. This single load primitive is ~46% of all stall and the entire reason occupancy/latency-hiding matters here.

**#2 — `softmax_kernel.py:41` — 12.20M / 38.4% / barrier (block-reduce sync).**
Line 41 is the `@flyc.kernel def softmax_kernel(...)` definition line; the stall attributed here is the two-level **block reduction** machinery — `wave_reduce` (`shuffle_xor`), the LDS write/read of partial sums, and the two `gpu.barrier()` calls (lines 85, 97). The trace shows this as a blend of `s_barrier` (8.96% + 2.78% + 2.38%), `s_waitcnt vmcnt(N)` (the loads feeding the reduction must land before reducing), and `v_permlane32_swap_b32` (the wave shuffle). So §2-#2 is really "the cost of synchronizing the row reduction across 4 waves × 256 threads," and much of its `s_waitcnt` weight is **downstream of the same slow scalar loads** in #1 — the barriers stall because the loads they depend on haven't completed.

**#3 — `softmax_kernel.py:194` — 4.67M / 14.7% / VMEM-store (scalar store).**
```python
view = fx.slice(divided, (None, index))
fx.copy_atom_call(copy_atom_s, r, view)   # <- line 194
```
The normalize-and-store epilogue, again **scalar 16-bit** (`BufferCopy16b`). Same defect as #1 on the write side: one bf16 element per instruction instead of a vectorized store.

**Artifacts / non-bottlenecks:**
- `flydsl/expr/rocdl/universal.py:144` (151.6K, 0.48%, LDS/SMEM-wait) is the **source-loc-granularity collapse** (FlyDSL issue #587 / PR #593) — a FlyDSL intrinsic helper, not a real hot site. At 0.48% it is noise; ignore it.
- Lines 84/90/96/99 (LDS/other, all < 0.05%) are the block-reduce LDS slot writes/reads — negligible, consistent with only 4 ds_read/4 ds_write in the mix.

---

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Enable a real vectorized (128-bit) load/store path. *(addresses VMEM-load 45.9% + VMEM-store 14.6%)*
**Root cause:** §1's dominant bucket and §2-#1/#3. The kernel runs the scalar generic path doing **2 B/lane** `buffer_load_ushort` / `buffer_store_short`, because the fast path at line 104 is gated by a hard-`False` (`if const_expr(False and ...)`). For aligned shapes like 8192-wide bf16 (`8192 % (256·8) == 0`), the existing `_load_vec` / `_store_vec` (`BufferCopy128b`, VEC_WIDTH=8 → 16 B/lane) would move **8× the bytes per instruction**, collapsing both the instruction count and the number of VMCNT-queue entries.

**Concrete change:** drop the dead `False and` guard so the fast path activates on alignment:
```python
# softmax_kernel.py:104  — currently dead code
if const_expr(False and N >= tile_cols and N % tile_cols == 0):
# ->
if const_expr(N >= tile_cols and N % tile_cols == 0):
```
Verify the fast path compiles for bf16 (the `_load_vec`/`_store_vec` + `BufferCopy128b` machinery on lines 116-163 is already written for it). This is the single highest-leverage change: it directly attacks ~60% of total stall (VMEM-load + VMEM-store) and is the same lever AIter/CK use to saturate HBM.

**Wiki grounding:** `technique-vectorized-loads` (*Vectorized & Non-Temporal Loads (128-bit) to Saturate HBM*, archs gfx942/gfx950) — "the single most reliable lever is to issue 128-bit (16-byte) loads instead of scalar 32-bit loads … one instruction issue slot, one address calculation, one VMCNT entry." Here the scalar loads are even worse (16-bit), so the win is larger. The same technique covers the vectorized store. Implemented in `pr-aiter-2394`, `pr-triton-729`, `pr-composable_kernel-1430`.

**Expected gain:** large. The store side (14.7%) should largely collapse; the load side becomes ~8× fewer, wider transactions that the VMCNT queue can pipeline. A scalar→vectorized switch on a bandwidth-bound kernel typically moves it toward the HBM roofline; given 5.25 TB/s today there is real headroom. **Effort: low** (delete two words, validate the pre-written fast path).

### #2 — Add buffer OOB guards so the fast path covers non-tile-aligned N too. *(robustness for the above)*
**Root cause:** the fast path only fires when `N % tile_cols == 0` (tile_cols = 256·8 = 2048). Non-aligned rows still fall to scalar. Use branchless buffer out-of-bounds guards (`buffer_load_dwordx4` past end returns 0) instead of per-element masking, so the wide path handles the tail without reverting to scalar.

**Wiki grounding:** `technique-buffer-oob-guard` — composes directly with `technique-vectorized-loads` ("a `buffer_load_dwordx4` past the end returns 0"). **Effort: medium.** **Gain:** extends the #1 win to arbitrary N; no benefit for this already-aligned 8192 capture, but removes the cliff.

### #3 — Push occupancy 5 → 6 waves/SIMD by trimming VGPRs. *(more loads in flight to hide HBM latency)*
**Root cause:** §1 — `arch_vgpr ≈ 48`, occupancy 5, and the analyzer says **6 waves needs VGPR ≤ 42**. The kernel **register-buffers the entire row across three passes** (max / exp+sum / normalize), holding the whole row resident — that is what pins VGPRs. With a latency-bound load stream, more concurrent waves is the classic latency-hiding lever.

**Concrete change:** after #1 lands, re-profile; if still load-latency-bound, reduce live VGPRs by not buffering the full row (recompute or stream a smaller working set), or shrink VEC_WIDTH, to drop arch_vgpr to ≤ 42 and unlock the 6th wave. Note this trades against ILP from the buffered row — measure both.

**Wiki grounding:** `technique-occupancy-tuning` (*Waves per SIMD vs ILP on CDNA*) and `technique-vgpr-budgeting`. **Effort: medium.** **Gain:** modest (+1 wave ≈ +20% latency-hiding slots), and only matters if the kernel is still latency- rather than throughput-bound after #1. Do this *after* #1, since vectorizing changes the VGPR picture.

> Why #1 first: it is the only change that touches the **45.9% top bucket** directly, it is nearly free to apply, and it likely makes #3 moot (wider loads need fewer in-flight waves to saturate HBM). #2 and #3 are follow-ups, not the headline fix.

---

## 4. Re-run

Captured via the standard ATT driver:

```bash
/opt/venv/bin/python /sgl-workspace/flydsl-prof/drivers/att_capture.py \
  --test /sgl-workspace/FlyDSL-lab/tests/kernels/test_softmax.py \
  --gpu 7 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_softmax \
  --tag big \
  --iter-range "[6, [8-8]]" \
  --cmd-override "python tests/kernels/test_softmax.py"
```

Shape is the harness default config in `tests/kernels/test_softmax.py` (line 148): `(32768, 8192, "bf16")`. Override with `ROCDSL_SOFTMAX_SHAPES="32768,8192,bf16"` if needed. Hotspot re-analysis: `hotspot_analyzer.py` on `results/att/test_softmax/att/` → `hotspot_big.txt`.
