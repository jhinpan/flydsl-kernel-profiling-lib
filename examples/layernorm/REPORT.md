# LayerNorm (fast vectorized path) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X (CDNA4) · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `layernorm_kernel_0` (device `@flyc.kernel layernorm_kernel`, ATT dispatch `ui_output_agent_22257_dispatch_23`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_layernorm/`

> Arch note: the analyzer prints `gfx942 (CDNA3)` — that is its hard-coded arch-detection default, not the real target. This capture ran on **gfx950 / MI350X / CDNA4** (160 KB LDS, combined VGPR pool, `v_permlane16_b32`). All occupancy/VGPR reasoning below uses the gfx950 model. This caveat applies once and is not repeated.

## Workload & headline

- **Shape captured:** `M=256, N=8192, bf16` (recipe shape; one block/row, 256 threads/block, grid = 256). N = 256·8·4 = 8192 selects the **fast vectorized path** (128-bit `BufferCopy`, register caching, pipelined γ/β). Confirmed by inst-mix: `mfma=0, buffer_load=12, buffer_store=4, ds_read=3, ds_write=2`.
- **Latency (sweep / HIP-event):** FlyDSL **24.1 µs**, bandwidth **1490 GB/s**. (TFLOPS N/A — LayerNorm is not a FLOP-bound op.)
- **Head-to-head:** baseline **AIter** (`aiter.ops.triton.norm.layer_norm`) at **24.7 µs** → **1.025× (FlyDSL marginally ahead)**, verdict **comparable**.
  - There is no `test_layernorm.py` entry in `baselines.json`; the manifest sweep is the only same-shape number available. Note the sweep timing was taken at the harness's own large shape (`32768×8192 bf16`), while the **ATT trace was captured at `256×8192`**. The wave-state breakdown is therefore representative of per-row behavior, not of the 24.1 µs end-to-end figure.

**Verdict:** Neck-and-neck with AIter Triton. At ~1490 GB/s this kernel is delivering a healthy fraction of MI350X HBM3e bandwidth, but it is **not bandwidth-bound on the sampled CU** — it is **stall-bound on LDS/SMEM waits (58%)**, and those waits come from the block-reduction tree, not from memory. Closing the gap on AIter means killing the `ds_*` cross-lane shuffles, not touching the copy path.

## 1. Wave-state / stall breakdown

ATT sampled one CU (`att_target_cu=1`), 16 waves, 456/457 instructions mapped (99.8%). 76.2K total cycles, 46.0K stalled = **60.3% of cycles are stall**.

| Stall type    | Cycles | %    |
|---------------|--------|------|
| **LDS/SMEM-wait** | **26.8K** | **58.2** |
| VMEM-wait     | 12.8K  | 27.9 |
| other         | 4.0K   | 8.6  |
| VMEM-load     | 952    | 2.1  |
| barrier       | 824    | 1.8  |
| VMEM-store    | 644    | 1.4  |

**#1 stall is LDS/SMEM-wait (58.2%)** — `s_waitcnt lgkmcnt(0)`. This kernel is **stall-bound, and specifically LDS-wait-bound.** VMEM-wait is a clear but secondary second (27.9%); actual memory *issue* stalls (VMEM-load + VMEM-store) are negligible (3.5% combined), and `mfma=0` confirms there is zero matrix work. So the cycles are not spent moving the 8192-element row in/out — they are spent waiting on the cross-lane reduction.

**Register pressure & occupancy:**
- `arch_vgpr ≈ 69` (alloc 72), `accum_vgpr = 0`. On gfx950's **combined** VGPR pool the 72-VGPR allocation is the single limiter (72 / 256).
- **Occupancy = 3 waves/SIMD.** Analyzer: *"4 waves requires max(arch,accum) ≤ 64"* — i.e. shave the allocation from 72 down to **≤ 64 VGPR** to step from 3→4 waves/SIMD.
- 69 VGPR is plausible for this body: 4 tiles × 8-wide bf16 vectors held live in registers (`in_local[]` register caching) plus the f32 accumulators and pipelined γ/β. The register caching that helps the load path is exactly what holds VGPR at 72.

## 2. Top instruction-level hotspots

| # | Stall | %tot | DomType | Source |
|---|-------|------|---------|--------|
| 1 | 22.8K | 49.6 | LDS/SMEM-wait | `flydsl/expr/rocdl/universal.py:144` |
| 2 | 21.4K | 46.6 | VMEM-wait | `kernels/layernorm_kernel.py:50` |
| 3 | 1.1K | 2.4 | VMEM-load | `kernels/layernorm_kernel.py:151` |
| 4 | 644 | 1.4 | VMEM-store | `kernels/layernorm_kernel.py:157` |

**Line 1 — `universal.py:144` (49.6%, LDS/SMEM-wait) — source-loc-collapse artifact, real stall.** `universal.py:144` is inside the FlyDSL `make_buffer_tensor` intrinsic helper (the `make_view`/buffer-descriptor builder), **not** the LDS reduction. This is the source-loc-granularity collapse documented in **FlyDSL #587 / PR #593**: debug line tables fold the `s_waitcnt lgkmcnt(0)` instructions onto the intrinsic helper's line instead of the kernel-body line that issued the dependent `ds_*` op. The instruction-level table makes the truth unambiguous — the top two single instructions are both `s_waitcnt lgkmcnt(0)` at **12.3K + 10.5K = 22.8K cycles (49.6%)**. So the *helper line is an attribution artifact, but the LDS wait itself is the genuine #1 bottleneck.* It is the wave/block-reduction waiting on the LDS crossbar.

  Root cause in the kernel body: `wave_reduce_add` (lines 69-75) does `log2(64)=6` iterations of `w.shuffle_xor(off, WARP_SIZE)`, and `block_reduce_add2` runs it on **two** values (sum and sum-of-squares) — plus a second reduction stage over `RED_SLOTS=4` slots through `s_sum`/`s_sumsq` LDS arrays with `gpu.barrier()` between. `shuffle_xor` lowers to `gpu.ShuffleOp(mode="xor")` → on CDNA, the `ds_bpermute`/`ds_swizzle` LDS crossbar. Each such op needs `s_waitcnt lgkmcnt(0)` before its result is consumed by the next `addf`, and the tree is a strict dependent chain (step k+1 needs step k). That dependent `ds_* → lgkmcnt(0) → addf` chain, ×6 steps ×2 values, with no independent work to hide behind it, is exactly the 58% LDS-wait bucket.

**Line 2 — `layernorm_kernel.py:50` (46.6%, VMEM-wait) — the kernel `@flyc.kernel` decorator line; this is the whole kernel body.** Line 50 is the decorator above `def layernorm_kernel`, so DWARF attributes everything that didn't get a finer line here. The instruction table shows what it really is: a ladder of `s_waitcnt vmcnt(N)` — `vmcnt(4)` 5.4K, `vmcnt(1)` 2.8K, `vmcnt(3)` 2.7K, `vmcnt(0)` 0.7K — i.e. **Pass-1 waiting for the 4 outstanding 128-bit `buffer_load`s to land before it can square-and-reduce.** The vmcnt ladder (4→3→1→0) is the compiler draining the load queue one tile at a time. These are real, but they are the *memory side* and are inherently overlappable; they rank below the LDS chain.

**Lines 3-4 — `layernorm_kernel.py:151 / :157` (2.4% / 1.4%).** `:151` is `fx.copy_atom_call(...)` inside `_load_vec` (the 128-bit vectorized load); `:157` is the `copy_atom_call` store inside `_store_vec`. These are the actual memory-issue points and they barely stall (3.8% combined) — the vectorized 128-bit BufferCopy path is doing its job; HBM is not the ceiling on this CU.

**Bottom line:** the dominant cost is the **dependent cross-lane reduction chain** (`ds_*` + `lgkmcnt(0)`), surfaced as 58% LDS-wait but mis-attributed by line table to the buffer-tensor helper. The copy path is already efficient.

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Replace the `shuffle_xor` reduction tree with `v_permlane16_b32` + DPP (kill the LDS crossbar)
**Root cause:** §1 #1 stall (LDS/SMEM-wait, 58%) = dependent `ds_bpermute`/`ds_swizzle` ops from `shuffle_xor` in `wave_reduce_add`, each gated by `s_waitcnt lgkmcnt(0)`.
**Change:** the kernel reduces over wave64 via `gpu.ShuffleOp(mode="xor")`, which routes every cross-lane step through the LDS crossbar. On **gfx950 (CDNA4)** the cross-16-lane-row steps can use **`v_permlane16_b32`** (pure VALU, no LDS, no `lgkmcnt`), and the intra-row steps can use **DPP** (`v_mov_b32_dpp`, also VALU). Per ROCmKernelWiki **`technique-wave-reduce`**: *"gfx950 adds `v_permlane16_b32` which replaces the `ds_*` cross-row step with a pure-ALU op."* Build the wave64 sum as DPP rows (steps 1-4) + `permlane16` cross-row (steps 5-6) instead of six `shuffle_xor`s. FlyDSL already exposes `permlane16_swap` and `ds_bpermute` in `flydsl/expr/rocdl.py` — wire `wave_reduce_add` to a permlane/DPP path under the existing `arch.startswith("gfx95")` guard (the kernel already branches on that for `cvt_pk_bf16`).
**Expected gain:** moves the bulk of the 58% LDS-wait bucket off the LDS path. The reductions become a VALU-only dependent chain (cheaper per step, no `lgkmcnt` drain). Realistic ceiling is the 27.9% VMEM-wait + residual reduction latency; a meaningful chunk of the 60% total-stall could be recovered. This is the single highest-leverage change and it directly targets the #1 bucket.
**Effort:** Medium — localized to the `wave_reduce_add`/`block_reduce_add2` helpers; gfx950-gated; correctness checkable against the existing torch ref in `test_layernorm.py`.
**Grounding:** ROCmKernelWiki `technique-wave-reduce` (gfx950 `v_permlane16` path; impls `pr-FlyDSL-447`, `pr-FlyDSL-524`, `pr-FlyDSL-450`, `pr-FlyDSL-300`); `kernel_types` explicitly lists `layernorm`.

### #2 — Single fused two-value reduction instead of two independent trees
**Root cause:** same LDS-wait bucket; `block_reduce_add2` runs `wave_reduce_add` **twice** (sum, sumsq), doubling the dependent `ds_*`/`lgkmcnt` chain length.
**Change:** pack `(sum, sumsq)` into one cross-lane reduction pass so the two trees share the same shuffle/permlane steps and interleave their `addf`s — giving the scheduler independent ALU work to hide the cross-lane latency, instead of two serial chains. Combine with #1 (permlane the packed pair).
**Expected gain:** roughly halves the number of serialized cross-lane round-trips in the reduction; meaningful on top of #1 since the trees are currently back-to-back dependent.
**Effort:** Low-Medium — restructure `block_reduce_add2`; no new intrinsics if layered on #1.

### #3 — Cut VGPR ≤ 64 to step occupancy 3 → 4 waves/SIMD
**Root cause:** §1 — `arch_vgpr ≈72` caps occupancy at 3 waves/SIMD; the analyzer states 4 waves needs `max(arch,accum) ≤ 64`.
**Change:** the 4-tile `in_local[]` register cache holds all four 8-wide bf16 input vectors live across Pass-1→Pass-2. Reloading the input in Pass-2 (or caching only 2 tiles) trades a few extra `buffer_load`s — which are *cheap* here (VMEM-load stall is only 2.1%) — for lower VGPR and a 4th wave. A 4th wave gives more in-flight waves to hide both the VMEM-wait (27.9%) and any residual reduction latency.
**Expected gain:** modest; higher occupancy helps latency hiding, but only after #1/#2 shorten the dependent chain (more waves do little if every wave is serially stalled on its own `lgkmcnt`). Sequence this *after* #1.
**Effort:** Low — drop/shrink the `in_local` cache; re-check VGPR via the build dump.
**Grounding:** ROCmKernelWiki `technique-occupancy-tuning` (waves/SIMD vs ILP on CDNA), `technique-vectorized-loads` (read-once input → stream, don't pin in registers).

### #4 — (Verify) γ/β software pipeline is actually overlapping
**Root cause:** Pass-2 prefetches `g_next`/`b_next` (lines 180-188) to hide γ/β load latency behind the normalize math. With the reduction chain dominating, this overlap may already be hidden — confirm it isn't adding to the VGPR pressure in §3 for no benefit. Low priority; measurement-gated.

## 4. Re-run

ATT capture (exact, via the bundle's driver; captured shape `256×8192 bf16`, target CU 1, iter range `[6,[8-8]]`):

```bash
/opt/venv/bin/python /sgl-workspace/flydsl-prof/drivers/att_capture.py \
  --test test_layernorm \
  --gpu 0 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_layernorm \
  --tag big \
  --iter-range "[6,[8-8]]" \
  --target-cu 1
```

Underlying invocation (from `/sgl-workspace/FlyDSL-lab`, the recipe the driver runs):

```bash
HIP_VISIBLE_DEVICES=0 ROCDSL_LAYERNORM_SHAPES="256,8192,bf16" \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
/opt/venv/bin/python tests/kernels/test_layernorm.py
```
