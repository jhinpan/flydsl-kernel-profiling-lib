# Vector Add (C = A + B, FP32) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed (branch `docs/update-compile-pipeline`), gfx950 / MI350X (CDNA4), ROCm 7.2.0, rocprofv3 1.1.0, captured 2026-06-01.
JIT kernel: `vecAddKernel_0` (dispatch_25, sampled on one CU). Bundle: `/sgl-workspace/flydsl-prof/results/att/test_vec_add/`.

> Arch caveat: the analyzer prints `gfx942 (CDNA3)` for this kernel — that is its arch-detection default, not a real read. The capture ran on **gfx950 / MI350X (CDNA4)**; all occupancy/VGPR reasoning below uses the gfx950 **combined VGPR pool** (no split arch/accum pool as on gfx942).

## Workload & headline

A textbook streaming triad: `C = A + B` over `SIZE = 256 × 4 × 10000 = 10.24M` FP32 elements (~41 MB per array, **~123 MB total traffic** = 2 reads + 1 write). One block per 1024-element tile, 256 threads/block, `vec_width=4` → each thread moves one 128-bit `float4`. No compute worth the name (one `v_add_f32` per lane), no LDS, no MFMA.

- **Achieved bandwidth: 6468 GB/s** (sweep `bandwidth_gbs`; the `tbps=6710.8` field is a stale/unit-mismatched duplicate — trust the GB/s figure). That is **~81% of the MI355X HBM3E ~8 TB/s ceiling** — right at the boundary the wiki calls the line between bandwidth-bound and instruction-bound.
- TFLOPS: N/A (one flop per 12 bytes; this is a pure bandwidth op).
- **Baseline head-to-head: not recorded.** There is no `test_vec_add.py` entry in `baselines.json`, and the sweep left `flydsl_us` / `baseline_us` / `speedup` null. The harness itself *does* time `torch.add` at the same size (`benchmark_pytorch_add`, line 89) and prints a BW ratio, but that number was not captured into the bundle.

**Verdict: bandwidth-bound and already near the roofline.** At ~81% of HBM3E peak with occupancy pinned at the hardware max (8 waves/SIMD), there is no stall bucket left to attack with a code change — the kernel is doing the right thing (128-bit buffer loads, max occupancy, trivial register footprint). The only honest "win" left is the last ~19% of roofline, which is dominated by grid/tail and L2-policy effects, not by anything visible in the kernel body. The missing baseline is the real gap in this bundle: get the `torch.add` ratio recorded.

## 1. Wave-state / stall breakdown

22 instructions mapped (95.7%), 3.73M total cycles on the sampled CU, **2.10M stalled (56.3%)**.

| Stall type     | Cycles | %     |
|----------------|--------|-------|
| **VMEM-wait**  | 1.73M  | **82.3%** |
| VMEM-load      | 178.1K | 8.5%  |
| VMEM-store     | 87.1K  | 4.1%  |
| LDS/SMEM-wait  | 69.7K  | 3.3%  |
| SMEM           | 29.5K  | 1.4%  |
| other          | 8.5K   | 0.4%  |

**#1 is VMEM-wait at 82.3%** — a single `s_waitcnt vmcnt(0)` that gates the two `buffer_load_dwordx4` operands before the add. This is the canonical signature of a **bandwidth-bound, latency-exposed** kernel: the wave issues both 128-bit loads, then sits on the counter until HBM returns the bytes. That is *expected* and *desirable* here — the stall is HBM round-trip latency, not a code defect. There is no LDS, no barrier, no MFMA contention to speak of (LDS/SMEM-wait 3.3% is the `s_waitcnt lgkmcnt(0)` for the scalar descriptor load, an artifact of `make_buffer_tensor`, not the data path).

**Register pressure & occupancy — already maxed.**
- `arch_vgpr ≈ 9` (allocated 16), `accum_vgpr = 0`. On gfx950's combined VGPR pool, 16 VGPRs is the granule floor.
- **Occupancy: 8 waves/SIMD** — the hardware ceiling on CDNA is 10 waves/SIMD, but the SGPR/scheduling floor effectively pins streaming kernels like this at 8. The limiting pool is `16 / 256`, i.e. VGPRs aren't the constraint at all; you have 240 VGPRs of headroom.
- 632 waves total across the grid (10000 blocks → plenty to saturate all CUs/XCDs).
- There is no `next_occ_step` in the manifest because **occupancy is already at the practical max** — you cannot buy latency-hiding with more waves here.

So: this is **bandwidth-bound**, latency-exposed by design, with occupancy and register footprint already optimal. The 56.3% stall figure is not a problem to fix; it is HBM latency being hidden by 8 resident waves, and the achieved 6.47 TB/s confirms the hiding is mostly working.

## 2. Top instruction-level hotspots

| # | %Total | DomType | Source | What runs there |
|---|--------|---------|--------|-----------------|
| 1 | 83.67% | VMEM-wait | `test_vec_add.py:28` | `s_waitcnt vmcnt(0)` — waits for both operand loads |
| 2 | 7.41%  | VMEM-load | `test_vec_add.py:64` | `buffer_load_dwordx4 v[0:3]` — load A (128-bit) |
| 3 | 4.14%  | VMEM-store | `test_vec_add.py:70` | `buffer_store_dwordx4 v[0:3]` — store C (128-bit) |
| 4 | 3.59%  | LDS/SMEM-wait | `flydsl/expr/rocdl/universal.py:144` | `s_waitcnt lgkmcnt(0)` — scalar descriptor setup |
| 5 | 1.19%  | VMEM-load | `test_vec_add.py:65` | `buffer_load_dwordx4 v[4:7]` — load B (128-bit) |

- **Line 28** is the `@flyc.kernel` decorator line — the source-locus for the kernel prologue/`s_waitcnt`. The 1.76M cycles parked here are the HBM read latency for A and B, surfaced at the wait that consumes them. This is the real cost, and it is intrinsic to the op: ~123 MB at ~6.5 TB/s is the whole runtime.
- **Lines 64 / 65 / 70** are exactly what you want to see: both operand loads and the store are **`*_dwordx4` (128-bit, `BufferCopy128b`)**. The kernel already follows wiki rule #1 — 16 B/lane, one VMCNT entry per load, 1 KiB/wave per instruction. No de-vectorization, no scalar fallback. The `copyAtom = make_copy_atom(BufferCopy128b(), Float32)` (line 58) plus `vec_width=4` is the correct setup.
- **Line 4 (`universal.py:144`)** is a **source-loc-granularity collapse artifact** (FlyDSL issue #587 / PR #593): the hot line maps into the `rocdl` intrinsic helper, not the kernel body. At 3.59% it is the scalar `s_load_dwordx2` / `lgkmcnt` of buffer-descriptor setup from `make_buffer_tensor`, a one-time prologue cost — do not treat it as a real bottleneck.

The instruction mix is the whole story: **2 buffer_load + 1 buffer_store, 0 LDS, 0 MFMA**. There is nothing in the body to optimize; the kernel is a clean 128-bit streaming triad.

## 3. Optimisation recommendations (ranked by expected impact)

This kernel is at ~81% of the HBM3E roofline with max occupancy and optimal vectorization. The honest assessment is that **the body is already correct** and there is little headroom. The recommendations below are ranked by what could realistically close the last ~19% — none are large, and #1 is the structural lever.

### 1. Convert to a persistent grid-stride kernel to recover the tail and amortize launch — *root cause: VMEM-wait latency exposed across 10000 short-lived blocks; tail + per-block setup leave the last wave under-occupied.*
Right now the launch is one block per 1024-element tile → **10000 blocks**. The dispatcher streams these onto ~256 CUs, so the final partial wave of blocks under-fills the machine (the tail effect), and every block re-pays V# descriptor setup (the `universal.py:144` / SMEM cost, ~1.4% SMEM + 3.3% lgkmcnt). A **persistent kernel** launches `num_cu × wgs_per_cu` resident blocks and grid-strides over tiles with a static, branchless `for (tile = blockIdx.x; tile < num_tiles; tile += grid_size)` loop. Each block then keeps several wide loads in flight across tile iterations (wiki rule #2: issue several `dwordx4`, one `s_waitcnt vmcnt(N)`), turning the per-block latency exposure into a steady-state pipeline that amortizes over many tiles.
- **Technique:** `technique-persistent-kernel` (grid-stride, static stride preferred over atomic counter for a memory-bound op — the wiki explicitly warns atomic counters serialize and cost more than the tail they remove). Pairs with `technique-vectorized-loads` (keep the VMCNT queue populated by unrolling the inner loop 2–4×).
- **Expected gain:** small — maybe a few % toward roofline by removing tail underutilization and amortizing descriptor setup. Not a 2× lever; this op is fundamentally HBM-bound.
- **Effort:** medium — rewrite the launch wrapper (`vecAdd`, line 73) to size the grid to `multiProcessorCount × wgs_per_cu` and add a grid-stride loop in `vecAddKernel`. Verify the ISA still shows clustered `*_dwordx4` before a single `s_waitcnt`.

### 2. Add non-temporal (streaming) hints to the read-once loads/store — *root cause: A, B, C are each touched exactly once; default L2-residency policy can evict useful lines for no reuse benefit.*
This is a pure single-pass triad — no operand is re-read. The wiki's vectorized-loads checklist (rule #6) says: tag read-once data non-temporal so it bypasses L2 replacement policy and does not pollute the cache. On the FlyDSL side this means a non-temporal / streaming variant of the buffer copy atom (sets the `slc`/non-temporal bit on `buffer_load_dwordx4` / `buffer_store_dwordx4`).
- **Technique:** `technique-vectorized-loads` (non-temporal section).
- **Expected gain:** marginal at this size (123 MB streams through L2 once regardless); the benefit is mostly avoiding cache pollution for a *co-running* workload, not this kernel in isolation. Measure `L2CacheHit` / `FETCH_SIZE` to confirm before/after.
- **Effort:** low-medium — depends on whether FlyDSL exposes a non-temporal `BufferCopy` atom; if not, this is a DSL feature ask, not a kernel edit.

### 3. Record the `torch.add` baseline into the bundle — *not an optimization, but the missing data point.*
The harness already times `torch.add` at the identical size and prints a BW ratio (`benchmark_pytorch_add`, line 89), but the sweep/baseline files captured nothing. Without it there is no head-to-head verdict for this kernel. Run the harness and log the ratio so the dashboard has an authoritative comparison (FlyDSL vs torch.add at the same shape, same `run_perftest` harness).
- **Effort:** trivial — run the command in §4 and parse the printed "Bandwidth ratio (FlyDSL / PyTorch)" line.

## 4. Re-run

ATT capture (FlyDSL kernel, single-CU instruction trace):

```bash
cd /sgl-workspace/flydsl-prof
/opt/venv/bin/python att_capture.py \
  --test /sgl-workspace/FlyDSL-lab/tests/kernels/test_vec_add.py \
  --kernel vecAddKernel_0 \
  --out results/att/test_vec_add
# shape: SIZE = 256 (threads/block) x 4 (vec_width) x 10000 = 10.24M FP32 elems
#        ~123 MB total traffic (2 reads + 1 write); block=256, grid=10000
```

Standalone benchmark (FlyDSL + torch.add head-to-head, FP32, vec_width=4):

```bash
cd /sgl-workspace/FlyDSL-lab
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
/opt/venv/bin/python tests/kernels/test_vec_add.py --benchmark --vec-width 4
```
