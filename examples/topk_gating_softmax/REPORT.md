# TopK Gating Softmax — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X / CDNA4 · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `topk_gating_softmax_kernel_0` · dispatch_37 (`ui_output_agent_23939_dispatch_37`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_topk_gating_softmax/`

> Arch note: the analyzer prints `gfx942 (CDNA3)` — that is its arch-detection default, not the truth. This capture is gfx950 / MI350X / CDNA4. gfx950 has a **combined** VGPR pool (no split arch/accum), which is what the occupancy reasoning below uses.

## Workload & headline

Fused MoE gating: `softmax(logits) → top-K argmax-mask → renormalize`, one token per sub-warp lane-group, no shared memory. Captured shape **16384 tokens × 128 experts × topk=6, bf16, renorm=True** (~4 MB read, ~0.6 MB write — a memory-bound op).

| | µs | vs FlyDSL |
|---|---|---|
| **FlyDSL** `topk_gating_softmax_kernel_0` | **30.86** | 1.00× |
| AIter `topk_softmax` (HIP, identical signature) | 6.69 | **4.6× faster** |
| AIter `topk_softmax_asm` (HSACO `topksoftmax_12x128x6_bf16.co`) | 23.04 | 1.34× faster |
| torch softmax+topk+renorm (naive ref) | 56.63 | 1.83× slower |

**Verdict: FlyDSL loses, badly.** It is 4.6× slower than the apples-to-apples AIter HIP kernel (speedup 0.22×) and ~0.75× the AIter asm path. It only beats the naive torch reference. The `flydsl_us=24.5` logged in `sweep_master.json` is at the smaller 1024×128 default shape and overstates standing; the authoritative matched-shape number is 30.86 µs (fluctuates 28–33 µs across runs). AIter's HIP kernel at 6.7 µs is near HBM-bandwidth-bound — that is the ceiling FlyDSL is ~4.6× short of, and the trace says why.

## 1. Wave-state / stall breakdown

1,162 instructions, 272.7K total cycles, **132.3K stalled (48.5%)**. The op is **stall-bound on cross-lane reduction latency**, not on memory bandwidth or compute.

| Stall type | Cycles | % of stalls |
|---|---|---|
| **other** | 66.5K | **50.3%** |
| **LDS/SMEM-wait** (`s_waitcnt lgkmcnt`) | 57.5K | **43.4%** |
| VMEM-wait (`s_waitcnt vmcnt`) | 7.2K | 5.4% |
| VMEM-store | 1.0K | 0.8% |
| VMEM-load | 72 | 0.1% |
| MFMA/FMA | 32 | 0.0% |

The two top buckets are one story. The instruction-level view (below) shows the "other" bucket is dominated by `v_exp_f32` and by `s_waitcnt vmcnt(1)`, while the 43.4% LDS/SMEM-wait is a wall of `s_waitcnt lgkmcnt(0)` / `lgkmcnt(1)` — **and there is no LDS in this kernel** (`ds_read: 0, ds_write: 0`, "No shared memory used"). Those LGKMCNT waits are the cross-lane `shuffle_xor` butterfly reductions: on CDNA `shuffle_xor` lowers to `ds_swizzle_b32` / `ds_bpermute_b32`, which retire on LGKMCNT exactly like an LDS op. So nearly half the kernel's stall time is threads stalling on the data-shuffle network during the reductions, not on memory.

Register / occupancy:
- **arch_vgpr ≈ 56** (alloc 56 of 256), accum_vgpr 0, limiting pool 56/256.
- **Occupancy 4 waves/SIMD.** Next step to 5 waves needs VGPR ≤ 51 (shave 5 registers).
- Instruction mix: MFMA 0, buffer_load 2, buffer_store 18, ds 0. Confirms: no matrix work, store-heavy (the per-k scalar writes), all reduction traffic rides the lane-shuffle path.

Occupancy is *not* the primary lever here. At 4 waves/SIMD the kernel has reasonable headroom; bumping to 5 waves would only help hide the latency that the reduction chain creates — better to remove the latency. (On gfx950's combined VGPR pool, the 56-register footprint is the only occupancy limiter; trimming to 51 is cheap if wanted, but secondary.)

## 2. Top instruction-level hotspots

| # | Stall | %Total | DomType | Source |
|---|---|---|---|---|
| 1 | 129.7K | 98.03% | other | `topk_gating_softmax_kernel.py:103` |
| 2 | 1.9K | 1.46% | LDS/SMEM-wait | `topk_gating_softmax_kernel.py:215` |
| 3 | 440 | 0.33% | other | `topk_gating_softmax_kernel.py:207` |
| 4 | 240 | 0.18% | other | `topk_gating_softmax_kernel.py:227` |

**Line 103 (98%) is a source-loc-collapse artifact** — it is the `@flyc.kernel` decorator / `def topk_gating_softmax_kernel(...)` line. The whole kernel body collapses onto it (FlyDSL issue #587 / PR #593). Here it collapses onto the kernel-def line itself rather than onto a `rocdl/universal.py` intrinsic helper (as rope/layernorm do), so the per-line table is useless — the **instruction-level** table is the truth:

- `s_waitcnt lgkmcnt(0)` — **22.5K cyc, 17.0%** (single hottest instruction). The barrier-style drain after a `shuffle_xor` step: the lane must have all peers' shuffled values in hand before `maximumf`/`addf`/the argmax compare. This is the `group_reduce` / `group_reduce_argmax` butterfly (lines 143–174), invoked for the per-token max (254), the sum (268), and **topk=6 separate argmax reductions** (299). Six serial argmax sweeps, each a full `log2(8)=3`-step butterfly over two values (val + idx), each step ending on an LGKMCNT drain.
- `s_waitcnt vmcnt(1)` — **7.1K cyc, 5.35%**. The gating-row load (`_load_atom_in`, line 207/208) feeding the first `maximumf`; the consumer stalls on the 128-bit buffer load.
- `v_exp_f32` ×2 — **5.7K + 1.7K cyc, ~5.6%**. The softmax `exp2` (line 264). Transcendental on the VALU; high-latency, and it sits between the max-reduce and the sum-reduce so its latency is exposed.
- Items 5–15: a long tail of `s_waitcnt lgkmcnt(1)` at ~1.0–1.4% each — the *other* butterfly steps. Summed, the LGKMCNT family is the dominant cost.

Lines 215 / 207 / 227 (the only real per-line hits, totaling <2%) are the actual `copy_atom_call` sites: 207 = input load, 215 = f32 weight store, 227 = i32 index/tei store. Their small LDS-wait share is the store path waiting on the lane-broadcast of the selected weight/index from lane 0.

**Root cause:** the algorithm does **K=6 independent full cross-lane argmax reductions** plus a max-reduce and a sum-reduce — eight butterfly passes per token — and every butterfly step lowers to a `ds_swizzle`/`ds_bpermute` that stalls on LGKMCNT. The reductions are serial (each depends on the previous, and topk iteration k+1 depends on k's mask). That dependency chain, not memory, is the 4.6× gap to AIter.

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Replace `shuffle_xor` butterflies with DPP / `v_permlane16` cross-lane ALU ops
**Root cause:** §1's top two buckets (50.3% other + 43.4% LDS-wait) are LGKMCNT drains on `ds_swizzle`/`ds_bpermute` emitted by `shuffle_xor` in `group_reduce` / `group_reduce_argmax`. The reduction group is only `THREADS_PER_TOKEN=8` lanes wide — entirely inside a 16-lane DPP row.

**Change:** lower the intra-group butterfly to DPP row ops (`v_mov_b32_dpp`, `dpp_ctrl` row_shr 1/2/4) and, where a 16-lane crossing is needed, to gfx950's `v_permlane16_b32` — a pure-ALU op that replaces the `ds_*` cross-row step. DPP/permlane do **not** retire on LGKMCNT; the per-step barrier disappears and the reduction collapses to back-to-back VALU with fixed wait-states the compiler hides. For an 8-lane group the entire max/sum/argmax tree stays in-row, so *all* eight butterfly passes become LGKMCNT-free. Ground: ROCmKernelWiki **`technique-wave-reduce`** — "gfx950 adds `v_permlane16_b32` which replaces the `ds_*` cross-row step with a pure-ALU op… removing the `LGKMCNT` dependency that `ds_bpermute` introduces" (impls: pr-FlyDSL-447, pr-FlyDSL-524, pr-FlyDSL-300). In FlyDSL this means adding a DPP-backed group-reduce primitive in `flydsl/expr/rocdl` and routing `group_reduce`/`group_reduce_argmax` through it instead of `.shuffle_xor`.

**Expected gain:** removes the bulk of the ~57K LGKMCNT-wait cycles and a large share of the "other" bucket → realistically halves stall cycles. Targets the 6.69 µs AIter HIP band. **Effort:** medium (new intrinsic + reduction rewrite; correctness preserved by keeping the same tree shape and tie-break).

### #2 — Fuse the K argmax passes / use threshold-or-sort instead of 6 serial argmax-mask sweeps
**Root cause:** the topk loop (lines 287–310) runs `topk=6` *serial* argmax reductions, each masking the winner then re-reducing — six dependent cross-lane sweeps. Even after #1 makes each sweep cheap, six dependent sweeps is six times the reduction depth.

**Change:** for VPT=16 elements/thread, each thread first does a *local* partial top-K in registers (no cross-lane), then a *single* cross-lane merge produces the group's top-6 (e.g. bitonic top-k merge over the 8-lane group, or a small register heap + one reduction). Collapses 6 group reductions → 1–2. This is what the AIter asm kernel effectively does.

**Expected gain:** cuts reduction *count* ~3–6×; compounds with #1. **Effort:** medium-high (rewrites Pass 4; argmax tie-break and renorm semantics must match the reference exactly).

### #3 — Hoist `v_exp_f32` and overlap it with the max-reduce; keep the input load 128-bit-wide
**Root cause:** `v_exp_f32` (line 264, ~5.6% stall) sits on the critical path between the max-reduce and the sum-reduce, and `s_waitcnt vmcnt(1)` (5.4%) exposes the gating load before the first compare.

**Change:** the exp2 is per-element and independent across a thread's VPT slots — issue all VPT `exp2` back-to-back so the VALU pipeline fills (the unrolled loop already allows this; ensure the scheduler isn't serializing on the shared `thread_sum` accumulator — use a small tree of partial sums). Confirm the input is one 128-bit `BufferCopy(128)` per atom (ATOM_BITS=128 for VPT=16 bf16, already the case) so the single `vmcnt` covers the whole row; ground: ROCmKernelWiki **`technique-vectorized-loads`**.

**Expected gain:** small-to-moderate (~5–10% of stalls), mostly latency hiding behind #1/#2. **Effort:** low.

### #4 — Trim VGPR to ≤51 for 5 waves/SIMD (only after #1/#2)
**Root cause:** occupancy 4 waves/SIMD, next step needs arch_vgpr ≤ 51 (currently 56). On gfx950's combined VGPR pool this is the sole occupancy limiter.

**Change:** the K selected weights/indices are held in registers across the whole topk loop (`selected_weights`/`selected_indices`, 6 entries each) plus the per-element `prob_list` (VPT=16). Spilling `selected_*` to LDS or recomputing in Pass 5 could shave 5 VGPRs. Ground: ROCmKernelWiki **`technique-occupancy-tuning`** / **`technique-vgpr-budgeting`**.

**Expected gain:** marginal on its own — extra occupancy only helps if there is still latency to hide, which #1/#2 remove. Do this last, and only if the post-#1 trace still shows exposed latency. **Effort:** low, but low value.

## 4. Re-run

```bash
# matched-shape baseline head-to-head (authoritative timing):
HIP_VISIBLE_DEVICES=7 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
  /opt/venv/bin/python \
  /sgl-workspace/flydsl-prof/results/baselines/test_topk_gating_softmax/baseline.py
# shape: 16384 tokens x 128 experts x topk6, bf16, renorm=True

# ATT capture (one CU, representative wave-state sample):
/opt/venv/bin/python /sgl-workspace/flydsl-prof/drivers/att_capture.py \
  --test /sgl-workspace/FlyDSL-lab/tests/kernels/test_topk_gating_softmax.py \
  --gpu 7 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_topk_gating_softmax \
  --tag big \
  --iter-range "6,8-8"
# kernel: topk_gating_softmax_kernel_0  ·  iter_range [6, [8-8]]
```

> The ATT trace samples one CU (`att_target_cu`) — a representative wave-state sample, not full-device timing. All µs/speedup figures come from HIP-event timing in the matched-shape baseline script, **not** from ATT.
