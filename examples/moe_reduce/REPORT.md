# MoE Reduce (topk sum) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X (CDNA4) · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `moe_reduction_kernel_0` (ATT dispatch_39, `ui_output_agent_42768_dispatch_39`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_moe_reduce/`

> Arch caveat: the analyzer labels this `gfx942 (CDNA3)` — that is its default arch-detection fallback, not a real read of the target. The capture is on **gfx950 / MI350X / CDNA4**, which has a **combined VGPR pool** (no split arch/accum), and all occupancy reasoning below uses that. The ATT trace samples **one CU** (`att_target_cu`); it is a representative wave-state sample, not device-wide timing. Latency/bandwidth come from HIP-event timing in the baseline harness, not from ATT.

## Workload & headline

`MoeReduce(X) = sum(X, dim=1)` — reduce a `[tokens, topk, model_dim]` tensor over the `topk` axis into `[tokens, model_dim]`. This is MoE stage-2 fan-in: combine the topk expert outputs per token. Pure streaming reduction — **no MFMA, no LDS, no data reuse**.

**Authoritative shape (baseline harness):** `tokens=16384, topk=8, model_dim=7168, f16` → X `[16384, 8, 7168]` → Y `[16384, 7168]`.

| Metric | FlyDSL | Baseline (torch.sum == aiter.moe_sum) |
|---|---|---|
| Latency | **382.7 µs** | 382.6 µs |
| Bandwidth | 5523.9 GB/s | 5524.5 GB/s |
| Speedup | — | **1.00×** |

**Verdict: bandwidth-bound, at the HBM ceiling. FlyDSL ties the best available baseline (1.00×).** Both move the same ~1.05 GB (8/9 read, 1/9 write) and both hit ~5.5 TB/s — within HIP-event jitter (3-run spread: FlyDSL 382.7/385.0/385.2 µs, torch.sum 382.6/386.1/383.6 µs). At topk=8 `aiter.moe_sum` has **no specialized kernel** — it falls through to `at::sum_out`, i.e. it *is* torch.sum (confirmed bit-identical). There is no faster CK / rocBLAS / dedicated-ASM moe-reduce for topk=8 in this build. Neither implementation can win because the arithmetic (7 f32 adds per output element) is trivial next to the bytes; the only thing that matters is byte movement, and that is already saturated.

> The ATT capture itself is a **different, much smaller shape** (decode-class: `iter_range [6,[8-8]]`, 4112 waves, `bandwidth_gbs` 26.86 in the manifest) — that low GB/s is the small-shape launch-overhead regime, not a kernel regression. The instruction mix and stall *taxonomy* it reveals are shape-independent and apply to the prefill shape too; only the headline latency/bandwidth above is from the prefill timing run.

## 1. Wave-state / stall breakdown

431 instructions, 59.01M total cycles, **45.69M stalled (77.4%)**.

| Stall type | Cycles | % of stalls | Meaning |
|---|---|---|---|
| **VMEM-load** | 32.23M | **70.5%** | issue/throughput pressure on `buffer_load_dwordx4` |
| **VMEM-wait** | 9.43M | **20.6%** | `s_waitcnt vmcnt(N)` blocking on loads to land |
| other | 2.53M | 5.5% | |
| VMEM-store | 1.05M | 2.3% | `buffer_store_dwordx4` of the result |
| LDS/SMEM-wait | 433.6K | 0.9% | (artifact, §2) |
| SMEM | 21.9K | 0.0% | |

**VMEM-load + VMEM-wait = 91.1% of all stalls.** This is the unmistakable signature of a **bandwidth-bound streaming kernel**: the SIMD spends almost all of its cycles either issuing wide loads or waiting on the `vmcnt` queue for HBM to return data. There is no compute to overlap (`MFMA: 0`), no LDS staging (`ds_read/ds_write: 0`) — the wave does nothing but pull bytes, add them, and write them back.

Instruction mix: **buffer_load: 72, buffer_store: 9, ds: 0, mfma: 0** — all loads/stores are 128-bit `dwordx4` (the optimal VMEM width on CDNA; see §3).

**Register pressure & occupancy (gfx950 combined pool):**
- `arch_vgpr ≈ 53` (alloc 56), `accum_vgpr 0` → limiting pool **56 / 256**.
- **occupancy: 4 waves/SIMD.**
- Next step: **5 waves/SIMD requires VGPR ≤ 51** (analyzer says max(arch,accum) ≤ 51; combined pool on gfx950).

4 waves/SIMD is on the low side, but for a memory-bound kernel occupancy is a second-order lever — what hides HBM latency here is the **depth of the in-flight load queue** (the `s_waitcnt vmcnt(7..2)` ladder in §2 shows ≥8 loads outstanding), not the number of resident waves. More waves would help only if the current outstanding-load depth is too shallow to cover latency; at 5.5 TB/s it clearly is not.

## 2. Top instruction-level hotspots

| # | Stall | %total | Type | Source |
|---|---|---|---|---|
| 1 | 33.55M | 73.4% | VMEM-load | `kernels/moe_gemm_2stage.py:3177` |
| 2 | 10.79M | 23.6% | VMEM-wait | `kernels/moe_gemm_2stage.py:3101` |
| 3 | 1.05M | 2.3% | VMEM-store | `kernels/moe_gemm_2stage.py:3211` |
| 4 | 162.6K | 0.36% | LDS/SMEM-wait | `flydsl/expr/typing.py:896` |
| 5 | 139.5K | 0.31% | LDS/SMEM-wait | `flydsl/expr/rocdl/universal.py:144` |

**#1 — line 3177, `fx.copy_atom_call(copy_atom, src, r)` (73.4%, VMEM-load).** This is the body of the topk loop (`for k in range_constexpr(topk)`), the 128-bit `BufferCopy128b` load of one thread's `VEC_WIDTH` slice of `X[token, k, :]` into rmem. The ASM is `buffer_load_dwordx4 v[0:3], v0, s[4:7], 0 offen` — the top instruction alone is 18.24M (39.9%), with the next loads (`v[4:7]`, `v[8:11]`, …) filling out the rest. **This is the real, irreducible hot line:** it is where every input byte enters the kernel. The stall here is throughput, not a bug — the loads are already maximally wide (16 B/lane = 1 KiB/wave64) and fully unrolled across topk and `n_sub`.

**#2 — line 3101, `s_waitcnt vmcnt(N)` (23.6%, VMEM-wait).** Line 3101 is the `@flyc.kernel def moe_reduction_kernel(...)` signature line — the waits collapse to the function header because the compiler hoists/schedules the `s_waitcnt` ladder there. The ASM confirms a full ladder `vmcnt(7), vmcnt(6), … vmcnt(2)` — the wave issues a batch of loads, then drains them one at a time before the f32 accumulate (line 3185, `acc_vecs[si] += vec_c`). This is the latency-hiding mechanism working as intended; the residual wait is the part of HBM latency the outstanding-load depth doesn't fully cover.

**#3 — line 3211, `fx.copy_atom_call(copy_atom, r_out, dst)` (2.3%, VMEM-store).** The single `buffer_store_dwordx4` of the accumulated result to `Y`. One write per ~8 reads — exactly the topk=8 read/write ratio. Cheap, as expected.

**#4/#5 — `typing.py:896` and `rocdl/universal.py:144` (0.67% combined, "LDS/SMEM-wait").** These are **source-loc-collapse artifacts** (FlyDSL issue #587 / PR #593): hot samples landing on FlyDSL intrinsic-helper Python lines rather than the kernel body, mislabeled "LDS/SMEM-wait" even though the kernel issues **zero** `ds_read`/`ds_write`. At 0.67% they are noise — ignore them; the real LDS traffic is nil.

## 3. Optimisation recommendations (ranked by expected impact)

> Honest framing: this kernel is **already at the HBM bandwidth ceiling and already ties the best baseline (1.00×)**. The dominant stall (91% VMEM load+wait) is the memory system streaming bytes — that is the *intended* steady state for a reduction, not a defect. There is **no large win available**; the entries below are the only levers, in descending order of plausibility, and #1/#2 are likely sub-1% at this shape.

**#1 — Add a non-temporal (streaming) hint to the loads. (root cause: §1 VMEM-load 70.5%; effort: low; expected: 0–3%.)**
Every byte of `X` is read exactly once and never reused, yet the current `BufferCopy128b` atom loads through the normal L2 residency policy. Tagging these as **non-temporal** lets the loads bypass L2 caching policy so they don't evict useful lines and cut L2 lookup overhead on a pure streaming pass. Grounded in **`technique-vectorized-loads`** (ROCmKernelWiki): "For data read once and never reused … add the non-temporal hint so the load bypasses L2 residency policy." Width is *already* optimal (128-bit `dwordx4`, the table-topping form), so NT is the only remaining knob on the load itself. Concretely: expose an `nt=`/`slc`/`glc` flag on `fx.rocdl.BufferCopy128b()` (or a streaming copy-atom variant) and use it for both the load at 3177 and the store at 3211. Caveat: at full HBM saturation the gain is small and shape-dependent — verify with a before/after bandwidth run, don't assume.

**#2 — Raise occupancy 4→5 waves/SIMD by trimming VGPR 56→≤51. (root cause: §1 occupancy/VGPR; effort: medium; expected: ~0–2%, likely none.)**
ATT reports 4 waves/SIMD at 56 VGPR; 5 waves needs ≤51. The accumulator state is `n_sub` × `copy_vec_width` f32 vectors (for f16: 1 × 8 = 8 f32 = 8 VGPRs) plus the fully-unrolled load registers across topk. Shrinking live VGPRs — e.g. accumulate into fewer live registers, or let the compiler reuse load regs more aggressively across the unrolled topk loop — could reach the 5-wave step. Grounded in **`technique-occupancy-tuning`** and **`technique-vgpr-budgeting`** (ROCmKernelWiki: ≤64 VGPR → 8 waves is the "latency-hiding sweet spot for memory-bound kernels"). **But:** the kernel already saturates HBM at 4 waves, so the extra wave has no bytes left to move — expect ~0 wall-clock change. Pursue only if a profiler shows the outstanding-load queue is latency-starved (it isn't at 5.5 TB/s).

**#3 — Leave it alone / fix the comparison, not the kernel. (effort: none.)**
The kernel ties `torch.sum`/`aiter.moe_sum` at the bandwidth ceiling. The only way to "beat" the baseline is to move fewer bytes, which a topk-8 elementwise sum cannot do (you must read all `tokens·topk·model_dim` inputs). The right engineering move is to confirm this is the ceiling (it is) and redirect effort to kernels that are *not* bandwidth-bound. If a win is ever needed here, it must come from **fusion** — folding this reduce into the producer (moe_gemm2 epilogue) so the topk outputs are summed in-register/LDS before they ever hit HBM, eliminating the `tokens·topk·model_dim` read entirely. That is an architectural change to the MoE stage-2 pipeline, not a tweak to this kernel.

**#4 — Fix the source-loc-collapse so future captures are honest. (effort: tracked upstream.)**
Lines 4/5 (`typing.py:896`, `universal.py:144`, mislabeled "LDS/SMEM-wait") are FlyDSL issue #587 / PR #593 granularity artifacts. Not a kernel change — but worth landing #593 so reduction captures stop attributing samples to intrinsic helpers and mislabeling them against a non-existent LDS path.

## 4. Re-run

ATT capture (one CU, decode-class shape — what produced this bundle):

```bash
/opt/venv/bin/python /sgl-workspace/flydsl-prof/drivers/att_capture.py \
  --test test_moe_reduce.py --gpu 6 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_moe_reduce \
  --tag big --target-cu 1
```

Authoritative prefill timing + baseline head-to-head (FlyDSL vs torch.sum == aiter.moe_sum):

```bash
cd /sgl-workspace/FlyDSL-lab && HIP_VISIBLE_DEVICES=6 \
  PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
  /opt/venv/bin/python tests/kernels/test_moe_reduce.py \
  --tokens 16384 --topk 8 --model_dim 7168 --dtype f16 --num_iters 50 --num_warmup 10
```

Shape: `X[16384, 8, 7168] f16 → Y[16384, 7168] f16`, reduce over topk=8.
Standalone aiter.moe_sum cross-check: `/sgl-workspace/flydsl-prof/results/baselines/test_moe_reduce/bench_aiter_moe_sum.py` (confirmed bit-identical to torch.sum at topk=8).
