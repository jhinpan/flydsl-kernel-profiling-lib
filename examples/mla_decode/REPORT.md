# MLA Decode (fp8/fp8 → bf16, m16x8) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X / CDNA4 · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `kn_mla_fwd_decode_m16x8_fp8_fp8_0` (dispatch_58, agent 41578)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_mla_decode/`

> Arch caveat: the analyzer prints `gfx942 (CDNA3)` for this kernel. That is the analyzer's arch-detection default, not the truth. This capture ran on **gfx950 / MI350X / CDNA4** with a **combined VGPR pool** — occupancy reasoning below uses that.

---

## Workload & headline

DeepSeek MLA decode, single tile: `b=1, ctx_len=64, nhead=128, qk=576 (kv_lora 512 + rope 64), v=512, page_size=1`, fp8 Q + fp8 KV, out bf16, `sm_scale=1/sqrt(576)`. Persistent-thread kernel, 256 threads/block, 4 warps.

| Metric | FlyDSL | Strongest baseline | Verdict |
|---|---|---|---|
| Decode stage1 latency | **12.40 µs** (p50; sweep recorded 12.23 µs) | aiter HipKittens CK `hk_mla_decode_fwd`: **11.19 µs** | **0.90× — FlyDSL ~10% SLOWER** |
| TFLOPS | 1.44 | 1.59 (hk) | — |
| vs aiter ASM `mla_decode_stage1_asm_fwd` | 12.40 µs | 12.03 µs | 0.97× (≈ tied) |

Apples-to-apples (harness `--bench_aiter`, all three time only stage1, all pass `cos_diff=1.44e-04`, `err_ratio=0.18%`). At this tiny single-tile shape FlyDSL is **not winning**: it loses ~10% to the hk CK path and is roughly tied with the aiter ASM path.

This is a **latency-bound, single-wave decode**. At `ctx_len=64` there is exactly one BLOCK_N=64 KV tile per work item; all three impls cluster at 11–12 µs and sit at 1.4–1.6 TFLOPS, two orders of magnitude below peak. There is no streaming inner loop to amortize — the runtime is dominated by **getting one KV tile from LDS into the MFMAs**, plus launch/scheduling overhead. The differences between impls are scheduling/pipelining quality, not compute throughput.

---

## 1. Wave-state / stall breakdown

ATT sampled one CU (`att_target_cu=1`), 32 waves, 4,948 of 4,949 instructions mapped (100%). **43.5K of 46.0K cycles are stalls — 94.6%.** This wave does almost nothing but wait.

| Stall type | Cycles | % | Meaning |
|---|---|---|---|
| **LDS/SMEM-wait** | **36.1K** | **83.0%** | `s_waitcnt lgkmcnt(0)` — front-end blocked on `ds_read` of K/V from LDS before MFMA can issue |
| VMEM-wait | 5.9K | 13.5% | `s_waitcnt vmcnt(0)` — waiting on the HBM→register / HBM→LDS KV loads |
| other | 1.0K | 2.4% | address-gen scalar ALU (lshrrev / readfirstlane / v_mov) |
| VMEM-load | 496 | 1.1% | the `buffer_load_dwordx2` issue itself |

**Bound type: stall-bound, specifically LDS-wait-bound.** 83% of all stall cycles are the wave parked on `lgkmcnt(0)` waiting for `ds_read` results. VMEM-wait (13.5%) is a distant second. Compute (MFMA issue) is invisible in the stall profile — the matrix core is idle waiting for operands, not saturated.

Instruction mix confirms an LDS-heavy operand-feed kernel: **ds_read 759, ds_write 127** vs **MFMA 688, buffer_load 130, buffer_store 64**. There are more LDS reads than MFMAs — every matrix issue is fed by a staged LDS round-trip, and there is no second wave to hide it behind.

**Register pressure & occupancy:**
- `arch_vgpr ≈ 251` (allocated 256), `accum_vgpr = 0`.
- Limiting pool: **256 / 256 VGPR → occupancy = 1 wave/SIMD.**
- Next step: `2 waves/SIMD requires VGPR ≤ 128` (analyzer: "next_occ_step waves=2, vgpr_budget=128").

On gfx950's combined VGPR pool, 251 VGPRs/wave is the hard occupancy cap. One wave per SIMD means **there is no other wave to swap in during the `lgkmcnt(0)` stalls** — the 83% LDS-wait is fully exposed. This is the core of the problem: a single-wave kernel must hide its LDS latency entirely through intra-wave pipelining (issue MFMAs from tile *t* while `ds_read` for tile *t+1* is in flight), and at one KV tile of work there is almost nothing to overlap with.

---

## 2. Top instruction-level hotspots

The source-line attribution collapses onto kernel-setup lines — a known FlyDSL source-loc-granularity artifact (issue #587 / PR #593). The ASM column is the real signal; read it, not the Python line.

| # | Stall | %tot | DomType | Source (collapsed) | What actually runs |
|---|---|---|---|---|---|
| 1 | 36.1K | 83.0% | LDS/SMEM-wait | `mla_fwd_decode_m16x8_fp8_fp8.py:396` | ASM = `s_waitcnt lgkmcnt(0)` |
| 2 | 6.9K | 15.8% | VMEM-wait | `…:309` | ASM = `s_waitcnt vmcnt(0)` + `v_lshrrev`/`v_readfirstlane`/`v_mov` |
| 3 | 496 | 1.1% | VMEM-load | `…:409` | `buffer_load_dwordx2 v[4:5], …` |

- **Line 396** (the report's #1, 83%) is the Python source `work_indptr_rsrc = create_buffer_resource(work_indptr)` — a one-line buffer-descriptor setup. It is **not** the bottleneck. The ASM (`s_waitcnt lgkmcnt(0)`) is the LDS-wait fence that gates the MFMA loop. The real bottleneck is the **`_load_k_from_lds` → `s_waitcnt lgkmcnt(0)` → `_mfma_fp8`** chain in `_process_kv_tile` (lines ~1060–1101 and the V/PV path ~1228–1242), where every K/V operand is read from LDS and the wave fences on it before issuing the matrix op. With occupancy=1 those fences stall the whole wave.
- **Line 309** (15.8%) is the `@flyc.kernel` decorator line; the ASM is `s_waitcnt vmcnt(0)` plus the thread-index address-gen (`v_lshrrev_b32 v2,6,v0` = `tid/WARP_SIZE`, `v_readfirstlane`). This is the wave waiting on the **HBM→LDS / HBM→register KV load** (`buffer_load_to_lds`, lines ~446–491) to land before it can start GEMM1 — the prologue VMEM latency that, again, has no other wave to hide behind.
- **Line 409** (1.1%) is genuinely line 409: `work_range = buffer_load(work_indptr_rsrc, …)` — the persistent-thread work-range fetch. Small, real, expected.

Net: **real bottleneck = exposed LDS→MFMA operand latency in the QK/PV MFMA chain (#1)**, secondary = exposed prologue HBM→LDS KV-load latency (#2). Both are exposed because occupancy is 1 wave/SIMD and the workload (one KV tile) gives almost nothing to overlap.

---

## 3. Optimisation recommendations (ranked by expected impact)

### 1. Software-pipeline the LDS→MFMA chain so `ds_read` of tile t+1 overlaps MFMA of tile t  ← biggest bucket (83% LDS-wait)
**Root cause:** §1 #1 — the wave fences on `s_waitcnt lgkmcnt(0)` before every MFMA, and with occupancy=1 there is no second wave to fill the gap. The matrix core sits idle waiting for LDS operands.
**Change:** restructure `_process_kv_tile` (the `range_constexpr` NOPE/ROPE/PV loops, lines ~1056–1101 / ~1228–1242) into an explicit prefetch-ahead pipeline: issue the `ds_read` for the next K/V sub-block (`_load_k_from_lds`), then issue the MFMAs that consume the *current* sub-block, and only `lgkmcnt` down to the count that keeps the next read in flight. The kernel already uses partial counts (`lgkmcnt=P_COMP_SUBS`, `wait_lgkm[step]`) — push that further so MFMAs never wait on a read they don't need yet. Goal: turn the 83% `lgkmcnt(0)` stall into overlapped issue.
**Wiki grounding:** `technique-mfma-pipelining` ("if the front stalls on an `s_waitcnt` waiting for the next K-slice from LDS before it can issue the next MFMA, the matrix core sits idle") and `technique-lds-double-buffering` — stage the next tile's operands into a second LDS region / register set while the current tile's MFMAs run. Both list MI350X/gfx950 support; FlyDSL PRs pr-FlyDSL-346 / pr-FlyDSL-579 already exercise this pattern in GEMM.
**Expected gain:** this is the 83% bucket. Even partial overlap on the single tile should close most of the 1.2 µs gap to the hk CK baseline; the CK/ASM baselines win precisely because they pipeline this better. Realistic target: match or beat 11.19 µs.
**Effort:** medium–high — it is a real restructuring of the inner MFMA loop, but the helpers (`_load_k_from_lds`, partial `_encode_waitcnt`) already exist.

### 2. Cut VGPR below 128 to reach 2 waves/SIMD and hide the residual LDS/VMEM latency
**Root cause:** §1 — occupancy=1 (`arch_vgpr≈251`, cap 256/256). A single wave cannot hide *any* memory latency by switching to another wave; a second wave would absorb the `lgkmcnt`/`vmcnt` stalls of the first.
**Change:** get VGPR ≤ 128 (analyzer's `next_occ_step`). 251 VGPRs is heavy for a decode tile — candidates: shrink the `oaccu` accumulator footprint, reduce the number of K/V sub-blocks held live simultaneously, reuse registers across the NOPE/ROPE/PV phases instead of keeping them all resident. This trades ILP within one wave for a second concurrent wave.
**Wiki grounding:** `technique-occupancy-tuning` and `technique-vgpr-budgeting` (gfx950 combined VGPR pool: `floor(VGPR_file_per_simd / vgprs_per_wave)` sets the slot count; 251→2 needs ≤128).
**Expected gain:** potentially large *if* achievable — a 2nd wave directly attacks the same 83% LDS-wait by giving the SIMD other work during fences. But halving VGPRs from 251 to 128 is aggressive for this register-heavy kernel and may force spills; treat as exploratory. Rank #2 because #1 is lower-risk and addresses the same stall.
**Effort:** high — large VGPR cuts on an MLA tile risk scratch spills (which would re-introduce VMEM-wait).

### 3. Confirm KV LDS staging is bank-conflict-free
**Root cause:** part of the 83% LDS-wait is `ds_read` *latency*, but bank conflicts would inflate it further. The V path uses `ds_read_b64_tr_b8` / volatile `ds_read_b32` (lines ~744–758) with a hand-built V3 LDS layout.
**Change:** verify the V3 KV LDS layout (264-byte slots, `KV_NUM_COLS` padding, lines ~146–181) produces conflict-free `ds_read`. The `_lds_load_volatile` comment notes it deliberately blocks `ds_read2` merging — confirm that is not leaving reads narrower/more-conflicted than necessary.
**Wiki grounding:** `technique-bank-conflict-avoidance` (padding / swizzle / `ds_read2`) and `technique-lds-swizzling`.
**Expected gain:** small-to-moderate; this is a refinement on top of #1, not a substitute. Only worth it if profiling after #1 still shows `ds_read` issue rate below expectation.
**Effort:** medium — requires LDS-layout inspection and a conflict-counter recapture.

> Honest note on ceiling: at `b=1, ctx_len=64` the absolute runtime is launch/scheduling-dominated (the manifest sweep and baselines.json both say so). The CK baseline's 1.2 µs edge is real but small in absolute terms. The recommendations above are most valuable as they generalize to larger `ctx_len` / batched decode, where the same LDS→MFMA pipeline runs many iterations and the 83% LDS-wait compounds.

---

## 4. Re-run

ATT capture (selects dispatch, builds `input_trace.yaml` with `kernel_iteration_range: "[6, [8-8]]"`, `att_target_cu: 1`):

```bash
/opt/venv/bin/python /sgl-workspace/flydsl-prof/drivers/att_capture.py \
  --test test_mla_decode.py \
  --gpu 7 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_mla_decode \
  --tag big \
  --iter-range "[6,[8-8]]" \
  --buffer 0x6000000 \
  --target-cu 1
```

Underlying workload command (shape `b=1, ctx_len=64`):

```bash
python tests/kernels/test_mla_decode.py -b 1 -c 64
```

Head-to-head vs aiter (authoritative baseline, `cwd=/sgl-workspace/FlyDSL-lab`):

```bash
HIP_VISIBLE_DEVICES=4 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
/opt/venv/bin/python tests/kernels/test_mla_decode.py -b 1 -c 64 --bench_aiter
```

Hotspot re-analysis: `hotspot_big.txt` in the bundle dir (hotspot_analyzer.py output).
