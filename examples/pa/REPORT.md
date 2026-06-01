# Paged-Attention Decode (FP8 KV, preshuffle) — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X (CDNA4) · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `pa_decode_ps_kernel_0` (ATT dispatch `ui_output_agent_10886_dispatch_209`)
Bundle: `/sgl-workspace/flydsl-prof/results/att/test_pa/`

> Arch caveat: the hotspot analyzer printed `gfx942 (CDNA3)`. That is the known analyzer arch-detection default, not the real target. This capture ran on **gfx950 / MI350X / CDNA4** — combined VGPR pool, 160 KB LDS. All occupancy reasoning below uses the gfx950 combined-pool model. (Stated once; not repeated.)

## Workload & headline

- **Op**: paged-attention decode, FP8 (e4m3) K/V cache, B-preshuffle layout, split-K partials + separate reduce.
- **Captured shape**: decode workload, 13 dispatches, 56 waves on the sampled CU, 936 instructions (99.9% mapped).
- **Latency** (HIP-event sweep, full device): FlyDSL **169.5 µs** vs baseline **80.6 µs** → **0.476×** (FlyDSL ~2.1× slower). TFLOPS/bandwidth not recorded for this op.
- **Baseline**: AIter Gluon `pa_decode_gluon` (Triton Gluon paged-attention, JIT-compiled at runtime; invoked via `run_gluon_ps()` at `test_pa.py:802–823`). No CK paged-attention baseline exists; Gluon is the strongest comparable. There is no `test_pa.py` entry in `baselines.json`, so the sweep number above is the authoritative head-to-head.
- **Verdict, blunt**: FlyDSL loses ~2:1 here. The capture shows why — this is a **stall-bound** kernel running at **occupancy 1 wave/SIMD**, and 65.7% of all cycles are pure stall, split almost evenly between waiting on LDS-staged softmax/P operands and waiting on FP8 K/V loads from HBM. There is essentially no latency hiding: one wave per SIMD cannot cover its own memory latency, so the matrix core idles.

## 1. Wave-state / stall breakdown

Total cycles 106.6K, total stalls **70.0K (65.7%)**. The kernel spends two of every three cycles stalled.

| Stall type      | Cycles | % of stall | Bound |
|-----------------|--------|-----------:|-------|
| **LDS/SMEM-wait** | 25.3K | **36.2%** | `s_waitcnt lgkmcnt` — LDS reads + scalar loads |
| VMEM-wait        | 16.3K | 23.3% | `s_waitcnt vmcnt` — waiting on K/V loads to retire |
| other            | 15.8K | 22.5% | ALU / scalar setup |
| VMEM-load        | 9.2K  | 13.1% | load issue itself |
| barrier          | 2.5K  | 3.6%  | `s_barrier` between QK-write and PV-read phases |
| LDS / MFMA / VMEM-store | <1.3K total | <1.5% | — |

**Classification: stall-bound, sub-class wait-bound.** #1 is LDS/SMEM-wait (36.2%), but VMEM-wait + VMEM-load together are 36.4% — the two are the same disease. The kernel issues a memory op, then stalls on the `s_waitcnt` because there is no independent work queued behind it. MFMA-stall is 0.4%: the matrix core is starved, not busy. Only 3.6% on barriers, so the QK→PV LDS round-trip is not itself the cost — the cost is having nothing to overlap it with.

**Register pressure & occupancy** (the root enabler of the stalls):

- arch_vgpr **~175 (alloc 176)**, accum_vgpr ~9 (alloc 16), limiting pool **176 / 256**.
- **occupancy = 1 wave/SIMD.**
- Next step: **2 waves/SIMD requires VGPR ≤ 128** (analyzer: `next_occ_step waves=2, vgpr_budget=128`).

On gfx950 the VGPR pool is combined, so the 176-VGPR allocation is the single hard cap. At 176 VGPRs only one wave fits per SIMD, and a single wave has no sibling to switch to while it waits on `vmcnt`/`lgkmcnt`. That is the entire story: **the VGPR footprint forces occupancy 1, occupancy 1 makes every memory wait a dead stall, and the stalls are 65.7% of runtime.** Drop the live VGPR count to ≤128 and a second wave covers the other's latency.

Inst mix (32 MFMA, 14 buffer_load, 18 ds_read, 8 ds_write, 3 buffer_store) is reasonable for FP8 paged attention — the problem is scheduling/occupancy, not instruction selection.

## 2. Top instruction-level hotspots

The hot source lines are heavily affected by **source-loc collapse**: the top two entries point at `pa_decode_fp8.py:1170` (the `@flyc.kernel def pa_decode_ps_kernel(...)` *signature* line) and `:1215` (`wi_rsrc = create_buffer_resource(work_indptr_ptr)`, a one-time setup line). Neither is the literal hot code — the debug-line table sinks many traced instructions onto the def line and onto early setup lines. The instruction-level table tells the truth; read it together with the source.

**#1 — `:1170` collapse sink, 39.1K (55.83%), VMEM-wait.** The instruction breakdown attributed here is the inner-loop body: `s_waitcnt vmcnt(0)` (10.9K, 29.8% of total — the single hottest instruction), plus `s_waitcnt vmcnt(1/2/3)`, `global_load_dwordx4 v[72:75]/v[82:85]` (the FP8 K and V tile loads), `v_pk_mul_f32` (apply K/V dequant scale), and two `s_barrier`s. This is the main per-block loop (`for ib ... in range(...)` at `pa_decode_fp8.py:1570`) and its `_process_block_split` callee: load the next K/V page tile from HBM, dequantize, do QK^T and P·V MFMA. **Why it stalls:** the loads are 128-bit `global_load_dwordx4` straight into VGPRs, and with one wave per SIMD the `s_waitcnt vmcnt(0)` immediately after has nothing to overlap — the wave parks until HBM returns. This is the VMEM-wait bucket.

**#2 — `:1215` collapse sink, 21.3K (30.48%), LDS/SMEM-wait.** The hot instruction here is `s_waitcnt lgkmcnt(0)` (20.9K, 29.8% of total). This is the wait on LDS reads of the staged softmax operands and the packed P (probability) tile before the P·V MFMA — the `fx.Vector.load(..., logits_lds_i64, ...)` P-operand loads (`pa_decode_fp8.py:890/925`) and the softmax max/sum `vector.load_op(..., softmax_lds_f32, ...)` (`:790/795/843`). The QK result and softmax are staged through LDS (`logits_lds`, `softmax_lds`) with a `gpu.barrier()` between the write phase and the read phase (`:1023/1025`). **Why it stalls:** same root cause — the `ds_read`s feed the matrix core, but with occupancy 1 there is no second wave to hide the LDS latency, so the `lgkmcnt(0)` is a hard stall. This is the LDS/SMEM-wait bucket, the #1 stall type.

**#3 — `:1209` 2.3K (3.23%), LDS/SMEM-wait.** Another `s_waitcnt lgkmcnt(0)`, attributed to the buffer-resource setup region; same waiting-on-LDS character as #2.

**#4 / #9 — `:1577` (1.4K) and `:1574` (432), VMEM-load.** These are the *real* hot lines, not collapse artifacts: `buffer_load` of `next_phys_block` / `phys_block` from `kv_page_indices` — the page-table indirection (`buffer_ops.buffer_load(kpi_rsrc, kv_start + ...)`). Paged attention pays an extra dependent scalar load per block to resolve the physical KV page before it can address the K/V tile. The code already prefetches `next_phys_block` one iteration ahead, which is why these are small — but each is a dependent VMEM load on the critical path.

**#5 — `:1220` 1.3K (1.87%), other.** `v_min_u32 v86, 7, v5` — index clamping for the work decomposition. ALU, minor.

**#6/#7/#10 — `:537/:531/:895`.** `:531/:537` are the per-token K/V scale scalar `buffer_load`s in `_load_kv_scale_scalars` (per-token-KV dequant path). `:895` is `rocdl.mfma_f32_16x16x32_fp8_fp8` — the P·V matrix op itself, only 844 cycles of VMEM-wait charged to it, confirming the MFMA is *not* the bottleneck; it's waiting on inputs.

**Bottom line:** the two giant entries (#1, #2) are collapse sinks but their instruction-level content is genuine — they are the `vmcnt`/`lgkmcnt` waits of the main K/V + softmax loop. The real, non-artifact hot lines (`:1577/:1574`) confirm the page-table indirection adds dependent latency. Everything points at the same fix.

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Cut VGPR live-range to ≤128 to reach occupancy 2 (addresses the #1 LDS/SMEM-wait + VMEM-wait buckets) — HIGH impact, MEDIUM effort

**Root cause:** arch_vgpr 176 caps the kernel at 1 wave/SIMD. With one wave, every `s_waitcnt lgkmcnt(0)` (25.3K, 36.2%) and `s_waitcnt vmcnt(0)` (16.3K, 23.3%) is a dead stall because there is no sibling wave to swap in. Together that is **59.5% of all cycles** spent waiting on memory the hardware *could* hide with a second resident wave.

**Concrete change:** the analyzer says 2 waves/SIMD needs VGPR ≤ 128 — a 48-register cut. Targets, in order of likely yield:
- Shrink the loop-carried state tuple in the `range(_loop_start_g, _loop_stop_g, ...)` loop (`pa_decode_fp8.py:1570`). It carries `running_max, running_sum, out0, out1, k_flat, scale_scalars` across iterations. The prefetched `k_flat` / `next_scale_scalars` keep two K-tiles' worth of registers live simultaneously — that double-buffering in VGPRs is exactly what inflates the footprint. Stage the prefetched K tile through LDS instead of VGPRs (see #2) to drop the carried `k_flat`.
- Reduce the QK/PV register-blocking factor (`VTLOOP`/`VHELOOP` accumulator tiles) so fewer `f32x4` accumulators are live at once.
- Verify with `FLYDSL_DUMP_IR=1` and the rocm `--save-temps` register report that the cut actually lands ≤128; the cliff is sharp (129 VGPRs = still 1 wave).

**Expected gain:** going 1→2 waves/SIMD typically recovers a large fraction of pure memory-wait stall. With ~59% of cycles in `vmcnt`/`lgkmcnt` waits, even partial hiding could close most of the 2.1× gap to AIter Gluon. This is the highest-leverage change.

**Wiki grounding:** `technique-vgpr-budgeting` and `technique-occupancy-tuning` — both document the CDNA wave64 VGPR-file-per-SIMD divisor and the ILP-vs-occupancy trade; on a latency-bound decode kernel the occupancy side wins. See PRs `pr-FlyDSL-591`, `pr-sglang-25898`.

### #2 — LDS double-buffer the K/V tile loads so MFMA overlaps the next HBM load (addresses VMEM-wait 23.3% + VMEM-load 13.1%) — HIGH impact, MEDIUM-HIGH effort

**Root cause:** the per-block loop loads the K/V tile into VGPRs, then `s_waitcnt vmcnt(0)` immediately, then computes. The prefetch of `next_phys_block` hides the *page-index* latency but not the *tile-data* latency — the `global_load_dwordx4` of the actual FP8 K/V bytes still stalls the matrix core (instruction #1: `vmcnt(0)`, 10.9K cycles).

**Concrete change:** stage K/V tiles through a double LDS buffer: while the MFMA consumes buffer A, issue the `global_load`→LDS for buffer B, and only `s_waitcnt` on B at the *start* of the next iteration. This decouples load latency from the compute and, as a bonus, moves the prefetched tile out of VGPRs (feeding directly into #1's register cut). On gfx950, use the wider LDS copy / direct-to-LDS path.

**Expected gain:** converts the serial load→wait→compute chain into an overlapped pipeline; targets the combined 36.4% VMEM stall. Synergistic with #1 (and partly substitutes for it by lowering VGPR pressure).

**Wiki grounding:** `technique-lds-double-buffering` (gfx950-listed; tags direct-to-lds, async-pipeline, s-waitcnt; implemented by `pr-FlyDSL-346`) and `technique-mfma-pipelining` (interleave loads and MFMA issue so the matrix core never waits on `s_waitcnt`; `pr-FlyDSL-579`).

### #3 — Vectorize / batch the page-table index loads (addresses the real hot lines `:1577/:1574`, VMEM-load) — MEDIUM impact, LOW effort

**Root cause:** each block iteration issues a dependent scalar `buffer_load` of the physical page index (`kpi_rsrc`) before it can address the K/V tile — paged attention's indirection tax. `:1577` (next) + `:1574` (current) are the largest *non-collapse* hot lines.

**Concrete change:** the loop already prefetches the next index. Go further: batch-load a small window of page indices (e.g. 4) into VGPRs/LDS once per several iterations, so the per-iteration index load is removed from the critical path entirely. Use a 128-bit `buffer_load_dwordx4` to pull 4 indices in one transaction.

**Expected gain:** removes ~2K cycles of dependent VMEM-load latency from the inner loop; small absolute but it's on the critical path and the fix is cheap.

**Wiki grounding:** `technique-vectorized-loads` (128-bit loads to amortize transactions / fewer `s_waitcnt` retirements).

### #4 — Confirm the source-loc collapse and don't over-trust the line table — N/A impact, documentation

Lines `:1170` and `:1215` are the kernel signature and a setup line, not 86% of the runtime. This is FlyDSL's source-loc-granularity collapse (issue #587 / PR #593) — when fixed, the line table will re-attribute these cycles to the real loop body and confirm the conclusions above. Until then, trust the **instruction-level** table (`s_waitcnt vmcnt/lgkmcnt`, `global_load_dwordx4`, `mfma`) over the source-line table for this kernel.

## 4. Re-run

```bash
cd /sgl-workspace/FlyDSL-lab
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
/opt/venv/bin/python /sgl-workspace/flydsl-prof/att_capture.py \
  --test tests/kernels/test_pa.py \
  --stem test_pa \
  --kernel pa_decode_ps_kernel_0 \
  --iteration-range "[5, [6-6]]" \
  --out /sgl-workspace/flydsl-prof/results/att/test_pa
```

ATT samples a single CU (`att_target_cu`) — it is a representative wave-state sample, not full-device timing. The 169.5 µs / 0.476× numbers come from the HIP-event sweep, not from ATT.
