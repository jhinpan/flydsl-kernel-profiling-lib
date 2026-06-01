# Fused QK-RoPE + Reshape-and-Cache — rocprofv3 / ATT Instruction-Level Analysis

FlyDSL 0.1.9.dev594 @ 18c5a7ed · gfx950 / MI350X (CDNA4) · ROCm 7.2.0 · captured 2026-06-01
JIT kernel: `fused_qk_rope_reshape_and_cache_0` · bundle: `/sgl-workspace/flydsl-prof/results/att/test_fused_rope_cache/`

> Arch caveat: the analyzer header prints `gfx942 (CDNA3)`. That is its arch-detection default, not the real target. This capture is **gfx950 / MI350X / CDNA4**, which has a **single combined VGPR pool** (no split arch/accum) — relevant for the occupancy reasoning below.

---

## Workload & headline

- **Op**: fused Q/K RoPE rotation + reshape-and-cache (writes Q_out, K_out, key_cache, value_cache) in a single launch. `head_dim=64`, NeoX, flash KV layout, bf16, `VEC_WIDTH=1` so **one wavefront of 64 lanes, one lane per head element**.
- **Latency (HIP-event sweep)**: FlyDSL **219.6 µs** vs **AIter (Triton `fused_qk_rope_reshape_and_cache`) 37.5 µs** → **0.17× (FlyDSL ~5.9× slower)**. No TFLOPS/bandwidth recorded (memory-trivial elementwise kernel).
- No `test_fused_rope_cache.py` entry exists in `baselines.json`, so the AIter number above comes from the sweep and is the only head-to-head available.
- **Verdict: FlyDSL is badly behind here.** This is not a roofline problem — the kernel moves almost no data. It is a **launch/structure problem**: a single 64-lane wavefront per (head, token) block doing a serial chain of buffer-descriptor builds and a cross-lane shuffle, fenced by full-barrier waitcounts. The grid is wide but each block is one tiny wave that spends most of its life stalled on `s_waitcnt`.

---

## 1. Wave-state / stall breakdown

ATT sampled **att_target_cu = 1**, one representative wave-state sample (not full-device timing). 207/208 instructions mapped (99.5%). Total 17.9K cycles, **12.4K stalled (69.5%)**.

| Stall type    | Cycles | %     | Note |
|---------------|--------|-------|------|
| **LDS/SMEM-wait** | 9.9K | **79.7%** | `s_waitcnt lgkmcnt(0)` — scalar/LGKM fence, **the bottleneck** |
| VMEM-wait     | 2.3K | 18.6% | `s_waitcnt vmcnt(N)` waiting on the Q/K/V global loads |
| VMEM-load     | 128  | 1.0%  | actual `buffer_load` issue |
| VMEM-store    | 64   | 0.5%  | `buffer_store` to cache/out |
| other         | 16   | 0.1%  | |

**Instruction mix**: `MFMA: 0, buffer_load: 7, buffer_store: 6, ds_read: 0, ds_write: 0`.

Note the apparent contradiction: 80% of stalls are tagged "LDS/SMEM-wait" yet `ds_read = ds_write = 0`. There is **no LDS staging buffer** in this kernel. The LGKM (LDS/GDS/Konstant/Message) counter is also bumped by **(a) the `ds_bpermute` cross-lane shuffle** used for the RoPE rotary pair, and **(b) scalar SMEM loads** that fill buffer-descriptor SGPRs. The `s_waitcnt lgkmcnt(0)` fences are draining those, not real LDS bank traffic. So the dominant cost is **a serial dependency chain of buffer-descriptor construction + one cross-lane shuffle, each gated by a full lgkmcnt(0) barrier** — classic latency exposure, not bandwidth.

**Register pressure & occupancy**: arch_vgpr ≈ 12 (alloc 16), accum_vgpr 0, limiting pool **16 / 256**, **occupancy 8 waves/SIMD** — already the max the hardware will schedule. This kernel is emphatically **not** VGPR- or occupancy-limited. More waves cannot help: the problem is the per-wave instruction stream is a stall chain, and (with `BLOCK_THREADS = WARP_SIZE = 64`) each workgroup is a *single* wavefront, so there is no intra-block latency hiding at all.

**Bound classification: stall-bound (latency-bound).** 69.5% of cycles are dead, dominated by serialized `lgkmcnt(0)` waits with full occupancy and near-zero compute.

---

## 2. Top instruction-level hotspots

| # | Stall | %tot | Dom | Source |
|---|-------|------|-----|--------|
| 1 | 4.6K | 36.9% | LDS/SMEM | `flydsl/expr/rocdl/universal.py:144` |
| 2 | 2.8K | 22.2% | LDS/SMEM | `fused_rope_cache_kernel.py:182` |
| 3 | 2.5K | 19.8% | VMEM | `fused_rope_cache_kernel.py:105` |
| 4 | 2.3K | 18.3% | LDS/SMEM | `fused_rope_cache_kernel.py:214` |
| 5-9 | <=136 | <=1.1% | mixed | lines 429/141/148/187/405 (the actual loads/stores — cheap) |

**#1 — `universal.py:144` (36.9%, LDS/SMEM-wait) — partly an artifact, partly real.**
Line 144 is `make_ptr(buf_ptr_ty, [ptr, Int16(0), num_records_bytes, Int32(flags)])` inside `make_buffer_tensor` — i.e. **buffer-descriptor (V#) construction**. Every `fx.rocdl.make_buffer_tensor(Q/K/V/Q_out/K_out/Cos/Sin/KeyCache/ValueCache)` call funnels through here, so the trace collapses the stall of *many* distinct descriptor builds onto one library line. This is the **source-loc-granularity collapse** (FlyDSL issue #587 / PR #593): treat #1 as "buffer-descriptor setup across the whole kernel," not one bottleneck instruction. The stall itself is real, though — the `s_waitcnt lgkmcnt(0)` here drains the scalar SMEM loads that materialize the descriptor fields before the descriptor can be used. With ~5 descriptors built back-to-back, that scalar-load -> fence -> use pattern serializes.

**#2 — `fused_rope_cache_kernel.py:182` (22.2%, LDS/SMEM-wait): `s_waitcnt lgkmcnt(0)` after the position load.**
Line 182 is `pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)` immediately followed by the position scalar load (line 187). The whole RoPE math is **data-dependent on `pos_val`** (it indexes Cos/Sin: lines 200-205). So the wave issues the position load and then *fully fences* before it can build the cos/sin rows — a serial latency bubble with nothing to overlap it (single wave per block).

**#3 — `fused_rope_cache_kernel.py:105` (19.8%, VMEM-wait): `s_waitcnt vmcnt(N)`.**
Line 105 is the `@flyc.kernel def fused_qk_rope_reshape_and_cache(...)` signature; the trace attributes the kernel's VMEM fences to it. The detailed instruction list shows these are `vmcnt(0)/vmcnt(1)` waits — the wave loading Q/K/V vectors (`load_vec`, lines 217/239/268) and blocking on them before the multiply. Again exposed latency, not bandwidth: only 7 `buffer_load`s total.

**#4 — `fused_rope_cache_kernel.py:214` (18.3%, LDS/SMEM-wait): `s_waitcnt lgkmcnt(0)` in the Q path.**
Line 214 is `qo_row = fx.slice(Q_out_buf, (pid_t, head_idx, None))` / surrounding Q-out descriptor + the `ds_bpermute_pair` shuffle (line 220) feeding the rotary math. The `lgkmcnt(0)` here is the **cross-lane `ds_bpermute`** result fence — the wave can't finish `q_rot_e` until the pair lane's value arrives.

**#5-9 are negligible** (<=1.1% each): these are the *actual* `buffer_load`/`buffer_store` instructions (lines 141/148/187/429). The real memory traffic is essentially free; the kernel is dominated entirely by the fences in front of it.

---

## 3. Optimisation recommendations (ranked by expected impact)

### #1 — Hide the descriptor + position + shuffle latency by running more than one wave per block (the 80% lgkmcnt bucket)
**Root cause (ties to §1, the 79.7% LDS/SMEM-wait + §2 hotspots #1/#2/#4):** every block is a *single* 64-lane wavefront (`BLOCK_THREADS = WARP_SIZE`). A single wave has no sibling to interleave while it sits on `s_waitcnt lgkmcnt(0)` for descriptor scalar loads, the position load, and the `ds_bpermute` result. With occupancy already at 8 waves/SIMD and VGPR at 16/256, **the hardware has tons of spare capacity to overlap — the kernel just doesn't expose it.**
**Concrete change:** process **multiple (head, token) pairs per workgroup** so each block is several wavefronts (e.g. fold the `head_idx`/`pid_t` grid so one workgroup covers N heads -> `BLOCK_THREADS = N*WARP_SIZE`). Independent waves then cover each other's `lgkmcnt`/`vmcnt` bubbles. Equivalently, batch several tokens per wave so the per-token descriptor/position latency is amortized over more in-flight work.
**Expected gain:** large. 69.5% of cycles are stall; even partial overlap of the lgkmcnt chain should cut latency multiple-fold and is the most direct path toward closing the 5.9x gap to AIter.
**Effort:** medium (grid/launch restructuring + indexing changes in the kernel body).
**Wiki grounding:** `technique-occupancy-tuning` (*Occupancy Tuning — Waves per SIMD vs ILP on CDNA*) — the relevant lever here is **using the available occupancy headroom to hide fixed per-wave latency**, not adding VGPRs (we have none to spare back; the issue is one-wave-per-block under-subscription).

### #2 — Hoist and share buffer descriptors; stop rebuilding V# per tensor inline
**Root cause (§2 hotspot #1, universal.py:144, 36.9%):** repeated `make_buffer_tensor` / `create_buffer_resource` calls each emit a scalar-load -> `lgkmcnt(0)` -> descriptor-build chain, serialized. The kernel already comments that it shares cos/sin and frees K SGPRs — extend that discipline.
**Concrete change:** build each buffer descriptor **once at the top, before the data-dependent fences**, and reuse; avoid interleaving descriptor construction with the position-dependent path so the scalar loads for *all* descriptors can be in flight together and drained by a single fence rather than N serial ones. Where Q_out/K_out/KeyCache descriptors can be hoisted above the `pos_val` dependency, do so to overlap their scalar-load latency with the VMEM position load.
**Expected gain:** medium — directly attacks the #1 line; should compress the lgkmcnt chain even before #1's restructuring lands.
**Effort:** low-medium (reorder within kernel body).
**Note:** part of #1's 36.9% is the **#587/#593 source-loc collapse** — confirm the real win with a fresh capture, since the line will keep aggregating many descriptors.

### #3 — Remove the per-element `ds_bpermute` fence from the rotary-pair fetch
**Root cause (§2 hotspot #4, line 214/220):** for NeoX RoPE the rotary pair is `tid XOR vecs_per_half`. The current path does a `ds_bpermute` cross-lane shuffle (LGKM-fenced) per element to fetch the pair value. At `VEC_WIDTH=1` that's one shuffle + one `lgkmcnt(0)` on the critical path of every output element.
**Concrete change:** with multiple waves in flight (#1) the shuffle latency hides for free; additionally consider fetching the rotary pair via a **second strided VMEM load** of the already-resident row instead of a cross-lane shuffle, removing the LGKM dependency entirely (trading a cheap extra VMEM load — we are far from bandwidth-bound — for dropping a serializing fence). Validate which is cheaper on a re-capture.
**Expected gain:** small-medium (folds into #1).
**Effort:** low.
**Wiki grounding:** `technique-wave-reduce` documents `ds_bpermute_b32` cross-lane cost characteristics on gfx950 and confirms it is LGKM-counted.

### #4 — Confirm the kernel isn't relaunch-bound across the 3761 calls
**Root cause:** the harness issued **3761 calls** (avg 4.1 µs each by the device timer). If the 219.6 µs sweep figure includes per-launch overhead that AIter amortizes differently, some of the gap is dispatch, not kernel body.
**Concrete change:** verify launch-count parity vs AIter in the sweep harness; if FlyDSL launches per-(head,token) where AIter launches per-batch, #1's block-fusion also reduces launch count.
**Effort:** low (harness inspection). **Gain:** unknown until measured — flag, don't assume.

---

## 4. Re-run

ATT capture config (`input_trace.yaml`): `att_target_cu: 1`, `att_shader_engine_mask: 0xf`, `att_simd_select: 0xf`, `att_buffer_size: 0x6000000`, `kernel_iteration_range: "[6, [8-8]]"`, regex `fused_qk_rope_reshape_and_cache_0`.

```bash
# Workload (the harness the capture wrapped):
cd /sgl-workspace/FlyDSL-lab && \
PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab \
  python tests/kernels/test_fused_rope_cache.py

# Reproduce the ATT bundle (rope op, this stem):
/opt/venv/bin/python /sgl-workspace/flydsl-prof/att_capture.py \
  --test test_fused_rope_cache \
  --kernel-regex fused_qk_rope_reshape_and_cache_0 \
  --iter-range "[6, [8-8]]" \
  --att-target-cu 1 \
  --outdir /sgl-workspace/flydsl-prof/results/att/test_fused_rope_cache

# Re-decode the hotspot view:
/opt/venv/bin/python /sgl-workspace/FlyDSL-lab/.claude/skills/kernel-trace-analysis/hotspot_analyzer.py \
  /sgl-workspace/flydsl-prof/results/att/test_fused_rope_cache/att/ui_output_agent_4253_dispatch_8781
```
