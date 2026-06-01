# FlyDSL Kernel Atlas — Sweep Findings (MI350X / gfx950)

*rocprofv3 ATT sweep of every major FlyDSL kernel on 8× AMD Instinct MI350X (CDNA4, gfx950).
FlyDSL `0.1.9.dev594` @ `18c5a7ed`, ROCm 7.2.0, rocprofv3 1.1.0. Captured 2026-06-01.*

**17 kernels** profiled at the instruction level (95–100 % source-mapped ATT), **15** with a
matched-shape AIter / CK / hipBLASLt head-to-head. Interactive view: **[docs/ dashboard](https://jhinpan.github.io/flydsl-kernel-profiling/)**.

This is the bird's-eye read. Each kernel has a per-instruction writeup under
[`examples/<kernel>/REPORT.md`](examples/).

---

## Scoreboard

| verdict | kernels |
|---|---|
| **FlyDSL wins** (≥1.05×) | **Softmax 2.05×** (vs AIter), **HGEMM Split-K 1.66×** (vs PyTorch/hipBLASLt), **MoE GEMM 2-stage 1.11×** (vs CK — stage-2 atomic is **1.30×**) |
| **Parity** (±5 %) | LayerNorm 1.02×, Per-Token Quant 0.96×, MoE Reduction 1.00× (both bandwidth-bound, at the HBM roofline) |
| **Headroom** (FlyDSL slower) | **RoPE 0.17×**, **TopK-Gating 0.22×**, **Paged-Attn decode 0.48×**, **Block-Scale Preshuffle GEMM 0.66×**, **Preshuffle GEMM 0.77×**, MoE Block-Scale 0.82×, RMSNorm 0.89×, MLA decode 0.90×, Flash-Attn 0.92× |

> **Preshuffle GEMM v2** is an internal win: the layout-API rewrite hits **767 TFLOPS, 1.20× over the manual v1** at 4096×5120×8192 bf16.

> **Note — GEMMs re-measured at compute-bound shapes (2026-06-01 follow-up).** The first sweep used the harnesses' tiny default M=16 shapes (launch-bound, ~0.88× within noise). Re-run at saturating shapes the gap is **real and larger**: Preshuffle GEMM 4096³ fp8 → **0.77×** vs untuned AIter-CK (1347 vs 1760 TFLOPS); Block-Scale Preshuffle M=4096 → **0.66×** vs a **tuned** AIter-CK (869 vs 1322 TFLOPS). Part of the block-scale gap is tuning parity — CK loads per-shape tuned configs while FlyDSL runs a fixed schedule. FP8 row-scale GEMM (`fp8_gemm_4wave`) **fails to compile at every shape tried** (M=16, 4096³, 8192³) with `missing _reusable_slot_spec` — config-independent, likely a harness/usage issue on our side, so no PR yet.

One kernel — **FP8 Row-Scale GEMM** (`fp8_gemm_4wave`) — currently **fails to compile**:
`flyc.compile(): missing _reusable_slot_spec` on the fast-dispatch path. That's a real regression, not a perf result.

---

## The three patterns that explain the headroom

### 1. Register pressure is capping occupancy on the attention/GEMM kernels
The kernels FlyDSL loses on are almost all **occupancy 1–3 waves/SIMD, throttled by VGPR**:

| kernel | arch_vgpr | occ (waves/SIMD) | dominant stall | vs baseline |
|---|---:|---:|---|---:|
| MLA decode (fp8) | **251** | 1 | LDS/SMEM-wait | 0.90× |
| Flash Attention | 243 | 1 | VMEM-store | 0.92× |
| MoE Block-Scale s1 | 203 | 2 | VMEM-wait | 0.82× |
| Paged-Attn decode | 175 | 1 | LDS/SMEM-wait | 0.48× |
| Preshuffle GEMM | 161 | 3 | VMEM-wait | 0.77× (4096³) |

At 1 wave/SIMD there is no second wave to hide global-load and MFMA latency behind — the
stall percentages (58–83 % of cycles) are the direct consequence. On gfx950's **combined**
VGPR pool, every register spilled below the next occupancy step (typically VGPR ≤ 128 for
2 waves) is paid back as exposed memory latency. **The single highest-leverage lever across
the attention stack is cutting live VGPRs to buy a second wave.**

### 2. RoPE and TopK-Gating are structural, not occupancy-bound
RoPE+cache is **5.9× slower than AIter** (219 µs vs 37 µs) yet runs at **occupancy 8 with only
12 VGPRs** — it is not register- or occupancy-limited. The hot samples collapse onto a FlyDSL
intrinsic (`flydsl/expr/rocdl/universal.py:144`), the source-loc-granularity artifact from
issue #587 — which hides a **serialized LDS/SMEM-wait dependency chain**: the kernel is doing
too many small, dependent ops per element instead of a vectorized fused pass. TopK-Gating
(0.22×, stall type "other") is the same shape of problem — a butterfly/selection that AIter
fuses and FlyDSL doesn't. **These two are algorithmic rewrites, and they're the biggest wins
on the board.**

### 3. Bandwidth-bound kernels are already at the roofline
Softmax (2.05×), LayerNorm (1.02×), RMSNorm (0.89×), Quant (0.96×), MoE-Reduction (1.00×) are
all VMEM-wait/-load dominated and sit at 4.4–6.5 TB/s — i.e. at MI350X HBM bandwidth. Softmax
*beats* AIter by 2× here; RMSNorm trails by 11 % despite a near-identical structure (occ 4,
VGPR 60, one step from 5 waves at VGPR ≤ 51). For this class the ceiling is bytes moved, and
the FlyDSL-vs-AIter gap is about vectorization width and tail handling, not compute.

---

## Where FlyDSL is genuinely strong
- **Softmax** — 2.05× over AIter's Triton softmax at 32768×8192 bf16, 5.2 TB/s. The vectorized
  reduction is the reference for what "good" looks like in this repo.
- **MoE GEMM stage-2 (atomic)** — 1.30× over CK; the 2-stage fused path is a real win.
- **HGEMM Split-K** — 1.66× over PyTorch/hipBLASLt on a skinny (M=32) decode-shaped GEMM.

## Optimization targets, ranked by expected payoff
1. **RoPE + KV-cache** (0.17×) — fuse the per-element op chain; kill the serial LDS round-trips. Biggest absolute gap.
2. **TopK-Gating Softmax** (0.22×) — match AIter's fused softmax+topk+renorm; current path serializes.
3. **Paged-Attn decode** (0.48×, occ 1, VGPR 175) — cut register pressure to reach 2 waves; LDS-wait bound.
4. **MoE Block-Scale stage-1** (0.82×, VGPR 203) — register diet + better global-load overlap.
5. **Fix FP8 row-scale GEMM compile** — restore the fast-dispatch path (`_reusable_slot_spec`).
6. **RMSNorm** (0.89×) — shave VGPRs from 60→≤51 for a 5th wave; tighten the tail.

---

*Method: each kernel run standalone, GPU-pinned across 8 cards; FlyDSL HIP-event timing vs the
strongest available non-FlyDSL impl at the same shape; rocprofv3 ATT (`att_target_cu`, AsmDebug,
fresh debug cache) for the instruction-level wave-state + stall taxonomy, decoded with
`hotspot_analyzer.py`. Drivers and raw results: `flydsl-prof/` (timing) + `examples/*/` (bundles).
Multi-GPU comm kernels (custom all-reduce, shmem) are deferred — they need ≥2 ranks and print no single-kernel perf.*
