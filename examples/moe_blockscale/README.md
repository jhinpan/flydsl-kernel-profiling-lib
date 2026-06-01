# moe_blockscale — FlyDSL kernel ATT bundle

**bandwidth-bound (41% VMEM-wait + 36% VMEM-load = 78% of stalls), occ 2/SIMD @ ~203 VGPR; FlyDSL 53.8us vs CK 44.0us (0.818x, slower) — MFMA-scale starved on an under-prefetched FP8 operand/scale load chain.**

- **Kernel (JIT):** `mfma_moe1_bs_fp8_f16_cshuffle_t16x256x128_wpe2_abi8`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 53.8 µs (FlyDSL) vs CK 44.0 µs → 0.818× 
- **ATT:** 886 ISA instructions, 99.9% source-mapped · 4 waves sampled · occ 2 waves/SIMD · top stall **VMEM-wait** (83% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
moe_blockscale/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_39345_dispatch_668/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_39345_dispatch_668/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_39345_dispatch_668 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_moe_blockscale.py --gpu 0 --outdir out --tag big
```
