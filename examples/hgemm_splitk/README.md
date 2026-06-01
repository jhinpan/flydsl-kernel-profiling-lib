# hgemm_splitk — FlyDSL kernel ATT bundle

**latency-bound (56% VMEM-wait, MFMA only 2%), occ 2/SIMD @ 117 VGPR; FlyDSL 7.0µs = 25 TFLOPS, 1.66x vs PyTorch 11.6µs — but only 84 workgroups on 256 CUs, so SPLIT_K=14 starves each block to ~2 K-iters and the 2-stage pipeline can't hide load latency.**

- **Kernel (JIT):** `hgemm_bf16_32x64x256_W1x2x2_S2_BT_BLDS1_AS1_SPK14_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 7.0 µs (FlyDSL) vs PyTorch 11.6 µs → 1.657× 
- **ATT:** 522 ISA instructions, 99.8% source-mapped · 8 waves sampled · occ 2 waves/SIMD · top stall **VMEM-wait** (71% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
hgemm_splitk/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_4716_dispatch_319/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_4716_dispatch_319/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_4716_dispatch_319 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_hgemm_splitk.py --gpu 0 --outdir out --tag big
```
