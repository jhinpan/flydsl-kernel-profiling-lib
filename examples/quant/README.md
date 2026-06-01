# quant — FlyDSL kernel ATT bundle

**bandwidth-bound (56.8% VMEM stalls, 77.5% total), occ 5/SIMD; FlyDSL 16.74µs vs aiter 16.05µs (0.96×) — near the HBM3E roofline, the recoverable budget is the ~23% barrier+LDS-wait from a two-barrier in-kernel block reduction.**

- **Kernel (JIT):** `quant_kernel_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** None µs (FlyDSL)
- **ATT:** 302 ISA instructions, 99.7% source-mapped · 260 waves sampled · occ 5 waves/SIMD · top stall **VMEM-wait** (78% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
quant/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_34678_dispatch_8/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_34678_dispatch_8/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_34678_dispatch_8 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_quant.py --gpu 0 --outdir out --tag big
```
