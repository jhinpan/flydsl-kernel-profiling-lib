# pa — FlyDSL kernel ATT bundle

**stall-bound (65.7% stall, 36% LDS-wait + 36% VMEM), occ 1/SIMD (VGPR 176, needs ≤128 for 2 waves); FlyDSL 169.5µs vs AIter Gluon 80.6µs (0.476×) — single resident wave can't hide its own K/V-load + softmax-LDS latency.**

- **Kernel (JIT):** `pa_decode_ps_kernel_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 169.5 µs (FlyDSL) vs AIter Gluon (pa_decode_gluon) 80.6 µs → 0.476× 
- **ATT:** 937 ISA instructions, 99.9% source-mapped · 56 waves sampled · occ 1 waves/SIMD · top stall **LDS/SMEM-wait** (66% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
pa/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_10886_dispatch_209/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_10886_dispatch_209/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_10886_dispatch_209 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_pa.py --gpu 0 --outdir out --tag big
```
