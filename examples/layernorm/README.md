# layernorm — FlyDSL kernel ATT bundle

**stall-bound (58% LDS/SMEM-wait), occ 3/SIMD (VGPR 72); FlyDSL 24.1µs ≈ AIter 24.7µs (1.03×) — dependent ds_* cross-lane reduction tree (shuffle_xor) is the ceiling, not HBM (~1490 GB/s).**

- **Kernel (JIT):** `layernorm_kernel_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 24.1 µs (FlyDSL) vs AIter 24.7 µs → 1.025× 
- **ATT:** 457 ISA instructions, 99.8% source-mapped · 16 waves sampled · occ 3 waves/SIMD · top stall **LDS/SMEM-wait** (60% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
layernorm/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_22257_dispatch_23/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_22257_dispatch_23/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_22257_dispatch_23 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_layernorm.py --gpu 0 --outdir out --tag big
```
