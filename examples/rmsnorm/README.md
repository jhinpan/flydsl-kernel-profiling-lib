# rmsnorm — FlyDSL kernel ATT bundle

**bandwidth-bound (41% VMEM-wait), occ 4/SIMD @60 VGPR; FlyDSL 25.1µs vs AIter 22.4µs (0.89×, slower) — single vmcnt(0) load-drain before the block reduction is the ceiling.**

- **Kernel (JIT):** `rmsnorm_kernel_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 25.1 µs (FlyDSL) vs AIter 22.4 µs → 0.892× 
- **ATT:** 338 ISA instructions, 99.7% source-mapped · 64 waves sampled · occ 4 waves/SIMD · top stall **VMEM-wait** (72% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
rmsnorm/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_19276_dispatch_18/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_19276_dispatch_18/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_19276_dispatch_18 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_rmsnorm.py --gpu 0 --outdir out --tag big
```
