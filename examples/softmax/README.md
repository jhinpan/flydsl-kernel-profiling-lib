# softmax — FlyDSL kernel ATT bundle

**stall-bound (76% stalled, 46% VMEM-load), occ 5/SIMD; FlyDSL 271.8µs vs AIter triton 558µs (2.05x win) at 32768x8192 bf16 — but the fast vectorized path is dead-coded (False-gated), so it runs scalar 16-bit loads, leaving the HBM roofline on the table.**

- **Kernel (JIT):** `softmax_kernel_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 271.8 µs (FlyDSL) vs AIter 558.0 µs → 2.053× 
- **ATT:** 732 ISA instructions, 99.9% source-mapped · 2056 waves sampled · occ 5 waves/SIMD · top stall **VMEM-load** (76% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
softmax/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_47920_dispatch_13/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_47920_dispatch_13/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_47920_dispatch_13 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_softmax.py --gpu 0 --outdir out --tag big
```
