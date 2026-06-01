# moe_gemm — FlyDSL kernel ATT bundle

**stall-bound (55% VMEM-wait, 70% VMEM total), occ 1/SIMD @155 VGPR; FlyDSL stage-1 70.8us = CK 71.1us (1.00x parity, 1.11x on full 2-stage) — load pipeline is unpipelined (vmcnt(1) stores) and matrix cores are starved.**

- **Kernel (JIT):** `moe_gemm1_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 142.9 µs (FlyDSL)
- **ATT:** 997 ISA instructions, 99.9% source-mapped · 24 waves sampled · occ 1 waves/SIMD · top stall **VMEM-wait** (91% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
moe_gemm/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_56573_dispatch_87/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_56573_dispatch_87/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_56573_dispatch_87 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_moe_gemm.py --gpu 0 --outdir out --tag big
```
