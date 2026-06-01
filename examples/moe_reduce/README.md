# moe_reduce — FlyDSL kernel ATT bundle

**bandwidth-bound (91% VMEM load+wait), occ 4/SIMD @ 56 VGPR; FlyDSL 382.7µs ≈ torch.sum/aiter.moe_sum 382.6µs (1.00×) at the ~5.5 TB/s HBM ceiling — already optimal, no win available.**

- **Kernel (JIT):** `moe_reduction_kernel_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** None µs (FlyDSL)
- **ATT:** 432 ISA instructions, 99.8% source-mapped · 4112 waves sampled · occ 4 waves/SIMD · top stall **VMEM-load** (77% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
moe_reduce/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_42768_dispatch_39/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_42768_dispatch_39/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_42768_dispatch_39 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_moe_reduce.py --gpu 0 --outdir out --tag big
```
