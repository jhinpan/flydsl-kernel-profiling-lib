# topk_gating_softmax — FlyDSL kernel ATT bundle

**stall-bound (43% LGKMCNT-wait + 50% "other"), occ 4/SIMD; FlyDSL 30.9µs vs AIter-HIP 6.7µs (0.22×) — K=6 serial shuffle_xor argmax butterflies (ds_swizzle/ds_bpermute on LGKMCNT) are the ceiling, not memory.**

- **Kernel (JIT):** `topk_gating_softmax_kernel_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 24.5 µs (FlyDSL)
- **ATT:** 1163 ISA instructions, 99.9% source-mapped · 32 waves sampled · occ 4 waves/SIMD · top stall **other** (48% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
topk_gating_softmax/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_23939_dispatch_37/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_23939_dispatch_37/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_23939_dispatch_37 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_topk_gating_softmax.py --gpu 0 --outdir out --tag big
```
