# mla_decode — FlyDSL kernel ATT bundle

**stall-bound (83% LDS/SMEM-wait, lgkmcnt(0)), occ 1/SIMD @ VGPR≈251; FlyDSL 12.40µs vs aiter-hk-CK 11.19µs (0.90×) — exposed LDS→MFMA operand-feed latency on a single-wave decode is the ceiling.**

- **Kernel (JIT):** `kn_mla_fwd_decode_m16x8_fp8_fp8_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 12.23 µs (FlyDSL)
- **ATT:** 4949 ISA instructions, 100.0% source-mapped · 32 waves sampled · occ 1 waves/SIMD · top stall **LDS/SMEM-wait** (95% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
mla_decode/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_41578_dispatch_58/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_41578_dispatch_58/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_41578_dispatch_58 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_mla_decode.py --gpu 0 --outdir out --tag big
```
