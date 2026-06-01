# preshuffle_gemm — FlyDSL kernel ATT bundle

**compute-bound re-test @4096³ fp8: FlyDSL 1347 TFLOPS (102µs) vs AIter-CK 1760 TFLOPS (78µs) → 0.77× — the M=16 sweep was launch-bound/noise; at saturation the gap is real. CK's bpreshuffle K-loop out-overlaps FlyDSL's.**

- **Kernel (JIT):** `kernel_gemm_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 102.0 µs (FlyDSL) vs AIter CK a8w8 bpreshuffle (untuned default config) 78.1 µs → 0.766× 
- **ATT:** 864 ISA instructions, 99.9% source-mapped · 68 waves sampled · occ 3 waves/SIMD · top stall **VMEM-wait** (83% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
preshuffle_gemm/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_58177_dispatch_135/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_58177_dispatch_135/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_58177_dispatch_135 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_preshuffle_gemm.py --gpu 0 --outdir out --tag big
```
