# blockscale_preshuffle_gemm — FlyDSL kernel ATT bundle

**compute-bound re-test @M=4096: FlyDSL 869 TFLOPS (156µs) vs AIter TUNED-CK 1322 TFLOPS (102µs) → 0.66× — gap WIDENS vs M=16 because CK loaded a per-shape tuned config and FlyDSL runs a fixed untuned schedule.**

- **Kernel (JIT):** `bs_gemm_bf16_direct_t16x64x256`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 155.7 µs (FlyDSL) vs AIter CK a8w8_blockscale_bpreshuffle (TUNED config) 102.3 µs → 0.66× 
- **ATT:** 391 ISA instructions, 99.7% source-mapped · 4 waves sampled · occ 6 waves/SIMD · top stall **VMEM-wait** (92% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
blockscale_preshuffle_gemm/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_14065_dispatch_120/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_14065_dispatch_120/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_14065_dispatch_120 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_blockscale_preshuffle_gemm.py --gpu 0 --outdir out --tag big
```
