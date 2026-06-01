# preshuffle_gemm_v2 — FlyDSL kernel ATT bundle

**internal v2-vs-v1 @4096×5120×8192 bf16: v2 layout-API 767 TFLOPS (448µs) = 1.20× over v1 manual (638 TFLOPS, 538µs) — the layout-API refactor is a real FlyDSL-internal win (no external baseline).**

- **Kernel (JIT):** `kernel_gemm_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 447.9 µs (FlyDSL) vs FlyDSL v1 (manual, pre-layout-API) 538.2 µs → 1.202× 
- **ATT:** 429 ISA instructions, 99.8% source-mapped · 12 waves sampled · occ 2 waves/SIMD · top stall **LDS/SMEM-wait** (81% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
preshuffle_gemm_v2/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_63371_dispatch_15/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_63371_dispatch_15/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_63371_dispatch_15 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test bench_preshuffle_gemm_v2.py --gpu 0 --outdir out --tag big
```
