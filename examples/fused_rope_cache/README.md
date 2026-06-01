# fused_rope_cache — FlyDSL kernel ATT bundle

**stall-bound (80% lgkmcnt s_waitcnt), occ 8/SIMD but 1 wave/block; FlyDSL 219.6µs vs AIter 37.5µs (0.17x) — serialized buffer-descriptor + position + ds_bpermute fence chain in a single 64-lane wave is the ceiling, not bandwidth.**

- **Kernel (JIT):** `fused_qk_rope_reshape_and_cache_0`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** 219.6 µs (FlyDSL) vs AIter 37.5 µs → 0.171× 
- **ATT:** 208 ISA instructions, 99.5% source-mapped · 4 waves sampled · occ 8 waves/SIMD · top stall **LDS/SMEM-wait** (70% of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
fused_rope_cache/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/ui_output_agent_4253_dispatch_8781/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/ui_output_agent_4253_dispatch_8781/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py ui_output_agent_4253_dispatch_8781 --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test test_fused_rope_cache.py --gpu 0 --outdir out --tag big
```
