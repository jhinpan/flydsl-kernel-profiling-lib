# tools/ — the MI350X sweep harness (reproducible)

The deterministic pipeline that produced the **2026-06-01 gfx950 sweep** (17 kernels,
ATT + baselines + reports + dashboard). All drivers are recipe-driven and GPU-pinned.

## Pipeline
```
recipes.json ──► run_sweep.py ──► (per-kernel timing JSON) ──► consolidate_sweep.py ──► sweep_master.json
   (per kernel)     run_pool.py drives N kernels over 8 GPUs (free-GPU pool, no contention)

recipes.json ──► att_capture.py ──► results/att/<k>/{att/ui_output_agent_*, capture.json, hotspot_big.txt}
   per AGENTS.md: fresh debug cache ► rocprofv3 --stats discovery ► input_trace.yaml ► ATT capture
   ► empty-shell cleanup ► source-map verify ► hotspot_analyzer.py
   run_pool.py --att drives all kernels over 8 GPUs

baselines.json ◄── (AIter / CK / hipBLASLt at matched shapes, via each harness's built-in compare flag)
build_manifest.py  ──► analysis_manifest.json   (capture + hotspot + sweep + paths, per kernel)
(per-kernel REPORT.md written from the manifest + hotspot + ROCmKernelWiki)
export_dashboard_data.py ──► ../docs/data/kernels.json (+ data.js)   merges sweep + baselines + headlines
build_bundles.py ──► ../examples/<kernel>/   assembles the canonical bundle (preserves existing examples)
```

## Re-run one kernel
```bash
ROOT=/sgl-workspace/FlyDSL-lab            # the built FlyDSL worktree (editable install)
export PYTHONPATH=$ROOT/build-fly/python_packages:$ROOT
# timing + AIter baseline:
python tools/run_sweep.py  --test test_rmsnorm.py --gpu 0
# full ATT bundle:
python tools/att_capture.py --test test_rmsnorm.py --gpu 0 --outdir out/rmsnorm --tag big
python tools/hotspot_analyzer.py out/rmsnorm/att/ui_output_agent_*_dispatch_* --topk 15 --mode both
```

## Notes / gotchas baked in
- **PYTHONPATH** must lead with `build-fly/python_packages` (native MLIR libs live there, not in `python/`).
- ATT needs a **fresh `FLYDSL_RUNTIME_CACHE_DIR` + `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`** for source mapping (cold debug cache; see AGENTS.md gotcha 5).
- `att_capture.py` auto-derives `kernel_iteration_range` from the discovered call count and cleans empty-shell dispatch dirs.
- **Small grids miss `att_target_cu`** → enlarge the workload shape (`--cmd-override`) or sweep `--target-cu`. Affected the first capture of topk_gating / preshuffle_gemm / moe_reduce / moe_blockscale; all fixed by enlargement.
- The analyzer prints `gfx942` by default; the real arch is **gfx950** (confirmed by the `out_gfx950_code_object_id_*.out` HSACO names).
- `data/` holds the raw aggregates (`recipes`, `sweep_master`, `baselines`, `headlines`, `analysis_manifest`) the dashboard is built from.
