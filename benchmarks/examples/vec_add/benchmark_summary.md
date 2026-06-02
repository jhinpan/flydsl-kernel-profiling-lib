# Benchmark Summary: vec_add

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 12 (sources: synthetic=7, model_config=5)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=48.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 12 |
| FlyDSL correct + timed | 12 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 12 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 0.95x  (n=12) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter_triton | 0.96x  (n=12) |
| vs triton | 0.95x  (n=12) |
| vs pytorch | 0.96x  (n=12) |
| worst hot shape | 0.77x  (n=7340032 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 7 | 0.97x |
| model_config | 5 | 0.92x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-V3 | 2 | 1.00x |
| Kimi-K2 | 1 | 0.77x |
| Qwen3 | 2 | 0.94x |
| micro | 7 | 0.97x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| n=33554432 | synthetic | fp32 | 101.52 | aiter_triton | 102.64 | 1.01x |
| n=67108864 | synthetic | fp32 | 165.72 | triton | 166.52 | 1.00x |
| n=262144 | synthetic | fp32 | 13.24 | pytorch | 13.28 | 1.00x |
| n=917504 | model_config | fp32 | 13.72 | pytorch | 13.72 | 1.00x |
| n=1048576 | synthetic | fp32 | 13.88 | triton | 13.88 | 1.00x |
| n=14680064 | model_config | fp32 | 55.52 | pytorch | 55.36 | 1.00x |
| n=327680 | model_config | fp32 | 13.36 | aiter_triton | 13.32 | 1.00x |
| n=1024 | synthetic | fp32 | 13.08 | triton | 13.04 | 1.00x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| n=7340032 | model_config | fp32 | 32.92 | aiter_triton | 25.32 | 0.77x | tuning_gap |
| n=4194304 | synthetic | fp32 | 23.28 | triton | 18.60 | 0.80x | tuning_gap |
| n=5242880 | model_config | fp32 | 22.64 | triton | 19.96 | 0.88x | tuning_gap |
| n=10240000 | synthetic | fp32 | 42.68 | triton | 41.80 | 0.98x | ok |
| n=1024 | synthetic | fp32 | 13.08 | triton | 13.04 | 1.00x | ok |
| n=327680 | model_config | fp32 | 13.36 | aiter_triton | 13.32 | 1.00x | ok |
| n=14680064 | model_config | fp32 | 55.52 | pytorch | 55.36 | 1.00x | ok |
| n=1048576 | synthetic | fp32 | 13.88 | triton | 13.88 | 1.00x | ok |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| n=1024 | 13.08 | 42.30 | 29.22 |
| n=262144 | 13.24 | 39.98 | 26.74 |
| n=327680 | 13.36 | 40.04 | 26.68 |
| n=1048576 | 13.88 | 40.32 | 26.44 |
| n=917504 | 13.72 | 39.60 | 25.88 |
| n=5242880 | 22.64 | 36.00 | 13.36 |

## Diagnosis

- `n=7340032` (fp32, vs-best 0.77x): **tuning_gap**
  - evidence: kernel-only vs-best 0.77x for args={'n': 7340032}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `n=4194304` (fp32, vs-best 0.80x): **tuning_gap**
  - evidence: kernel-only vs-best 0.80x for args={'n': 4194304}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `n=5242880` (fp32, vs-best 0.88x): **tuning_gap**
  - evidence: kernel-only vs-best 0.88x for args={'n': 5242880}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `n=10240000` (fp32, vs-best 0.98x): **ok**
  - evidence: kernel-only vs-best 0.98x (near parity or better)
  - likely fix: none
- `n=1024` (fp32, vs-best 1.00x): **ok**
  - evidence: kernel-only vs-best 1.00x (near parity or better)
  - likely fix: none
- `n=327680` (fp32, vs-best 1.00x): **ok**
  - evidence: kernel-only vs-best 1.00x (near parity or better)
  - likely fix: none

## Promotion Decision

**tune_needed** — geomean 0.95x

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 12/12.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops vec_add
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op vec_add --shape-ledger benchmarks/examples/vec_add/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/vec_add/baseline_matrix.yaml \
  --out benchmarks/examples/vec_add --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/vec_add/shape_ledger.jsonl \
  --results benchmarks/examples/vec_add/benchmark_results.jsonl --out benchmarks/examples/vec_add/benchmark_summary.md \
  --kernel vec_add
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
