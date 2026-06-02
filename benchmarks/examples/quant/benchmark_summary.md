# Benchmark Summary: quant

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 12 (sources: model_config=12)
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
| unweighted geomean vs best | 0.89x  (n=12) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.97x  (n=12) |
| vs aiter_triton | 0.90x  (n=12) |
| vs pytorch | 6.80x  (n=12) |
| worst hot shape | 0.70x  (M=32768,N=4096 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 12 | 0.89x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-V3 | 3 | 1.00x |
| Kimi-K2 | 1 | 0.74x |
| Qwen3 | 3 | 0.96x |
| flydsl_test | 2 | 0.81x |
| flydsl_test_default | 1 | 0.83x |
| stress | 2 | 0.83x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=1024,N=16384 | model_config | fp16 | 19.80 | aiter | 20.92 | 1.06x |
| M=64,N=8192 | model_config | fp16 | 13.40 | aiter_triton | 13.36 | 1.00x |
| M=128,N=2560 | model_config | fp16 | 13.24 | aiter_triton | 13.12 | 0.99x |
| M=512,N=6144 | model_config | fp16 | 13.76 | aiter | 13.52 | 0.98x |
| M=256,N=7168 | model_config | fp16 | 13.68 | aiter | 13.36 | 0.98x |
| M=4096,N=7168 | model_config | fp16 | 28.24 | aiter_triton | 27.52 | 0.97x |
| M=2048,N=4096 | model_config | fp16 | 16.76 | aiter | 15.52 | 0.93x |
| M=2048,N=2560 | model_config | fp16 | 15.28 | aiter | 13.96 | 0.91x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=32768,N=4096 | model_config | fp16 | 107.08 | aiter_triton | 74.80 | 0.70x | tuning_gap |
| M=8192,N=8192 | model_config | fp16 | 64.60 | aiter_triton | 45.32 | 0.70x | tuning_gap |
| M=16384,N=7168 | model_config | fp16 | 97.44 | aiter_triton | 72.00 | 0.74x | tuning_gap |
| M=4096,N=8192 | model_config | fp16 | 35.16 | aiter_triton | 29.24 | 0.83x | tuning_gap |
| M=2048,N=2560 | model_config | fp16 | 15.28 | aiter | 13.96 | 0.91x | tuning_gap |
| M=2048,N=4096 | model_config | fp16 | 16.76 | aiter | 15.52 | 0.93x | tuning_gap |
| M=4096,N=7168 | model_config | fp16 | 28.24 | aiter_triton | 27.52 | 0.97x | ok |
| M=256,N=7168 | model_config | fp16 | 13.68 | aiter | 13.36 | 0.98x | ok |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=256,N=7168 | 13.68 | 43.44 | 29.76 |
| M=128,N=2560 | 13.24 | 42.08 | 28.84 |
| M=2048,N=2560 | 15.28 | 43.44 | 28.16 |
| M=512,N=6144 | 13.76 | 41.38 | 27.62 |
| M=64,N=8192 | 13.40 | 40.64 | 27.24 |
| M=2048,N=4096 | 16.76 | 40.36 | 23.60 |

## Diagnosis

- `M=32768,N=4096` (fp16, vs-best 0.70x): **tuning_gap**
  - evidence: kernel-only vs-best 0.70x for args={'M': 32768, 'N': 4096}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=8192,N=8192` (fp16, vs-best 0.70x): **tuning_gap**
  - evidence: kernel-only vs-best 0.70x for args={'M': 8192, 'N': 8192}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=16384,N=7168` (fp16, vs-best 0.74x): **tuning_gap**
  - evidence: kernel-only vs-best 0.74x for args={'M': 16384, 'N': 7168}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=4096,N=8192` (fp16, vs-best 0.83x): **tuning_gap**
  - evidence: kernel-only vs-best 0.83x for args={'M': 4096, 'N': 8192}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=2048,N=2560` (fp16, vs-best 0.91x): **tuning_gap**
  - evidence: kernel-only vs-best 0.91x for args={'M': 2048, 'N': 2560}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=2048,N=4096` (fp16, vs-best 0.93x): **tuning_gap**
  - evidence: kernel-only vs-best 0.93x for args={'M': 2048, 'N': 4096}
  - likely fix: profile the hot shape and add an op-specific diagnosis

## Promotion Decision

**tune_needed** — sub-parity overall (geomean 0.89x); wins on its target regime but needs per-shape tuning + bug fixes before broad promotion

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 12/12.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops quant
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op quant --shape-ledger benchmarks/examples/quant/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/quant/baseline_matrix.yaml \
  --out benchmarks/examples/quant --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/quant/shape_ledger.jsonl \
  --results benchmarks/examples/quant/benchmark_results.jsonl --out benchmarks/examples/quant/benchmark_summary.md \
  --kernel quant
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
