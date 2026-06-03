# Benchmark Summary: preshuffle_gemm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 14 (sources: synthetic=6, model_config=8)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=42.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 14 |
| FlyDSL correct + timed | 14 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 14 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 0.72x  (n=14) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.72x  (n=14) |
| vs pytorch | 6.76x  (n=14) |
| worst hot shape | 0.26x  (M=4096,N=8192,K=8192 vs aiter) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 6 | 0.71x |
| model_config | 8 | 0.73x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-V3 | 3 | 0.61x |
| Kimi-K2 | 2 | 0.82x |
| Qwen | 3 | 0.80x |
| generic | 6 | 0.71x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=33,N=1024,K=2048 | synthetic | fp8 | 14.60 | aiter | 15.48 | 1.06x |
| M=16,N=7168,K=7168 | model_config | fp8 | 20.44 | aiter | 21.36 | 1.05x |
| M=16,N=7168,K=7168 | model_config | fp8 | 20.96 | aiter | 21.12 | 1.01x |
| M=16,N=2560,K=2560 | model_config | fp8 | 15.64 | aiter | 15.48 | 0.99x |
| M=16,N=5120,K=8192 | synthetic | fp8 | 22.80 | aiter | 21.36 | 0.94x |
| M=5120,N=2048,K=8320 | synthetic | fp8 | 121.04 | aiter | 103.80 | 0.86x |
| M=128,N=2560,K=2560 | model_config | fp8 | 19.88 | aiter | 16.68 | 0.84x |
| M=9728,N=8192,K=8320 | synthetic | fp8 | 814.21 | aiter | 650.25 | 0.80x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=4096,N=8192,K=8192 | synthetic | fp8 | 989.25 | aiter | 252.28 | 0.26x | tuning_gap |
| M=2048,N=7168,K=7168 | model_config | fp8 | 336.80 | aiter | 121.56 | 0.36x | tuning_gap |
| M=2048,N=2560,K=2560 | model_config | fp8 | 55.60 | aiter | 33.96 | 0.61x | tuning_gap |
| M=128,N=7168,K=2048 | model_config | fp8 | 28.76 | aiter | 17.68 | 0.61x | tuning_gap |
| M=256,N=7168,K=2048 | model_config | fp8 | 30.20 | aiter | 20.12 | 0.67x | tuning_gap |
| M=5120,N=5120,K=8320 | synthetic | fp8 | 294.52 | aiter | 217.76 | 0.74x | tuning_gap |
| M=9728,N=8192,K=8320 | synthetic | fp8 | 814.21 | aiter | 650.25 | 0.80x | tuning_gap |
| M=128,N=2560,K=2560 | model_config | fp8 | 19.88 | aiter | 16.68 | 0.84x | tuning_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=5120,N=2048,K=8320 | 121.04 | 118.58 | -2.46 |
| M=2048,N=2560,K=2560 | 55.60 | 52.08 | -3.52 |
| M=256,N=7168,K=2048 | 30.20 | 22.25 | -7.95 |
| M=128,N=7168,K=2048 | 28.76 | 20.32 | -8.44 |
| M=128,N=2560,K=2560 | 19.88 | 11.00 | -8.88 |
| M=16,N=7168,K=7168 | 20.44 | 11.42 | -9.02 |

## Diagnosis

- `M=4096,N=8192,K=8192` (fp8, vs-best 0.26x): **tuning_gap**
  - evidence: kernel-only vs-best 0.26x for args={'K': 8192, 'M': 4096, 'N': 8192}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=2048,N=7168,K=7168` (fp8, vs-best 0.36x): **tuning_gap**
  - evidence: kernel-only vs-best 0.36x for args={'K': 7168, 'M': 2048, 'N': 7168}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=2048,N=2560,K=2560` (fp8, vs-best 0.61x): **tuning_gap**
  - evidence: kernel-only vs-best 0.61x for args={'K': 2560, 'M': 2048, 'N': 2560}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=128,N=7168,K=2048` (fp8, vs-best 0.61x): **tuning_gap**
  - evidence: kernel-only vs-best 0.61x for args={'K': 2048, 'M': 128, 'N': 7168}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=256,N=7168,K=2048` (fp8, vs-best 0.67x): **tuning_gap**
  - evidence: kernel-only vs-best 0.67x for args={'K': 2048, 'M': 256, 'N': 7168}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=5120,N=5120,K=8320` (fp8, vs-best 0.74x): **tuning_gap**
  - evidence: kernel-only vs-best 0.74x for args={'K': 8320, 'M': 5120, 'N': 5120}
  - likely fix: profile the hot shape and add an op-specific diagnosis

## Promotion Decision

**tune_needed** — sub-parity overall (geomean 0.72x); wins on its target regime but needs per-shape tuning + bug fixes before broad promotion

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 14/14.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops preshuffle_gemm
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op preshuffle_gemm --shape-ledger benchmarks/examples/preshuffle_gemm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/preshuffle_gemm/baseline_matrix.yaml \
  --out benchmarks/examples/preshuffle_gemm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/preshuffle_gemm/shape_ledger.jsonl \
  --results benchmarks/examples/preshuffle_gemm/benchmark_results.jsonl --out benchmarks/examples/preshuffle_gemm/benchmark_summary.md \
  --kernel preshuffle_gemm
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
