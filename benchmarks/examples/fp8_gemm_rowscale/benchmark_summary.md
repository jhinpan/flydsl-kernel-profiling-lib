# Benchmark Summary: fp8_gemm_rowscale

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 14 (sources: model_config=12, synthetic=2)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=69.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 14 |
| FlyDSL correct + timed | 13 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 1 |
| measured FlyDSL-vs-baseline pairs | 13 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 0.69x  (n=13) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.90x  (n=13) |
| vs aiter_triton | 1.05x  (n=13) |
| vs gluon | 7.58x  (n=13) |
| vs pytorch | 0.69x  (n=13) |
| worst hot shape | 0.26x  (M=16,N=5120,K=8320 vs pytorch) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 2 | 1.04x |
| model_config | 11 | 0.64x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| deepseek-v3 | 5 | 0.75x |
| kimi-k2 | 3 | 0.68x |
| qwen3-32b | 3 | 0.45x |
| synthetic | 2 | 1.04x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=5120,N=5120,K=8320,8wave | model_config | fp8_e4m3 | 212.20 | pytorch | 234.96 | 1.11x |
| M=5120,N=5120,K=8320 | model_config | fp8_e4m3 | 233.16 | pytorch | 249.64 | 1.07x |
| M=9728,N=8192,K=8320 | synthetic | fp8_e4m3 | 565.93 | pytorch | 593.81 | 1.05x |
| M=8192,N=8192,K=8192 | synthetic | fp8_e4m3 | 472.72 | pytorch | 491.76 | 1.04x |
| M=8192,N=7168,K=7168,preshuffle_b | model_config | fp8_e4m3 | 414.32 | pytorch | 387.68 | 0.94x |
| M=2048,N=7168,K=2048 | model_config | fp8_e4m3 | 47.44 | pytorch | 43.72 | 0.92x |
| M=4096,N=7168,K=7168 | model_config | fp8_e4m3 | 206.64 | pytorch | 190.28 | 0.92x |
| M=512,N=2112,K=7168 | model_config | fp8_e4m3 | 38.32 | pytorch | 30.60 | 0.80x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=16,N=5120,K=8320 | model_config | fp8_e4m3 | 94.16 | pytorch | 24.68 | 0.26x | tuning_gap |
| M=128,N=2560,K=2560 | model_config | fp8_e4m3 | 52.20 | pytorch | 18.24 | 0.35x | tuning_gap |
| M=1,N=2560,K=2560 | model_config | fp8_e4m3 | 52.64 | aiter | 19.48 | 0.37x | tuning_gap |
| M=256,N=7168,K=7168 | model_config | fp8_e4m3 | 97.40 | pytorch | 36.92 | 0.38x | tuning_gap |
| M=4096,N=2560,K=2560 | model_config | fp8_e4m3 | 54.60 | pytorch | 39.24 | 0.72x | tuning_gap |
| M=512,N=2112,K=7168 | model_config | fp8_e4m3 | 38.32 | pytorch | 30.60 | 0.80x | tuning_gap |
| M=4096,N=7168,K=7168 | model_config | fp8_e4m3 | 206.64 | pytorch | 190.28 | 0.92x | tuning_gap |
| M=2048,N=7168,K=2048 | model_config | fp8_e4m3 | 47.44 | pytorch | 43.72 | 0.92x | tuning_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=4096,N=7168,K=7168 | 206.64 | 205.60 | -1.04 |
| M=2048,N=7168,K=2048 | 47.44 | 44.02 | -3.42 |
| M=4096,N=2560,K=2560 | 54.60 | 49.88 | -4.72 |
| M=256,N=7168,K=7168 | 97.40 | 92.26 | -5.14 |
| M=128,N=2560,K=2560 | 52.20 | 46.20 | -6.00 |
| M=512,N=2112,K=7168 | 38.32 | 31.82 | -6.50 |

## Diagnosis

- `M=16,N=5120,K=8320` (fp8_e4m3, vs-best 0.26x): **tuning_gap**
  - evidence: kernel-only vs-best 0.26x for args={'K': 8320, 'M': 16, 'N': 5120}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=128,N=2560,K=2560` (fp8_e4m3, vs-best 0.35x): **tuning_gap**
  - evidence: kernel-only vs-best 0.35x for args={'K': 2560, 'M': 128, 'N': 2560}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=1,N=2560,K=2560` (fp8_e4m3, vs-best 0.37x): **tuning_gap**
  - evidence: kernel-only vs-best 0.37x for args={'K': 2560, 'M': 1, 'N': 2560}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=256,N=7168,K=7168` (fp8_e4m3, vs-best 0.38x): **tuning_gap**
  - evidence: kernel-only vs-best 0.38x for args={'K': 7168, 'M': 256, 'N': 7168}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=4096,N=2560,K=2560` (fp8_e4m3, vs-best 0.72x): **tuning_gap**
  - evidence: kernel-only vs-best 0.72x for args={'K': 2560, 'M': 4096, 'N': 2560}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=512,N=2112,K=7168` (fp8_e4m3, vs-best 0.80x): **tuning_gap**
  - evidence: kernel-only vs-best 0.80x for args={'K': 7168, 'M': 512, 'N': 2112}
  - likely fix: profile the hot shape and add an op-specific diagnosis

## Promotion Decision

**rewrite_needed** — well below parity (geomean 0.69x); structural rework needed

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 13/14.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops fp8_gemm_rowscale
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op fp8_gemm_rowscale --shape-ledger benchmarks/examples/fp8_gemm_rowscale/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/fp8_gemm_rowscale/baseline_matrix.yaml \
  --out benchmarks/examples/fp8_gemm_rowscale --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/fp8_gemm_rowscale/shape_ledger.jsonl \
  --results benchmarks/examples/fp8_gemm_rowscale/benchmark_results.jsonl --out benchmarks/examples/fp8_gemm_rowscale/benchmark_summary.md \
  --kernel fp8_gemm_rowscale
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
