# Benchmark Summary: blockscale_preshuffle_gemm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 14 (sources: model_config=14)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=53.

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
| unweighted geomean vs best | 0.94x  (n=14) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.96x  (n=11) |
| vs aiter_triton | 1.05x  (n=14) |
| vs pytorch | 6.23x  (n=14) |
| worst hot shape | 0.50x  (M=1,N=2112,K=7168,out=bf16 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 14 | 0.94x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| deepseek-v3 | 10 | 0.91x |
| kimi-k2 | 2 | 0.84x |
| qwen3 | 2 | 1.21x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=2048,N=2560,K=2560,out=bf16 | model_config | fp8 | 42.80 | aiter_triton | 65.24 | 1.52x |
| M=1024,N=2112,K=7168,out=bf16 | model_config | fp8 | 69.36 | aiter_triton | 93.44 | 1.35x |
| M=1024,N=3072,K=1536,out=bf16 | model_config | fp8 | 25.40 | aiter_triton | 32.72 | 1.29x |
| M=7,N=3072,K=1536,out=bf16 | model_config | fp8 | 16.08 | aiter_triton | 16.52 | 1.03x |
| M=16,N=2560,K=2560,out=bf16 | model_config | fp8 | 19.88 | aiter_triton | 19.20 | 0.97x |
| M=16,N=7168,K=2304,out=bf16 | model_config | fp8 | 19.68 | aiter | 18.96 | 0.96x |
| M=16,N=7168,K=2304,out=bf16 | model_config | fp8 | 20.36 | aiter_triton | 19.20 | 0.94x |
| M=256,N=3072,K=1536,out=fp16 | model_config | fp8 | 19.00 | aiter | 17.28 | 0.91x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=1,N=2112,K=7168,out=bf16 | model_config | fp8 | 35.44 | aiter_triton | 17.88 | 0.50x | tuning_gap |
| M=1024,N=7168,K=2304,out=bf16 | model_config | fp8 | 52.72 | aiter | 39.72 | 0.75x | tuning_gap |
| M=33,N=2112,K=7168,out=bf16 | model_config | fp8 | 28.80 | aiter_triton | 22.08 | 0.77x | tuning_gap |
| M=4096,N=7168,K=2304,out=bf16 | model_config | fp8 | 136.12 | aiter | 106.44 | 0.78x | tuning_gap |
| M=64,N=7168,K=2304,out=bf16 | model_config | fp8 | 21.44 | aiter | 18.96 | 0.88x | tuning_gap |
| M=64,N=3072,K=1536,out=bf16 | model_config | fp8 | 17.92 | aiter_triton | 16.08 | 0.90x | tuning_gap |
| M=256,N=3072,K=1536,out=fp16 | model_config | fp8 | 19.00 | aiter | 17.28 | 0.91x | tuning_gap |
| M=16,N=7168,K=2304,out=bf16 | model_config | fp8 | 20.36 | aiter_triton | 19.20 | 0.94x | tuning_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=1024,N=7168,K=2304,out=bf16 | 52.72 | 50.54 | -2.18 |
| M=1024,N=2112,K=7168,out=bf16 | 69.36 | 65.68 | -3.68 |
| M=2048,N=2560,K=2560,out=bf16 | 42.80 | 38.20 | -4.60 |
| M=16,N=7168,K=2304,out=bf16 | 19.68 | 14.20 | -5.48 |
| M=33,N=2112,K=7168,out=bf16 | 28.80 | 21.52 | -7.28 |
| M=4096,N=7168,K=2304,out=bf16 | 136.12 | 128.64 | -7.48 |

## Diagnosis

- `M=1,N=2112,K=7168,out=bf16` (fp8, vs-best 0.50x): **tuning_gap**
  - evidence: kernel-only vs-best 0.50x for args={'K': 7168, 'M': 1, 'N': 2112, 'out_dtype': 'bf16'}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=1024,N=7168,K=2304,out=bf16` (fp8, vs-best 0.75x): **tuning_gap**
  - evidence: kernel-only vs-best 0.75x for args={'K': 2304, 'M': 1024, 'N': 7168, 'out_dtype': 'bf16'}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=33,N=2112,K=7168,out=bf16` (fp8, vs-best 0.77x): **tuning_gap**
  - evidence: kernel-only vs-best 0.77x for args={'K': 7168, 'M': 33, 'N': 2112, 'out_dtype': 'bf16'}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=4096,N=7168,K=2304,out=bf16` (fp8, vs-best 0.78x): **tuning_gap**
  - evidence: kernel-only vs-best 0.78x for args={'K': 2304, 'M': 4096, 'N': 7168, 'out_dtype': 'bf16'}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=64,N=7168,K=2304,out=bf16` (fp8, vs-best 0.88x): **tuning_gap**
  - evidence: kernel-only vs-best 0.88x for args={'K': 2304, 'M': 64, 'N': 7168, 'out_dtype': 'bf16'}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `M=64,N=3072,K=1536,out=bf16` (fp8, vs-best 0.90x): **tuning_gap**
  - evidence: kernel-only vs-best 0.90x for args={'K': 1536, 'M': 64, 'N': 3072, 'out_dtype': 'bf16'}
  - likely fix: profile the hot shape and add an op-specific diagnosis

## Promotion Decision

**tune_needed** — geomean 0.94x

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 14/14.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops blockscale_preshuffle_gemm
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op blockscale_preshuffle_gemm --shape-ledger benchmarks/examples/blockscale_preshuffle_gemm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/blockscale_preshuffle_gemm/baseline_matrix.yaml \
  --out benchmarks/examples/blockscale_preshuffle_gemm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/blockscale_preshuffle_gemm/shape_ledger.jsonl \
  --results benchmarks/examples/blockscale_preshuffle_gemm/benchmark_results.jsonl --out benchmarks/examples/blockscale_preshuffle_gemm/benchmark_summary.md \
  --kernel blockscale_preshuffle_gemm
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
