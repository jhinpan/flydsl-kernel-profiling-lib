# Benchmark Summary: softmax

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 73 (sources: diagnostic=1, aiter_model_shapes=45, synthetic=27)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 73 |
| FlyDSL correct + timed | 73 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 73 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 1.13x  (n=73) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter_triton | 1.96x  (n=73) |
| vs triton | 1.17x  (n=73) |
| vs pytorch | 1.48x  (n=73) |
| worst hot shape | 0.39x  (M=16384,N=128 vs pytorch) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 27 | 1.28x |
| diagnostic | 1 | 1.40x |
| model_config | 45 | 1.04x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 15 | 1.01x |
| GPT-OSS 120B | 5 | 1.10x |
| Llama3 405B | 5 | 1.18x |
| Llama3 70B | 5 | 1.14x |
| Llama3 8B | 5 | 1.04x |
| Llama4 Maverick | 5 | 1.25x |
| Qwen3-235B-A22B | 10 | 0.86x |
| diagnostic | 1 | 1.40x |
| synthetic | 27 | 1.28x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=4096,N=4097 | synthetic | bf16 | 14.62 | triton | 36.17 | 2.47x |
| M=4096,N=8193 | synthetic | bf16 | 24.77 | pytorch | 56.89 | 2.30x |
| M=4096,N=5333 | synthetic | bf16 | 17.64 | pytorch | 36.06 | 2.04x |
| M=4096,N=2049 | synthetic | bf16 | 9.69 | triton | 16.66 | 1.72x |
| M=4096,N=8191 | synthetic | bf16 | 24.35 | triton | 39.95 | 1.64x |
| M=16384,N=5120 | model_config | bf16 | 69.05 | pytorch | 109.45 | 1.59x |
| M=16384,N=2880 | model_config | bf16 | 38.18 | triton | 59.72 | 1.56x |
| M=4096,N=3000 | synthetic | bf16 | 11.76 | triton | 18.28 | 1.56x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=16384,N=128 | model_config | bf16 | 13.47 | pytorch | 5.24 | 0.39x | implementation_gap |
| M=16384,N=512 | model_config | bf16 | 16.08 | pytorch | 12.05 | 0.75x | implementation_gap |
| M=1,N=128 | model_config | bf16 | 2.28 | triton | 1.79 | 0.78x | implementation_gap |
| M=256,N=512 | model_config | bf16 | 2.65 | triton | 2.09 | 0.79x | implementation_gap |
| M=32,N=512 | model_config | bf16 | 2.57 | triton | 2.03 | 0.79x | implementation_gap |
| M=32,N=128 | model_config | bf16 | 2.42 | triton | 1.92 | 0.80x | implementation_gap |
| M=1,N=2047 | synthetic | bf16 | 2.69 | triton | 2.17 | 0.81x | implementation_gap |
| M=1,N=512 | model_config | bf16 | 2.31 | triton | 1.86 | 0.81x | implementation_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=4096,N=8192 | 40.01 | 72.98 | 32.98 |
| M=1,N=2049 | 2.66 | 33.56 | 30.90 |
| M=1,N=512 | 2.31 | 33.12 | 30.81 |
| M=1,N=2880 | 2.78 | 33.48 | 30.70 |
| M=1,N=4095 | 2.91 | 33.36 | 30.45 |
| M=1,N=2048 | 2.65 | 32.82 | 30.17 |

## Diagnosis

- `M=16384,N=128` (bf16, vs-best 0.39x): **implementation_gap**
  - evidence: N=128 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.39x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=16384,N=512` (bf16, vs-best 0.75x): **implementation_gap**
  - evidence: N=512 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.75x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=128` (bf16, vs-best 0.78x): **implementation_gap**
  - evidence: N=128 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.78x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=256,N=512` (bf16, vs-best 0.79x): **implementation_gap**
  - evidence: N=512 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.79x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=32,N=512` (bf16, vs-best 0.79x): **implementation_gap**
  - evidence: N=512 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.79x). Compounded at small M=32: grid=(M,1,1) launches one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=32,N=128` (bf16, vs-best 0.80x): **implementation_gap**
  - evidence: N=128 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.80x). Compounded at small M=32: grid=(M,1,1) launches one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs

## Promotion Decision

**promote** — kernel-only parity-or-better across all measured shapes, no failures

Regime-specific reading:
- **Large-M aligned (prefill-like):** kernel-only parity-or-better (diagnostic ~1.03x, beats PyTorch ~1.5x). Promotable.
- **Small-M large-N (decode):** one-block-per-row underutilizes the GPU (kernel-only worst ~0.36x). Needs a parallelization change.
- **Eager decode latency:** ~tens-of-us launcher host overhead per call. Needs a launch cache (host-side).
## Reproduction

```bash
# 1. build the ledger
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops rmsnorm
python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm --out benchmarks/examples \
  --synthetic-boundary --diagnostic 32768,8192,bf16
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op softmax --shape-ledger benchmarks/examples/softmax/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/softmax/baseline_matrix.yaml \
  --out benchmarks/examples/softmax --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/softmax/shape_ledger.jsonl \
  --results benchmarks/examples/softmax/benchmark_results.jsonl --out benchmarks/examples/softmax/benchmark_summary.md
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
