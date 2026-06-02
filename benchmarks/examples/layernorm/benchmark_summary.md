# Benchmark Summary: layernorm

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
| unweighted geomean vs best | 0.84x  (n=73) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.88x  (n=72) |
| vs aiter_triton | 1.05x  (n=72) |
| vs pytorch | 1.49x  (n=73) |
| worst hot shape | 0.38x  (M=1,N=16384 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 27 | 1.07x |
| diagnostic | 1 | 1.10x |
| model_config | 45 | 0.71x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 15 | 0.71x |
| GPT-OSS 120B | 5 | 0.71x |
| Llama3 405B | 5 | 0.50x |
| Llama3 70B | 5 | 1.11x |
| Llama3 8B | 5 | 0.64x |
| Llama4 Maverick | 5 | 0.66x |
| Qwen3-235B-A22B | 10 | 0.72x |
| diagnostic | 1 | 1.10x |
| synthetic | 27 | 1.07x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=4096,N=2049 | synthetic | bf16 | 10.42 | pytorch | 34.29 | 3.29x |
| M=4096,N=4097 | synthetic | bf16 | 19.93 | aiter | 46.89 | 2.35x |
| M=4096,N=8193 | synthetic | bf16 | 37.35 | aiter | 87.24 | 2.34x |
| M=4096,N=4095 | synthetic | bf16 | 19.81 | aiter | 43.65 | 2.20x |
| M=4096,N=5333 | synthetic | bf16 | 25.53 | aiter | 49.47 | 1.94x |
| M=4096,N=8191 | synthetic | bf16 | 37.33 | aiter | 56.14 | 1.50x |
| M=1,N=2049 | synthetic | bf16 | 3.23 | aiter | 4.00 | 1.24x |
| M=1,N=8192 | model_config | bf16 | 2.70 | aiter_triton | 3.09 | 1.15x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=1,N=16384 | model_config | bf16 | 11.36 | aiter_triton | 4.34 | 0.38x | implementation_gap |
| M=32,N=16384 | model_config | bf16 | 11.82 | aiter_triton | 4.66 | 0.39x | implementation_gap |
| M=256,N=16384 | model_config | bf16 | 12.69 | aiter_triton | 6.12 | 0.48x | tuning_gap |
| M=1,N=12288 | synthetic | bf16 | 8.91 | aiter_triton | 4.31 | 0.48x | implementation_gap |
| M=16384,N=512 | model_config | bf16 | 14.76 | aiter | 7.52 | 0.51x | implementation_gap |
| M=1,N=7168 | model_config | bf16 | 5.99 | aiter | 3.07 | 0.51x | implementation_gap |
| M=32,N=7168 | model_config | bf16 | 6.31 | aiter | 3.28 | 0.52x | implementation_gap |
| M=256,N=7168 | model_config | bf16 | 6.69 | aiter | 3.81 | 0.57x | implementation_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=1,N=2880 | 3.73 | 58.54 | 54.81 |
| M=1,N=2047 | 2.97 | 57.16 | 54.19 |
| M=1,N=512 | 2.27 | 56.20 | 53.93 |
| M=1,N=8192 | 2.70 | 56.40 | 53.70 |
| M=1,N=2048 | 3.12 | 56.50 | 53.38 |
| M=1,N=128 | 1.86 | 54.78 | 52.92 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 55us host launch overhead (kernel 3.7us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `M=1,N=16384` (bf16, vs-best 0.38x): **implementation_gap**
  - evidence: aligned N=16384 but small M=1: FlyDSL launches grid=(M,1,1) -> one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied; kernel-only vs-best 0.38x).
  - likely fix: parallelize across N (split-N / persistent blocks) for small M so occupancy is not capped at M
- `M=32,N=16384` (bf16, vs-best 0.39x): **implementation_gap**
  - evidence: aligned N=16384 but small M=32: FlyDSL launches grid=(M,1,1) -> one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied; kernel-only vs-best 0.39x).
  - likely fix: parallelize across N (split-N / persistent blocks) for small M so occupancy is not capped at M
- `M=256,N=16384` (bf16, vs-best 0.48x): **tuning_gap**
  - evidence: aligned large-M shape but vs-best 0.48x; fixed FlyDSL schedule vs tuned baseline, no structural cause evident
  - likely fix: add a per-shape tuned schedule (block size, vector width, waves); capture rocprofv3 to confirm
- `M=1,N=12288` (bf16, vs-best 0.48x): **implementation_gap**
  - evidence: aligned N=12288 but small M=1: FlyDSL launches grid=(M,1,1) -> one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied; kernel-only vs-best 0.48x).
  - likely fix: parallelize across N (split-N / persistent blocks) for small M so occupancy is not capped at M
- `M=16384,N=512` (bf16, vs-best 0.51x): **implementation_gap**
  - evidence: N=512 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.51x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=7168` (bf16, vs-best 0.51x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.51x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs

## Promotion Decision

**tune_needed** — sub-parity overall (geomean 0.84x); wins on its target regime but needs per-shape tuning + bug fixes before broad promotion

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
  --op layernorm --shape-ledger benchmarks/examples/layernorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/layernorm/baseline_matrix.yaml \
  --out benchmarks/examples/layernorm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/layernorm/shape_ledger.jsonl \
  --results benchmarks/examples/layernorm/benchmark_results.jsonl --out benchmarks/examples/layernorm/benchmark_summary.md
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
