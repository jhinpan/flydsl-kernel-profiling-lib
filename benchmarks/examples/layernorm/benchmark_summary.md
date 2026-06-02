# Benchmark Summary: layernorm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 73 (sources: diagnostic=1, aiter_model_shapes=45, synthetic=27)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=290.

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
| unweighted geomean vs best | 0.88x  (n=73) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.90x  (n=72) |
| vs aiter_triton | 1.00x  (n=72) |
| vs pytorch | 1.20x  (n=73) |
| worst hot shape | 0.52x  (M=32,N=16384 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 27 | 1.00x |
| diagnostic | 1 | 1.02x |
| model_config | 45 | 0.81x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 15 | 0.83x |
| GPT-OSS 120B | 5 | 0.81x |
| Llama3 405B | 5 | 0.59x |
| Llama3 70B | 5 | 0.99x |
| Llama3 8B | 5 | 0.76x |
| Llama4 Maverick | 5 | 0.77x |
| Qwen3-235B-A22B | 10 | 0.84x |
| diagnostic | 1 | 1.02x |
| synthetic | 27 | 1.00x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=4096,N=2049 | synthetic | bf16 | 24.28 | pytorch | 51.60 | 2.13x |
| M=4096,N=8193 | synthetic | bf16 | 55.44 | aiter | 114.92 | 2.07x |
| M=4096,N=4097 | synthetic | bf16 | 32.92 | aiter | 59.44 | 1.81x |
| M=4096,N=4095 | synthetic | bf16 | 34.04 | aiter | 57.68 | 1.69x |
| M=4096,N=5333 | synthetic | bf16 | 39.04 | aiter | 60.84 | 1.56x |
| M=4096,N=8191 | synthetic | bf16 | 54.12 | aiter | 66.92 | 1.24x |
| M=131072,N=8192 | synthetic | bf16 | 851.89 | aiter | 894.73 | 1.05x |
| M=2048,N=8192 | model_config | bf16 | 26.08 | aiter | 26.56 | 1.02x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=32,N=16384 | model_config | bf16 | 29.52 | aiter_triton | 15.44 | 0.52x | implementation_gap |
| M=1,N=16384 | model_config | bf16 | 30.44 | aiter_triton | 16.40 | 0.54x | implementation_gap |
| M=256,N=16384 | model_config | bf16 | 31.00 | aiter_triton | 16.80 | 0.54x | tuning_gap |
| M=1,N=12288 | synthetic | bf16 | 28.32 | aiter_triton | 16.76 | 0.59x | implementation_gap |
| M=16384,N=512 | model_config | bf16 | 32.40 | aiter | 19.36 | 0.60x | implementation_gap |
| M=16384,N=16384 | model_config | bf16 | 373.16 | aiter_triton | 239.48 | 0.64x | tuning_gap |
| M=32,N=7168 | model_config | bf16 | 20.08 | aiter | 13.36 | 0.67x | implementation_gap |
| M=1,N=7168 | model_config | bf16 | 20.28 | aiter | 13.56 | 0.67x | implementation_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=32,N=512 | 13.28 | 56.50 | 43.22 |
| M=2048,N=512 | 13.68 | 55.94 | 42.26 |
| M=1,N=128 | 13.00 | 55.02 | 42.02 |
| M=32,N=128 | 13.28 | 55.26 | 41.98 |
| M=256,N=512 | 13.32 | 54.62 | 41.30 |
| M=256,N=128 | 13.36 | 54.60 | 41.24 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 43us host launch overhead (kernel 13.3us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `M=32,N=16384` (bf16, vs-best 0.52x): **implementation_gap**
  - evidence: aligned N=16384 but small M=32: FlyDSL launches grid=(M,1,1) -> one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied; kernel-only vs-best 0.52x).
  - likely fix: parallelize across N (split-N / persistent blocks) for small M so occupancy is not capped at M
- `M=1,N=16384` (bf16, vs-best 0.54x): **implementation_gap**
  - evidence: aligned N=16384 but small M=1: FlyDSL launches grid=(M,1,1) -> one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied; kernel-only vs-best 0.54x).
  - likely fix: parallelize across N (split-N / persistent blocks) for small M so occupancy is not capped at M
- `M=256,N=16384` (bf16, vs-best 0.54x): **tuning_gap**
  - evidence: aligned large-M shape but vs-best 0.54x; fixed FlyDSL schedule vs tuned baseline, no structural cause evident
  - likely fix: add a per-shape tuned schedule (block size, vector width, waves); capture rocprofv3 to confirm
- `M=1,N=12288` (bf16, vs-best 0.59x): **implementation_gap**
  - evidence: aligned N=12288 but small M=1: FlyDSL launches grid=(M,1,1) -> one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied; kernel-only vs-best 0.59x).
  - likely fix: parallelize across N (split-N / persistent blocks) for small M so occupancy is not capped at M
- `M=16384,N=512` (bf16, vs-best 0.60x): **implementation_gap**
  - evidence: N=512 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.60x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=16384,N=16384` (bf16, vs-best 0.64x): **tuning_gap**
  - evidence: aligned large-M shape but vs-best 0.64x; fixed FlyDSL schedule vs tuned baseline, no structural cause evident
  - likely fix: add a per-shape tuned schedule (block size, vector width, waves); capture rocprofv3 to confirm

## Promotion Decision

**tune_needed** — sub-parity overall (geomean 0.88x); wins on its target regime but needs per-shape tuning + bug fixes before broad promotion

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 73/73.
- Norm conclusions are shape-regime dependent; keep the fast-path/generic-path split visible.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops layernorm
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op layernorm --shape-ledger benchmarks/examples/layernorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/layernorm/baseline_matrix.yaml \
  --out benchmarks/examples/layernorm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/layernorm/shape_ledger.jsonl \
  --results benchmarks/examples/layernorm/benchmark_results.jsonl --out benchmarks/examples/layernorm/benchmark_summary.md \
  --kernel layernorm
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
