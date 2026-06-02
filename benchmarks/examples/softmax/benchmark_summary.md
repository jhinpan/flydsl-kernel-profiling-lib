# Benchmark Summary: softmax

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 73 (sources: diagnostic=1, aiter_model_shapes=45, synthetic=27)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=292.

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
| unweighted geomean vs best | 1.08x  (n=73) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter_triton | 1.53x  (n=73) |
| vs triton | 1.09x  (n=73) |
| vs pytorch | 1.27x  (n=73) |
| worst hot shape | 0.65x  (M=16384,N=128 vs pytorch) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 27 | 1.13x |
| diagnostic | 1 | 1.48x |
| model_config | 45 | 1.04x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 15 | 1.05x |
| GPT-OSS 120B | 5 | 1.08x |
| Llama3 405B | 5 | 1.05x |
| Llama3 70B | 5 | 1.08x |
| Llama3 8B | 5 | 1.00x |
| Llama4 Maverick | 5 | 1.12x |
| Qwen3-235B-A22B | 10 | 0.96x |
| diagnostic | 1 | 1.48x |
| synthetic | 27 | 1.13x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=4096,N=4097 | synthetic | bf16 | 26.44 | triton | 49.28 | 1.86x |
| M=4096,N=8193 | synthetic | bf16 | 47.00 | triton | 73.96 | 1.57x |
| M=4096,N=5333 | synthetic | bf16 | 32.28 | triton | 50.36 | 1.56x |
| M=32768,N=8192 | diagnostic | bf16 | 217.40 | pytorch | 321.16 | 1.48x |
| M=131072,N=8192 | synthetic | bf16 | 809.85 | pytorch | 1192.49 | 1.47x |
| M=16384,N=1536 | model_config | bf16 | 39.12 | triton | 56.04 | 1.43x |
| M=16384,N=5120 | model_config | bf16 | 90.60 | pytorch | 124.76 | 1.38x |
| M=4096,N=3000 | synthetic | bf16 | 23.08 | triton | 30.88 | 1.34x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=16384,N=128 | model_config | bf16 | 26.56 | pytorch | 17.20 | 0.65x | implementation_gap |
| M=16384,N=512 | model_config | bf16 | 31.88 | triton | 26.60 | 0.83x | implementation_gap |
| M=1,N=16384 | model_config | bf16 | 16.72 | triton | 15.32 | 0.92x | implementation_gap |
| M=1,N=7168 | model_config | bf16 | 14.68 | triton | 13.84 | 0.94x | implementation_gap |
| M=1,N=4096 | model_config | bf16 | 14.00 | triton | 13.24 | 0.95x | implementation_gap |
| M=1,N=8191 | synthetic | bf16 | 14.68 | triton | 13.96 | 0.95x | ok |
| M=1,N=3000 | synthetic | bf16 | 14.04 | triton | 13.36 | 0.95x | ok |
| M=1,N=8192 | synthetic | bf16 | 14.84 | triton | 14.16 | 0.95x | ok |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=256,N=128 | 13.08 | 33.34 | 20.26 |
| M=1,N=4096 | 13.72 | 33.90 | 20.18 |
| M=32,N=5120 | 13.52 | 32.96 | 19.44 |
| M=32,N=128 | 13.28 | 32.66 | 19.38 |
| M=1,N=128 | 13.20 | 32.46 | 19.26 |
| M=1,N=512 | 13.48 | 32.68 | 19.20 |

## Diagnosis

- `M=16384,N=128` (bf16, vs-best 0.65x): **implementation_gap**
  - evidence: FlyDSL softmax currently reports the generic scalar path; kernel-only vs-best 0.65x for args={'M': 16384, 'N': 128}
  - likely fix: profile and re-enable/fix the vectorized softmax path before promotion
- `M=16384,N=512` (bf16, vs-best 0.83x): **implementation_gap**
  - evidence: FlyDSL softmax currently reports the generic scalar path; kernel-only vs-best 0.83x for args={'M': 16384, 'N': 512}
  - likely fix: profile and re-enable/fix the vectorized softmax path before promotion
- `M=1,N=16384` (bf16, vs-best 0.92x): **implementation_gap**
  - evidence: FlyDSL softmax currently reports the generic scalar path; kernel-only vs-best 0.92x for args={'M': 1, 'N': 16384}
  - likely fix: profile and re-enable/fix the vectorized softmax path before promotion
- `M=1,N=7168` (bf16, vs-best 0.94x): **implementation_gap**
  - evidence: FlyDSL softmax currently reports the generic scalar path; kernel-only vs-best 0.94x for args={'M': 1, 'N': 7168}
  - likely fix: profile and re-enable/fix the vectorized softmax path before promotion
- `M=1,N=4096` (bf16, vs-best 0.95x): **implementation_gap**
  - evidence: FlyDSL softmax currently reports the generic scalar path; kernel-only vs-best 0.95x for args={'M': 1, 'N': 4096}
  - likely fix: profile and re-enable/fix the vectorized softmax path before promotion
- `M=1,N=8191` (bf16, vs-best 0.95x): **ok**
  - evidence: kernel-only vs-best 0.95x (near parity or better)
  - likely fix: none

## Promotion Decision

**promote** — overall kernel-only geomean is parity-or-better, no FlyDSL hard failures

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 73/73.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops softmax
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op softmax --shape-ledger benchmarks/examples/softmax/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/softmax/baseline_matrix.yaml \
  --out benchmarks/examples/softmax --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/softmax/shape_ledger.jsonl \
  --results benchmarks/examples/softmax/benchmark_results.jsonl --out benchmarks/examples/softmax/benchmark_summary.md \
  --kernel softmax
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
