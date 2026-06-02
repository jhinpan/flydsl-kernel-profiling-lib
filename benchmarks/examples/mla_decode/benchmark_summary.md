# Benchmark Summary: mla_decode

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 14 (sources: model_config=14)
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
| unweighted geomean vs best | 0.94x  (n=14) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.94x  (n=14) |
| vs pytorch | 19.36x  (n=14) |
| worst hot shape | 0.88x  (batch=1,ctx_len=64,decode_qlen=1 vs aiter) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 14 | 0.94x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 6 | 0.93x |
| DeepSeek-V3 | 5 | 0.95x |
| Kimi-K2 | 3 | 0.93x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| batch=4,ctx_len=2048,decode_qlen=1 | model_config | fp8 | 40.28 | aiter | 41.16 | 1.02x |
| batch=128,ctx_len=1024,decode_qlen=1 | model_config | fp8 | 86.16 | aiter | 84.16 | 0.98x |
| batch=33,ctx_len=2333,decode_qlen=1 | model_config | fp8 | 64.92 | aiter | 62.80 | 0.97x |
| batch=32,ctx_len=8192,decode_qlen=1 | model_config | fp8 | 117.12 | aiter | 112.56 | 0.96x |
| batch=8,ctx_len=4096,decode_qlen=1 | model_config | fp8 | 50.68 | aiter | 48.28 | 0.95x |
| batch=64,ctx_len=2048,decode_qlen=1 | model_config | fp8 | 83.52 | aiter | 79.32 | 0.95x |
| batch=8,ctx_len=8192,decode_qlen=1 | model_config | fp8 | 63.32 | aiter | 59.68 | 0.94x |
| batch=16,ctx_len=4096,decode_qlen=1 | model_config | fp8 | 67.36 | aiter | 63.48 | 0.94x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| batch=1,ctx_len=64,decode_qlen=1 | model_config | fp8 | 28.88 | aiter | 25.36 | 0.88x | tuning_gap |
| batch=1,ctx_len=63,decode_qlen=1 | model_config | fp8 | 29.32 | aiter | 25.84 | 0.88x | tuning_gap |
| batch=1,ctx_len=128,decode_qlen=1 | model_config | fp8 | 30.40 | aiter | 27.12 | 0.89x | tuning_gap |
| batch=1,ctx_len=32768,decode_qlen=1 | model_config | fp8 | 86.40 | aiter | 79.60 | 0.92x | tuning_gap |
| batch=1,ctx_len=1024,decode_qlen=1 | model_config | fp8 | 33.32 | aiter | 30.80 | 0.92x | tuning_gap |
| batch=2,ctx_len=16384,decode_qlen=1 | model_config | fp8 | 62.36 | aiter | 58.64 | 0.94x | tuning_gap |
| batch=16,ctx_len=4096,decode_qlen=1 | model_config | fp8 | 67.36 | aiter | 63.48 | 0.94x | tuning_gap |
| batch=8,ctx_len=8192,decode_qlen=1 | model_config | fp8 | 63.32 | aiter | 59.68 | 0.94x | tuning_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| batch=1,ctx_len=128,decode_qlen=1 | 30.40 | 309.00 | 278.60 |
| batch=1,ctx_len=64,decode_qlen=1 | 28.88 | 306.90 | 278.02 |
| batch=1,ctx_len=63,decode_qlen=1 | 29.32 | 302.94 | 273.62 |
| batch=1,ctx_len=1024,decode_qlen=1 | 33.32 | 305.38 | 272.06 |
| batch=4,ctx_len=2048,decode_qlen=1 | 40.28 | 301.20 | 260.92 |
| batch=8,ctx_len=4096,decode_qlen=1 | 50.68 | 299.66 | 248.98 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 279us host launch overhead (kernel 30.4us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `batch=1,ctx_len=64,decode_qlen=1` (fp8, vs-best 0.88x): **tuning_gap**
  - evidence: kernel-only vs-best 0.88x for args={'batch': 1, 'ctx_len': 64, 'decode_qlen': 1}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `batch=1,ctx_len=63,decode_qlen=1` (fp8, vs-best 0.88x): **tuning_gap**
  - evidence: kernel-only vs-best 0.88x for args={'batch': 1, 'ctx_len': 63, 'decode_qlen': 1}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `batch=1,ctx_len=128,decode_qlen=1` (fp8, vs-best 0.89x): **tuning_gap**
  - evidence: kernel-only vs-best 0.89x for args={'batch': 1, 'ctx_len': 128, 'decode_qlen': 1}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `batch=1,ctx_len=32768,decode_qlen=1` (fp8, vs-best 0.92x): **tuning_gap**
  - evidence: kernel-only vs-best 0.92x for args={'batch': 1, 'ctx_len': 32768, 'decode_qlen': 1}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `batch=1,ctx_len=1024,decode_qlen=1` (fp8, vs-best 0.92x): **tuning_gap**
  - evidence: kernel-only vs-best 0.92x for args={'batch': 1, 'ctx_len': 1024, 'decode_qlen': 1}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `batch=2,ctx_len=16384,decode_qlen=1` (fp8, vs-best 0.94x): **tuning_gap**
  - evidence: kernel-only vs-best 0.94x for args={'batch': 2, 'ctx_len': 16384, 'decode_qlen': 1}
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
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops mla_decode
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op mla_decode --shape-ledger benchmarks/examples/mla_decode/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/mla_decode/baseline_matrix.yaml \
  --out benchmarks/examples/mla_decode --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/mla_decode/shape_ledger.jsonl \
  --results benchmarks/examples/mla_decode/benchmark_results.jsonl --out benchmarks/examples/mla_decode/benchmark_summary.md \
  --kernel mla_decode
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
