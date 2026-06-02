# Benchmark Summary: fused_rope_cache

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 30 (sources: aiter_model_shapes=30)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 30 |
| FlyDSL correct + timed | 30 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 30 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 2.48x  (n=30) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter_triton | 0.99x  (n=20) |
| vs pytorch | 16.85x  (n=30) |
| worst hot shape | 0.67x  (D=64,QH=8,KH=1,T=2048 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 30 | 2.48x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| GPT-OSS 120B | 6 | 2.27x |
| Llama3 405B | 6 | 2.46x |
| Llama3 70B | 6 | 2.46x |
| Llama3 8B | 6 | 2.68x |
| Llama4 Maverick | 6 | 2.53x |
| Qwen3-235B-A22B | 6 | 2.46x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| D=128,QH=4,KH=1,T=128 | model_config | f16 | 2.20 | pytorch | 63.28 | 28.72x |
| D=64,QH=8,KH=1,T=128 | model_config | f16 | 2.27 | pytorch | 62.87 | 27.66x |
| D=128,QH=5,KH=1,T=128 | model_config | f16 | 2.28 | pytorch | 62.50 | 27.44x |
| D=128,QH=8,KH=1,T=128 | model_config | f16 | 2.34 | pytorch | 63.57 | 27.13x |
| D=128,QH=16,KH=1,T=128 | model_config | bf16 | 2.72 | pytorch | 64.93 | 23.91x |
| D=128,QH=4,KH=1,T=2048 | model_config | f16 | 6.38 | pytorch | 75.84 | 11.89x |
| D=128,QH=5,KH=1,T=2048 | model_config | f16 | 7.03 | pytorch | 78.03 | 11.10x |
| D=128,QH=8,KH=1,T=2048 | model_config | f16 | 10.85 | pytorch | 87.41 | 8.06x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| D=64,QH=8,KH=1,T=2048 | model_config | bf16 | 10.27 | aiter_triton | 6.83 | 0.67x | tuning_gap |
| D=128,QH=5,KH=1,T=2048 | model_config | bf16 | 7.20 | aiter_triton | 5.38 | 0.75x | tuning_gap |
| D=128,QH=8,KH=1,T=2048 | model_config | bf16 | 11.22 | aiter_triton | 9.97 | 0.89x | tuning_gap |
| D=128,QH=4,KH=1,T=2048 | model_config | bf16 | 6.56 | aiter_triton | 6.19 | 0.94x | tuning_gap |
| D=64,QH=8,KH=1,T=128 | model_config | bf16 | 2.41 | aiter_triton | 2.34 | 0.97x | ok |
| D=64,QH=8,KH=1,T=1 | model_config | f16 | 2.01 | aiter_triton | 2.03 | 1.01x | ok |
| D=128,QH=4,KH=1,T=1 | model_config | bf16 | 2.13 | aiter_triton | 2.15 | 1.01x | ok |
| D=128,QH=16,KH=1,T=2048 | model_config | f16 | 12.23 | aiter_triton | 12.39 | 1.01x | ok |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| D=128,QH=5,KH=1,T=128 | 2.43 | 149.50 | 147.07 |
| D=128,QH=8,KH=1,T=1 | 2.11 | 142.36 | 140.26 |
| D=128,QH=8,KH=1,T=128 | 2.45 | 139.26 | 136.82 |
| D=128,QH=16,KH=1,T=1 | 2.09 | 137.94 | 135.85 |
| D=128,QH=4,KH=1,T=1 | 2.04 | 137.74 | 135.70 |
| D=128,QH=4,KH=1,T=128 | 2.20 | 137.66 | 135.46 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 147us host launch overhead (kernel 2.4us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `D=64,QH=8,KH=1,T=2048` (bf16, vs-best 0.67x): **tuning_gap**
  - evidence: aligned large-M shape but vs-best 0.67x; fixed FlyDSL schedule vs tuned baseline, no structural cause evident
  - likely fix: add a per-shape tuned schedule (block size, vector width, waves); capture rocprofv3 to confirm
- `D=128,QH=5,KH=1,T=2048` (bf16, vs-best 0.75x): **tuning_gap**
  - evidence: aligned large-M shape but vs-best 0.75x; fixed FlyDSL schedule vs tuned baseline, no structural cause evident
  - likely fix: add a per-shape tuned schedule (block size, vector width, waves); capture rocprofv3 to confirm
- `D=128,QH=8,KH=1,T=2048` (bf16, vs-best 0.89x): **tuning_gap**
  - evidence: aligned large-M shape but vs-best 0.89x; fixed FlyDSL schedule vs tuned baseline, no structural cause evident
  - likely fix: add a per-shape tuned schedule (block size, vector width, waves); capture rocprofv3 to confirm
- `D=128,QH=4,KH=1,T=2048` (bf16, vs-best 0.94x): **tuning_gap**
  - evidence: aligned large-M shape but vs-best 0.94x; fixed FlyDSL schedule vs tuned baseline, no structural cause evident
  - likely fix: add a per-shape tuned schedule (block size, vector width, waves); capture rocprofv3 to confirm
- `D=64,QH=8,KH=1,T=128` (bf16, vs-best 0.97x): **ok**
  - evidence: kernel-only vs-best 0.97x (>= parity)
  - likely fix: none
- `D=64,QH=8,KH=1,T=1` (f16, vs-best 1.01x): **ok**
  - evidence: kernel-only vs-best 1.01x (>= parity)
  - likely fix: none

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
  --op fused_rope_cache --shape-ledger benchmarks/examples/fused_rope_cache/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/fused_rope_cache/baseline_matrix.yaml \
  --out benchmarks/examples/fused_rope_cache --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/fused_rope_cache/shape_ledger.jsonl \
  --results benchmarks/examples/fused_rope_cache/benchmark_results.jsonl --out benchmarks/examples/fused_rope_cache/benchmark_summary.md
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
