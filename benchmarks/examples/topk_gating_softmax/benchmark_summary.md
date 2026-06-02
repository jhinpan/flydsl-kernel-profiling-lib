# Benchmark Summary: topk_gating_softmax

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 16 (sources: model_config=16)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=61.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 16 |
| FlyDSL correct + timed | 13 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 3 |
| measured FlyDSL-vs-baseline pairs | 13 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 1.61x  (n=13) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 1.61x  (n=13) |
| vs aiter_triton | 2.30x  (n=13) |
| vs pytorch | 2.62x  (n=13) |
| worst hot shape | 1.29x  (num_tokens=1,num_experts=256,topk=8 vs aiter) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 13 | 1.61x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 6 | 1.53x |
| Llama4-class | 2 | 1.73x |
| Mixtral-8x22B-class | 3 | 1.57x |
| Mixtral-8x7B | 2 | 1.80x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| num_tokens=256,num_experts=8,topk=2 | model_config | fp32 | 13.12 | aiter | 24.56 | 1.87x |
| num_tokens=16384,num_experts=256,topk=8 | model_config | bf16 | 21.44 | aiter | 38.44 | 1.79x |
| num_tokens=512,num_experts=64,topk=2 | model_config | bf16 | 13.28 | aiter | 23.40 | 1.76x |
| num_tokens=2048,num_experts=8,topk=2 | model_config | bf16 | 13.20 | aiter | 22.80 | 1.73x |
| num_tokens=2048,num_experts=64,topk=2 | model_config | bf16 | 13.40 | aiter | 22.68 | 1.69x |
| num_tokens=1024,num_experts=128,topk=6 | model_config | bf16 | 15.08 | aiter | 24.92 | 1.65x |
| num_tokens=128,num_experts=128,topk=6 | model_config | fp16 | 15.24 | aiter | 24.48 | 1.61x |
| num_tokens=2048,num_experts=256,topk=8 | model_config | bf16 | 16.60 | aiter | 25.68 | 1.55x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| num_tokens=1,num_experts=256,topk=8 | model_config | bf16 | 16.96 | aiter | 21.96 | 1.29x | ok |
| num_tokens=1,num_experts=128,topk=6 | model_config | bf16 | 15.48 | aiter | 22.40 | 1.45x | ok |
| num_tokens=256,num_experts=256,topk=8 | model_config | bf16 | 16.76 | aiter | 25.32 | 1.51x | ok |
| num_tokens=2048,num_experts=256,topk=8 | model_config | fp16 | 16.72 | aiter | 25.56 | 1.53x | ok |
| num_tokens=2048,num_experts=256,topk=8 | model_config | fp32 | 16.80 | aiter | 25.92 | 1.54x | ok |
| num_tokens=2048,num_experts=256,topk=8 | model_config | bf16 | 16.60 | aiter | 25.68 | 1.55x | ok |
| num_tokens=128,num_experts=128,topk=6 | model_config | fp16 | 15.24 | aiter | 24.48 | 1.61x | ok |
| num_tokens=1024,num_experts=128,topk=6 | model_config | bf16 | 15.08 | aiter | 24.92 | 1.65x | ok |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| num_tokens=1,num_experts=256,topk=8 | 16.96 | 67.80 | 50.84 |
| num_tokens=2048,num_experts=64,topk=2 | 13.40 | 62.66 | 49.26 |
| num_tokens=2048,num_experts=8,topk=2 | 13.20 | 62.14 | 48.94 |
| num_tokens=512,num_experts=64,topk=2 | 13.28 | 61.20 | 47.92 |
| num_tokens=256,num_experts=8,topk=2 | 13.12 | 60.78 | 47.66 |
| num_tokens=256,num_experts=256,topk=8 | 16.76 | 63.54 | 46.78 |

## Diagnosis

- `num_tokens=1,num_experts=256,topk=8` (bf16, vs-best 1.29x): **ok**
  - evidence: kernel-only vs-best 1.29x (near parity or better)
  - likely fix: none
- `num_tokens=1,num_experts=128,topk=6` (bf16, vs-best 1.45x): **ok**
  - evidence: kernel-only vs-best 1.45x (near parity or better)
  - likely fix: none
- `num_tokens=256,num_experts=256,topk=8` (bf16, vs-best 1.51x): **ok**
  - evidence: kernel-only vs-best 1.51x (near parity or better)
  - likely fix: none
- `num_tokens=2048,num_experts=256,topk=8` (fp16, vs-best 1.53x): **ok**
  - evidence: kernel-only vs-best 1.53x (near parity or better)
  - likely fix: none
- `num_tokens=2048,num_experts=256,topk=8` (fp32, vs-best 1.54x): **ok**
  - evidence: kernel-only vs-best 1.54x (near parity or better)
  - likely fix: none
- `num_tokens=2048,num_experts=256,topk=8` (bf16, vs-best 1.55x): **ok**
  - evidence: kernel-only vs-best 1.55x (near parity or better)
  - likely fix: none

## Promotion Decision

**promote** — overall kernel-only geomean is parity-or-better, no FlyDSL hard failures

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 13/16.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops topk_gating_softmax
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op topk_gating_softmax --shape-ledger benchmarks/examples/topk_gating_softmax/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/topk_gating_softmax/baseline_matrix.yaml \
  --out benchmarks/examples/topk_gating_softmax --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/topk_gating_softmax/shape_ledger.jsonl \
  --results benchmarks/examples/topk_gating_softmax/benchmark_results.jsonl --out benchmarks/examples/topk_gating_softmax/benchmark_summary.md \
  --kernel topk_gating_softmax
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
