# Benchmark Summary: moe_reduce

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 16 (sources: model_config=16)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=48.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 16 |
| FlyDSL correct + timed | 16 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 16 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 1.03x  (n=16) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs triton | 1.06x  (n=16) |
| vs pytorch | 1.20x  (n=16) |
| worst hot shape | 0.90x  (tokens=16384,topk=6,model_dim=5120 vs pytorch) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 16 | 1.03x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| deepseek-v3 | 6 | 1.06x |
| ep-k6 | 4 | 1.00x |
| kimi-k2 | 3 | 1.02x |
| qwen3-moe | 3 | 1.00x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| tokens=65,topk=8,model_dim=7168 | model_config | bf16 | 13.76 | pytorch | 15.36 | 1.12x |
| tokens=65,topk=8,model_dim=7168 | model_config | bf16 | 13.40 | pytorch | 14.76 | 1.10x |
| tokens=129,topk=8,model_dim=7168,mask | model_config | f16 | 14.36 | triton | 15.64 | 1.09x |
| tokens=65,topk=8,model_dim=2560 | model_config | bf16 | 13.32 | triton | 14.24 | 1.07x |
| tokens=129,topk=6,model_dim=5120,mask | model_config | f16 | 13.68 | triton | 14.56 | 1.06x |
| tokens=1,topk=8,model_dim=7168 | model_config | bf16 | 13.44 | triton | 14.20 | 1.06x |
| tokens=65,topk=6,model_dim=5120 | model_config | f16 | 13.48 | triton | 14.20 | 1.05x |
| tokens=32769,topk=8,model_dim=7168 | model_config | bf16 | 825.33 | pytorch | 863.93 | 1.05x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| tokens=16384,topk=6,model_dim=5120 | model_config | f16 | 267.80 | pytorch | 241.36 | 0.90x | tuning_gap |
| tokens=16384,topk=8,model_dim=7168 | model_config | bf16 | 431.20 | pytorch | 396.12 | 0.92x | tuning_gap |
| tokens=16384,topk=8,model_dim=2560 | model_config | bf16 | 180.00 | pytorch | 166.96 | 0.93x | tuning_gap |
| tokens=5,topk=6,model_dim=5120 | model_config | f16 | 13.28 | triton | 13.12 | 0.99x | ok |
| tokens=1,topk=8,model_dim=2560 | model_config | bf16 | 13.00 | triton | 13.20 | 1.02x | ok |
| tokens=5,topk=8,model_dim=7168 | model_config | f32 | 13.96 | pytorch | 14.28 | 1.02x | ok |
| tokens=5,topk=8,model_dim=7168 | model_config | bf16 | 13.12 | triton | 13.52 | 1.03x | ok |
| tokens=1,topk=8,model_dim=7168 | model_config | bf16 | 13.16 | triton | 13.68 | 1.04x | ok |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| tokens=5,topk=8,model_dim=7168 | 13.12 | 52.90 | 39.78 |
| tokens=1,topk=8,model_dim=7168 | 13.44 | 52.48 | 39.04 |
| tokens=65,topk=8,model_dim=7168 | 13.40 | 51.56 | 38.16 |
| tokens=1,topk=8,model_dim=7168 | 13.16 | 50.80 | 37.64 |
| tokens=65,topk=6,model_dim=5120 | 13.48 | 50.24 | 36.76 |
| tokens=1,topk=8,model_dim=2560 | 13.00 | 49.34 | 36.34 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 40us host launch overhead (kernel 13.1us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `tokens=16384,topk=6,model_dim=5120` (f16, vs-best 0.90x): **tuning_gap**
  - evidence: kernel-only vs-best 0.90x for args={'model_dim': 5120, 'tokens': 16384, 'topk': 6}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `tokens=16384,topk=8,model_dim=7168` (bf16, vs-best 0.92x): **tuning_gap**
  - evidence: kernel-only vs-best 0.92x for args={'model_dim': 7168, 'tokens': 16384, 'topk': 8}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `tokens=16384,topk=8,model_dim=2560` (bf16, vs-best 0.93x): **tuning_gap**
  - evidence: kernel-only vs-best 0.93x for args={'model_dim': 2560, 'tokens': 16384, 'topk': 8}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `tokens=5,topk=6,model_dim=5120` (f16, vs-best 0.99x): **ok**
  - evidence: kernel-only vs-best 0.99x (near parity or better)
  - likely fix: none
- `tokens=1,topk=8,model_dim=2560` (bf16, vs-best 1.02x): **ok**
  - evidence: kernel-only vs-best 1.02x (near parity or better)
  - likely fix: none
- `tokens=5,topk=8,model_dim=7168` (f32, vs-best 1.02x): **ok**
  - evidence: kernel-only vs-best 1.02x (near parity or better)
  - likely fix: none

## Promotion Decision

**promote** — overall kernel-only geomean is parity-or-better, no FlyDSL hard failures

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 16/16.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops moe_reduce
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op moe_reduce --shape-ledger benchmarks/examples/moe_reduce/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/moe_reduce/baseline_matrix.yaml \
  --out benchmarks/examples/moe_reduce --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/moe_reduce/shape_ledger.jsonl \
  --results benchmarks/examples/moe_reduce/benchmark_results.jsonl --out benchmarks/examples/moe_reduce/benchmark_summary.md \
  --kernel moe_reduce
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
