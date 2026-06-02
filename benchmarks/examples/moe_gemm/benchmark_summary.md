# Benchmark Summary: moe_gemm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 29 (sources: aiter_model_shapes=25, synthetic=4)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=3.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 29 |
| FlyDSL correct + timed | 3 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 6 |
| FlyDSL unsupported | 20 |
| measured FlyDSL-vs-baseline pairs | 3 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 157.59x  (n=3) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs pytorch | 157.59x  (n=3) |
| worst hot shape | 16.58x  (tokens=256,E=4,model_dim=1024,inter_dim=256,topk=2 vs pytorch) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 1 | 16.58x |
| model_config | 2 | 485.92x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 2 | 485.92x |
| synthetic-smallest-passing | 1 | 16.58x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| tokens=32,E=256,model_dim=7168,inter_dim=256,topk=8 | model_config | fp8 | 43.16 | pytorch | 37021.54 | 857.77x |
| tokens=1,E=256,model_dim=7168,inter_dim=256,topk=8 | model_config | fp8 | 43.96 | pytorch | 12100.92 | 275.27x |
| tokens=256,E=4,model_dim=1024,inter_dim=256,topk=2 | synthetic | fp8 | 45.16 | pytorch | 748.59 | 16.58x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| tokens=256,E=4,model_dim=1024,inter_dim=256,topk=2 | synthetic | fp8 | 45.16 | pytorch | 748.59 | 16.58x | ok |
| tokens=1,E=256,model_dim=7168,inter_dim=256,topk=8 | model_config | fp8 | 43.96 | pytorch | 12100.92 | 275.27x | ok |
| tokens=32,E=256,model_dim=7168,inter_dim=256,topk=8 | model_config | fp8 | 43.16 | pytorch | 37021.54 | 857.77x | ok |

## FlyDSL hard failures (crash / incorrect)

| shape | model | stage | dtype | status | reason |
|---|---|---|---|---|---|
| tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8 | DeepSeek-R1 | model_config | fp8 | incorrect | max_abs=1.786e-01 max_rel=4.071e+05 rtol=0.15 atol=0.15 |
| tokens=2048,E=256,model_dim=7168,inter_dim=256,topk=8 | DeepSeek-R1 | model_config | fp8 | incorrect | max_abs=1.958e-01 max_rel=2.070e+06 rtol=0.15 atol=0.15 |
| tokens=16384,E=256,model_dim=7168,inter_dim=256,topk=8 | DeepSeek-R1 | model_config | fp8 | incorrect | max_abs=2.103e-01 max_rel=2.606e+10 rtol=0.15 atol=0.15 |
| tokens=32,E=8,model_dim=6144,inter_dim=4096,topk=2 | synthetic-profiled-a | synthetic | fp8 | incorrect | max_abs=2.853e-01 max_rel=1.162e+05 rtol=0.15 atol=0.15 |
| tokens=256,E=8,model_dim=6144,inter_dim=4096,topk=2 | synthetic-profiled-a | synthetic | fp8 | incorrect | max_abs=3.178e-01 max_rel=3.748e+05 rtol=0.15 atol=0.15 |
| tokens=2048,E=8,model_dim=6144,inter_dim=4096,topk=2 | synthetic-profiled-a | synthetic | fp8 | incorrect | max_abs=3.445e-01 max_rel=1.221e+06 rtol=0.15 atol=0.15 |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| tokens=16384,E=256,model_dim=7168,inter_dim=256,topk=8 | 318.00 | 2861.50 | 2543.50 |
| tokens=2048,E=8,model_dim=6144,inter_dim=4096,topk=2 | 142.04 | 857.91 | 715.87 |
| tokens=2048,E=256,model_dim=7168,inter_dim=256,topk=8 | 73.68 | 654.97 | 581.28 |
| tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8 | 48.52 | 360.82 | 312.30 |
| tokens=32,E=256,model_dim=7168,inter_dim=256,topk=8 | 43.16 | 264.48 | 221.32 |
| tokens=256,E=8,model_dim=6144,inter_dim=4096,topk=2 | 65.36 | 258.56 | 193.20 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 2544us host launch overhead (kernel 318.0us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `tokens=256,E=4,model_dim=1024,inter_dim=256,topk=2` (fp8, vs-best 16.58x): **ok**
  - evidence: kernel-only vs-best 16.58x (near parity or better)
  - likely fix: none
- `tokens=1,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8, vs-best 275.27x): **ok**
  - evidence: kernel-only vs-best 275.27x (near parity or better)
  - likely fix: none
- `tokens=32,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8, vs-best 857.77x): **ok**
  - evidence: kernel-only vs-best 857.77x (near parity or better)
  - likely fix: none
- `tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8, vs-best 906.07x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=2048,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8, vs-best 625.02x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=16384,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8, vs-best 216.15x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=32,E=8,model_dim=6144,inter_dim=4096,topk=2` (fp8, vs-best 54.81x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=256,E=8,model_dim=6144,inter_dim=4096,topk=2` (fp8, vs-best 50.59x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=2048,E=8,model_dim=6144,inter_dim=4096,topk=2` (fp8, vs-best 56.89x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing

## Promotion Decision

**tune_needed** — correct measured rows are parity-or-better (geomean 157.59x), but hard failures/incorrect rows block promotion

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 3/29.
- Hard FlyDSL failures must be fixed or explicitly scoped out before broad promotion.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops moe_gemm
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op moe_gemm --shape-ledger benchmarks/examples/moe_gemm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/moe_gemm/baseline_matrix.yaml \
  --out benchmarks/examples/moe_gemm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/moe_gemm/shape_ledger.jsonl \
  --results benchmarks/examples/moe_gemm/benchmark_results.jsonl --out benchmarks/examples/moe_gemm/benchmark_summary.md \
  --kernel moe_gemm
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
