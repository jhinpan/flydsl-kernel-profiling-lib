# Benchmark Summary: moe_blockscale

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 12 (sources: model_config=12)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- No CUDA-graph timing rows found.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 12 |
| FlyDSL correct + timed | 0 |
| FlyDSL failed/oom | 2 |
| FlyDSL incorrect | 10 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 0 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | -x  (n=0) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|

## FlyDSL hard failures (crash / incorrect)

| shape | model | stage | dtype | status | reason |
|---|---|---|---|---|---|
| tokens=32,E=256,model_dim=7168,inter_dim=2048,topk=8 | deepseek-v3 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=128,E=256,model_dim=7168,inter_dim=2048,topk=8 | deepseek-v3 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=2048,E=256,model_dim=7168,inter_dim=2048,topk=8 | deepseek-v3 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=32,E=384,model_dim=7168,inter_dim=2048,topk=8 | kimi-k2 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=128,E=384,model_dim=7168,inter_dim=2048,topk=8 | kimi-k2 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=4096,E=384,model_dim=7168,inter_dim=2048,topk=8 | kimi-k2 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=16,E=8,model_dim=7168,inter_dim=256,topk=2 | deepseek-v3 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=32,E=8,model_dim=7168,inter_dim=256,topk=2 | deepseek-v3 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8 | deepseek-v3 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=64,E=384,model_dim=7168,inter_dim=256,topk=8 | kimi-k2 | model_config | fp8 | incorrect | max_abs=nan max_rel=nan rtol=0.2 atol=0.2 |
| tokens=1024,E=256,model_dim=7168,inter_dim=2048,topk=8 | deepseek-v3 | model_config | fp8 | failed | input/reference build: HIP out of memory. Tried to allocate 28.00 GiB. GPU 0 has a total capacity of 287.98 GiB of which |
| tokens=512,E=384,model_dim=7168,inter_dim=2048,topk=8 | kimi-k2 | model_config | fp8 | failed | input/reference build: HIP out of memory. Tried to allocate 42.00 GiB. GPU 0 has a total capacity of 287.98 GiB of which |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| tokens=4096,E=384,model_dim=7168,inter_dim=2048,topk=8 | 203.36 | 7700.63 | 7497.26 |
| tokens=2048,E=256,model_dim=7168,inter_dim=2048,topk=8 | 108.88 | 4360.16 | 4251.28 |
| tokens=128,E=384,model_dim=7168,inter_dim=2048,topk=8 | 30.00 | 2677.04 | 2647.04 |
| tokens=128,E=256,model_dim=7168,inter_dim=2048,topk=8 | 28.96 | 1949.26 | 1920.30 |
| tokens=32,E=384,model_dim=7168,inter_dim=2048,topk=8 | 23.80 | 1377.75 | 1353.95 |
| tokens=32,E=256,model_dim=7168,inter_dim=2048,topk=8 | 25.72 | 1326.57 | 1300.85 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 7497us host launch overhead (kernel 203.4us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `tokens=32,E=256,model_dim=7168,inter_dim=2048,topk=8` (fp8, vs-best 3948.23x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=128,E=256,model_dim=7168,inter_dim=2048,topk=8` (fp8, vs-best 5418.99x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=2048,E=256,model_dim=7168,inter_dim=2048,topk=8` (fp8, vs-best 1543.18x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=32,E=384,model_dim=7168,inter_dim=2048,topk=8` (fp8, vs-best 4887.10x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=128,E=384,model_dim=7168,inter_dim=2048,topk=8` (fp8, vs-best 7140.91x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=4096,E=384,model_dim=7168,inter_dim=2048,topk=8` (fp8, vs-best 1263.55x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=16,E=8,model_dim=7168,inter_dim=256,topk=2` (fp8, vs-best 80.74x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=32,E=8,model_dim=7168,inter_dim=256,topk=2` (fp8, vs-best 97.20x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8, vs-best 3201.16x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=64,E=384,model_dim=7168,inter_dim=256,topk=8` (fp8, vs-best 3969.64x): **baseline_unfair_or_unmatched**
  - evidence: FlyDSL output failed correctness vs fp32 reference
  - likely fix: fix correctness before trusting timing
- `tokens=1024,E=256,model_dim=7168,inter_dim=2048,topk=8` (fp8): **failed** — input/reference build: HIP out of memory. Tried to allocate 28.00 GiB. GPU 0 has a total capacity of 287.98 GiB of which 18.02 GiB is free. Of the allocated mem
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `tokens=512,E=384,model_dim=7168,inter_dim=2048,topk=8` (fp8): **failed** — input/reference build: HIP out of memory. Tried to allocate 42.00 GiB. GPU 0 has a total capacity of 287.98 GiB of which 4.02 GiB is free. Of the allocated memo
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support

## Promotion Decision

**blocked** — no correct+timed FlyDSL-vs-baseline pairs; fix coverage/failures before a speedup verdict

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 0/12.
- Hard FlyDSL failures must be fixed or explicitly scoped out before broad promotion.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops moe_blockscale
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op moe_blockscale --shape-ledger benchmarks/examples/moe_blockscale/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/moe_blockscale/baseline_matrix.yaml \
  --out benchmarks/examples/moe_blockscale --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/moe_blockscale/shape_ledger.jsonl \
  --results benchmarks/examples/moe_blockscale/benchmark_results.jsonl --out benchmarks/examples/moe_blockscale/benchmark_summary.md \
  --kernel moe_blockscale
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
