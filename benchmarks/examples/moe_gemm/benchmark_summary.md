# Benchmark Summary: moe_gemm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 29 (sources: aiter_model_shapes=25, synthetic=4)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 29 |
| FlyDSL correct + timed | 0 |
| FlyDSL failed/oom | 9 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 20 |
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
| tokens=1,E=256,model_dim=7168,inter_dim=256,topk=8 | DeepSeek-R1 | model_config | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |
| tokens=32,E=256,model_dim=7168,inter_dim=256,topk=8 | DeepSeek-R1 | model_config | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |
| tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8 | DeepSeek-R1 | model_config | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |
| tokens=2048,E=256,model_dim=7168,inter_dim=256,topk=8 | DeepSeek-R1 | model_config | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |
| tokens=16384,E=256,model_dim=7168,inter_dim=256,topk=8 | DeepSeek-R1 | model_config | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |
| tokens=256,E=4,model_dim=1024,inter_dim=256,topk=2 | synthetic-smallest-p | synthetic | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |
| tokens=32,E=8,model_dim=6144,inter_dim=4096,topk=2 | synthetic-profiled-a | synthetic | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |
| tokens=256,E=8,model_dim=6144,inter_dim=4096,topk=2 | synthetic-profiled-a | synthetic | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |
| tokens=2048,E=8,model_dim=6144,inter_dim=4096,topk=2 | synthetic-profiled-a | synthetic | fp8 | failed | ValueError: stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1). |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|

## Diagnosis

- `tokens=1,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `tokens=32,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `tokens=256,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `tokens=2048,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `tokens=16384,E=256,model_dim=7168,inter_dim=256,topk=8` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `tokens=256,E=4,model_dim=1024,inter_dim=256,topk=2` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `tokens=32,E=8,model_dim=6144,inter_dim=4096,topk=2` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `tokens=256,E=8,model_dim=6144,inter_dim=4096,topk=2` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `tokens=2048,E=8,model_dim=6144,inter_dim=4096,topk=2` (fp8): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path

## Promotion Decision

**rewrite_needed** — well below parity (geomean 0.00x); structural rework needed

Regime-specific reading:
- **Large-M aligned (prefill-like):** kernel-only parity-or-better (diagnostic ~1.03x, beats PyTorch ~1.5x). Promotable.
- **Small-M large-N (decode):** one-block-per-row underutilizes the GPU (kernel-only worst ~0.36x). Needs a parallelization change.
- **Eager decode latency:** ~tens-of-us launcher host overhead per call. Needs a launch cache (host-side).
- **Large-M small-N:** hard crash (block size > AMDGPU max_flat_workgroup_size). Must-fix bug.

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
  --op moe_gemm --shape-ledger benchmarks/examples/moe_gemm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/moe_gemm/baseline_matrix.yaml \
  --out benchmarks/examples/moe_gemm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/moe_gemm/shape_ledger.jsonl \
  --results benchmarks/examples/moe_gemm/benchmark_results.jsonl --out benchmarks/examples/moe_gemm/benchmark_summary.md
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
