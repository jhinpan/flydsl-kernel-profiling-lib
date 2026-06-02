# Benchmark Summary: rmsnorm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 159 (sources: sglang_trace=78, atom_workload=8, diagnostic=1, aiter_model_shapes=45, synthetic=27)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=787.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 159 |
| FlyDSL correct + timed | 153 |
| FlyDSL failed/oom | 6 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 153 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 0.88x  (n=153) |
| production-weighted geomean vs best | 0.82x |
| vs aiter | 0.92x  (n=152) |
| vs aiter_triton | 0.97x  (n=152) |
| vs triton | 0.99x  (n=153) |
| vs pytorch | 1.10x  (n=153) |
| worst hot shape | 0.58x  (M=16384,N=7168 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| prefill | 51 | 0.85x |
| decode | 32 | 0.88x |
| synthetic | 27 | 0.97x |
| diagnostic | 1 | 0.90x |
| model_config | 42 | 0.87x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 53 | 0.83x |
| GPT-OSS 120B | 5 | 0.83x |
| Llama3 405B | 5 | 0.86x |
| Llama3 70B | 5 | 0.94x |
| Llama3 8B | 5 | 0.96x |
| Llama4 Maverick | 5 | 0.74x |
| Qwen3-235B-A22B | 9 | 0.97x |
| Qwen3-4B | 43 | 0.89x |
| diagnostic | 1 | 0.90x |
| synthetic | 27 | 0.97x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=4096,N=8193 | synthetic | bf16 | 52.52 | pytorch | 106.40 | 2.03x |
| M=4096,N=8191 | synthetic | bf16 | 51.84 | aiter | 83.64 | 1.61x |
| M=4096,N=5333 | synthetic | bf16 | 37.08 | triton | 57.40 | 1.55x |
| M=4096,N=4097 | synthetic | bf16 | 33.72 | triton | 40.88 | 1.21x |
| M=4096,N=4095 | synthetic | bf16 | 30.88 | triton | 36.60 | 1.19x |
| M=4096,N=4096 | synthetic | bf16 | 23.52 | triton | 25.52 | 1.08x |
| M=131072,N=8192 | synthetic | bf16 | 778.81 | aiter_triton | 803.21 | 1.03x |
| M=2048,N=8192 | model_config | bf16 | 22.12 | triton | 22.48 | 1.02x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=16384,N=7168 | model_config | bf16 | 160.88 | aiter_triton | 93.36 | 0.58x | implementation_gap |
| M=8192,N=7168 | prefill | bf16 | 90.08 | aiter_triton | 53.08 | 0.59x | implementation_gap |
| M=16384,N=7168 | prefill | bf16 | 160.44 | aiter_triton | 94.64 | 0.59x | implementation_gap |
| M=1,N=7168 | model_config | bf16 | 21.68 | aiter_triton | 12.88 | 0.59x | implementation_gap |
| M=1,N=7168 | decode | bf16 | 20.84 | aiter | 13.04 | 0.63x | implementation_gap |
| M=131072,N=2560 | prefill | bf16 | 422.88 | aiter | 273.80 | 0.65x | implementation_gap |
| M=65536,N=2560 | prefill | bf16 | 220.96 | aiter | 146.80 | 0.66x | implementation_gap |
| M=512,N=7168 | prefill | bf16 | 21.48 | triton | 14.28 | 0.66x | implementation_gap |

## FlyDSL hard failures (crash / incorrect)

| shape | model | stage | dtype | status | reason |
|---|---|---|---|---|---|
| M=16384,N=128 | Qwen3-235B-A22B | model_config | bf16 | failed | ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add kno |
| M=16384,N=512 | DeepSeek-R1 | model_config | bf16 | failed | ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add kno |
| M=16384,N=1536 | DeepSeek-R1 | model_config | bf16 | failed | ValueError: launch block size 512x1x1 = 512 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known |
| M=16384,N=512 | DeepSeek-R1 | prefill | bf16 | failed | ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add kno |
| M=16384,N=1536 | DeepSeek-R1 | prefill | bf16 | failed | ValueError: launch block size 512x1x1 = 512 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known |
| M=16384,N=128 | Qwen3-4B | prefill | bf16 | failed | ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add kno |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=1,N=512 | 13.60 | 50.06 | 36.46 |
| M=1,N=512 | 13.08 | 46.84 | 33.76 |
| M=4,N=512 | 13.16 | 46.72 | 33.56 |
| M=1,N=1536 | 13.52 | 46.74 | 33.22 |
| M=8,N=1536 | 14.16 | 47.22 | 33.06 |
| M=8,N=512 | 13.16 | 46.06 | 32.90 |

## Diagnosis

- `M=16384,N=7168` (bf16, vs-best 0.58x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.58x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=8192,N=7168` (bf16, vs-best 0.59x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.59x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=16384,N=7168` (bf16, vs-best 0.59x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.59x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=7168` (bf16, vs-best 0.59x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.59x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=7168` (bf16, vs-best 0.63x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.63x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=131072,N=2560` (bf16, vs-best 0.65x): **implementation_gap**
  - evidence: N=2560 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.65x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=16384,N=128` (bf16): **failed** — ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known_block_size=[1024, 1, 1] to @kernel fo
  - classification: **flydsl_codegen_gap**
  - likely fix: annotate known_block_size or lower the generated workgroup size for this path
- `M=16384,N=512` (bf16): **failed** — ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known_block_size=[1024, 1, 1] to @kernel fo
  - classification: **flydsl_codegen_gap**
  - likely fix: annotate known_block_size or lower the generated workgroup size for this path
- `M=16384,N=1536` (bf16): **failed** — ValueError: launch block size 512x1x1 = 512 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known_block_size=[512, 1, 1] to @kernel for k
  - classification: **flydsl_codegen_gap**
  - likely fix: annotate known_block_size or lower the generated workgroup size for this path
- `M=16384,N=512` (bf16): **failed** — ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known_block_size=[1024, 1, 1] to @kernel fo
  - classification: **flydsl_codegen_gap**
  - likely fix: annotate known_block_size or lower the generated workgroup size for this path
- `M=16384,N=1536` (bf16): **failed** — ValueError: launch block size 512x1x1 = 512 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known_block_size=[512, 1, 1] to @kernel for k
  - classification: **flydsl_codegen_gap**
  - likely fix: annotate known_block_size or lower the generated workgroup size for this path
- `M=16384,N=128` (bf16): **failed** — ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known_block_size=[1024, 1, 1] to @kernel fo
  - classification: **flydsl_codegen_gap**
  - likely fix: annotate known_block_size or lower the generated workgroup size for this path

## Promotion Decision

**tune_needed** — sub-parity correct measured rows (geomean 0.88x) + hard failures on a shape class; needs per-shape tuning + bug fixes before broad promotion

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 153/159.
- Hard FlyDSL failures must be fixed or explicitly scoped out before broad promotion.
- Norm conclusions are shape-regime dependent; keep the fast-path/generic-path split visible.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops rmsnorm
python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm --out benchmarks/examples \
  --synthetic-boundary --diagnostic 32768,8192,bf16
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op rmsnorm --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
  --out benchmarks/examples/rmsnorm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl --out benchmarks/examples/rmsnorm/benchmark_summary.md \
  --kernel rmsnorm
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
