# Benchmark Summary: rmsnorm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 159 (sources: sglang_trace=78, atom_workload=8, diagnostic=1, aiter_model_shapes=45, synthetic=27)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.

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
| unweighted geomean vs best | 0.71x  (n=153) |
| production-weighted geomean vs best | 0.65x |
| vs aiter | 0.80x  (n=152) |
| vs aiter_triton | 0.94x  (n=152) |
| vs triton | 0.81x  (n=153) |
| vs pytorch | 1.27x  (n=153) |
| worst hot shape | 0.34x  (M=8192,N=128 vs triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| prefill | 51 | 0.66x |
| decode | 32 | 0.64x |
| synthetic | 27 | 0.95x |
| diagnostic | 1 | 1.04x |
| model_config | 42 | 0.71x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 53 | 0.61x |
| GPT-OSS 120B | 5 | 0.61x |
| Llama3 405B | 5 | 0.85x |
| Llama3 70B | 5 | 0.87x |
| Llama3 8B | 5 | 0.92x |
| Llama4 Maverick | 5 | 0.54x |
| Qwen3-235B-A22B | 9 | 0.88x |
| Qwen3-4B | 43 | 0.69x |
| diagnostic | 1 | 1.04x |
| synthetic | 27 | 0.95x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=4096,N=8193 | synthetic | bf16 | 35.95 | aiter | 67.12 | 1.87x |
| M=4096,N=5333 | synthetic | bf16 | 24.09 | triton | 33.43 | 1.39x |
| M=4096,N=4097 | synthetic | bf16 | 17.85 | triton | 24.57 | 1.38x |
| M=4096,N=12288 | synthetic | bf16 | 29.75 | aiter_triton | 40.15 | 1.35x |
| M=4096,N=8191 | synthetic | bf16 | 35.14 | aiter | 47.19 | 1.34x |
| M=2048,N=16384 | model_config | bf16 | 21.74 | aiter_triton | 25.75 | 1.18x |
| M=4096,N=4095 | synthetic | bf16 | 17.25 | triton | 20.35 | 1.18x |
| M=2048,N=8192 | model_config | bf16 | 11.33 | triton | 12.51 | 1.10x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=8192,N=128 | prefill | bf16 | 5.85 | triton | 2.00 | 0.34x | implementation_gap |
| M=1,N=7168 | model_config | bf16 | 5.72 | triton | 2.04 | 0.36x | implementation_gap |
| M=1,N=7168 | prefill | bf16 | 5.72 | triton | 2.06 | 0.36x | implementation_gap |
| M=1,N=7168 | decode | bf16 | 5.70 | triton | 2.07 | 0.36x | implementation_gap |
| M=32,N=7168 | model_config | bf16 | 6.17 | triton | 2.27 | 0.37x | implementation_gap |
| M=8,N=7168 | decode | bf16 | 5.96 | triton | 2.20 | 0.37x | implementation_gap |
| M=4,N=7168 | decode | bf16 | 5.92 | triton | 2.20 | 0.37x | implementation_gap |
| M=8,N=7168 | prefill | bf16 | 5.98 | triton | 2.22 | 0.37x | implementation_gap |

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
| M=2048,N=5120 | 12.20 | 66.72 | 54.52 |
| M=1,N=512 | 2.11 | 50.84 | 48.73 |
| M=1,N=512 | 2.10 | 47.82 | 45.73 |
| M=8,N=512 | 2.23 | 47.76 | 45.53 |
| M=64,N=128 | 2.14 | 47.48 | 45.34 |
| M=16,N=2560 | 3.43 | 48.52 | 45.09 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 55us host launch overhead (kernel 12.2us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `M=8192,N=128` (bf16, vs-best 0.34x): **implementation_gap**
  - evidence: N=128 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.34x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=7168` (bf16, vs-best 0.36x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.36x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=7168` (bf16, vs-best 0.36x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.36x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=7168` (bf16, vs-best 0.36x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.36x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=32,N=7168` (bf16, vs-best 0.37x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.37x). Compounded at small M=32: grid=(M,1,1) launches one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=8,N=7168` (bf16, vs-best 0.37x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.37x). Compounded at small M=8: grid=(M,1,1) launches one workgroup per row, so only ~8 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=16384,N=128` (bf16): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=512` (bf16): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=1536` (bf16): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=512` (bf16): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=1536` (bf16): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=128` (bf16): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path

## Promotion Decision

**tune_needed** — sub-parity overall (geomean 0.71x) + hard failures on a shape class; wins on its target regime but needs per-shape tuning + bug fixes before broad promotion

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
  --op rmsnorm --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
  --out benchmarks/examples/rmsnorm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl --out benchmarks/examples/rmsnorm/benchmark_summary.md
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
