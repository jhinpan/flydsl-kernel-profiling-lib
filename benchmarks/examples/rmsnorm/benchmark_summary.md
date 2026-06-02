# Benchmark Summary: rmsnorm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 117 (sources: sglang_trace=36, atom_workload=8, diagnostic=1, aiter_model_shapes=45, synthetic=27)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 117 |
| FlyDSL correct + timed | 113 |
| FlyDSL failed/oom | 4 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 113 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 0.76x  (n=113) |
| production-weighted geomean vs best | 0.70x |
| vs aiter | 0.85x  (n=112) |
| vs aiter_triton | 1.01x  (n=112) |
| vs triton | 0.86x  (n=113) |
| vs pytorch | 1.36x  (n=113) |
| worst hot shape | 0.34x  (M=8192,N=128 vs triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| prefill | 26 | 0.70x |
| decode | 17 | 0.69x |
| synthetic | 27 | 0.94x |
| diagnostic | 1 | 1.03x |
| model_config | 42 | 0.71x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 13 | 0.62x |
| GPT-OSS 120B | 5 | 0.61x |
| Llama3 405B | 5 | 0.85x |
| Llama3 70B | 5 | 0.87x |
| Llama3 8B | 5 | 0.91x |
| Llama4 Maverick | 5 | 0.55x |
| Qwen3-235B-A22B | 9 | 0.88x |
| Qwen3-4B | 43 | 0.69x |
| diagnostic | 1 | 1.03x |
| synthetic | 27 | 0.94x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=4096,N=8193 | synthetic | bf16 | 36.00 | aiter | 68.88 | 1.91x |
| M=4096,N=4097 | synthetic | bf16 | 17.74 | triton | 24.49 | 1.38x |
| M=4096,N=5333 | synthetic | bf16 | 24.41 | triton | 33.39 | 1.37x |
| M=4096,N=12288 | synthetic | bf16 | 30.14 | aiter_triton | 40.36 | 1.34x |
| M=4096,N=8191 | synthetic | bf16 | 35.88 | aiter | 46.93 | 1.31x |
| M=2048,N=16384 | model_config | bf16 | 21.69 | aiter_triton | 25.39 | 1.17x |
| M=4096,N=4095 | synthetic | bf16 | 17.48 | triton | 20.37 | 1.17x |
| M=131072,N=8192 | synthetic | bf16 | 738.84 | aiter | 817.42 | 1.11x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=8192,N=128 | prefill | bf16 | 5.86 | triton | 2.01 | 0.34x | implementation_gap |
| M=1,N=7168 | model_config | bf16 | 5.70 | triton | 2.06 | 0.36x | implementation_gap |
| M=32,N=7168 | model_config | bf16 | 6.17 | triton | 2.26 | 0.37x | implementation_gap |
| M=1,N=5120 | model_config | bf16 | 4.57 | triton | 2.02 | 0.44x | implementation_gap |
| M=256,N=7168 | model_config | bf16 | 6.34 | triton | 2.84 | 0.45x | implementation_gap |
| M=32,N=5120 | model_config | bf16 | 4.96 | triton | 2.25 | 0.45x | implementation_gap |
| M=256,N=5120 | model_config | bf16 | 5.07 | triton | 2.61 | 0.51x | implementation_gap |
| M=4096,N=128 | prefill | bf16 | 3.71 | triton | 1.92 | 0.52x | implementation_gap |

## FlyDSL hard failures (crash / incorrect)

| shape | model | stage | dtype | status | reason |
|---|---|---|---|---|---|
| M=16384,N=128 | Qwen3-235B-A22B | model_config | bf16 | failed | ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add kno |
| M=16384,N=512 | DeepSeek-R1 | model_config | bf16 | failed | ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add kno |
| M=16384,N=1536 | DeepSeek-R1 | model_config | bf16 | failed | ValueError: launch block size 512x1x1 = 512 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add known |
| M=16384,N=128 | Qwen3-4B | prefill | bf16 | failed | ValueError: launch block size 1024x1x1 = 1024 threads exceeds the AMDGPU default max_flat_workgroup_size of 256. Add kno |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=1,N=128 | 1.88 | 48.22 | 46.34 |
| M=256,N=128 | 2.18 | 48.08 | 45.90 |
| M=1,N=2048 | 2.00 | 46.98 | 44.98 |
| M=2,N=128 | 1.88 | 46.32 | 44.44 |
| M=1024,N=128 | 2.30 | 46.72 | 44.43 |
| M=1,N=2047 | 2.98 | 47.00 | 44.03 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 46us host launch overhead (kernel 1.9us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `M=8192,N=128` (bf16, vs-best 0.34x): **implementation_gap**
  - evidence: N=128 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.34x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=7168` (bf16, vs-best 0.36x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.36x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=32,N=7168` (bf16, vs-best 0.37x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.37x). Compounded at small M=32: grid=(M,1,1) launches one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=5120` (bf16, vs-best 0.44x): **implementation_gap**
  - evidence: N=5120 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.44x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=256,N=7168` (bf16, vs-best 0.45x): **implementation_gap**
  - evidence: N=7168 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.45x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=32,N=5120` (bf16, vs-best 0.45x): **implementation_gap**
  - evidence: N=5120 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.45x). Compounded at small M=32: grid=(M,1,1) launches one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied).
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
- `M=16384,N=128` (bf16): **failed** — 
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path

## Promotion Decision

**tune_needed** — sub-parity overall (geomean 0.76x) + hard failures on a shape class; wins on its target regime but needs per-shape tuning + bug fixes before broad promotion

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
