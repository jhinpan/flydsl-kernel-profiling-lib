# Benchmark Summary: fused_rope_cache

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 30 (sources: aiter_model_shapes=30)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=80.

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
| unweighted geomean vs best | 1.72x  (n=30) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter_triton | 0.97x  (n=20) |
| vs pytorch | 5.16x  (n=30) |
| worst hot shape | 0.84x  (D=64,QH=8,KH=1,T=2048 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 30 | 1.72x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| GPT-OSS 120B | 6 | 1.68x |
| Llama3 405B | 6 | 1.70x |
| Llama3 70B | 6 | 1.71x |
| Llama3 8B | 6 | 1.77x |
| Llama4 Maverick | 6 | 1.76x |
| Qwen3-235B-A22B | 6 | 1.71x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| D=128,QH=4,KH=1,T=128 | model_config | f16 | 13.76 | pytorch | 85.32 | 6.20x |
| D=128,QH=16,KH=1,T=128 | model_config | bf16 | 14.24 | pytorch | 88.16 | 6.19x |
| D=128,QH=8,KH=1,T=128 | model_config | f16 | 13.68 | pytorch | 84.32 | 6.16x |
| D=128,QH=5,KH=1,T=128 | model_config | f16 | 13.76 | pytorch | 84.80 | 6.16x |
| D=64,QH=8,KH=1,T=128 | model_config | f16 | 13.64 | pytorch | 83.44 | 6.12x |
| D=128,QH=5,KH=1,T=2048 | model_config | f16 | 18.24 | pytorch | 98.96 | 5.43x |
| D=128,QH=4,KH=1,T=2048 | model_config | f16 | 18.48 | pytorch | 98.44 | 5.33x |
| D=128,QH=8,KH=1,T=2048 | model_config | f16 | 22.16 | pytorch | 100.44 | 4.53x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| D=64,QH=8,KH=1,T=2048 | model_config | bf16 | 22.16 | aiter_triton | 18.68 | 0.84x | tuning_gap |
| D=128,QH=5,KH=1,T=2048 | model_config | bf16 | 18.12 | aiter_triton | 16.64 | 0.92x | tuning_gap |
| D=128,QH=8,KH=1,T=2048 | model_config | bf16 | 23.20 | aiter_triton | 21.32 | 0.92x | tuning_gap |
| D=128,QH=4,KH=1,T=2048 | model_config | bf16 | 18.76 | aiter_triton | 17.64 | 0.94x | tuning_gap |
| D=128,QH=16,KH=1,T=1 | model_config | f16 | 14.08 | aiter_triton | 13.72 | 0.97x | ok |
| D=128,QH=5,KH=1,T=1 | model_config | f16 | 13.88 | aiter_triton | 13.60 | 0.98x | ok |
| D=64,QH=8,KH=1,T=128 | model_config | bf16 | 14.04 | aiter_triton | 13.76 | 0.98x | ok |
| D=128,QH=5,KH=1,T=128 | model_config | bf16 | 14.28 | aiter_triton | 14.00 | 0.98x | ok |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| D=128,QH=5,KH=1,T=128 | 14.28 | 149.96 | 135.68 |
| D=128,QH=16,KH=1,T=1 | 13.68 | 139.86 | 126.18 |
| D=128,QH=4,KH=1,T=128 | 13.88 | 139.40 | 125.52 |
| D=64,QH=8,KH=1,T=1 | 13.72 | 138.14 | 124.42 |
| D=128,QH=4,KH=1,T=1 | 13.68 | 137.28 | 123.60 |
| D=128,QH=16,KH=1,T=1 | 14.08 | 137.58 | 123.50 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 136us host launch overhead (kernel 14.3us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `D=64,QH=8,KH=1,T=2048` (bf16, vs-best 0.84x): **tuning_gap**
  - evidence: single fused RoPE/cache launch is below the best baseline (0.84x) for args={'apply_scale': False, 'block_size': 16, 'flash_layout': True, 'head_dim': 64, 'max_pos': 8192, 'num_heads': 8, 'num_kv_heads': 1, 'pos_dtype': 'i32', 'positions': True, 'reuse_freqs_front_part': True, 'rotate_style': 'neox', 'seq_len': 2048, 'two_inputs': True}
  - likely fix: capture rocprofv3 and inspect occupancy, vectorization, and memory-store behavior
- `D=128,QH=5,KH=1,T=2048` (bf16, vs-best 0.92x): **tuning_gap**
  - evidence: single fused RoPE/cache launch is below the best baseline (0.92x) for args={'apply_scale': False, 'block_size': 16, 'flash_layout': True, 'head_dim': 128, 'max_pos': 8192, 'num_heads': 5, 'num_kv_heads': 1, 'pos_dtype': 'i32', 'positions': True, 'reuse_freqs_front_part': True, 'rotate_style': 'neox', 'seq_len': 2048, 'two_inputs': True}
  - likely fix: capture rocprofv3 and inspect occupancy, vectorization, and memory-store behavior
- `D=128,QH=8,KH=1,T=2048` (bf16, vs-best 0.92x): **tuning_gap**
  - evidence: single fused RoPE/cache launch is below the best baseline (0.92x) for args={'apply_scale': False, 'block_size': 16, 'flash_layout': True, 'head_dim': 128, 'max_pos': 8192, 'num_heads': 8, 'num_kv_heads': 1, 'pos_dtype': 'i32', 'positions': True, 'reuse_freqs_front_part': True, 'rotate_style': 'neox', 'seq_len': 2048, 'two_inputs': True}
  - likely fix: capture rocprofv3 and inspect occupancy, vectorization, and memory-store behavior
- `D=128,QH=4,KH=1,T=2048` (bf16, vs-best 0.94x): **tuning_gap**
  - evidence: single fused RoPE/cache launch is below the best baseline (0.94x) for args={'apply_scale': False, 'block_size': 16, 'flash_layout': True, 'head_dim': 128, 'max_pos': 8192, 'num_heads': 4, 'num_kv_heads': 1, 'pos_dtype': 'i32', 'positions': True, 'reuse_freqs_front_part': True, 'rotate_style': 'neox', 'seq_len': 2048, 'two_inputs': True}
  - likely fix: capture rocprofv3 and inspect occupancy, vectorization, and memory-store behavior
- `D=128,QH=16,KH=1,T=1` (f16, vs-best 0.97x): **ok**
  - evidence: kernel-only vs-best 0.97x (near parity or better)
  - likely fix: none
- `D=128,QH=5,KH=1,T=1` (f16, vs-best 0.98x): **ok**
  - evidence: kernel-only vs-best 0.98x (near parity or better)
  - likely fix: none

## Promotion Decision

**promote** — overall kernel-only geomean is parity-or-better, no FlyDSL hard failures

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 30/30.
- Incorrect baselines are excluded from vs-best aggregates; check the coverage matrix before promoting.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
# fused_rope_cache rows are derived from rope model_config rows; use the checked-in ledger here.
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op fused_rope_cache --shape-ledger benchmarks/examples/fused_rope_cache/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/fused_rope_cache/baseline_matrix.yaml \
  --out benchmarks/examples/fused_rope_cache --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/fused_rope_cache/shape_ledger.jsonl \
  --results benchmarks/examples/fused_rope_cache/benchmark_results.jsonl --out benchmarks/examples/fused_rope_cache/benchmark_summary.md \
  --kernel fused_rope_cache
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
