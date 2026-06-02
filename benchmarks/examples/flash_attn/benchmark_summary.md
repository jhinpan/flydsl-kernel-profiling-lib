# Benchmark Summary: flash_attn

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 16 (sources: synthetic=9, model_config=7)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=62.

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
| unweighted geomean vs best | 0.87x  (n=16) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter_triton | 0.94x  (n=16) |
| vs aiter_asm | 0.98x  (n=14) |
| vs pytorch | 1.42x  (n=16) |
| worst hot shape | 0.71x  (B=1,S=2048,H=8,D=128,causal vs aiter_asm) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| synthetic | 9 | 0.89x |
| model_config | 7 | 0.84x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-V3 | 3 | 0.85x |
| Kimi-K2 | 2 | 0.83x |
| Qwen3 | 2 | 0.83x |
| synthetic | 9 | 0.89x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| B=1,S=2048,H=64,D=128,causal | synthetic | bf16 | 168.76 | aiter_triton | 175.40 | 1.04x |
| B=32,S=8192,H=8,D=128,causal | model_config | bf16 | 7131.78 | aiter_triton | 7005.30 | 0.98x |
| B=1,S=4096,H=64,D=128,causal | synthetic | fp16 | 590.09 | aiter_triton | 565.81 | 0.96x |
| B=1,S=1024,H=64,D=128,causal | synthetic | bf16 | 69.88 | aiter_asm | 66.68 | 0.95x |
| B=16,S=8192,H=16,D=128,causal | model_config | bf16 | 7175.50 | aiter_triton | 6839.78 | 0.95x |
| B=4,S=8192,H=64,D=128,causal | synthetic | bf16 | 7799.71 | aiter_triton | 7240.02 | 0.93x |
| B=1,S=8192,H=64,D=128,causal | synthetic | bf16 | 2044.50 | aiter_triton | 1876.22 | 0.92x |
| B=8,S=8192,H=32,D=128,causal | model_config | bf16 | 7891.15 | aiter_triton | 7037.10 | 0.89x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| B=1,S=2048,H=8,D=128,causal | model_config | bf16 | 97.16 | aiter_asm | 68.80 | 0.71x | tuning_gap |
| B=1,S=2048,H=16,D=128,causal | model_config | bf16 | 101.32 | aiter_asm | 74.00 | 0.73x | tuning_gap |
| B=1,S=2048,H=32,D=128,noncausal | synthetic | bf16 | 146.40 | aiter_asm | 108.24 | 0.74x | tuning_gap |
| B=1,S=2048,H=32,D=128,causal | model_config | bf16 | 116.24 | aiter_asm | 90.00 | 0.77x | tuning_gap |
| B=8,S=512,H=64,D=128,causal | synthetic | bf16 | 162.24 | aiter_triton | 132.52 | 0.82x | tuning_gap |
| B=8,S=512,H=64,D=128,noncausal | synthetic | fp16 | 189.92 | aiter_triton | 161.40 | 0.85x | tuning_gap |
| B=1,S=128,H=64,D=128,causal | synthetic | bf16 | 21.40 | aiter_asm | 18.76 | 0.88x | tuning_gap |
| B=1,S=4096,H=32,D=128,causal | model_config | bf16 | 337.40 | aiter_triton | 299.00 | 0.89x | tuning_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| B=1,S=128,H=64,D=128,causal | 21.40 | 70.60 | 49.20 |
| B=8,S=512,H=64,D=128,noncausal | 189.92 | 188.44 | -1.48 |
| B=1,S=2048,H=8,D=128,causal | 97.16 | 93.28 | -3.88 |
| B=1,S=2048,H=16,D=128,causal | 101.32 | 97.02 | -4.30 |
| B=1,S=2048,H=32,D=128,noncausal | 146.40 | 141.92 | -4.48 |
| B=1,S=1024,H=64,D=128,causal | 69.88 | 64.74 | -5.14 |

## Diagnosis

- `B=1,S=2048,H=8,D=128,causal` (bf16, vs-best 0.71x): **tuning_gap**
  - evidence: kernel-only vs-best 0.71x for args={'batch': 1, 'causal': True, 'head_dim': 128, 'num_heads': 8, 'seq_len': 2048}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `B=1,S=2048,H=16,D=128,causal` (bf16, vs-best 0.73x): **tuning_gap**
  - evidence: kernel-only vs-best 0.73x for args={'batch': 1, 'causal': True, 'head_dim': 128, 'num_heads': 16, 'seq_len': 2048}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `B=1,S=2048,H=32,D=128,noncausal` (bf16, vs-best 0.74x): **tuning_gap**
  - evidence: kernel-only vs-best 0.74x for args={'batch': 1, 'causal': False, 'head_dim': 128, 'num_heads': 32, 'seq_len': 2048}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `B=1,S=2048,H=32,D=128,causal` (bf16, vs-best 0.77x): **tuning_gap**
  - evidence: kernel-only vs-best 0.77x for args={'batch': 1, 'causal': True, 'head_dim': 128, 'num_heads': 32, 'seq_len': 2048}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `B=8,S=512,H=64,D=128,causal` (bf16, vs-best 0.82x): **tuning_gap**
  - evidence: kernel-only vs-best 0.82x for args={'batch': 8, 'causal': True, 'head_dim': 128, 'num_heads': 64, 'seq_len': 512}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `B=8,S=512,H=64,D=128,noncausal` (fp16, vs-best 0.85x): **tuning_gap**
  - evidence: kernel-only vs-best 0.85x for args={'batch': 8, 'causal': False, 'head_dim': 128, 'num_heads': 64, 'seq_len': 512}
  - likely fix: profile the hot shape and add an op-specific diagnosis

## Promotion Decision

**tune_needed** — sub-parity overall (geomean 0.87x); wins on its target regime but needs per-shape tuning + bug fixes before broad promotion

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 16/16.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops flash_attn
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op flash_attn --shape-ledger benchmarks/examples/flash_attn/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/flash_attn/baseline_matrix.yaml \
  --out benchmarks/examples/flash_attn --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/flash_attn/shape_ledger.jsonl \
  --results benchmarks/examples/flash_attn/benchmark_results.jsonl --out benchmarks/examples/flash_attn/benchmark_summary.md \
  --kernel flash_attn
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
