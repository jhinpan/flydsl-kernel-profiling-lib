# Benchmark Summary: pa_decode

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 12 (sources: model_config=12)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=22.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 12 |
| FlyDSL correct + timed | 12 |
| FlyDSL failed/oom | 0 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 0 |
| measured FlyDSL-vs-baseline pairs | 12 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 0.49x  (n=12) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs gluon | 0.49x  (n=12) |
| vs pytorch | 132.41x  (n=12) |
| worst hot shape | 0.04x  (hq=16,hkv=1,d=128,ctx=8192,b=128,ql=1,blk=1024,quant=per_token vs gluon) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 12 | 0.49x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek/Kimi GQA decode (TP shard -> hkv=1) | 1 | 0.85x |
| Qwen-like (hidden 2560, 8 q-heads/shard) long-ctx decode | 1 | 0.88x |
| sliding-window model (e.g. Mistral-style) | 1 | 0.04x |
| sliding-window model long window | 1 | 0.07x |
| test_pa normal_accuracy | 8 | 0.74x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| hq=8,hkv=1,d=128,ctx=8192,b=128,ql=1,blk=1024,quant=per_token | model_config | fp8 | 90.56 | gluon | 79.40 | 0.88x |
| hq=8,hkv=1,d=128,ctx=4096,b=16,ql=1,blk=1024,quant=per_token | model_config | fp8 | 24.64 | gluon | 20.92 | 0.85x |
| hq=8,hkv=1,d=128,ctx=1027,b=3,ql=2,blk=1024,quant=per_token | model_config | fp8 | 21.96 | gluon | 18.20 | 0.83x |
| hq=16,hkv=1,d=128,ctx=1027,b=3,ql=1,blk=1024,quant=per_token | model_config | fp8 | 22.44 | gluon | 18.52 | 0.83x |
| hq=8,hkv=1,d=128,ctx=1027,b=3,ql=1,blk=1024,quant=per_token | model_config | fp8 | 22.00 | gluon | 18.00 | 0.82x |
| hq=8,hkv=1,d=128,ctx=1027,b=81,ql=1,blk=1024,quant=per_token | model_config | fp8 | 30.40 | gluon | 23.64 | 0.78x |
| hq=16,hkv=1,d=128,ctx=1027,b=81,ql=1,blk=1024,quant=per_tensor | model_config | fp8 | 31.16 | gluon | 23.40 | 0.75x |
| hq=8,hkv=1,d=128,ctx=1027,b=81,ql=2,blk=1024,quant=per_tensor | model_config | fp8 | 31.36 | gluon | 23.20 | 0.74x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| hq=16,hkv=1,d=128,ctx=8192,b=128,ql=1,blk=1024,quant=per_token | model_config | fp8 | 437.40 | gluon | 18.20 | 0.04x | tuning_gap |
| hq=16,hkv=1,d=128,ctx=8192,b=128,ql=4,blk=1024,quant=per_token | model_config | fp8 | 431.14 | gluon | 31.72 | 0.07x | tuning_gap |
| hq=16,hkv=1,d=128,ctx=1027,b=3,ql=4,blk=1024,quant=per_tensor | model_config | fp8 | 32.48 | gluon | 17.80 | 0.55x | tuning_gap |
| hq=8,hkv=1,d=128,ctx=1027,b=3,ql=4,blk=1024,quant=per_token | model_config | fp8 | 26.20 | gluon | 18.08 | 0.69x | tuning_gap |
| hq=8,hkv=1,d=128,ctx=1027,b=81,ql=2,blk=1024,quant=per_tensor | model_config | fp8 | 31.36 | gluon | 23.20 | 0.74x | tuning_gap |
| hq=16,hkv=1,d=128,ctx=1027,b=81,ql=1,blk=1024,quant=per_tensor | model_config | fp8 | 31.16 | gluon | 23.40 | 0.75x | tuning_gap |
| hq=8,hkv=1,d=128,ctx=1027,b=81,ql=1,blk=1024,quant=per_token | model_config | fp8 | 30.40 | gluon | 23.64 | 0.78x | tuning_gap |
| hq=8,hkv=1,d=128,ctx=1027,b=3,ql=1,blk=1024,quant=per_token | model_config | fp8 | 22.00 | gluon | 18.00 | 0.82x | tuning_gap |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| hq=8,hkv=1,d=128,ctx=1027,b=3,ql=1,blk=1024,quant=per_token | 22.00 | 264.76 | 242.76 |
| hq=16,hkv=1,d=128,ctx=1027,b=3,ql=1,blk=1024,quant=per_token | 22.44 | 262.80 | 240.36 |
| hq=8,hkv=1,d=128,ctx=1027,b=3,ql=2,blk=1024,quant=per_token | 21.96 | 262.06 | 240.10 |
| hq=8,hkv=1,d=128,ctx=4096,b=16,ql=1,blk=1024,quant=per_token | 24.64 | 259.38 | 234.74 |
| hq=8,hkv=1,d=128,ctx=1027,b=3,ql=4,blk=1024,quant=per_token | 26.20 | 256.98 | 230.78 |
| hq=8,hkv=1,d=128,ctx=1027,b=81,ql=1,blk=1024,quant=per_token | 30.40 | 256.06 | 225.66 |

**Eager verdict:** launch_or_roofline_limited — eager call adds 243us host launch overhead (kernel 22.0us) -- the @flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode (mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)
  - likely fix: add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)

## Diagnosis

- `hq=16,hkv=1,d=128,ctx=8192,b=128,ql=1,blk=1024,quant=per_token` (fp8, vs-best 0.04x): **tuning_gap**
  - evidence: kernel-only vs-best 0.04x for args={'batch': 128, 'block_size': 1024, 'context_length': 8192, 'head_size': 128, 'num_kv_heads': 1, 'num_q_heads': 16, 'quant_mode': 'per_token', 'query_length': 1, 'sliding_window': 128}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `hq=16,hkv=1,d=128,ctx=8192,b=128,ql=4,blk=1024,quant=per_token` (fp8, vs-best 0.07x): **tuning_gap**
  - evidence: kernel-only vs-best 0.07x for args={'batch': 128, 'block_size': 1024, 'context_length': 8192, 'head_size': 128, 'num_kv_heads': 1, 'num_q_heads': 16, 'quant_mode': 'per_token', 'query_length': 4, 'sliding_window': 1023}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `hq=16,hkv=1,d=128,ctx=1027,b=3,ql=4,blk=1024,quant=per_tensor` (fp8, vs-best 0.55x): **tuning_gap**
  - evidence: kernel-only vs-best 0.55x for args={'batch': 3, 'block_size': 1024, 'context_length': 1027, 'head_size': 128, 'num_kv_heads': 1, 'num_q_heads': 16, 'quant_mode': 'per_tensor', 'query_length': 4, 'sliding_window': 0}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `hq=8,hkv=1,d=128,ctx=1027,b=3,ql=4,blk=1024,quant=per_token` (fp8, vs-best 0.69x): **tuning_gap**
  - evidence: kernel-only vs-best 0.69x for args={'batch': 3, 'block_size': 1024, 'context_length': 1027, 'head_size': 128, 'num_kv_heads': 1, 'num_q_heads': 8, 'quant_mode': 'per_token', 'query_length': 4, 'sliding_window': 0}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `hq=8,hkv=1,d=128,ctx=1027,b=81,ql=2,blk=1024,quant=per_tensor` (fp8, vs-best 0.74x): **tuning_gap**
  - evidence: kernel-only vs-best 0.74x for args={'batch': 81, 'block_size': 1024, 'context_length': 1027, 'head_size': 128, 'num_kv_heads': 1, 'num_q_heads': 8, 'quant_mode': 'per_tensor', 'query_length': 2, 'sliding_window': 0}
  - likely fix: profile the hot shape and add an op-specific diagnosis
- `hq=16,hkv=1,d=128,ctx=1027,b=81,ql=1,blk=1024,quant=per_tensor` (fp8, vs-best 0.75x): **tuning_gap**
  - evidence: kernel-only vs-best 0.75x for args={'batch': 81, 'block_size': 1024, 'context_length': 1027, 'head_size': 128, 'num_kv_heads': 1, 'num_q_heads': 16, 'quant_mode': 'per_tensor', 'query_length': 1, 'sliding_window': 0}
  - likely fix: profile the hot shape and add an op-specific diagnosis

## Promotion Decision

**rewrite_needed** — well below parity (geomean 0.49x); structural rework needed

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 12/12.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops pa_decode
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op pa_decode --shape-ledger benchmarks/examples/pa_decode/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/pa_decode/baseline_matrix.yaml \
  --out benchmarks/examples/pa_decode --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/pa_decode/shape_ledger.jsonl \
  --results benchmarks/examples/pa_decode/benchmark_results.jsonl --out benchmarks/examples/pa_decode/benchmark_summary.md \
  --kernel pa_decode
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
