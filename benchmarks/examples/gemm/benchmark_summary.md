# Benchmark Summary: gemm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 265 (sources: aiter_model_shapes=265)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.
- Graph cache state: l2_flushed_graph=134.

## Coverage

| Category | Count |
|---|---:|
| total shapes | 265 |
| FlyDSL correct + timed | 30 |
| FlyDSL failed/oom | 195 |
| FlyDSL incorrect | 0 |
| FlyDSL unsupported | 40 |
| measured FlyDSL-vs-baseline pairs | 30 |

## Overall Speedup (kernel-only, vs best available)

| Aggregate | value |
|---|---:|
| unweighted geomean vs best | 0.50x  (n=30) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.89x  (n=14) |
| vs aiter_triton | 0.59x  (n=30) |
| vs hipblaslt | 0.55x  (n=30) |
| vs pytorch | 0.55x  (n=30) |
| worst hot shape | 0.26x  (M=1,N=256,K=7168 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 30 | 0.50x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 5 | 0.41x |
| GPT-OSS 120B | 15 | 0.62x |
| Llama4 Maverick | 5 | 0.38x |
| Qwen3-235B-A22B | 5 | 0.43x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=1,N=2880,K=512 | model_config | bf16 | 13.48 | aiter_triton | 13.56 | 1.01x |
| M=32,N=2880,K=512 | model_config | bf16 | 13.56 | aiter_triton | 13.60 | 1.00x |
| M=2048,N=640,K=2880 | model_config | bf16 | 31.84 | aiter | 29.80 | 0.94x |
| M=16384,N=256,K=7168 | model_config | bf16 | 115.88 | pytorch | 104.60 | 0.90x |
| M=256,N=2880,K=512 | model_config | bf16 | 17.96 | aiter_triton | 14.36 | 0.80x |
| M=16384,N=128,K=2880 | model_config | bf16 | 49.72 | pytorch | 37.44 | 0.75x |
| M=16384,N=640,K=2880 | model_config | bf16 | 114.40 | pytorch | 76.08 | 0.67x |
| M=256,N=640,K=2880 | model_config | bf16 | 30.40 | aiter_triton | 18.40 | 0.61x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=1,N=256,K=7168 | model_config | bf16 | 60.64 | aiter_triton | 15.96 | 0.26x | algorithm_gap |
| M=32,N=256,K=7168 | model_config | bf16 | 61.08 | aiter_triton | 16.60 | 0.27x | algorithm_gap |
| M=256,N=256,K=7168 | model_config | bf16 | 62.32 | hipblaslt | 19.96 | 0.32x | algorithm_gap |
| M=1,N=128,K=4096 | model_config | bf16 | 43.48 | aiter | 14.40 | 0.33x | algorithm_gap |
| M=1,N=128,K=5120 | model_config | bf16 | 46.36 | aiter_triton | 15.44 | 0.33x | algorithm_gap |
| M=32,N=128,K=5120 | model_config | bf16 | 47.16 | aiter_triton | 15.96 | 0.34x | algorithm_gap |
| M=2048,N=128,K=5120 | model_config | bf16 | 66.84 | hipblaslt | 23.60 | 0.35x | algorithm_gap |
| M=16384,N=2880,K=512 | model_config | bf16 | 235.72 | hipblaslt | 84.20 | 0.36x | algorithm_gap |

## FlyDSL hard failures (crash / incorrect)

| shape | model | stage | dtype | status | reason |
|---|---|---|---|---|---|
| M=1,N=512,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=512,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=512,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=512,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=512,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=768,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=768,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=768,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=768,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=768,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=1280,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=1280,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=1280,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=1280,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=1280,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=2112,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=2112,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=2112,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=2112,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=2112,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=2304,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=2304,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=2304,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=2304,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=2304,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=3072,K=1536 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=3072,K=1536 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=3072,K=1536 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=3072,K=1536 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=3072,K=1536 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=3584,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=3584,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=3584,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=3584,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=3584,K=4096 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=4096,K=1792 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=4096,K=512 | DeepSeek-R1\|Llama3 8 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=4096,K=512 | DeepSeek-R1\|Llama3 8 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=4096,K=1792 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=4096,K=512 | DeepSeek-R1\|Llama3 8 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=4096,K=1792 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=4096,K=1792 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=4096,K=512 | DeepSeek-R1\|Llama3 8 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=4096,K=1792 | Llama3 8B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=4096,K=512 | DeepSeek-R1\|Llama3 8 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=4608,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=4608,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=4608,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=4608,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=4608,K=7168 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=7168,K=256 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=7168,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=7168,K=2304 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=7168,K=2048 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=7168,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=7168,K=256 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=7168,K=2048 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=7168,K=2304 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=7168,K=2048 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=7168,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=7168,K=2304 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=7168,K=256 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=7168,K=2304 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=7168,K=2048 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=7168,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=7168,K=256 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=7168,K=256 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=7168,K=2304 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=7168,K=8192 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=7168,K=2048 | DeepSeek-R1 | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=8192,K=3584 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=8192,K=1024 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=8192,K=1024 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=8192,K=3584 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=8192,K=3584 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=8192,K=1024 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=8192,K=3584 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=8192,K=1024 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=8192,K=3584 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=8192,K=1024 | Llama3 70B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=13312,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=13312,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=13312,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=13312,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=13312,K=16384 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=16384,K=6656 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=16384,K=2048 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=16384,K=2048 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=32,N=16384,K=6656 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=16384,K=6656 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=256,N=16384,K=2048 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=16384,K=2048 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=2048,N=16384,K=6656 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=16384,K=6656 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=16384,N=16384,K=2048 | Llama3 405B | model_config | fp4 | failed | input/reference build: 'fp4' |
| M=1,N=768,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=768,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=768,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=768,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=768,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=896,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=896,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=896,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=896,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=896,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=1152,K=4096 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=1152,K=4096 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=1152,K=4096 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=1152,K=4096 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=1152,K=4096 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=1280,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=1280,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=1280,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=1280,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=1280,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=2048,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=2048,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=2048,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=2048,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=2048,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=2304,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=2304,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=2304,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=2304,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=2304,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=3584,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=3584,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=3584,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=3584,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=3584,K=4096 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=4096,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=4096,K=512 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=4096,K=1024 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=4096,K=1792 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=4096,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=4096,K=1792 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=4096,K=512 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=4096,K=1024 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=4096,K=512 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=4096,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=4096,K=1024 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=4096,K=1792 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=4096,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=4096,K=512 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=4096,K=1792 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=4096,K=1024 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=4096,K=1024 | Qwen3-235B-A22B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=4096,K=512 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=4096,K=1792 | Llama3 8B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=4096,K=5120 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=5120,K=2048 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=5120,K=1024 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=5120,K=640 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=5120,K=1024 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=5120,K=640 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=5120,K=2048 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=5120,K=2048 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=5120,K=1024 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=5120,K=640 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=5120,K=2048 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=5120,K=1024 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=5120,K=640 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=5120,K=1024 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=5120,K=2048 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=5120,K=640 | Llama4 Maverick | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=7168,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=7168,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=7168,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=7168,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=7168,K=8192 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=8192,K=3584 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=8192,K=1024 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=8192,K=3584 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=8192,K=1024 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=8192,K=1024 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=8192,K=3584 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=8192,K=3584 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=8192,K=1024 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=8192,K=1024 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=8192,K=3584 | Llama3 70B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=13312,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=13312,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=13312,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=13312,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=13312,K=16384 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=16384,K=6656 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=1,N=16384,K=2048 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=16384,K=6656 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=32,N=16384,K=2048 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=16384,K=2048 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=256,N=16384,K=6656 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=16384,K=6656 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=2048,N=16384,K=2048 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=16384,K=2048 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |
| M=16384,N=16384,K=6656 | Llama3 405B | model_config | int8 | failed | input/reference build: 'int8' |

## Eager vs kernel-only (host launch overhead)

FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every call; on short shapes this host overhead dwarfs the kernel. This is a launcher (host-side) issue, distinct from kernel speed.

| shape | FlyDSL kernel us | FlyDSL eager us | host overhead us |
|---|---:|---:|---:|
| M=256,N=128,K=5120 | 49.12 | 56.34 | 7.22 |
| M=16384,N=256,K=7168 | 115.88 | 115.32 | -0.56 |
| M=16384,N=640,K=2880 | 114.40 | 111.68 | -2.72 |
| M=32,N=128,K=5120 | 47.16 | 41.18 | -5.98 |
| M=16384,N=128,K=4096 | 76.20 | 70.22 | -5.98 |
| M=1,N=128,K=5120 | 46.36 | 39.96 | -6.40 |

## Diagnosis

- `M=1,N=256,K=7168` (bf16, vs-best 0.26x): **algorithm_gap**
  - evidence: hgemm_splitk is slower than the best GEMM baseline (0.26x) for args={'K': 7168, 'M': 1, 'N': 256}
  - likely fix: retune tile/split-K selection or route this shape to a stronger GEMM backend
- `M=32,N=256,K=7168` (bf16, vs-best 0.27x): **algorithm_gap**
  - evidence: hgemm_splitk is slower than the best GEMM baseline (0.27x) for args={'K': 7168, 'M': 32, 'N': 256}
  - likely fix: retune tile/split-K selection or route this shape to a stronger GEMM backend
- `M=256,N=256,K=7168` (bf16, vs-best 0.32x): **algorithm_gap**
  - evidence: hgemm_splitk is slower than the best GEMM baseline (0.32x) for args={'K': 7168, 'M': 256, 'N': 256}
  - likely fix: retune tile/split-K selection or route this shape to a stronger GEMM backend
- `M=1,N=128,K=4096` (bf16, vs-best 0.33x): **algorithm_gap**
  - evidence: hgemm_splitk is slower than the best GEMM baseline (0.33x) for args={'K': 4096, 'M': 1, 'N': 128}
  - likely fix: retune tile/split-K selection or route this shape to a stronger GEMM backend
- `M=1,N=128,K=5120` (bf16, vs-best 0.33x): **algorithm_gap**
  - evidence: hgemm_splitk is slower than the best GEMM baseline (0.33x) for args={'K': 5120, 'M': 1, 'N': 128}
  - likely fix: retune tile/split-K selection or route this shape to a stronger GEMM backend
- `M=32,N=128,K=5120` (bf16, vs-best 0.34x): **algorithm_gap**
  - evidence: hgemm_splitk is slower than the best GEMM baseline (0.34x) for args={'K': 5120, 'M': 32, 'N': 128}
  - likely fix: retune tile/split-K selection or route this shape to a stronger GEMM backend
- `M=1,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=1,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=32,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=256,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=2048,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support
- `M=16384,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **measurement_issue**
  - likely fix: exclude unsupported dtype rows from this op run, or implement input/reference support

## Promotion Decision

**rewrite_needed** — well below parity (geomean 0.50x) + hard failures on a shape class; structural rework needed

Reading:
- Correct+timed FlyDSL-vs-baseline pairs: 30/265.
- Hard FlyDSL failures must be fixed or explicitly scoped out before broad promotion.
- Only supported bf16/f16 rows should drive the hgemm_splitk verdict; quantized rows need separate adapters.

## Reproduction

```bash
# 1. use the checked-in shape ledger, or refresh model_config rows when this op is importer-backed
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --ops gemm
# 2. run (env.sh sets the FlyDSL build-tree PYTHONPATH/LD that also unblocks aiter)
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op gemm --shape-ledger benchmarks/examples/gemm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/gemm/baseline_matrix.yaml \
  --out benchmarks/examples/gemm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/gemm/shape_ledger.jsonl \
  --results benchmarks/examples/gemm/benchmark_results.jsonl --out benchmarks/examples/gemm/benchmark_summary.md \
  --kernel gemm
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
