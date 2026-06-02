# Benchmark Summary: gemm

## Scope

- GPU: AMD Instinct MI350X  |  Arch: gfx950  |  ROCm: 7.2.0
- torch: 2.9.1+rocm7.2.0.git7e1940d4  |  triton: 3.6.0
- FlyDSL commit: 7255fff8  |  AITER commit: 32e1e6d76  |  SGLang commit: b6f71d585
- Shapes: 265 (sources: aiter_model_shapes=265)
- Headline metric: **kernel-only** (CUDA-graph) median speedup vs best available baseline. Eager/host-overhead reported separately.

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
| unweighted geomean vs best | 0.37x  (n=30) |
| production-weighted geomean vs best | n/a (no weights yet — add a serving trace) |
| vs aiter | 0.98x  (n=14) |
| vs aiter_triton | 0.42x  (n=30) |
| vs hipblaslt | 0.43x  (n=30) |
| vs pytorch | 0.43x  (n=30) |
| worst hot shape | 0.10x  (M=1,N=256,K=7168 vs aiter_triton) |

## Stage Split (kernel-only vs best)

| Stage | Shapes | Geomean vs best |
|---|---:|---:|
| model_config | 30 | 0.37x |

## Model Split (kernel-only vs best)

| Model | Shapes | Geomean vs best |
|---|---:|---:|
| DeepSeek-R1 | 5 | 0.27x |
| GPT-OSS 120B | 15 | 0.47x |
| Llama4 Maverick | 5 | 0.29x |
| Qwen3-235B-A22B | 5 | 0.32x |

## Top Wins (kernel-only)

| shape | stage | dtype | FlyDSL us | best baseline | baseline us | speedup |
|---|---|---|---:|---|---:|---:|
| M=16384,N=128,K=4096 | model_config | bf16 | 31.95 | aiter_triton | 36.49 | 1.14x |
| M=16384,N=256,K=7168 | model_config | bf16 | 83.01 | aiter_triton | 93.47 | 1.13x |
| M=16384,N=128,K=2880 | model_config | bf16 | 24.83 | hipblaslt | 27.72 | 1.12x |
| M=1,N=2880,K=512 | model_config | bf16 | 2.52 | aiter_triton | 2.63 | 1.04x |
| M=16384,N=128,K=5120 | model_config | bf16 | 42.00 | aiter_triton | 42.16 | 1.00x |
| M=2048,N=640,K=2880 | model_config | bf16 | 20.96 | pytorch | 20.27 | 0.97x |
| M=32,N=2880,K=512 | model_config | bf16 | 2.68 | aiter_triton | 2.45 | 0.91x |
| M=256,N=2880,K=512 | model_config | bf16 | 5.56 | aiter_triton | 4.00 | 0.72x |

## Top Regressions (kernel-only) + diagnosis

| shape | stage | dtype | FlyDSL us | best | baseline us | speedup | classification |
|---|---|---|---:|---|---:|---:|---|
| M=1,N=256,K=7168 | model_config | bf16 | 43.47 | aiter_triton | 4.39 | 0.10x | implementation_gap |
| M=32,N=256,K=7168 | model_config | bf16 | 43.66 | aiter_triton | 5.97 | 0.14x | implementation_gap |
| M=32,N=128,K=5120 | model_config | bf16 | 31.48 | aiter_triton | 5.10 | 0.16x | implementation_gap |
| M=1,N=128,K=4096 | model_config | bf16 | 25.87 | aiter_triton | 4.56 | 0.18x | implementation_gap |
| M=1,N=128,K=5120 | model_config | bf16 | 31.27 | aiter_triton | 5.57 | 0.18x | implementation_gap |
| M=256,N=256,K=7168 | model_config | bf16 | 44.02 | hipblaslt | 8.58 | 0.19x | implementation_gap |
| M=256,N=128,K=5120 | model_config | bf16 | 31.96 | aiter_triton | 6.50 | 0.20x | implementation_gap |
| M=32,N=128,K=4096 | model_config | bf16 | 26.13 | aiter_triton | 5.47 | 0.21x | implementation_gap |

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
| M=16384,N=128,K=5120 | 42.00 | 86.76 | 44.76 |
| M=16384,N=128,K=4096 | 31.95 | 70.26 | 38.31 |
| M=16384,N=256,K=7168 | 83.01 | 114.72 | 31.71 |
| M=2048,N=128,K=5120 | 32.33 | 59.88 | 27.55 |
| M=256,N=128,K=5120 | 31.96 | 56.16 | 24.20 |
| M=16384,N=2880,K=512 | 202.00 | 221.78 | 19.78 |

## Diagnosis

- `M=1,N=256,K=7168` (bf16, vs-best 0.10x): **implementation_gap**
  - evidence: N=256 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.10x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=32,N=256,K=7168` (bf16, vs-best 0.14x): **implementation_gap**
  - evidence: N=256 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.14x). Compounded at small M=32: grid=(M,1,1) launches one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=32,N=128,K=5120` (bf16, vs-best 0.16x): **implementation_gap**
  - evidence: N=128 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.16x). Compounded at small M=32: grid=(M,1,1) launches one workgroup per row, so only ~32 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=128,K=4096` (bf16, vs-best 0.18x): **implementation_gap**
  - evidence: N=128 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.18x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=128,K=5120` (bf16, vs-best 0.18x): **implementation_gap**
  - evidence: N=128 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.18x). Compounded at small M=1: grid=(M,1,1) launches one workgroup per row, so only ~1 of the ~256 CUs are used (under-occupied).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=256,N=256,K=7168` (bf16, vs-best 0.19x): **implementation_gap**
  - evidence: N=256 misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic scalar path; per-block efficiency loss (kernel-only vs-best 0.19x).
  - likely fix: vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs
- `M=1,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=512,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=768,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=1280,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=2112,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=2304,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=3072,K=1536` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=3584,K=4096` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=4096,K=1792` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=4096,K=512` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=4608,K=7168` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=7168,K=256` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=7168,K=2304` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=7168,K=8192` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=7168,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=8192,K=3584` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=8192,K=1024` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=13312,K=16384` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=16384,K=6656` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=16384,K=2048` (fp4): **failed** — input/reference build: 'fp4'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=768,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=896,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=1152,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=1280,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=2048,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=2304,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=3584,K=4096` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=4096,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=4096,K=512` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=4096,K=1792` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=4096,K=5120` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=5120,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=5120,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=5120,K=640` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=7168,K=8192` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=8192,K=1024` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=8192,K=3584` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=13312,K=16384` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=1,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=32,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=256,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=2048,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=16384,K=2048` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path
- `M=16384,N=16384,K=6656` (int8): **failed** — input/reference build: 'int8'
  - classification: **flydsl_codegen_gap** (illegal launch config)
  - likely fix: annotate known_block_size / clamp block size to <=256 in the large-M small-N path

## Promotion Decision

**rewrite_needed** — well below parity (geomean 0.37x); structural rework needed

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
  --op gemm --shape-ledger benchmarks/examples/gemm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/gemm/baseline_matrix.yaml \
  --out benchmarks/examples/gemm --warmup-iters 20 --repeat-iters 60
# 3. reports
python -m benchmarks.reports.summarize_results --shape-ledger benchmarks/examples/gemm/shape_ledger.jsonl \
  --results benchmarks/examples/gemm/benchmark_results.jsonl --out benchmarks/examples/gemm/benchmark_summary.md
```

Raw artifacts: `shape_ledger.jsonl`, `benchmark_results.jsonl`, `benchmark_results.csv`, `coverage_matrix.md`, `profiles/`
