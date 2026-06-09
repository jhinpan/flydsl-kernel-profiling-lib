# FlyDSL Flash Attention MI350X Current Benchmark

- timestamp_utc: `2026-06-08T16:17:51Z`
- host: `nimrodmi350`
- GPU: `AMD Instinct MI350X` visible_count=1 arch=`gfx950`
- HIP_VISIBLE_DEVICES: `2`
- FlyDSL: `/sgl-workspace/FlyDSL-lab@48170e30482d`
- timing: warm event timing, median of repeats; output allocation excluded for FlyDSL, preallocated `out=` attempted for AITER

## Coverage

- result rows: 198
- ok rows: 171
- failed/unsupported rows: 27

## Best Provider Per Shape

| shape | B | S | H/Hkv | D | dtype | causal | auto path | best | us | TFLOPS | auto/CK | dualwave/generic_m256 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| boundary_B1_S128_H32 | 1 | 128 | 32/32 | 128 | bf16 | causal | generic_m128 | aiter_asm | 10.8 | 12.5 | 0.14x |  |
| boundary_B1_S256_H32 | 1 | 256 | 32/32 | 128 | bf16 | causal | generic_m128 | aiter_ck | 12.2 | 44.1 | 0.14x |  |
| boundary_B1_S384_H32 | 1 | 384 | 32/32 | 128 | bf16 | causal | generic_m128 | aiter_ck | 14.9 | 80.9 | 0.18x |  |
| flydsl_default_B1_S384_H64_Hkv64 | 1 | 384 | 64/64 | 128 | bf16 | causal | generic_m128 | aiter_ck | 18.9 | 127.8 | 0.23x |  |
| boundary_B1_S512_H32 | 1 | 512 | 32/32 | 128 | bf16 | causal | dualwave_gfx950 | aiter_ck | 18.1 | 118.5 | 0.17x | 0.75x |
| fp16_S512 | 1 | 512 | 32/32 | 128 | fp16 | causal | dualwave_gfx950 | aiter_ck | 18.5 | 116.0 | 0.17x | 0.75x |
| boundary_B1_S1024_H32 | 1 | 1024 | 32/32 | 128 | bf16 | causal | dualwave_gfx950 | aiter_ck | 35.7 | 240.9 | 0.33x | 0.75x |
| flydsl_default_B1_S1024_H64_Hkv64 | 1 | 1024 | 64/64 | 128 | bf16 | causal | dualwave_gfx950 | aiter_ck | 46.6 | 368.8 | 0.42x | 0.75x |
| boundary_B1_S2048_H32 | 1 | 2048 | 32/32 | 128 | bf16 | causal | dualwave_gfx950 | aiter_ck | 70.3 | 488.5 | 0.65x | 0.99x |
| fp16_S2048 | 1 | 2048 | 32/32 | 128 | fp16 | causal | dualwave_gfx950 | aiter_ck | 76.0 | 452.1 | 0.69x | 0.98x |
| flydsl_default_B1_S2048_H64_Hkv64 | 1 | 2048 | 64/64 | 128 | bf16 | causal | dualwave_gfx950 | auto | 111.1 | 618.4 | 1.09x | 1.49x |
| boundary_B1_S4096_H32 | 1 | 4096 | 32/32 | 128 | bf16 | causal | dualwave_gfx950 | auto | 172.3 | 797.8 | 1.23x | 1.64x |
| flydsl_default_B1_S4096_H64_Hkv64 | 1 | 4096 | 64/64 | 128 | bf16 | causal | dualwave_gfx950 | dualwave | 333.5 | 824.1 | 1.22x | 1.64x |
| wide_h128 | 1 | 4096 | 128/8 | 128 | bf16 | causal | dualwave_gfx950 | dualwave | 626.6 | 877.4 | 1.34x | 1.69x |
| llama3_8b_prefill | 1 | 8192 | 32/8 | 128 | bf16 | causal | dualwave_gfx950 | dualwave | 600.6 | 915.3 | 1.29x | 1.83x |
| llama3_70b_prefill | 1 | 8192 | 64/8 | 128 | bf16 | causal | dualwave_gfx950 | dualwave | 1182.8 | 929.6 | 1.34x | 1.76x |
| D64_nocausal | 2 | 2048 | 32/32 | 64 | bf16 | nocausal | generic_m256 | aiter_ck | 145.8 | 471.2 |  |  |
| D64_causal | 2 | 2048 | 32/32 | 64 | bf16 | causal | generic_m256 | aiter_ck | 83.3 | 412.6 |  |  |
| D96_nocausal | 2 | 2048 | 32/32 | 96 | bf16 | nocausal | generic_m256 | aiter_ck | 176.3 | 584.8 |  |  |
| D96_causal | 2 | 2048 | 32/32 | 96 | bf16 | causal | generic_m256 | aiter_ck | 112.0 | 460.1 |  |  |
| D128_nocausal | 2 | 2048 | 32/32 | 128 | bf16 | nocausal | dualwave_gfx950 | dualwave | 159.4 | 862.3 | 1.16x | 1.59x |
| D128_causal | 2 | 2048 | 32/32 | 128 | bf16 | causal | dualwave_gfx950 | auto | 112.1 | 613.3 | 1.07x | 1.58x |
| boundary_B2_S8192_H32 | 2 | 8192 | 32/32 | 128 | bf16 | causal | dualwave_gfx950 | auto | 1233.9 | 891.1 | 1.25x | 1.70x |
| fp16_S8192 | 2 | 8192 | 32/32 | 128 | fp16 | causal | dualwave_gfx950 | dualwave | 1318.8 | 833.7 | 1.24x | 1.61x |
| qwen_gqa_h28 | 4 | 4096 | 28/4 | 128 | bf16 | causal | dualwave_gfx950 | auto | 574.4 | 837.4 | 1.27x | 1.65x |
| llama3_8b_batch | 4 | 4096 | 32/8 | 128 | bf16 | causal | dualwave_gfx950 | dualwave | 648.2 | 848.1 | 1.26x | 1.64x |
| llama3_70b_batch | 4 | 4096 | 64/8 | 128 | bf16 | causal | dualwave_gfx950 | auto | 1261.1 | 871.9 | 1.35x | 1.73x |
| flydsl_default_B4_S8192_H64_Hkv64 | 4 | 8192 | 64/64 | 128 | bf16 | causal | dualwave_gfx950 | dualwave | 4904.0 | 896.8 | 1.32x | 1.62x |
| flydsl_default_B8_S128_H64_Hkv64 | 8 | 128 | 64/64 | 128 | bf16 | causal | generic_m128 | aiter_asm | 21.4 | 100.1 | 0.28x |  |
| flydsl_default_B8_S256_H64_Hkv64 | 8 | 256 | 64/64 | 128 | bf16 | causal | generic_m128 | aiter_asm | 39.6 | 217.0 | 0.55x |  |
| flydsl_default_B16_S8192_H16_Hkv16 | 16 | 8192 | 16/16 | 128 | bf16 | causal | dualwave_gfx950 | dualwave | 4771.6 | 921.7 | 1.33x | 1.62x |
| flydsl_default_B16_S8192_H64_Hkv8 | 16 | 8192 | 64/8 | 128 | bf16 | causal | dualwave_gfx950 | dualwave | 18546.0 | 948.6 | 1.37x | 1.72x |
| flydsl_default_B32_S8192_H8_Hkv8 | 32 | 8192 | 8/8 | 128 | bf16 | causal | dualwave_gfx950 | auto | 4817.9 | 912.8 | 1.31x | 1.63x |

## Provider Geomean Ratios

Ratios use per-shape pairs only. `>1` means the left provider is faster.

- `auto` / `aiter_ck` geomean: **0.676x** over 29 shapes
- `auto` / `aiter_asm` geomean: **0.944x** over 26 shapes
- `dualwave` / `generic_m256` geomean: **1.377x** over 23 shapes
- `dualwave` / `aiter_ck` geomean: **0.901x** over 23 shapes
- `generic_m256` / `aiter_ck` geomean: **0.527x** over 29 shapes
- `generic_m128` / `aiter_ck` geomean: **0.560x** over 31 shapes

## Failures And Unsupported Rows

| shape | provider | status | reason |
|---|---|---|---|
| boundary_B1_S128_H32 | dualwave | unsupported | dualwave dispatcher contract is S>=384 and S%256==0 |
| boundary_B1_S256_H32 | dualwave | unsupported | dualwave dispatcher contract is S>=384 and S%256==0 |
| boundary_B1_S384_H32 | dualwave | unsupported | dualwave dispatcher contract is S>=384 and S%256==0 |
| flydsl_default_B8_S128_H64_Hkv64 | dualwave | unsupported | dualwave dispatcher contract is S>=384 and S%256==0 |
| flydsl_default_B8_S256_H64_Hkv64 | dualwave | unsupported | dualwave dispatcher contract is S>=384 and S%256==0 |
| flydsl_default_B1_S384_H64_Hkv64 | dualwave | unsupported | dualwave dispatcher contract is S>=384 and S%256==0 |
| D64_causal | generic_m256 | unsupported | skipped after observed GPU memory fault for generic_m256 with D<128 |
| D64_causal | dualwave | unsupported | dualwave is D=128 only |
| D64_causal | auto | unsupported | skipped: auto would dispatch to generic_m256, which faulted for D<128 |
| D64_causal | aiter_asm | failed | invalid argument for fmha_fwd |
| D64_nocausal | generic_m256 | unsupported | skipped after observed GPU memory fault for generic_m256 with D<128 |
| D64_nocausal | dualwave | unsupported | dualwave is D=128 only |
| D64_nocausal | auto | unsupported | skipped: auto would dispatch to generic_m256, which faulted for D<128 |
| D64_nocausal | aiter_asm | failed | invalid argument for fmha_fwd |
| D96_causal | generic_m128 | failed | AssertionError:  |
| D96_causal | generic_m256 | unsupported | skipped after observed GPU memory fault for generic_m256 with D<128 |
| D96_causal | dualwave | unsupported | dualwave is D=128 only |
| D96_causal | auto | unsupported | skipped: auto would dispatch to generic_m256, which faulted for D<128 |
| D96_causal | aiter_asm | failed | invalid argument for fmha_fwd |
| D96_nocausal | generic_m128 | failed | AssertionError:  |
| D96_nocausal | generic_m256 | unsupported | skipped after observed GPU memory fault for generic_m256 with D<128 |
| D96_nocausal | dualwave | unsupported | dualwave is D=128 only |
| D96_nocausal | auto | unsupported | skipped: auto would dispatch to generic_m256, which faulted for D<128 |
| D96_nocausal | aiter_asm | failed | invalid argument for fmha_fwd |
| fp16_S512 | aiter_asm | unsupported | aiter asm v3 is bf16 only |
| fp16_S2048 | aiter_asm | unsupported | aiter asm v3 is bf16 only |
| fp16_S8192 | aiter_asm | unsupported | aiter asm v3 is bf16 only |
