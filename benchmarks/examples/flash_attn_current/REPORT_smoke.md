# FlyDSL Flash Attention MI350X Current Benchmark

- timestamp_utc: `2026-06-08T16:15:23Z`
- host: `nimrodmi350`
- GPU: `AMD Instinct MI350X` visible_count=1 arch=`gfx950`
- HIP_VISIBLE_DEVICES: `1`
- FlyDSL: `/sgl-workspace/FlyDSL-lab@48170e30482d`
- timing: warm event timing, median of repeats; output allocation excluded for FlyDSL, preallocated `out=` attempted for AITER

## Coverage

- result rows: 18
- ok rows: 17
- failed/unsupported rows: 1

## Best Provider Per Shape

| shape | B | S | H/Hkv | D | dtype | causal | auto path | best | us | TFLOPS | auto/CK | dualwave/generic_m256 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| small_generic | 1 | 256 | 8/8 | 128 | bf16 | causal | generic_m128 | aiter_ck | 13.9 | 9.7 | 0.15x |  |
| dualwave_gate | 1 | 512 | 8/8 | 128 | bf16 | causal | dualwave_gfx950 | aiter_ck | 18.7 | 28.7 | 0.16x | 0.74x |
| diagnostic_old | 1 | 2048 | 32/32 | 128 | bf16 | causal | dualwave_gfx950 | aiter_ck | 71.7 | 479.0 | 0.60x | 1.11x |

## Provider Geomean Ratios

Ratios use per-shape pairs only. `>1` means the left provider is faster.

- `auto` / `aiter_ck` geomean: **0.245x** over 3 shapes
- `auto` / `aiter_asm` geomean: **0.312x** over 3 shapes
- `dualwave` / `generic_m256` geomean: **0.905x** over 2 shapes
- `dualwave` / `aiter_ck` geomean: **0.311x** over 2 shapes
- `generic_m256` / `aiter_ck` geomean: **0.260x** over 3 shapes
- `generic_m128` / `aiter_ck` geomean: **0.280x** over 3 shapes

## Failures And Unsupported Rows

| shape | provider | status | reason |
|---|---|---|---|
| small_generic | dualwave | unsupported | dualwave dispatcher contract is S>=384 and S%256==0 |
