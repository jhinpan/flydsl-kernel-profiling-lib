# FlyDSL Flash Attention MI350X Current Benchmark

- timestamp_utc: `2026-06-08T16:20:55Z`
- host: `nimrodmi350`
- GPU: `AMD Instinct MI350X` visible_count=1 arch=`gfx950`
- HIP_VISIBLE_DEVICES: `3`
- FlyDSL: `/sgl-workspace/FlyDSL-lab@48170e30482d`
- timing: warm event timing, median of repeats; output allocation excluded for FlyDSL, preallocated `out=` attempted for AITER

## Coverage

- result rows: 22
- ok rows: 22
- failed/unsupported rows: 0

## Best Provider Per Shape

| shape | B | S | H/Hkv | D | dtype | causal | auto path | best | us | TFLOPS | auto/CK | dualwave/generic_m256 |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| config_diag_2048 | 1 | 2048 | 32/32 | 128 | bf16 | causal | dualwave_gfx950 | aiter_ck | 80.9 | 424.9 | 0.72x | 0.95x |
| config_gqa_8192 | 4 | 8192 | 64/8 | 128 | bf16 | causal | dualwave_gfx950 | dualwave_no_setprio | 4456.0 | 987.0 | 1.41x | 1.79x |

## Provider Geomean Ratios

Ratios use per-shape pairs only. `>1` means the left provider is faster.

- `auto` / `aiter_ck` geomean: **1.010x** over 2 shapes
- `auto` / `aiter_asm` geomean: **1.338x** over 2 shapes
- `dualwave` / `generic_m256` geomean: **1.308x** over 2 shapes
- `dualwave` / `aiter_ck` geomean: **0.989x** over 2 shapes
- `generic_m256` / `aiter_ck` geomean: **0.756x** over 2 shapes
- `generic_m128` / `aiter_ck` geomean: **0.795x** over 2 shapes

## Failures And Unsupported Rows

| shape | provider | status | reason |
|---|---|---|---|
