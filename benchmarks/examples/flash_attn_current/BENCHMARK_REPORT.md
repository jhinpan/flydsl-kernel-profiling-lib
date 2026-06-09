# FlyDSL Flash Attention MI350X Benchmark Report

Date: 2026-06-08  
Host: `nimrodmi350`  
GPU: `AMD Instinct MI350X`, arch `gfx950`  
FlyDSL checkout: `/sgl-workspace/FlyDSL-lab@48170e30482daa0117eae4951106a4208f99f4b1`  
Benchmark harness: `/sgl-workspace/flydsl-kernel-profiling@ec880afff62518675630ef10b913b9bb06cd5611`

## Executive Summary

Current FlyDSL flash attention on MI350X has two real implementation families:

- `flash_attn_generic.py`: generic MFMA32 kernel with `BLOCK_M=128` or `BLOCK_M=256`.
- `flash_attn_gfx950.py`: gfx950-only dual-wave software-pipelined fast path for `D=128`, bf16/fp16.

The high-level `auto` launcher first tries the gfx950 dual-wave path for eligible runtime sequence lengths, then falls back to generic auto dispatch. This is the main reason performance changes sharply across sequence-length and head-count regimes.

Top-line results from the full run:

- Full sweep: 198 provider rows across 33 shapes, 171 successful rows.
- `auto / aiter_ck` geomean: `0.676x` over 29 paired shapes.
- For long model-like prefill/GQA shapes, `auto / aiter_ck` is consistently faster, about `1.26x-1.37x`.
- For short and mid shapes, especially `B=1,H=32,S<=2048`, `auto` is much slower than AITER CK, mostly because it routes into dualwave too early.
- `dualwave / generic_m256` geomean: `1.377x` over 23 paired shapes, but this hides short-shape losses.
- The D64/D96 coverage found correctness/robustness gaps:
  - `generic_m256` with `D<128` triggered a GPU memory access fault in the first full run.
  - `generic_m128` with `D=96` raises `AssertionError`.
  - Current auto would choose the unsafe `generic_m256` path for D64/D96 large enough shapes, so the dispatch contract needs tightening if D<128 is intended to be supported.

Most actionable recommendation: change the dualwave auto gate from only `S>=384 && S%256==0` to a work/CTA threshold as well. The data points to `B * H * ceil(S / 256) >= 512` as a good first gate to test, because dualwave loses hard at 64/128/256 CTAs and starts winning around 512 CTAs.

## Artifacts

All artifacts are under:

`/sgl-workspace/flydsl-kernel-profiling/benchmarks/examples/flash_attn_current/`

Important files:

| file | purpose |
|---|---|
| `results_full.csv` | full benchmark result table |
| `results_full.jsonl` | full benchmark raw structured rows |
| `REPORT_full.md` | generated per-shape summary for full suite |
| `full_run2.log` | successful full benchmark log |
| `full_run.log` | first full run log; contains the D64 `generic_m256` GPU fault |
| `results_config.csv` | config/toggle sweep result table |
| `REPORT_config.md` | generated config sweep summary |
| `config_run.log` | config sweep log |
| `metadata_full.json` | full-run environment metadata |
| `metadata_config.json` | config-run environment metadata |

Harness:

`/sgl-workspace/flydsl-kernel-profiling/benchmarks/special/flash_attn_mi350_current.py`

## Methodology

The benchmark compares:

- `generic_m128`: forced generic `BLOCK_M=128`.
- `generic_m256`: forced generic `BLOCK_M=256`.
- `dualwave`: forced gfx950 dual-wave software-pipelined kernel.
- `auto`: current exported FlyDSL builder/dispatcher.
- `aiter_ck`: AITER CK FMHA baseline.
- `aiter_asm`: AITER ASM/v3 FMHA baseline where supported.

The config suite also compares:

- `generic_m128_n32`
- `generic_m128_n128`
- `dualwave_no_lazy`
- `dualwave_no_setprio`
- `dualwave_no_stagger`

Timing policy:

- HIP event timing.
- Median of 3 repeats.
- 10 warmup iterations and 30 measured iterations for full/config runs.
- FlyDSL output tensor allocation excluded from timed region.
- AITER `out=` was preallocated where the API accepted it.
- Correctness was checked against PyTorch SDPA only when the score matrix was below the configured threshold. Large long-sequence rows are marked `not_checked` to avoid huge reference workspaces.

Reproduce:

```bash
cd /sgl-workspace/flydsl-kernel-profiling
export FLYDSL_LAB=/sgl-workspace/FlyDSL-lab
export PYTHONPATH=/sgl-workspace/FlyDSL-lab/build-fly/python_packages:/sgl-workspace/FlyDSL-lab:/sgl-workspace/aiter:${PYTHONPATH:-}

HIP_VISIBLE_DEVICES=2 python3 benchmarks/special/flash_attn_mi350_current.py \
  --suite full \
  --out benchmarks/examples/flash_attn_current \
  --warmup 10 --iters 30 --repeats 3 --verify

HIP_VISIBLE_DEVICES=3 python3 benchmarks/special/flash_attn_mi350_current.py \
  --suite config \
  --out benchmarks/examples/flash_attn_current \
  --warmup 10 --iters 30 --repeats 3 --verify
```

## Source Analysis

### Generic Kernel

Code pointer:

`/sgl-workspace/FlyDSL-lab/kernels/flash_attn_generic.py:4-20`

The file-level contract says the generic path supports `BLOCK_M=128` or `256`, `BLOCK_N=64`, and requires `head_dim % 32 == 0`, `head_dim >= 64`, `seq_len % 128 == 0`.

Relevant excerpt:

```python
# Tile shape: BLOCK_M=128 or 256 (auto-selected), BLOCK_N=64.
# BLOCK_M=128: 4 waves (256 threads), BLOCK_M=256: 8 waves (512 threads).
# ...
# For H>=32, both M=128 and M=256 variants are built and dispatched at runtime.
# Requires: head_dim % 32 == 0, head_dim >= 64, seq_len % 128 == 0.
```

The high-level generic builder constructs dualwave first when possible:

`/sgl-workspace/FlyDSL-lab/kernels/flash_attn_generic.py:114-158`

```python
# DUALWAVE_SWP fast path (gfx950 D=128 bf16/f16)
# Runtime dispatch additionally requires seq_len >= 384 and seq_len % 256 == 0.
if block_m is None and head_dim == 128 and dtype_str in ("bf16", "f16") and gpu_arch.startswith("gfx950"):
    _dualwave_swp_launch = build_flash_attn_dualwave_swp_module(...)

...
if S_int >= 384 and S_int % 256 == 0:
    return _dualwave_swp_launch(*args, **kwargs)
return _fallback(*args, **kwargs)
```

Then generic auto builds M128 and M256 when `num_heads >= 32`:

`/sgl-workspace/FlyDSL-lab/kernels/flash_attn_generic.py:165-216`

```python
if block_m is None and num_heads >= 32:
    _launcher_m128 = build_flash_attn_func_module_primary(... block_m=128 ...)
    _launcher_m256 = build_flash_attn_func_module_primary(... block_m=256 ...)
    _BS_THRESHOLD = 4096 * num_heads

    def _auto_launch(*args, **kwargs):
        B = args[4] if len(args) > 4 else kwargs.get("batch_size", 1)
        S = args[5] if len(args) > 5 else kwargs.get("seq_len", 128)
        bs = (B if isinstance(B, int) else 1) * (S if isinstance(S, int) else 128)
        if bs * num_heads >= _BS_THRESHOLD:
            return _launcher_m256(*args, **kwargs)
        return _launcher_m128(*args, **kwargs)

    return _wrap_with_dualwave_swp(_auto_launch)
```

Because `_BS_THRESHOLD = 4096 * num_heads`, the generic M128/M256 switch is equivalent to `B*S >= 4096`. The `num_heads` factor cancels out.

### gfx950 Dualwave Kernel

Code pointer:

`/sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:4-10`

```python
"""Dual-wave, software-pipelined flash-attention kernel for gfx950 (D=128, bf16/fp16).

Dispatched only when gpu_arch >= gfx950, head_dim == 128, dtype in (bf16, fp16),
and (at runtime) seq_len % 256 == 0 and seq_len >= 384.
"""
```

Builder contract:

`/sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:76-97`

```python
if not gpu_arch.startswith("gfx950"):
    raise RuntimeError(...)
if head_dim != 128:
    raise RuntimeError(...)
if dtype_str not in ("bf16", "f16"):
    raise RuntimeError(...)
```

Tile and LDS constants:

`/sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:103-150`

```python
BLOCK_M = 256
BLOCK_N = 64
NUM_WAVES = 8
BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 512
ROWS_PER_WAVE = 32
...
LDS_KV_TOTAL_SIZE = NUM_PREFETCH_K * DUALWAVE_SWP_KV_PER_BUFFER  # 68096 B
```

Tunable optimization switches:

`/sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:183-190`

```python
DUALWAVE_SWP_RESCALE_THRESHOLD = 8.0
DUALWAVE_SWP_LAZY_RESCALE = bool(dualwave_swp_lazy_rescale)
DUALWAVE_SWP_SETPRIO = bool(dualwave_swp_setprio)
DUALWAVE_SWP_ENABLE_STAGGER = bool(dualwave_swp_enable_stagger)
```

## CI And Benchmark Coverage

The primary GitHub workflow runs tests and benchmarks on single-GPU runners including `linux-flydsl-mi355-1`:

`/sgl-workspace/FlyDSL-lab/.github/workflows/flydsl.yaml:44-63`

The CI benchmark step runs:

`/sgl-workspace/FlyDSL-lab/.github/workflows/flydsl.yaml:195-205`

```bash
BENCH_LOG_DIR=/tmp/flydsl_bench_current bash scripts/run_benchmark.sh 2>&1 | tee /tmp/bench_current.out
python3 scripts/benchmark_output_to_csv.py /tmp/bench_current.out /tmp/bench_current.csv
```

It then builds baselines from recent `main` and latest tag, and compares:

`/sgl-workspace/FlyDSL-lab/.github/workflows/flydsl.yaml:303-330`

```bash
python3 scripts/compare_benchmark.py /tmp/bench_main.csv /tmp/bench_current.csv
python3 scripts/compare_benchmark.py /tmp/bench_latest_tag.csv /tmp/bench_current.csv
```

The repo benchmark script has an environment-overridable flash attention shape list:

`/sgl-workspace/FlyDSL-lab/scripts/run_benchmark.sh:72-81`

```bash
DEFAULT_FLASH_ATTN_FUNC_SHAPES='
32,8192,8,8,128,bf16,true
16,8192,16,16,128,bf16,true
4,8192,64,64,128,bf16,true
4,8192,64,8,128,bf16,true
'
FLASH_ATTN_FUNC_SHAPES="${FLASH_ATTN_FUNC_SHAPES:-${DEFAULT_FLASH_ATTN_FUNC_SHAPES}}"
```

And invokes only the exported auto builder:

`/sgl-workspace/FlyDSL-lab/scripts/run_benchmark.sh:603-652`

```bash
python3 tests/kernels/test_flash_attn_fwd.py \
  --batch "$batch" \
  --seq_len "$seq_len" \
  --num_heads "$heads" \
  --num_kv_heads "$kv_heads" \
  --head_dim "$head_dim" \
  --dtype "$dtype" \
  "${causal_flag}" \
  --warmup 10 \
  --iters 100
```

The test file itself compares FlyDSL against PyTorch SDPA and can compare with AITER CK/ASM:

`/sgl-workspace/FlyDSL-lab/tests/kernels/test_flash_attn_fwd.py:35-50`

```python
from kernels.flash_attn_generic import build_flash_attn_func_module

FLASH_ATTN_FUNC_KERNEL_CONFIG = {
    "waves_per_eu": int(os.getenv("FLYDSL_WAVES_PER_EU", "2")),
    "dualwave_swp_lazy_rescale": os.getenv("FLYDSL_DUALWAVE_SWP_LAZY_RESCALE", "1") == "1",
    "dualwave_swp_setprio": os.getenv("FLYDSL_DUALWAVE_SWP_SETPRIO", "1") == "1",
    "dualwave_swp_enable_stagger": os.getenv("FLYDSL_DUALWAVE_SWP_STAGGER", "1") == "1",
}
```

Current CI is useful for regression gating, but it does not separate `generic_m128`, `generic_m256`, and `dualwave`. That separation is necessary to tune dispatch thresholds.

## ATOM Coverage

FlyDSL has an ATOM nightly workflow, but it is model-serving accuracy oriented rather than a flash-attn microbenchmark:

`/sgl-workspace/FlyDSL-lab/.github/workflows/flydsl-atom-integration.yaml:23-55`

The model matrix currently includes:

- `DeepSeek-R1-0528-MXFP4`, TP8, `linux-flydsl-mi355-8`
- `Kimi-K2.5-MXFP4`, TP4, `linux-flydsl-mi355-8`
- `gpt-oss-120b`, single MI355 runner

It runs ATOM launch and accuracy:

`/sgl-workspace/FlyDSL-lab/.github/workflows/flydsl-atom-integration.yaml:329-350`

```bash
bash .github/scripts/atom_test.sh launch "$model_path" ...
bash .github/scripts/atom_test.sh accuracy "$model_path"
```

In the profiling repo, the ATOM workload importer intentionally treats ISL/OSL/concurrency as serving anchors, not kernel shapes:

`/sgl-workspace/flydsl-kernel-profiling/benchmarks/shape_ledgers/atom_workload_importer.py:1-23`

```python
# ISL / OSL / concurrency are serving-workload metadata, not kernel shapes.
# ...
# ONLY when a concrete model config is supplied (--model-config a HF config.json)
# AND a kernel is named (--op rmsnorm) does it derive real kernel shapes.
```

Currently it can derive only `rmsnorm`:

`/sgl-workspace/flydsl-kernel-profiling/benchmarks/shape_ledgers/atom_workload_importer.py:61-69`

```python
_DERIVERS = {
    "rmsnorm": {
        "prefill": lambda isl, osl, c: isl * c,
        "decode": lambda isl, osl, c: c,
    },
}
```

For this report, model-like flash-attn shapes were therefore added manually: Llama-style MHA/GQA and Qwen-style GQA shapes at prefill-like sequence lengths. A proper next step is to add a flash-attn deriver that reads HF config keys such as `num_attention_heads`, `num_key_value_heads`, and `hidden_size/head_dim`.

## Full Sweep Results

Coverage:

| metric | value |
|---|---:|
| result rows | 198 |
| ok rows | 171 |
| unsupported rows | 21 |
| failed rows | 6 |

Provider status:

| provider | ok | unsupported | failed |
|---|---:|---:|---:|
| `generic_m128` | 31 | 0 | 2 |
| `generic_m256` | 29 | 4 | 0 |
| `dualwave` | 23 | 10 | 0 |
| `auto` | 29 | 4 | 0 |
| `aiter_ck` | 33 | 0 | 0 |
| `aiter_asm` | 26 | 3 | 4 |

Provider geomean ratios:

| ratio | geomean | paired shapes | meaning |
|---|---:|---:|---|
| `auto / aiter_ck` | 0.676x | 29 | auto slower overall |
| `auto / aiter_asm` | 0.944x | 26 | near parity overall |
| `dualwave / generic_m256` | 1.377x | 23 | dualwave faster where supported |
| `dualwave / aiter_ck` | 0.901x | 23 | dualwave still behind CK overall due short shapes |
| `generic_m256 / aiter_ck` | 0.527x | 29 | generic M256 well behind CK |
| `generic_m128 / aiter_ck` | 0.560x | 31 | generic M128 well behind CK |

Auto vs CK by stage:

| stage | auto/CK geomean | shapes |
|---|---:|---:|
| boundary | 0.345x | 8 |
| CI/default | 0.760x | 10 |
| dtype | 0.525x | 3 |
| model-like | 1.308x | 6 |
| tile shape | 1.115x | 2 |

Auto vs CK by sequence-length bucket:

| S bucket | auto/CK geomean | shapes |
|---|---:|---:|
| `S <= 512` | 0.210x | 8 |
| `1024 <= S <= 2048` | 0.701x | 7 |
| `S >= 4096` | 1.294x | 14 |

Representative rows:

| shape | B | S | H/Hkv | D | dtype | best | best TFLOPS | auto TFLOPS | CK TFLOPS | auto/CK |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| `boundary_B1_S512_H32` | 1 | 512 | 32/32 | 128 | bf16 | `aiter_ck` | 118.5 | 19.8 | 118.5 | 0.17x |
| `boundary_B1_S2048_H32` | 1 | 2048 | 32/32 | 128 | bf16 | `aiter_ck` | 488.5 | 315.4 | 488.5 | 0.65x |
| `flydsl_default_B1_S2048_H64_Hkv64` | 1 | 2048 | 64/64 | 128 | bf16 | `auto` | 618.4 | 618.4 | 567.2 | 1.09x |
| `boundary_B1_S4096_H32` | 1 | 4096 | 32/32 | 128 | bf16 | `auto` | 797.8 | 797.8 | 650.2 | 1.23x |
| `llama3_8b_prefill` | 1 | 8192 | 32/8 | 128 | bf16 | `dualwave` | 915.3 | 908.0 | 704.5 | 1.29x |
| `llama3_70b_prefill` | 1 | 8192 | 64/8 | 128 | bf16 | `dualwave` | 929.6 | 925.3 | 688.7 | 1.34x |
| `flydsl_default_B16_S8192_H64_Hkv8` | 16 | 8192 | 64/8 | 128 | bf16 | `dualwave` | 948.6 | 948.1 | 690.6 | 1.37x |
| `D128_nocausal` | 2 | 2048 | 32/32 | 128 | bf16 | `dualwave` | 862.3 | 842.9 | 725.1 | 1.16x |

Worst auto/CK rows:

| shape | auto/CK | reason |
|---|---:|---|
| `boundary_B1_S128_H32` | 0.142x | short generic path, launch/overhead dominated |
| `boundary_B1_S256_H32` | 0.145x | short generic path |
| `boundary_B1_S512_H32` | 0.167x | dualwave dispatch too early |
| `fp16_S512` | 0.171x | dualwave dispatch too early |
| `boundary_B1_S384_H32` | 0.181x | generic fallback because S not multiple of 256 |
| `boundary_B1_S1024_H32` | 0.328x | dualwave still too little work |

Best auto/CK rows:

| shape | auto/CK | note |
|---|---:|---|
| `flydsl_default_B16_S8192_H64_Hkv8` | 1.373x | long GQA, dualwave strong |
| `llama3_70b_batch` | 1.346x | long GQA batch |
| `llama3_70b_prefill` | 1.344x | long GQA prefill |
| `wide_h128` | 1.341x | many Q heads |
| `flydsl_default_B16_S8192_H16_Hkv16` | 1.328x | long MHA |
| `flydsl_default_B4_S8192_H64_Hkv64` | 1.318x | long MHA |

## Dispatch Threshold Finding

Current dualwave runtime gate:

```python
if S_int >= 384 and S_int % 256 == 0:
    return _dualwave_swp_launch(*args, **kwargs)
```

This is too permissive. It routes `B=1,S=512,H=32,D=128` to dualwave even though:

| provider | us | TFLOPS |
|---|---:|---:|
| `generic_m128` | 81.1 | 26.5 |
| `generic_m256` | 80.8 | 26.6 |
| `dualwave` | 108.3 | 19.8 |
| `auto` | 108.6 | 19.8 |
| `aiter_ck` | 18.1 | 118.5 |

The same pattern persists at `B=1,S=1024,H=32`; dualwave becomes competitive around higher total CTA counts:

| shape | dualwave CTAs estimate `B*H*ceil(S/256)` | dualwave/generic_m256 | auto/CK |
|---|---:|---:|---:|
| `B1_S512_H32` | 64 | 0.75x | 0.17x |
| `B1_S1024_H32` | 128 | 0.75x | 0.33x |
| `B1_S2048_H32` | 256 | 0.99x | 0.65x |
| `B1_S2048_H64` | 512 | 1.49x | 1.09x |
| `B1_S4096_H32` | 512 | 1.64x | 1.23x |

Recommended first dispatch experiment:

```python
num_q_tiles = (S_int + 255) // 256
dualwave_ctas = B_int * num_heads * num_q_tiles
if S_int >= 384 and S_int % 256 == 0 and dualwave_ctas >= 512:
    return _dualwave_swp_launch(*args, **kwargs)
```

This is not a final proof, but it matches the observed MI350X knee much better than `S` alone.

## Config Sweep Results

Two shapes were used:

- `config_diag_2048`: `B=1,S=2048,H=32,Hkv=32,D=128,bf16,causal`
- `config_gqa_8192`: `B=4,S=8192,H=64,Hkv=8,D=128,bf16,causal`

Diagnostic shape:

| provider | us | TFLOPS | note |
|---|---:|---:|---|
| `aiter_ck` | 80.9 | 424.9 | best overall |
| `generic_m128_n128` | 94.3 | 364.4 | best FlyDSL direct path |
| `generic_m128_n32` | 102.2 | 336.1 | |
| `generic_m128` | 103.3 | 332.6 | |
| `generic_m256` | 111.4 | 308.4 | |
| `auto` | 112.0 | 306.6 | routes dualwave, loses here |
| `dualwave` | 116.8 | 294.1 | |
| `aiter_asm` | 98.2 | 349.9 | |

Large GQA shape:

| provider | us | TFLOPS | note |
|---|---:|---:|---|
| `dualwave_no_setprio` | 4456.0 | 987.0 | best, but only 0.6% faster than default |
| `dualwave` | 4481.5 | 981.4 | default fast path |
| `auto` | 4483.6 | 980.9 | expected, routes dualwave |
| `dualwave_no_stagger` | 5107.3 | 861.1 | about 12% slower |
| `dualwave_no_lazy` | 5132.0 | 857.0 | about 13% slower |
| `generic_m128` | 7847.9 | 560.4 | |
| `generic_m256` | 8038.7 | 547.1 | |
| `aiter_ck` | 6332.0 | 694.6 | |
| `aiter_asm` | 9162.7 | 480.0 | |

Config conclusions:

- Keep lazy rescale enabled.
- Keep stagger enabled.
- `setprio` is close to neutral on the tested large shape. It was slightly slower than `no_setprio`, but the delta is small enough that more shapes are needed before disabling it.
- For the small/mid diagnostic shape, `generic_m128_n128` is clearly better than current auto.

## Robustness Findings

### D64 Generic M256 Fault

The first full run reached:

`D64_causal B=2 S=2048 H=32/32 D=64 bf16 causal provider=generic_m256`

and then the runtime reported:

```text
Memory access fault by GPU node-3 ... Reason: Unknown.
```

That log is preserved in `full_run.log`. The second run skipped `generic_m256` and auto rows for `D<128` after this fault to allow the rest of the matrix to finish.

The source-level file header says generic requires only `head_dim >= 64` and divisible by 32, so either:

- `D=64` is intended to work and `generic_m256` has a real bug, or
- `generic_m256` actually has a narrower D contract and the builder/dispatcher should reject D<128.

### D96 Generic M128 Assertion

Rows:

- `D96_causal`, `generic_m128`: `AssertionError`
- `D96_nocausal`, `generic_m128`: `AssertionError`

Again, the public file-level contract currently appears broader than actual behavior.

### AITER ASM Coverage

AITER ASM/v3 failed for D64/D96 and was skipped for fp16:

- D64/D96: `invalid argument for fmha_fwd`
- fp16: harness marks ASM v3 as bf16-only

This does not affect FlyDSL correctness directly, but it means CK is the more consistent external baseline for these shapes.

## Recommendations

1. Tighten the dualwave auto dispatch gate.

   Add a work/CTA threshold in addition to `S>=384 && S%256==0`. First candidate:

   ```python
   dualwave_ctas = B * num_heads * ceil(S / 256)
   use_dualwave = S >= 384 and S % 256 == 0 and dualwave_ctas >= 512
   ```

2. Fix or explicitly disable `generic_m256` for `D<128`.

   Current auto can route D64/D96 large shapes to M256. The observed D64 GPU fault makes this unsafe.

3. Clarify D96 support.

   If D96 should work under the documented `head_dim % 32 == 0` contract, fix the assertion. If not, reject it early with a clear message.

4. Add split-provider benchmark coverage to CI or nightly.

   Existing CI benchmarks only the exported auto builder. Add a small matrix that forces M128/M256/dualwave for:

   - `B1,S512,H32,D128`
   - `B1,S2048,H32,D128`
   - `B1,S2048,H64,D128`
   - `B1,S4096,H32,D128`
   - one large GQA shape

   This will catch threshold regressions.

5. Add ATOM flash-attn shape derivation.

   Extend `atom_workload_importer.py` with a `flash_attn` deriver that reads:

   - `num_attention_heads`
   - `num_key_value_heads`
   - `hidden_size`
   - optional `head_dim`

   Then map ATOM anchors into prefill attention shapes. Decode attention is a different kernel regime and should be handled separately.

6. Update the profiling framework's historical flash-attn provider.

   The current FlyDSL code split old `flash_attn_func` into `flash_attn_generic` plus `flash_attn_gfx950`. The new harness in this directory handles that split; the older profiling provider should be either replaced or explicitly marked historical.

## Bottom Line

FlyDSL dualwave is already strong on MI350X for the long prefill/GQA regime we care about most: it beats AITER CK by roughly `1.26x-1.37x` on the tested model-like long shapes and reaches roughly `900-980 TFLOPS` in the best rows.

The main performance issue is not the long-shape kernel body; it is dispatch. Current auto sends too-small shapes into dualwave, where it loses badly. The main correctness/robustness issue is the D64/D96 generic path contract, especially `generic_m256` with `D<128`.
