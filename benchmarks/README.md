# `benchmarks/` — FlyDSL multi-shape kernel benchmark

A multi-shape benchmark **layer** that lives beside the diagnostic rocprofv3/ATT
trace bundles in `examples/<kernel>/`. The ATT bundles answer "where does this
one kernel stall on one diagnostic shape." This layer answers a different
question: **across the shapes that actually occur in models and serving, is
FlyDSL's kernel faster than the field — and if not, why?**

The two layers share the `examples/<op>/` tree but never collide. ATT work owns
`REPORT.md`, the per-example `README.md`, `att_viewer/`, `compute_viewer/`,
`source/`. This layer owns `shape_ledger.jsonl`, `baseline_matrix.yaml`,
`benchmark_results.{jsonl,csv}`, `coverage_matrix.md`, `benchmark_summary.md`.
New files here **never clobber** the ATT bundle's `REPORT.md` or `README.md`.

## Directory tree

```
benchmarks/
├── README.md                  ← this file
├── env.sh                     ← source before any GPU runner (build-tree PYTHONPATH/LD)
├── bench                      ← wrapper: sources env.sh then `exec python "$@"`
├── common.py                  ← spine: stable_shape_id, read/write_jsonl, provenance,
│                                timing (eager + cudagraph), measure_both, speedup, geomean
├── ops.py                     ← per-op inputs + fp32 reference + roofline (one Op per op_type)
├── validate.py                ← validate rows against schemas/ (jsonschema or fallback)
├── schemas/
│   ├── shape_ledger.schema.json
│   ├── benchmark_result.schema.json
│   └── baseline_matrix.schema.yaml
├── shape_ledgers/             ← importers that build shape_ledger.jsonl (CPU-only)
│   ├── README.md              ← shape sources + per-importer CLI + upsert semantics
│   ├── ledger_io.py           ← idempotent upsert_ledger (dedup by shape_id, replace by kind)
│   ├── aiter_model_shapes_importer.py
│   └── manual_shape_importer.py   (synthetic-boundary + diagnostic + manual)
├── providers/                 ← one adapter per (provider, op_type)
│   ├── base.py                ← ProviderAdapter contract + load_entrypoint
│   ├── flydsl.py              ← candidate: build_rmsnorm_module
│   ├── pytorch.py             ← F.rms_norm (also the dtype-matched reference provider)
│   ├── aiter.py               ← compiled module_rmsnorm (CK/HIP/ASM, records >8192 branch)
│   ├── aiter_triton.py        ← aiter.ops.triton.normalization.rmsnorm
│   ├── triton.py              ← sglang standalone one-pass kernel
│   └── aiter_ck / aiter_asm / ck / gluon / hipblaslt   ← honest support-detection stubs
├── runners/
│   └── multishape_runner.py   ← orchestrator: inputs once/shape, ref, every provider
├── reports/                   ← CPU-only, from ledger + results
│   ├── analysis.py            ← join + per-shape speedups + best-baseline + aggregates
│   ├── summarize_results.py   ← benchmark_summary.md (headline + splits + decision)
│   ├── coverage_matrix.py     ← coverage_matrix.md (per-shape × per-provider status)
│   ├── render_markdown_report.py  ← render both at once
│   ├── weighted_summary.py    ← weighted vs unweighted aggregate
│   └── classify_bottleneck.py ← rule-based gap classification
└── examples/<op>/             ← artifacts land here (rmsnorm is fully wired)
```

## `env.sh` — the recipe and WHY

```bash
# (env.sh, verbatim intent)
export FLYDSL_LAB=/sgl-workspace/FlyDSL-lab
export PYTHONPATH="$FLYDSL_LAB/build-fly/python_packages:$FLYDSL_LAB:<repo>:$PYTHONPATH"
export LD_LIBRARY_PATH="$FLYDSL_LAB/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH"
export SGLANG_USE_AITER=0
```

**Always launch GPU runners via `benchmarks/env.sh` (or the `benchmarks/bench`
wrapper).** Importers and report generators are pure-data and run on a CPU-only
box without it; anything that imports `flydsl`, `aiter`, `triton`, or `torch`
needs it.

Why it is load-bearing on this node:

- `flydsl` is pip-installed editable, but the editable tree has **no compiled
  `_mlir` extension**. The built `.so` lives in
  `FLYDSL-lab/build-fly/python_packages`. Putting that tree first on
  `PYTHONPATH` makes `import flydsl.*` resolve to the built copy.
- That same path **also unblocks `import aiter`** — `aiter/__init__` imports
  `flydsl.expr` transitively, so without the build tree `import aiter` raises
  `ModuleNotFoundError: flydsl._mlir`. This is the non-obvious part: the FlyDSL
  build tree is what makes the AITER baselines importable at all.
- The `_mlir` `.so` needs its `_mlir_libs` dir on `LD_LIBRARY_PATH`, and the
  dynamic loader reads `LD_LIBRARY_PATH` **at process exec** — so it must be set
  before python starts. `env.sh` does this; `common.bootstrap_env()` only covers
  the `PYTHONPATH` half (so imports resolve), while `common.flydsl_runtime_ok()`
  reports whether the native half actually loaded.
- `SGLANG_USE_AITER=0` lets the standalone sglang Triton kernel import without
  forcing the aiter path.

Verified node recipe: GPU **MI350X gfx950**, **ROCm 7.2**, **torch
2.9.1+rocm**, **triton 3.6**.

## Methodology — kernel-only vs eager

The PRIMARY metric is **kernel-only CUDA-graph time** (`common.benchmark_cudagraph`).
It pays host launch overhead + JIT/autotune + allocation ONCE at capture, then
replays N unrolled launches and divides — leaving pure device time. This is the
fair metric across providers, especially on short shapes where Python launch
overhead would otherwise dominate.

Reported separately:

- **eager event time** (`common.benchmark`, L2-flush + loop amortization), and
- **`host_overhead_us = eager_median - graph_median`** — surfaced as a
  first-class signal. FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every
  call, so it has high per-call host overhead; on short/decode shapes that can
  be tens of µs. This is a **launcher (host-side)** problem, distinct from kernel
  speed, and is reported as a separate eager verdict — it is mitigated when
  serving captures decode in a CUDA/hipgraph (as SGLang does).

`common.measure_both` returns both; the result row carries `median_us` (primary),
`eager_median_us`, `graph_median_us`, `host_overhead_us`, and `timing_method`.

**Speedups.** `speedup = baseline_median / flydsl_median` (>1 => FlyDSL faster).
A shape's headline is its kernel-only speedup vs the **best available** correct
baseline (`speedup_vs_best`). Aggregates: **unweighted geomean** over measured
shapes, **weighted geomean** using `baseline_time_weight` (preferred) or
`traffic_weight` from the ledger, plus **per-baseline**, **per-stage**, and
**per-model** splits. Weighted numbers print `n/a` until a serving trace
populates weights — synthetic/diagnostic shapes alone never carry production
weight.

A measurement is flagged unstable (`stable=False`) when p90/p10 > 1.2; unstable
hot shapes are re-measured, not trusted.

## rmsnorm worked example (end to end)

```bash
# 0. (one-time) inspect the wired matrix
cat benchmarks/examples/rmsnorm/baseline_matrix.yaml

# 1. build the ledger (CPU-only; idempotent upsert by source.kind)
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --dtype bf16 --ops rmsnorm
python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm \
  --out benchmarks/examples --synthetic-boundary --diagnostic 32768,8192,bf16

# 2. run on the GPU — via the bench wrapper so the build-tree PYTHONPATH/LD
#    (which also unblocks `import aiter`) is set BEFORE python starts
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op rmsnorm \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
  --out benchmarks/examples/rmsnorm --warmup-iters 25 --repeat-iters 100
#   -> benchmark_results.jsonl + benchmark_results.csv

# 3. reports (CPU-only)
python -m benchmarks.reports.render_markdown_report \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl \
  --out benchmarks/examples/rmsnorm --kernel rmsnorm
#   -> coverage_matrix.md + benchmark_summary.md
python -m benchmarks.reports.weighted_summary \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl
```

The runner streams `[i/n] <shape_id> M=..,N=.. <dtype> (<stage>)` per shape and
keeps every provider in the output with an explicit `benchmark_status`
(`ok | failed | oom | unsupported | incorrect | not_configured`) — nothing is
silently dropped. Correctness is recorded inline on each row (`correct` +
`correctness_error`); there is no separate correctness runner.

## Where artifacts land

Everything for one kernel lands in `examples/<op>/`:

| File | Producer |
|---|---|
| `shape_ledger.jsonl` | importers (`shape_ledgers/*`) |
| `baseline_matrix.yaml` | authored once per kernel |
| `benchmark_results.jsonl` / `.csv` | `runners/multishape_runner.py` |
| `coverage_matrix.md` | `reports/coverage_matrix.py` |
| `benchmark_summary.md` | `reports/summarize_results.py` |
| `profiles/` (planned) | profiler gate (`runners/profiler_runner.py`) |

These sit next to — and never overwrite — the ATT bundle's `REPORT.md`,
`README.md`, `att_viewer/`, `compute_viewer/`, `source/`.

## What is wired vs planned

- **Wired:** `rmsnorm` end-to-end (5 enabled providers + 5 honest stubs), the
  AITER model-shapes importer, the manual/synthetic/diagnostic importer, the
  multishape runner, and all report generators.
- **Planned (schema slots reserved, reports reference them):** the profiler gate
  runner (`runners/profiler_runner.py`, fires when kernel-only
  `speedup_vs_best < 0.90`); serving-trace importers
  (`sglang_trace_importer.py`, `atom_workload_importer.py`) that populate
  `weight.*` and `source.kind in {sglang_trace, atom_workload}`; a regression
  diff helper over a pinned ledger.

## See also

- `benchmarks/shape_ledgers/README.md` — shape sources + importer CLIs + upsert.
- `.claude/skills/flydsl-kernel-multishape-benchmark/SKILL.md` — the agent contract.
- Repo top-level `AGENTS.md` — the ATT/rocprofv3 diagnostic layer this sits beside.
