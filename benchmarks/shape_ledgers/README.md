# `shape_ledgers/` — shape sources + importers

Importers build the per-op `examples/<op>/shape_ledger.jsonl` that the
benchmark runner consumes. Every importer is **pure-data** (no torch/GPU) and
goes through one idempotent merge (`ledger_io.upsert_ledger`), so the ledger is
the union of several independent sources that never clobber each other.

Each row is validated against `benchmarks/schemas/shape_ledger.schema.json` and
keyed by a stable `shape_id` (`common.stable_shape_id` → `sha1:<16 hex>` over the
identity fields `op_type, model, stage, dtype, layout, args`). Same shape →
same id across runs and importers.

## Shape sources and which `source.kind` each owns

The schema's `source.kind` enum is the contract. Each importer owns one or more
kinds; `upsert_ledger` replaces only the kinds an importer declares.

| Source | `source.kind` | Owner | What it is |
|---|---|---|---|
| AITER model-config sweep | `aiter_model_shapes` | `aiter_model_shapes_importer.py` | model→kernel shapes from AITER's `model_shapes.json`, crossed with a token sweep; `stage=model_config` (attention rows get prefill/decode/mixed from a `comment`) |
| Synthetic boundary probe | `synthetic` | `manual_shape_importer.py` | canned shapes around the FlyDSL fast-path cliff (N≥2048 & N%2048==0 & 16-bit); `stage=synthetic` |
| Diagnostic ATT shape | `diagnostic` | `manual_shape_importer.py` | the exact shape the existing rocprofv3/ATT capture used; `stage=diagnostic` |
| Manual list | `manual` (or per-entry `source_kind`) | `manual_shape_importer.py` | an explicit JSON/YAML list of shapes |
| Serving trace (SGLang) | `sglang_trace` | `sglang_trace_importer.py` *(planned)* | real prefill/decode shapes + traffic/time weights from a live SGLang run |
| Atom workload | `atom_workload` | `atom_workload_importer.py` *(planned)* | canonical serving workload mix → weighted shapes |

`sglang_trace` / `atom_workload` are reserved in the schema and referenced by
the weighted report; the importers that emit them are the next to add. They are
the only sources that populate `weight.{occurrences, traffic_weight,
baseline_time_weight}`, which is what makes the production-weighted geomean
meaningful — until then the weighted aggregate prints `n/a`, by design.

## Importer CLIs (exactly as implemented)

### `aiter_model_shapes_importer.py` — `source.kind=aiter_model_shapes`

`model_shapes.json` is `model → kernel_key → [shape_dict]` and carries only the
kernel-class fields (e.g. `N` for rmsnorm; `N,K,TP_dim` for gemm). `M`/batch/seq
are NOT in the file — `bench_models.py` synthesizes `M = batch * seq` at run
time. This importer mirrors that: it crosses the model-fixed fields with a
configurable token sweep, applies the same TP sharding as `bench_models.py`
(`shard = max(v // tp, 1)`), and labels the rows `stage=model_config` (attention
rows take prefill/decode/mixed from an optional `comment`).

```bash
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples \
  --tp 8 --gpu MI350X --arch gfx950 --dtype bf16 \
  --m-values 1,32,256,2048,16384 \
  --attn-batch 1,16,128 --attn-seq 1024,8192 \
  --ops rmsnorm            # comma list of op_types/kernels, or 'all'
```

Writes `<out>/<op_type>/shape_ledger.jsonl` for each op. TP rules (verified
against `bench_models.py`): rmsnorm never sharded; gemm/batched_gemm shard the
single dim named by `TP_dim`; moe always shards `Dim2`; rope shards
heads, `seq_len = M`; attention shards `hq+hkv`.

### `manual_shape_importer.py` — `source.kind ∈ {synthetic, diagnostic, manual}`

Three independent sources, each upserted without touching the others:

```bash
# synthetic boundary probe + the existing ATT diagnostic shape (rmsnorm)
python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm \
  --out benchmarks/examples \
  --synthetic-boundary \
  --diagnostic 32768,8192,bf16        # "M,N,dtype"

# an explicit list (JSON or YAML; per-entry source_kind allowed)
python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm \
  --out benchmarks/examples --manual-file my_shapes.yaml
```

- `--synthetic-boundary` probes the fast-path cliff: alignment edges
  (`N ∈ {2047,2048,2049,4095,4096,4097,8191,8192,8193}`), non-power-of-2
  (`3000,5333,12288`), tiny single-token decode, very large prefill, and
  f16/f32 dtype variants — at `M ∈ {1, 4096}`. (rmsnorm only today.)
- `--diagnostic "M,N,dtype"` adds the one shape the rocprofv3/ATT capture used.
- `--manual-file` takes a list of `{op_type?, model?, stage?, dtype, args,
  source_kind?, notes?}`.

## Dedup / upsert semantics (`ledger_io.upsert_ledger`)

Re-running any importer is safe and never clobbers another source:

1. Read the existing ledger (if any).
2. **Drop** existing rows whose `source.kind` is in the importer's
   `replace_kinds` set. Rows from every OTHER kind are kept verbatim.
3. Add the new rows; dedup by `shape_id` (last writer wins).
4. Validate every row against the schema; raise on any error.
5. Rewrite the file, stably sorted by `(op_type, stage, dtype, N, M, shape_id)`
   so diffs are clean.

So `aiter_model_shapes_importer` (`replace_kinds={aiter_model_shapes}`) can be
re-run without disturbing the `synthetic` / `diagnostic` / `manual` rows the
manual importer added, and vice-versa. The importer's own dedup
(`aiter_model_shapes_importer.dedup`) additionally merges rows that share
`(op_type, stage, dtype, args)` across models, joining the model names with `|`
(e.g. `Llama3 8B|Qwen3-235B-A22B`) so the same physical shape is one row.

## See also

- `benchmarks/README.md` — directory tree, env recipe, methodology, worked example.
- `benchmarks/schemas/shape_ledger.schema.json` — the row contract.
- `.claude/skills/flydsl-kernel-multishape-benchmark/SKILL.md` — the agent contract.
