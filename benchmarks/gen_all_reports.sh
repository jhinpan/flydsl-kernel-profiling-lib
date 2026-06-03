#!/bin/bash
# Regenerate coverage_matrix.md + benchmark_summary.md (+ weighted geomean) for
# every kernel that has a benchmark_results.jsonl. CPU-only.
source /sgl-workspace/flydsl-kernel-profiling/benchmarks/env.sh
cd /sgl-workspace/flydsl-kernel-profiling
ALL=(rmsnorm layernorm softmax gemm fused_rope_cache moe_gemm \
     vec_add quant topk_gating_softmax moe_reduce preshuffle_gemm \
     blockscale_preshuffle_gemm fp8_gemm_rowscale moe_blockscale \
     flash_attn pa_decode mla_decode)
for k in "${ALL[@]}"; do
  d="benchmarks/examples/$k"
  [ -f "$d/benchmark_results.jsonl" ] || { echo "SKIP $k (no results)"; continue; }
  python -m benchmarks.reports.render_markdown_report \
    --shape-ledger "$d/shape_ledger.jsonl" \
    --results "$d/benchmark_results.jsonl" \
    --out "$d" --kernel "$k" > /dev/null 2>&1 \
    && echo "report OK: $k" || echo "report FAIL: $k"
done
