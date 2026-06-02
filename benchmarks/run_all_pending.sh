#!/bin/bash
# Run all 11 campaign-expansion kernels across the 8-GPU pool, each kernel
# exclusive on its GPU (clean cold-cache timing). Waves of 8.
source /sgl-workspace/flydsl-kernel-profiling/benchmarks/env.sh
cd /sgl-workspace/flydsl-kernel-profiling
KERNELS=(vec_add quant topk_gating_softmax moe_reduce preshuffle_gemm \
         blockscale_preshuffle_gemm fp8_gemm_rowscale moe_blockscale \
         flash_attn pa_decode mla_decode)
mkdir -p traces/runlogs
: > traces/runlogs/_status.log
i=0
for k in "${KERNELS[@]}"; do
  gpu=$((i % 8))
  (
    export HIP_VISIBLE_DEVICES=$gpu
    timeout 1500 python -m benchmarks.runners.multishape_runner \
      --op "$k" \
      --shape-ledger "benchmarks/examples/$k/shape_ledger.jsonl" \
      --baseline-matrix "benchmarks/examples/$k/baseline_matrix.yaml" \
      --out "benchmarks/examples/$k" > "traces/runlogs/$k.log" 2>&1
    echo "DONE $k gpu=$gpu rc=$?" >> traces/runlogs/_status.log
  ) &
  i=$((i + 1))
  if (( i % 8 == 0 )); then wait; fi
done
wait
echo "ALL_DONE" >> traces/runlogs/_status.log
