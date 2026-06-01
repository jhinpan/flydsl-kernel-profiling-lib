window.KERNEL_DATA = {
  "provenance": {
    "flydsl_version": "0.1.9.dev594",
    "flydsl_commit": "18c5a7ed",
    "flydsl_branch": "docs/update-compile-pipeline",
    "gpu": "AMD Instinct MI350X (gfx950, CDNA4)",
    "gpu_count": 8,
    "rocm": "7.2.0",
    "rocprofv3": "1.1.0",
    "capture_date": "2026-06-01"
  },
  "generated": "2026-06-01",
  "summary": {
    "total": 18,
    "with_att": 17,
    "with_baseline": 16,
    "wins": 4,
    "headroom": 9,
    "categories": {
      "attention": 3,
      "elementwise": 1,
      "gemm": 6,
      "moe": 2,
      "norm": 2,
      "quant": 1,
      "reduction": 2,
      "rope": 1
    }
  },
  "headroom_ranking": [
    "test_fused_rope_cache",
    "test_topk_gating_softmax",
    "test_pa",
    "test_blockscale_preshuffle_gemm",
    "test_preshuffle_gemm",
    "test_moe_blockscale",
    "test_rmsnorm",
    "test_mla_decode",
    "test_flash_attn_func"
  ],
  "win_ranking": [
    "test_softmax",
    "test_hgemm_splitk",
    "bench_preshuffle_gemm_v2",
    "test_moe_gemm"
  ],
  "kernels": [
    {
      "name": "Flash Attention",
      "test": "test_flash_attn_func.py",
      "stem": "test_flash_attn_func",
      "example": "flash_attn_func",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/flash_attn_func/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/flash_attn_func",
      "op_category": "attention",
      "kernels": [
        "flash_attn_func_kernel"
      ],
      "jit_kernel": "flash_attn_func_kernel_0",
      "flydsl_us": 9.9,
      "baseline_us": 9.1,
      "baseline_name": "aiter.mha_fwd (CK FMHA), softmax_scale=1/sqrt(128), causal=True",
      "speedup_vs_baseline": 0.919,
      "verdict": "FlyDSL slower",
      "tflops": 3.1,
      "baseline_tflops": 3.7,
      "tbps": null,
      "bandwidth_gbs": null,
      "extra_baselines": {
        "ck_S128_us": 9.1,
        "asm_fmha_v3_S128_us": 9.3,
        "flydsl_S128_us": 9.9,
        "ck_S2048_us": 49.7,
        "asm_fmha_v3_S2048_us": 56.9,
        "flydsl_S2048_us": 69.3
      },
      "baseline_source": "harness_flag",
      "baseline_notes": "Used the harness's BUILT-IN --compare flag (fastest path): runs FlyDSL vs CK (aiter.mha_fwd) vs ASM (aiter.fmha_v3_fwd) in one process on GPU 3, MI350X gfx950, ROCm 7.2. All three match PyTorch SDPA ref to MaxErr 3.91e-03 (bf16).\n\nMATCHING SHAPE B=1 S=128 H=8 D=128 (sweep_master records flydsl_us=11.0, tflops=3.1): FlyDSL is latency-bound at ~9.9-10.5us (bounces run-to-run), CK steady at 9.1us, ASM 9.3us. CK is the strongest comparable. FlyDSL does NOT win here: it sits at 86-92% of CK TFLOPS, i.e. CK is ~1.0-1.15x faster. At this tiny shape everyone clusters near the ~9us launch-overhead floor so the gap is small.\n\nSTEADY-STATE B=1 S=2048 H=8 D=128 (fairer compute compare): FlyDSL 69.3us/124 TFLOPS, CK 49.7us/173 TFLOPS, ASM 56.9us/151 TFLOPS. CK is 1.39x faster than FlyDSL (FlyDSL=71.7% of CK TFLOPS); ASM 1.22x faster. The gap WIDENS with sequence length, indicating FlyDSL FMHA is throughput-limited vs CK rather than merely overhead-limited.\n\nspeedup_flydsl_vs_baseline=0.919 reported at the matching S=128 shape (baseline_us 9.1 / flydsl_us 9.9). At S=2048 it is 49.7/69.3=0.717. In both regimes the external CK baseline beats FlyDSL, so FlyDSL does not truly win on this kernel on MI350X.\n\nArtifacts: /sgl-workspace/flydsl-prof/results/baselines/test_flash_attn_func/BASELINE_NOTES.md and fmha_perf_compare_MI350X_S2048.csv (CSV holds the last --compare run only). Harness: /sgl-workspace/FlyDSL-lab/tests/kernels/test_flash_attn_func.py (run_aiter_bench with backend ck/asm).",
      "calls": 108,
      "waves": 4,
      "ins": 2023,
      "mapped_pct": 100.0,
      "occupancy": "1",
      "arch_vgpr": "243",
      "stall_pct_total": 54.9,
      "top_stall_type": "VMEM-store",
      "stall_breakdown": [
        {
          "type": "VMEM-store",
          "stall": "11.0K",
          "pct": 29.0
        },
        {
          "type": "other",
          "stall": "8.2K",
          "pct": 21.6
        },
        {
          "type": "VMEM-wait",
          "stall": "8.1K",
          "pct": 21.4
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "5.9K",
          "pct": 15.7
        },
        {
          "type": "MFMA/FMA",
          "stall": "1.8K",
          "pct": 4.8
        },
        {
          "type": "barrier",
          "stall": "1.8K",
          "pct": 4.7
        },
        {
          "type": "VMEM-load",
          "stall": "716",
          "pct": 1.9
        },
        {
          "type": "LDS",
          "stall": "336",
          "pct": 0.9
        }
      ],
      "top_source_lines": [
        {
          "stall": "30.2K",
          "pct_total": 79.92,
          "domtype": "VMEM-store",
          "source": "kspace/FlyDSL-lab/kernels/flash_attn_func.py:257"
        },
        {
          "stall": "5.3K",
          "pct_total": 13.95,
          "domtype": "LDS/SMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/flash_attn_func.py:283"
        },
        {
          "stall": "2.3K",
          "pct_total": 6.08,
          "domtype": "LDS/SMEM-wait",
          "source": "d-fly/python_packages/flydsl/expr/numeric.py:872"
        },
        {
          "stall": "16",
          "pct_total": 0.04,
          "domtype": "LDS/SMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/flash_attn_func.py:514"
        }
      ],
      "inst_mix": {
        "mfma": 64,
        "buffer_load": 20,
        "buffer_store": 0,
        "ds_read": 96,
        "ds_write": 0
      },
      "next_occ_step": {
        "waves": 2,
        "vgpr_budget": 128
      },
      "headline": "stall-bound (55% of cycles), occ 1/SIMD @243 VGPR; FlyDSL 9.9\u00b5s vs CK 9.1\u00b5s (0.92x, loses) and 0.72x at S=2048 \u2014 the scalar uncoalesced O-store epilogue (29% VMEM-store stalls, 80% of attributed cycles) is the ceiling.",
      "top_recommendation": "Coalesce + widen the O writeback epilogue (lines 1132-1142): replace 16 scalar global_store_dwordx2/D-chunk with contiguous dwordx4 vectorized stores and a single trailing vmcnt(0).",
      "bound_type": "stall-bound",
      "status": "ok",
      "ck_candidate": "aiter.mha_fwd with softmax_scale=1/sqrt(head_dim), causal=True",
      "aiter_comparable": "aiter.mha_fwd (CK backend) or aiter.fmha_v3_fwd (ASM backend, bf16 only)",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "MLA Decode (fp8)",
      "test": "test_mla_decode.py",
      "stem": "test_mla_decode",
      "example": "mla_decode",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/mla_decode/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/mla_decode",
      "op_category": "attention",
      "kernels": [
        "kn_mla_fwd_decode_m16x8_fp8_fp8"
      ],
      "jit_kernel": "kn_mla_fwd_decode_m16x8_fp8_fp8_0",
      "flydsl_us": 12.4,
      "baseline_us": 11.19,
      "baseline_name": "aiter.hk_mla_decode_fwd (HipKittens CK MLA decode stage1)",
      "speedup_vs_baseline": 0.902,
      "verdict": "FlyDSL slower",
      "tflops": 1.46,
      "baseline_tflops": 1.59,
      "tbps": 0.02,
      "bandwidth_gbs": null,
      "extra_baselines": {
        "aiter.mla_decode_stage1_asm_fwd (mla_a8w8_qh128_m32x4_n16x2 ASM)": 12.03
      },
      "baseline_source": "harness_flag",
      "baseline_notes": "Used the harness's built-in --bench_aiter flag, which benchmarks FlyDSL and TWO aiter MLA-decode impls in one run at the identical shape, so it is a true apples-to-apples comparison. All three time ONLY the decode stage1 kernel (mla_reduce_v1 is excluded from every timed region) and all pass the same correctness check (cos_diff=1.44e-04, err_ratio=0.18%). Single clean run on GPU 4 (MI350X, gfx950).\n\nMeasured this run (us_p50 / TFLOPS):\n- FlyDSL kn_mla_fwd_decode_m16x8_fp8_fp8: 12.40 us / 1.44  (range 12.39-12.41; sweep_master.json recorded 12.23 us / 1.46 -- consistent within noise)\n- aiter.hk_mla_decode_fwd (HipKittens CK): 11.19 us / 1.59  (range 11.15-11.57) <- STRONGEST baseline\n- aiter.mla_decode_stage1_asm_fwd (ASM kernel mla_a8w8_qh128_m32x4_n16x2_msk0_ps.co): 12.03 us / 1.48  (range 12.02-12.05)\n\nVerdict: at this tiny single-tile shape (b=1, ctx_len=64) FlyDSL is NOT winning. The best external baseline (aiter HipKittens CK) is 11.19 us vs FlyDSL 12.40 us => speedup_flydsl_vs_baseline = 11.19/12.40 = 0.90 (FlyDSL ~10% SLOWER). FlyDSL is roughly tied with the aiter ASM path (12.40 vs 12.03, ~3% slower). This is a latency-bound, memory-bound decode at very small ctx_len so all three are clustered near 11-12 us and far below peak (1.4-1.6 TFLOPS); differences are launch/scheduling-dominated, not compute. Note: hk_mla required a one-time 71s JIT build of module_hk_mla on first call (already cached after this run). speedup recomputed against the strongest baseline (hk) = 0.902; vs the asm baseline it would be 12.03/12.40 = 0.97.",
      "calls": 363,
      "waves": 32,
      "ins": 4949,
      "mapped_pct": 100.0,
      "occupancy": "1",
      "arch_vgpr": "251",
      "stall_pct_total": 94.6,
      "top_stall_type": "LDS/SMEM-wait",
      "stall_breakdown": [
        {
          "type": "LDS/SMEM-wait",
          "stall": "36.1K",
          "pct": 83.0
        },
        {
          "type": "VMEM-wait",
          "stall": "5.9K",
          "pct": 13.5
        },
        {
          "type": "other",
          "stall": "1.0K",
          "pct": 2.4
        },
        {
          "type": "VMEM-load",
          "stall": "496",
          "pct": 1.1
        }
      ],
      "top_source_lines": [
        {
          "stall": "36.1K",
          "pct_total": 83.03,
          "domtype": "LDS/SMEM-wait",
          "source": "ab/./kernels/mla_fwd_decode_m16x8_fp8_fp8.py:396"
        },
        {
          "stall": "6.9K",
          "pct_total": 15.83,
          "domtype": "VMEM-wait",
          "source": "ab/./kernels/mla_fwd_decode_m16x8_fp8_fp8.py:309"
        },
        {
          "stall": "496",
          "pct_total": 1.14,
          "domtype": "VMEM-load",
          "source": "ab/./kernels/mla_fwd_decode_m16x8_fp8_fp8.py:409"
        }
      ],
      "inst_mix": {
        "mfma": 688,
        "buffer_load": 130,
        "buffer_store": 64,
        "ds_read": 759,
        "ds_write": 127
      },
      "next_occ_step": {
        "waves": 2,
        "vgpr_budget": 128
      },
      "headline": "stall-bound (83% LDS/SMEM-wait, lgkmcnt(0)), occ 1/SIMD @ VGPR\u2248251; FlyDSL 12.40\u00b5s vs aiter-hk-CK 11.19\u00b5s (0.90\u00d7) \u2014 exposed LDS\u2192MFMA operand-feed latency on a single-wave decode is the ceiling.",
      "top_recommendation": "Software-pipeline the LDS\u2192MFMA chain in _process_kv_tile (prefetch next K/V sub-block, partial lgkmcnt, overlap ds_read with MFMA issue) to attack the 83% LDS-wait \u2014 technique-mfma-pipelining + technique-lds-double-buffering.",
      "bound_type": "stall-bound",
      "status": "ok",
      "ck_candidate": "hipBLASLt-based paged attention if available; else DeepSeek-V3 reference attention",
      "aiter_comparable": "aiter.hk_mla_decode_fwd + aiter.mla_decode_stage1_asm_fwd",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "Paged-Attn Decode (PS)",
      "test": "test_pa.py",
      "stem": "test_pa",
      "example": "pa",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/pa/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/pa",
      "op_category": "attention",
      "kernels": [
        "pa_decode_ps_kernel",
        "compile_pa_decode_sw",
        "compile_pa_decode_sw_reduce"
      ],
      "jit_kernel": "pa_decode_ps_kernel_0",
      "flydsl_us": 169.5,
      "baseline_us": 80.6,
      "baseline_name": "AIter Gluon (pa_decode_gluon)",
      "speedup_vs_baseline": 0.476,
      "verdict": "FlyDSL slower",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": null,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": 13,
      "waves": 56,
      "ins": 937,
      "mapped_pct": 99.9,
      "occupancy": "1",
      "arch_vgpr": "175",
      "stall_pct_total": 65.7,
      "top_stall_type": "LDS/SMEM-wait",
      "stall_breakdown": [
        {
          "type": "LDS/SMEM-wait",
          "stall": "25.3K",
          "pct": 36.2
        },
        {
          "type": "VMEM-wait",
          "stall": "16.3K",
          "pct": 23.3
        },
        {
          "type": "other",
          "stall": "15.8K",
          "pct": 22.5
        },
        {
          "type": "VMEM-load",
          "stall": "9.2K",
          "pct": 13.1
        },
        {
          "type": "barrier",
          "stall": "2.5K",
          "pct": 3.6
        },
        {
          "type": "LDS",
          "stall": "364",
          "pct": 0.5
        },
        {
          "type": "MFMA/FMA",
          "stall": "296",
          "pct": 0.4
        },
        {
          "type": "VMEM-store",
          "stall": "264",
          "pct": 0.4
        }
      ],
      "top_source_lines": [
        {
          "stall": "39.1K",
          "pct_total": 55.83,
          "domtype": "VMEM-wait",
          "source": "rkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:1170"
        },
        {
          "stall": "21.3K",
          "pct_total": 30.48,
          "domtype": "LDS/SMEM-wait",
          "source": "rkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:1215"
        },
        {
          "stall": "2.3K",
          "pct_total": 3.23,
          "domtype": "LDS/SMEM-wait",
          "source": "rkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:1209"
        },
        {
          "stall": "1.4K",
          "pct_total": 1.99,
          "domtype": "VMEM-load",
          "source": "rkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:1577"
        },
        {
          "stall": "1.3K",
          "pct_total": 1.87,
          "domtype": "other",
          "source": "rkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:1220"
        },
        {
          "stall": "1.0K",
          "pct_total": 1.47,
          "domtype": "VMEM-load",
          "source": "orkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:537"
        },
        {
          "stall": "844",
          "pct_total": 1.21,
          "domtype": "VMEM-wait",
          "source": "orkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:895"
        },
        {
          "stall": "544",
          "pct_total": 0.78,
          "domtype": "other",
          "source": "orkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:151"
        },
        {
          "stall": "432",
          "pct_total": 0.62,
          "domtype": "VMEM-load",
          "source": "rkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:1574"
        },
        {
          "stall": "252",
          "pct_total": 0.36,
          "domtype": "VMEM-load",
          "source": "orkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:531"
        },
        {
          "stall": "248",
          "pct_total": 0.35,
          "domtype": "VMEM-store",
          "source": "rkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:1624"
        },
        {
          "stall": "200",
          "pct_total": 0.29,
          "domtype": "other",
          "source": "orkspace/FlyDSL-lab/kernels/pa_decode_fp8.py:808"
        }
      ],
      "inst_mix": {
        "mfma": 32,
        "buffer_load": 14,
        "buffer_store": 3,
        "ds_read": 18,
        "ds_write": 8
      },
      "next_occ_step": {
        "waves": 2,
        "vgpr_budget": 128
      },
      "headline": "stall-bound (65.7% stall, 36% LDS-wait + 36% VMEM), occ 1/SIMD (VGPR 176, needs \u2264128 for 2 waves); FlyDSL 169.5\u00b5s vs AIter Gluon 80.6\u00b5s (0.476\u00d7) \u2014 single resident wave can't hide its own K/V-load + softmax-LDS latency.",
      "top_recommendation": "Cut VGPR live-range to \u2264128 (shrink the per-block loop-carried state, esp. the prefetched k_flat held in VGPRs) to reach occupancy 2/SIMD so a sibling wave hides the vmcnt/lgkmcnt waits.",
      "bound_type": "wait-bound",
      "status": "ok",
      "ck_candidate": "No CK baseline available; use AIter Gluon pa_decode_gluon as reference. For paged-attention performance comparison, AIter uses Triton Gluon compiled at runtime.",
      "aiter_comparable": "torch.ops.aiter.pa_decode_gluon (Gluon reference paged-attention kernel; invoked via run_gluon_ps() at test_pa.py:802\u2013823)",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "Vector Add",
      "test": "test_vec_add.py",
      "stem": "test_vec_add",
      "example": "vec_add",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/vec_add/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/vec_add",
      "op_category": "elementwise",
      "kernels": [
        "vecAddKernel"
      ],
      "jit_kernel": "vecAddKernel_0",
      "flydsl_us": null,
      "baseline_us": null,
      "baseline_name": null,
      "speedup_vs_baseline": null,
      "verdict": null,
      "tflops": null,
      "baseline_tflops": null,
      "tbps": 6710.8,
      "bandwidth_gbs": 6468.37,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": 24,
      "waves": 632,
      "ins": 23,
      "mapped_pct": 95.7,
      "occupancy": "8",
      "arch_vgpr": "9",
      "stall_pct_total": 56.3,
      "top_stall_type": "VMEM-wait",
      "stall_breakdown": [
        {
          "type": "VMEM-wait",
          "stall": "1.73M",
          "pct": 82.3
        },
        {
          "type": "VMEM-load",
          "stall": "178.1K",
          "pct": 8.5
        },
        {
          "type": "VMEM-store",
          "stall": "87.1K",
          "pct": 4.1
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "69.7K",
          "pct": 3.3
        },
        {
          "type": "SMEM",
          "stall": "29.5K",
          "pct": 1.4
        },
        {
          "type": "other",
          "stall": "8.5K",
          "pct": 0.4
        }
      ],
      "top_source_lines": [
        {
          "stall": "1.76M",
          "pct_total": 83.67,
          "domtype": "VMEM-wait",
          "source": "pace/FlyDSL-lab/tests/kernels/test_vec_add.py:28"
        },
        {
          "stall": "155.6K",
          "pct_total": 7.41,
          "domtype": "VMEM-load",
          "source": "pace/FlyDSL-lab/tests/kernels/test_vec_add.py:64"
        },
        {
          "stall": "87.1K",
          "pct_total": 4.14,
          "domtype": "VMEM-store",
          "source": "pace/FlyDSL-lab/tests/kernels/test_vec_add.py:70"
        },
        {
          "stall": "75.4K",
          "pct_total": 3.59,
          "domtype": "LDS/SMEM-wait",
          "source": "thon_packages/flydsl/expr/rocdl/universal.py:144"
        },
        {
          "stall": "25.1K",
          "pct_total": 1.19,
          "domtype": "VMEM-load",
          "source": "pace/FlyDSL-lab/tests/kernels/test_vec_add.py:65"
        }
      ],
      "inst_mix": {
        "mfma": 0,
        "buffer_load": 2,
        "buffer_store": 1,
        "ds_read": 0,
        "ds_write": 0
      },
      "next_occ_step": null,
      "headline": "bandwidth-bound (82% VMEM-wait), occ 8/SIMD (max), 9 VGPR; 6468 GB/s \u2248 81% of MI355X HBM3E ~8 TB/s \u2014 already-optimal 128-bit streaming triad, body has no headroom (no baseline recorded).",
      "top_recommendation": "Convert to a persistent grid-stride kernel (technique-persistent-kernel) to absorb the 10000-block tail and amortize per-block V# descriptor setup; expect only a few % toward roofline since the op is fundamentally HBM-bound.",
      "bound_type": "bandwidth-bound",
      "status": "ok",
      "ck_candidate": "Not applicable (vector add is elementwise; rocBLAS/CK target GEMM/reduction)",
      "aiter_comparable": "torch.add (PyTorch baseline already in test)",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "Preshuffle GEMM v2",
      "test": "bench_preshuffle_gemm_v2.py",
      "stem": "bench_preshuffle_gemm_v2",
      "example": "preshuffle_gemm_v2",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/preshuffle_gemm_v2/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/preshuffle_gemm_v2",
      "op_category": "gemm",
      "kernels": [
        "kernel_gemm"
      ],
      "jit_kernel": "kernel_gemm_0",
      "flydsl_us": 447.9,
      "baseline_us": 538.2,
      "baseline_name": "FlyDSL v1 (manual, pre-layout-API)",
      "speedup_vs_baseline": 1.202,
      "verdict": "FlyDSL faster",
      "tflops": 767.2,
      "baseline_tflops": 638.4,
      "tbps": null,
      "bandwidth_gbs": null,
      "extra_baselines": {},
      "baseline_source": "harness_flag",
      "baseline_notes": "INTERNAL comparison (v2 layout-API vs v1 manual), not vs an external lib. At 4096x5120x8192 bf16 the v2 refactor is 1.20x faster (767 vs 638 TFLOPS) \u2014 a real FlyDSL-internal win.",
      "calls": 54,
      "waves": 12,
      "ins": 429,
      "mapped_pct": 99.8,
      "occupancy": "2",
      "arch_vgpr": "97",
      "stall_pct_total": 81.1,
      "top_stall_type": "LDS/SMEM-wait",
      "stall_breakdown": [
        {
          "type": "LDS/SMEM-wait",
          "stall": "567.6K",
          "pct": 40.6
        },
        {
          "type": "VMEM-wait",
          "stall": "461.5K",
          "pct": 33.0
        },
        {
          "type": "barrier",
          "stall": "211.5K",
          "pct": 15.1
        },
        {
          "type": "MFMA/FMA",
          "stall": "87.1K",
          "pct": 6.2
        },
        {
          "type": "LDS",
          "stall": "35.0K",
          "pct": 2.5
        },
        {
          "type": "VMEM-load",
          "stall": "22.4K",
          "pct": 1.6
        },
        {
          "type": "other",
          "stall": "12.2K",
          "pct": 0.9
        },
        {
          "type": "VMEM-store",
          "stall": "220",
          "pct": 0.0
        }
      ],
      "top_source_lines": [
        {
          "stall": "901.7K",
          "pct_total": 64.52,
          "domtype": "LDS/SMEM-wait",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:309"
        },
        {
          "stall": "285.5K",
          "pct_total": 20.43,
          "domtype": "barrier",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:100"
        },
        {
          "stall": "131.4K",
          "pct_total": 9.4,
          "domtype": "VMEM-wait",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:311"
        },
        {
          "stall": "30.5K",
          "pct_total": 2.18,
          "domtype": "LDS",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:305"
        },
        {
          "stall": "20.1K",
          "pct_total": 1.44,
          "domtype": "VMEM-load",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:302"
        },
        {
          "stall": "11.1K",
          "pct_total": 0.8,
          "domtype": "VMEM-load",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:298"
        },
        {
          "stall": "9.4K",
          "pct_total": 0.67,
          "domtype": "LDS/SMEM-wait",
          "source": "thon_packages/flydsl/expr/rocdl/universal.py:144"
        },
        {
          "stall": "5.5K",
          "pct_total": 0.4,
          "domtype": "VMEM-wait",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:320"
        },
        {
          "stall": "1.1K",
          "pct_total": 0.08,
          "domtype": "other",
          "source": "ld-fly/python_packages/flydsl/expr/derived.py:45"
        },
        {
          "stall": "596",
          "pct_total": 0.04,
          "domtype": "VMEM-load",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:318"
        },
        {
          "stall": "368",
          "pct_total": 0.03,
          "domtype": "VMEM-store",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:414"
        },
        {
          "stall": "140",
          "pct_total": 0.01,
          "domtype": "VMEM-load",
          "source": "ace/FlyDSL-lab/kernels/preshuffle_gemm_v2.py:317"
        }
      ],
      "inst_mix": {
        "mfma": 64,
        "buffer_load": 24,
        "buffer_store": 32,
        "ds_read": 32,
        "ds_write": 10
      },
      "next_occ_step": {
        "waves": 3,
        "vgpr_budget": 85
      },
      "headline": "internal v2-vs-v1 @4096\u00d75120\u00d78192 bf16: v2 layout-API 767 TFLOPS (448\u00b5s) = 1.20\u00d7 over v1 manual (638 TFLOPS, 538\u00b5s) \u2014 the layout-API refactor is a real FlyDSL-internal win (no external baseline).",
      "top_recommendation": "Ship the v2 layout-API path as default; it is 1.20\u00d7 over the manual v1 at compute-bound shape \u2014 then chase the external CK gap separately (see preshuffle_gemm).",
      "bound_type": "stall-bound",
      "status": "ok",
      "ck_candidate": "rocBLAS or hipBLASLt fp16/bf16 128x5120x8192 tiled GEMM",
      "aiter_comparable": "false",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "Block-Scale Preshuffle GEMM",
      "test": "test_blockscale_preshuffle_gemm.py",
      "stem": "test_blockscale_preshuffle_gemm",
      "example": "blockscale_preshuffle_gemm",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/blockscale_preshuffle_gemm/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/blockscale_preshuffle_gemm",
      "op_category": "gemm",
      "kernels": [
        "bs_gemm_bf16_direct_t64x128x128",
        "rocdl.mfma_scale_f32_16x16x128_f8f6f4"
      ],
      "jit_kernel": "bs_gemm_bf16_direct_t16x64x256",
      "flydsl_us": 155.7,
      "baseline_us": 102.3,
      "baseline_name": "AIter CK a8w8_blockscale_bpreshuffle (TUNED config)",
      "speedup_vs_baseline": 0.66,
      "verdict": "FlyDSL slower",
      "tflops": 869.07,
      "baseline_tflops": 1321.97,
      "tbps": 1.389,
      "bandwidth_gbs": null,
      "extra_baselines": {
        "aiter_ck_bpreshuffle_run1_us": 9.7,
        "aiter_ck_bpreshuffle_run2_us": 10.1
      },
      "baseline_source": "harness_flag",
      "baseline_notes": "COMPUTE-BOUND re-test. Gap WIDENS vs M=16 (0.88x->0.66x) because AIter loaded a TUNED blockscale config (dsv3/qwen tables) while FlyDSL runs an untuned fixed schedule. FlyDSL 869 TFLOPS vs tuned-CK 1322 TFLOPS. Needs split-K + per-shape tile tuning to close.",
      "calls": 25,
      "waves": 4,
      "ins": 391,
      "mapped_pct": 99.7,
      "occupancy": "6",
      "arch_vgpr": "73",
      "stall_pct_total": 91.8,
      "top_stall_type": "VMEM-wait",
      "stall_breakdown": [
        {
          "type": "VMEM-wait",
          "stall": "63.4K",
          "pct": 76.6
        },
        {
          "type": "barrier",
          "stall": "6.0K",
          "pct": 7.3
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "6.0K",
          "pct": 7.2
        },
        {
          "type": "VMEM-load",
          "stall": "4.7K",
          "pct": 5.6
        },
        {
          "type": "other",
          "stall": "2.4K",
          "pct": 2.9
        },
        {
          "type": "LDS",
          "stall": "192",
          "pct": 0.2
        },
        {
          "type": "MFMA/FMA",
          "stall": "84",
          "pct": 0.1
        },
        {
          "type": "VMEM-store",
          "stall": "28",
          "pct": 0.0
        }
      ],
      "top_source_lines": [
        {
          "stall": "44.2K",
          "pct_total": 53.49,
          "domtype": "VMEM-wait",
          "source": "SL-lab/kernels/blockscale_preshuffle_gemm.py:138"
        },
        {
          "stall": "30.4K",
          "pct_total": 36.72,
          "domtype": "VMEM-wait",
          "source": "yDSL-lab/kernels/mfma_preshuffle_pipeline.py:585"
        },
        {
          "stall": "2.9K",
          "pct_total": 3.46,
          "domtype": "LDS/SMEM-wait",
          "source": "SL-lab/kernels/blockscale_preshuffle_gemm.py:541"
        },
        {
          "stall": "2.6K",
          "pct_total": 3.12,
          "domtype": "VMEM-load",
          "source": "SL-lab/kernels/blockscale_preshuffle_gemm.py:481"
        },
        {
          "stall": "1.8K",
          "pct_total": 2.2,
          "domtype": "VMEM-load",
          "source": "lyDSL-lab/kernels/mfma_preshuffle_pipeline.py:83"
        },
        {
          "stall": "272",
          "pct_total": 0.33,
          "domtype": "VMEM-load",
          "source": "SL-lab/kernels/blockscale_preshuffle_gemm.py:489"
        },
        {
          "stall": "192",
          "pct_total": 0.23,
          "domtype": "LDS",
          "source": "SL-lab/kernels/blockscale_preshuffle_gemm.py:288"
        },
        {
          "stall": "176",
          "pct_total": 0.21,
          "domtype": "other",
          "source": "lyDSL-lab/kernels/mfma_preshuffle_pipeline.py:24"
        },
        {
          "stall": "132",
          "pct_total": 0.16,
          "domtype": "other",
          "source": "SL-lab/kernels/blockscale_preshuffle_gemm.py:211"
        },
        {
          "stall": "56",
          "pct_total": 0.07,
          "domtype": "other",
          "source": "SL-lab/kernels/blockscale_preshuffle_gemm.py:654"
        },
        {
          "stall": "8",
          "pct_total": 0.01,
          "domtype": "other",
          "source": "yDSL-lab/kernels/mfma_preshuffle_pipeline.py:530"
        }
      ],
      "inst_mix": {
        "mfma": 18,
        "buffer_load": 72,
        "buffer_store": 4,
        "ds_read": 36,
        "ds_write": 9
      },
      "next_occ_step": null,
      "headline": "compute-bound re-test @M=4096: FlyDSL 869 TFLOPS (156\u00b5s) vs AIter TUNED-CK 1322 TFLOPS (102\u00b5s) \u2192 0.66\u00d7 \u2014 gap WIDENS vs M=16 because CK loaded a per-shape tuned config and FlyDSL runs a fixed untuned schedule.",
      "top_recommendation": "Add per-shape tile/split-K tuning (FlyDSL has no autotune table here); the 0.66\u00d7 is partly a tuning-parity gap, not a codegen gap \u2014 match CK's tuned tile + GlobalSplitU.",
      "bound_type": "latency-bound",
      "status": "ok",
      "ck_candidate": "CK gemm A8W8 blockscale_bpreshuffle (weight layout 16x16, scale_block 128x128)",
      "aiter_comparable": "aiter.gemm_a8w8_blockscale_bpreshuffle",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "FP8 Row-Scale GEMM",
      "test": "test_fp8_gemm_rowscale.py",
      "stem": "test_fp8_gemm_rowscale",
      "example": "fp8_gemm_rowscale",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/fp8_gemm_rowscale/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/fp8_gemm_rowscale",
      "op_category": "gemm",
      "kernels": [
        "kernel_gemm (fp8_gemm_4wave)",
        "kernel_gemm (fp8_gemm_8wave)"
      ],
      "jit_kernel": null,
      "flydsl_us": null,
      "baseline_us": null,
      "baseline_name": null,
      "speedup_vs_baseline": null,
      "verdict": null,
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": null,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": null,
      "waves": null,
      "ins": null,
      "mapped_pct": null,
      "occupancy": null,
      "arch_vgpr": null,
      "stall_pct_total": null,
      "top_stall_type": null,
      "stall_breakdown": [],
      "top_source_lines": [],
      "inst_mix": {},
      "next_occ_step": null,
      "headline": null,
      "top_recommendation": null,
      "bound_type": null,
      "status": "compile_fail",
      "ck_candidate": "rocBLAS fp8_mm or CK scaled_gemm for FP8 with row-wise scales",
      "aiter_comparable": "torch._scaled_mm (with --vs_torch flag)",
      "capture_error": null,
      "has_bundle": false
    },
    {
      "name": "HGEMM Split-K",
      "test": "test_hgemm_splitk.py",
      "stem": "test_hgemm_splitk",
      "example": "hgemm_splitk",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/hgemm_splitk/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/hgemm_splitk",
      "op_category": "gemm",
      "kernels": [
        "hgemm_bf16_32x64x256_W1x2x2_S2_BT_BLDS1_AS1_SPK14"
      ],
      "jit_kernel": "hgemm_bf16_32x64x256_W1x2x2_S2_BT_BLDS1_AS1_SPK14_0",
      "flydsl_us": 7.0,
      "baseline_us": 11.6,
      "baseline_name": "PyTorch",
      "speedup_vs_baseline": 1.657,
      "verdict": "FlyDSL faster",
      "tflops": 25.09,
      "baseline_tflops": null,
      "tbps": 0.853,
      "bandwidth_gbs": null,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": 55,
      "waves": 8,
      "ins": 522,
      "mapped_pct": 99.8,
      "occupancy": "2",
      "arch_vgpr": "117",
      "stall_pct_total": 71.3,
      "top_stall_type": "VMEM-wait",
      "stall_breakdown": [
        {
          "type": "VMEM-wait",
          "stall": "32.5K",
          "pct": 55.8
        },
        {
          "type": "barrier",
          "stall": "8.3K",
          "pct": 14.2
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "7.7K",
          "pct": 13.3
        },
        {
          "type": "VMEM-load",
          "stall": "4.3K",
          "pct": 7.4
        },
        {
          "type": "other",
          "stall": "3.3K",
          "pct": 5.7
        },
        {
          "type": "MFMA/FMA",
          "stall": "1.2K",
          "pct": 2.0
        },
        {
          "type": "LDS",
          "stall": "868",
          "pct": 1.5
        }
      ],
      "top_source_lines": [
        {
          "stall": "48.6K",
          "pct_total": 83.46,
          "domtype": "VMEM-wait",
          "source": "workspace/FlyDSL-lab/kernels/hgemm_splitk.py:217"
        },
        {
          "stall": "6.5K",
          "pct_total": 11.2,
          "domtype": "LDS/SMEM-wait",
          "source": "d-fly/python_packages/flydsl/expr/numeric.py:872"
        },
        {
          "stall": "1.4K",
          "pct_total": 2.47,
          "domtype": "MFMA/FMA",
          "source": "-workspace/FlyDSL-lab/kernels/hgemm_splitk.py:80"
        },
        {
          "stall": "1.3K",
          "pct_total": 2.18,
          "domtype": "LDS",
          "source": "-workspace/FlyDSL-lab/kernels/tensor_shim.py:242"
        },
        {
          "stall": "180",
          "pct_total": 0.31,
          "domtype": "other",
          "source": "-workspace/FlyDSL-lab/kernels/tensor_shim.py:255"
        },
        {
          "stall": "84",
          "pct_total": 0.14,
          "domtype": "other",
          "source": "workspace/FlyDSL-lab/kernels/hgemm_splitk.py:487"
        },
        {
          "stall": "48",
          "pct_total": 0.08,
          "domtype": "other",
          "source": "workspace/FlyDSL-lab/kernels/hgemm_splitk.py:456"
        },
        {
          "stall": "32",
          "pct_total": 0.05,
          "domtype": "LDS/SMEM-wait",
          "source": "-workspace/FlyDSL-lab/kernels/tensor_shim.py:225"
        },
        {
          "stall": "32",
          "pct_total": 0.05,
          "domtype": "LDS/SMEM-wait",
          "source": "workspace/FlyDSL-lab/kernels/hgemm_splitk.py:449"
        },
        {
          "stall": "20",
          "pct_total": 0.03,
          "domtype": "other",
          "source": "workspace/FlyDSL-lab/kernels/hgemm_splitk.py:448"
        },
        {
          "stall": "8",
          "pct_total": 0.01,
          "domtype": "other",
          "source": "workspace/FlyDSL-lab/kernels/hgemm_splitk.py:738"
        },
        {
          "stall": "4",
          "pct_total": 0.01,
          "domtype": "other",
          "source": "workspace/FlyDSL-lab/kernels/hgemm_splitk.py:495"
        }
      ],
      "inst_mix": {
        "mfma": 32,
        "buffer_load": 24,
        "buffer_store": 2,
        "ds_read": 34,
        "ds_write": 16
      },
      "next_occ_step": {
        "waves": 3,
        "vgpr_budget": 85
      },
      "headline": "latency-bound (56% VMEM-wait, MFMA only 2%), occ 2/SIMD @ 117 VGPR; FlyDSL 7.0\u00b5s = 25 TFLOPS, 1.66x vs PyTorch 11.6\u00b5s \u2014 but only 84 workgroups on 256 CUs, so SPLIT_K=14 starves each block to ~2 K-iters and the 2-stage pipeline can't hide load latency.",
      "top_recommendation": "Reduce SPLIT_K (sweep down from 14 to ~4-7) so each block runs enough K-loop iterations to overlap global loads with MFMA, directly attacking the 63% memory-latency stall.",
      "bound_type": "latency-bound",
      "status": "ok",
      "ck_candidate": "rocBLAS hemm or hgemm (MI350X tuned); CK GEMM bf16 for comparison",
      "aiter_comparable": "torch.mm(a_bf16, b_bf16.T) via torch.cuda.Event timing",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "MoE GEMM (2-stage)",
      "test": "test_moe_gemm.py",
      "stem": "test_moe_gemm",
      "example": "moe_gemm",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/moe_gemm/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/moe_gemm",
      "op_category": "gemm",
      "kernels": [
        "moe_gemm1",
        "moe_gemm2"
      ],
      "jit_kernel": "moe_gemm1_0",
      "flydsl_us": 111.1,
      "baseline_us": 123.4,
      "baseline_name": "AIter CK MoE 2-stage (ck_moe_stage1_fwd + ck_moe_stage2_fwd), per_Token fp8, Silu",
      "speedup_vs_baseline": 1.11,
      "verdict": "FlyDSL faster",
      "tflops": 93.01,
      "baseline_tflops": 90.64,
      "tbps": null,
      "bandwidth_gbs": null,
      "extra_baselines": {
        "ck_stage1_us": 71.1,
        "ck_stage2_atomic_us": 52.3,
        "ck_stage2_reduce_us": 51.1,
        "flydsl_stage1_us": 70.8,
        "flydsl_stage2_atomic_us": 40.3,
        "flydsl_stage2_reduce_us": 47.2
      },
      "baseline_source": "harness_flag",
      "baseline_notes": "Baseline obtained via the harness built-in --compare_aiter_ck flag (FASTEST path). The baseline is aiter's CK MoE 2-stage: harness imports `from aiter.ops.moe_op import ck_moe_stage1_fwd` (line ~817) and the analogous ck_moe_stage2_fwd, with QuantType.per_Token + ActivationType.Silu, run via the same run_perftest harness as FlyDSL so timing/warmup are identical. Same shape as the FlyDSL run, so it is a true apples-to-apples comparison; the harness also asserts FlyDSL-vs-aiter output correctness (rtol/atol=0.25) before printing, so numbers are validated.\n\nPer-stage results (20 iters / 10 warmup, GPU 2, MI350-class gfx950), stable across two runs:\n- Stage1: FlyDSL 70.8 us (91.06 TFLOPS) vs CK 71.1 us (90.64 TFLOPS) -> ~1.00x, parity.\n- Stage2 atomic (FlyDSL's faster/default mode): FlyDSL 40.3 us (79.96 TFLOPS) vs CK 52.3 us (61.55 TFLOPS) -> 1.30x faster.\n- Stage2 reduce: FlyDSL 47.2 us vs CK 51.1 us -> 1.08x.\n\nHeadline (stage1 + best stage2-atomic) used for the top-level numbers: FlyDSL 70.8+40.3=111.1 us vs CK 71.1+52.3=123.4 us => 1.11x faster overall. baseline_tflops/flydsl_tflops report the stage1 figures (representative; per-stage TFLOPS are in the FLOPS prints above).\n\nDISCREPANCY vs sweep_master.json: the sweep recorded flydsl_total=142.9 us (s1=69.3, s2=73.6) and 93.01 TFLOPS with baseline_us=null. The s1 number matches; the s2=73.6 in the sweep is much slower than the current stage2-atomic (40.3 us) \u2014 the sweep evidently timed a slower stage2 variant. Re-measuring here, FlyDSL stage2 is substantially faster, so FlyDSL genuinely wins (stage2 1.30x, total 1.11x) and is at parity on stage1. Verdict: FlyDSL wins overall vs the strongest comparable non-FlyDSL impl (AIter CK).\n\nNo standalone script was needed (harness flag sufficed). FlyDSL harness: /sgl-workspace/FlyDSL-lab/tests/kernels/test_moe_gemm.py.",
      "calls": 18,
      "waves": 24,
      "ins": 997,
      "mapped_pct": 99.9,
      "occupancy": "1",
      "arch_vgpr": "155",
      "stall_pct_total": 91.0,
      "top_stall_type": "VMEM-wait",
      "stall_breakdown": [
        {
          "type": "VMEM-wait",
          "stall": "1.11M",
          "pct": 55.0
        },
        {
          "type": "VMEM-load",
          "stall": "304.7K",
          "pct": 15.0
        },
        {
          "type": "barrier",
          "stall": "263.9K",
          "pct": 13.0
        },
        {
          "type": "MFMA/FMA",
          "stall": "227.5K",
          "pct": 11.2
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "101.5K",
          "pct": 5.0
        },
        {
          "type": "other",
          "stall": "11.1K",
          "pct": 0.5
        },
        {
          "type": "LDS",
          "stall": "2.7K",
          "pct": 0.1
        },
        {
          "type": "VMEM-store",
          "stall": "132",
          "pct": 0.0
        }
      ],
      "top_source_lines": [
        {
          "stall": "639.6K",
          "pct_total": 31.59,
          "domtype": "VMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:880"
        },
        {
          "stall": "570.7K",
          "pct_total": 28.18,
          "domtype": "VMEM-wait",
          "source": "yDSL-lab/kernels/mfma_preshuffle_pipeline.py:585"
        },
        {
          "stall": "306.9K",
          "pct_total": 15.16,
          "domtype": "barrier",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:280"
        },
        {
          "stall": "306.4K",
          "pct_total": 15.13,
          "domtype": "VMEM-load",
          "source": "lyDSL-lab/kernels/mfma_preshuffle_pipeline.py:83"
        },
        {
          "stall": "124.9K",
          "pct_total": 6.17,
          "domtype": "MFMA/FMA",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:881"
        },
        {
          "stall": "34.1K",
          "pct_total": 1.68,
          "domtype": "LDS/SMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:363"
        },
        {
          "stall": "18.0K",
          "pct_total": 0.89,
          "domtype": "VMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:370"
        },
        {
          "stall": "9.1K",
          "pct_total": 0.45,
          "domtype": "LDS/SMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:442"
        },
        {
          "stall": "8.3K",
          "pct_total": 0.41,
          "domtype": "LDS/SMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:405"
        },
        {
          "stall": "2.9K",
          "pct_total": 0.15,
          "domtype": "other",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:785"
        },
        {
          "stall": "456",
          "pct_total": 0.02,
          "domtype": "other",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:616"
        },
        {
          "stall": "400",
          "pct_total": 0.02,
          "domtype": "other",
          "source": "kspace/FlyDSL-lab/kernels/moe_gemm_2stage.py:508"
        }
      ],
      "inst_mix": {
        "mfma": 256,
        "buffer_load": 96,
        "buffer_store": 16,
        "ds_read": 32,
        "ds_write": 8
      },
      "next_occ_step": {
        "waves": 2,
        "vgpr_budget": 128
      },
      "headline": "stall-bound (55% VMEM-wait, 70% VMEM total), occ 1/SIMD @155 VGPR; FlyDSL stage-1 70.8us = CK 71.1us (1.00x parity, 1.11x on full 2-stage) \u2014 load pipeline is unpipelined (vmcnt(1) stores) and matrix cores are starved.",
      "top_recommendation": "Deepen the global->LDS software pipeline (more buffer_loads in flight, relax vmcnt(N) so the LDS store waits only on its own tile) to attack the 55% VMEM-wait bucket \u2014 grounded in technique-mfma-pipelining + technique-lds-double-buffering.",
      "bound_type": "latency-bound",
      "status": "ok",
      "ck_candidate": "ck_moe_stage1_fwd,ck_moe_stage2_fwd",
      "aiter_comparable": "moe_gemm1,moe_gemm2",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "Preshuffle GEMM",
      "test": "test_preshuffle_gemm.py",
      "stem": "test_preshuffle_gemm",
      "example": "preshuffle_gemm",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/preshuffle_gemm/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/preshuffle_gemm",
      "op_category": "gemm",
      "kernels": [
        "preshuffle_gemm.kernel_gemm",
        "preshuffle_gemm_v2.kernel_gemm"
      ],
      "jit_kernel": "kernel_gemm_0",
      "flydsl_us": 102.0,
      "baseline_us": 78.1,
      "baseline_name": "AIter CK a8w8 bpreshuffle (untuned default config)",
      "speedup_vs_baseline": 0.766,
      "verdict": "FlyDSL slower",
      "tflops": 1347.41,
      "baseline_tflops": 1760.2,
      "tbps": 458.15,
      "bandwidth_gbs": null,
      "extra_baselines": {
        "aiter_gemm_a8w8_bpreshuffle_run1_iters20": 4.4,
        "aiter_run2_iters50": 4.3,
        "aiter_run3_iters50": 4.8,
        "aiter_run4_iters50": 4.2
      },
      "baseline_source": "harness_flag",
      "baseline_notes": "COMPUTE-BOUND re-test (the M=16 sweep was launch-bound / within noise). At 4096^3 the gap is REAL: FlyDSL 1347 TFLOPS vs AIter CK 1760 TFLOPS (0.77x). AIter used its UNTUNED default config here (no tuned entry for 4096^3), so this is FlyDSL vs untuned CK. Headroom is in the K-loop load pipeline depth.",
      "calls": 25,
      "waves": 68,
      "ins": 864,
      "mapped_pct": 99.9,
      "occupancy": "3",
      "arch_vgpr": "161",
      "stall_pct_total": 82.7,
      "top_stall_type": "VMEM-wait",
      "stall_breakdown": [
        {
          "type": "VMEM-wait",
          "stall": "1.79M",
          "pct": 33.7
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "1.18M",
          "pct": 22.3
        },
        {
          "type": "MFMA/FMA",
          "stall": "698.1K",
          "pct": 13.1
        },
        {
          "type": "LDS",
          "stall": "626.5K",
          "pct": 11.8
        },
        {
          "type": "VMEM-load",
          "stall": "439.2K",
          "pct": 8.3
        },
        {
          "type": "barrier",
          "stall": "366.0K",
          "pct": 6.9
        },
        {
          "type": "other",
          "stall": "109.3K",
          "pct": 2.1
        },
        {
          "type": "VMEM-store",
          "stall": "85.2K",
          "pct": 1.6
        }
      ],
      "top_source_lines": [
        {
          "stall": "2.76M",
          "pct_total": 52.02,
          "domtype": "VMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/preshuffle_gemm.py:313"
        },
        {
          "stall": "1.89M",
          "pct_total": 35.56,
          "domtype": "LDS/SMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/preshuffle_gemm.py:940"
        },
        {
          "stall": "233.6K",
          "pct_total": 4.4,
          "domtype": "VMEM-load",
          "source": "lyDSL-lab/kernels/mfma_preshuffle_pipeline.py:83"
        },
        {
          "stall": "219.2K",
          "pct_total": 4.13,
          "domtype": "VMEM-load",
          "source": "kspace/FlyDSL-lab/kernels/preshuffle_gemm.py:517"
        },
        {
          "stall": "87.2K",
          "pct_total": 1.64,
          "domtype": "VMEM-store",
          "source": "space/FlyDSL-lab/kernels/preshuffle_gemm.py:1167"
        },
        {
          "stall": "84.4K",
          "pct_total": 1.59,
          "domtype": "LDS/SMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/preshuffle_gemm.py:396"
        },
        {
          "stall": "15.0K",
          "pct_total": 0.28,
          "domtype": "VMEM-load",
          "source": "kspace/FlyDSL-lab/kernels/preshuffle_gemm.py:832"
        },
        {
          "stall": "10.4K",
          "pct_total": 0.2,
          "domtype": "SMEM",
          "source": "d-fly/python_packages/flydsl/expr/numeric.py:872"
        },
        {
          "stall": "3.9K",
          "pct_total": 0.07,
          "domtype": "LDS/SMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/preshuffle_gemm.py:397"
        },
        {
          "stall": "1.7K",
          "pct_total": 0.03,
          "domtype": "other",
          "source": "kspace/FlyDSL-lab/kernels/preshuffle_gemm.py:433"
        },
        {
          "stall": "1.5K",
          "pct_total": 0.03,
          "domtype": "other",
          "source": "yDSL-lab/kernels/mfma_preshuffle_pipeline.py:530"
        },
        {
          "stall": "1.3K",
          "pct_total": 0.03,
          "domtype": "VMEM-load",
          "source": "kspace/FlyDSL-lab/kernels/preshuffle_gemm.py:825"
        }
      ],
      "inst_mix": {
        "mfma": 64,
        "buffer_load": 42,
        "buffer_store": 64,
        "ds_read": 64,
        "ds_write": 32
      },
      "next_occ_step": null,
      "headline": "compute-bound re-test @4096\u00b3 fp8: FlyDSL 1347 TFLOPS (102\u00b5s) vs AIter-CK 1760 TFLOPS (78\u00b5s) \u2192 0.77\u00d7 \u2014 the M=16 sweep was launch-bound/noise; at saturation the gap is real. CK's bpreshuffle K-loop out-overlaps FlyDSL's.",
      "top_recommendation": "Deepen the K-loop software-pipeline (more outstanding vmcnt prefetch) so HBM B-loads hide behind MFMA at large K \u2014 technique-mfma-pipelining; FlyDSL is at occ 3/SIMD, register-headroom permitting.",
      "bound_type": "latency-bound",
      "status": "ok",
      "ck_candidate": "rocblas_gemm_ex or composable_kernels preshuffle fp8 variant",
      "aiter_comparable": "aiter.gemm_a8w8_bpreshuffle",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "MoE Block-Scale (2-stage)",
      "test": "test_moe_blockscale.py",
      "stem": "test_moe_blockscale",
      "example": "moe_blockscale",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/moe_blockscale/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/moe_blockscale",
      "op_category": "moe",
      "kernels": [
        "moe_blockscale_gemm1",
        "moe_blockscale_gemm2"
      ],
      "jit_kernel": "mfma_moe1_bs_fp8_f16_cshuffle_t16x256x128_wpe2_abi8",
      "flydsl_us": 53.8,
      "baseline_us": 44.0,
      "baseline_name": "CK",
      "speedup_vs_baseline": 0.818,
      "verdict": "FlyDSL slower",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": null,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": 9,
      "waves": 4,
      "ins": 886,
      "mapped_pct": 99.9,
      "occupancy": "2",
      "arch_vgpr": "203",
      "stall_pct_total": 82.8,
      "top_stall_type": "VMEM-wait",
      "stall_breakdown": [
        {
          "type": "VMEM-wait",
          "stall": "189.1K",
          "pct": 41.3
        },
        {
          "type": "VMEM-load",
          "stall": "166.0K",
          "pct": 36.3
        },
        {
          "type": "barrier",
          "stall": "54.2K",
          "pct": 11.9
        },
        {
          "type": "other",
          "stall": "21.2K",
          "pct": 4.6
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "13.4K",
          "pct": 2.9
        },
        {
          "type": "MFMA/FMA",
          "stall": "13.3K",
          "pct": 2.9
        },
        {
          "type": "VMEM-store",
          "stall": "68",
          "pct": 0.0
        },
        {
          "type": "LDS",
          "stall": "32",
          "pct": 0.0
        }
      ],
      "top_source_lines": [
        {
          "stall": "140.6K",
          "pct_total": 30.74,
          "domtype": "VMEM-wait",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:740"
        },
        {
          "stall": "112.2K",
          "pct_total": 24.52,
          "domtype": "VMEM-load",
          "source": "lyDSL-lab/kernels/mfma_preshuffle_pipeline.py:83"
        },
        {
          "stall": "91.1K",
          "pct_total": 19.91,
          "domtype": "barrier",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:196"
        },
        {
          "stall": "44.4K",
          "pct_total": 9.72,
          "domtype": "VMEM-load",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:674"
        },
        {
          "stall": "40.2K",
          "pct_total": 8.79,
          "domtype": "VMEM-wait",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:737"
        },
        {
          "stall": "14.4K",
          "pct_total": 3.16,
          "domtype": "VMEM-load",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:688"
        },
        {
          "stall": "3.8K",
          "pct_total": 0.84,
          "domtype": "LDS/SMEM-wait",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:306"
        },
        {
          "stall": "2.4K",
          "pct_total": 0.53,
          "domtype": "LDS/SMEM-wait",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:341"
        },
        {
          "stall": "1.6K",
          "pct_total": 0.36,
          "domtype": "VMEM-wait",
          "source": "yDSL-lab/kernels/mfma_preshuffle_pipeline.py:612"
        },
        {
          "stall": "1.6K",
          "pct_total": 0.35,
          "domtype": "other",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:450"
        },
        {
          "stall": "1.5K",
          "pct_total": 0.32,
          "domtype": "VMEM-load",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:686"
        },
        {
          "stall": "1.3K",
          "pct_total": 0.28,
          "domtype": "VMEM-wait",
          "source": "/FlyDSL-lab/kernels/moe_blockscale_2stage.py:273"
        }
      ],
      "inst_mix": {
        "mfma": 32,
        "buffer_load": 104,
        "buffer_store": 4,
        "ds_read": 12,
        "ds_write": 20
      },
      "next_occ_step": null,
      "headline": "bandwidth-bound (41% VMEM-wait + 36% VMEM-load = 78% of stalls), occ 2/SIMD @ ~203 VGPR; FlyDSL 53.8us vs CK 44.0us (0.818x, slower) \u2014 MFMA-scale starved on an under-prefetched FP8 operand/scale load chain.",
      "top_recommendation": "Deepen the global->LDS software pipeline (prefetch A/W tiles 1-2 K-stages ahead, vmcnt-gated double/triple buffering) so the mfma_scale at lines 737/740 stops blocking on operand loads (ROCmKernelWiki technique-lds-double-buffering + technique-mfma-pipelining).",
      "bound_type": "bandwidth-bound",
      "status": "ok",
      "ck_candidate": "ck_moe_stage1_fwd and ck_moe_stage2_fwd from aiter.ops.moe_op (blockscale)",
      "aiter_comparable": "aiter.fmoe_fp8_blockscale_g1u1 (fused 1-stage)",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "TopK Gating Softmax",
      "test": "test_topk_gating_softmax.py",
      "stem": "test_topk_gating_softmax",
      "example": "topk_gating_softmax",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/topk_gating_softmax/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/topk_gating_softmax",
      "op_category": "moe",
      "kernels": [
        "topk_gating_softmax_kernel"
      ],
      "jit_kernel": "topk_gating_softmax_kernel_0",
      "flydsl_us": 30.86,
      "baseline_us": 6.69,
      "baseline_name": "AIter aiter.topk_softmax (HIP fused softmax+topk+renorm; exact same signature as FlyDSL: topk_weights, topk_ids, token_expert_indices, gating, renormalize)",
      "speedup_vs_baseline": 0.217,
      "verdict": "FlyDSL slower",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": null,
      "extra_baselines": {
        "torch_softmax_topk_renorm": 56.63,
        "aiter_topk_softmax_asm": 23.04,
        "aiter_topk_softmax_hip": 6.69
      },
      "baseline_source": "standalone_script",
      "baseline_notes": "GPU 7 = MI350X / gfx950. FlyDSL harness (tests/kernels/test_topk_gating_softmax.py) has NO built-in aiter compare wired (ROCDSL_COMPARE_AITER only gates an extra FlyDSL self-timing; aiter_gpu_us is left None and the printed perf table has no aiter column). Its default sweep config is 1024x128xk6, NOT the requested 16384x128xk6; sweep_master.json logged flydsl_us=24.5 with baseline_us=null and aiter_comparable=\"None\". So I wrote a standalone matched-shape script.\n\nFound TWO direct AIter comparables, both run at the EXACT FlyDSL shape (16384x128xk6 bf16, renorm=True), measured with torch.cuda.Event (warmup=20, iters=100), two runs each:\n  - aiter.topk_softmax (HIP)     : 6.75 / 6.63 us  -> ~6.69 us  (STRONGEST, identical op signature to FlyDSL)\n  - aiter.topk_softmax_asm       : 22.43 / 23.65 us -> ~23 us   (auto-loaded the dedicated HSACO topksoftmax_12x128x6_bf16.co; (128,6) is in the gfx950 asm whitelist)\n  - torch softmax+topk+renorm    : 56.68 / 56.58 us -> ~56.6 us\n  - FlyDSL topk_gating_softmax   : 28.31 / 33.41 us -> ~30.9 us median (fluctuates 28-33us across runs)\n\nVerdict: FlyDSL LOSES badly to the apples-to-apples AIter HIP kernel (~4.6x slower; speedup 0.22x) and is also slightly slower than the AIter asm path (~0.75x). FlyDSL only beats the naive torch reference (~1.8x). This is a memory-bound MoE-gating op (16384x128 bf16 = 4MB read, ~0.6MB write); the aiter HIP kernel at 6.7us is near memory-bandwidth-bound, so the 24.5us figure in sweep_master.json (at the smaller 1024x128 default shape) overstates FlyDSL's standing. Conclusion: FlyDSL does NOT win here. Script saved at /sgl-workspace/flydsl-prof/results/baselines/test_topk_gating_softmax/baseline.py.",
      "calls": 112,
      "waves": 32,
      "ins": 1163,
      "mapped_pct": 99.9,
      "occupancy": "4",
      "arch_vgpr": "56",
      "stall_pct_total": 48.5,
      "top_stall_type": "other",
      "stall_breakdown": [
        {
          "type": "other",
          "stall": "66.5K",
          "pct": 50.3
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "57.5K",
          "pct": 43.4
        },
        {
          "type": "VMEM-wait",
          "stall": "7.2K",
          "pct": 5.4
        },
        {
          "type": "VMEM-store",
          "stall": "1.0K",
          "pct": 0.8
        },
        {
          "type": "VMEM-load",
          "stall": "72",
          "pct": 0.1
        },
        {
          "type": "MFMA/FMA",
          "stall": "32",
          "pct": 0.0
        }
      ],
      "top_source_lines": [
        {
          "stall": "129.7K",
          "pct_total": 98.03,
          "domtype": "other",
          "source": "SL-lab/kernels/topk_gating_softmax_kernel.py:103"
        },
        {
          "stall": "1.9K",
          "pct_total": 1.46,
          "domtype": "LDS/SMEM-wait",
          "source": "SL-lab/kernels/topk_gating_softmax_kernel.py:215"
        },
        {
          "stall": "440",
          "pct_total": 0.33,
          "domtype": "other",
          "source": "SL-lab/kernels/topk_gating_softmax_kernel.py:207"
        },
        {
          "stall": "240",
          "pct_total": 0.18,
          "domtype": "other",
          "source": "SL-lab/kernels/topk_gating_softmax_kernel.py:227"
        }
      ],
      "inst_mix": {
        "mfma": 0,
        "buffer_load": 2,
        "buffer_store": 18,
        "ds_read": 0,
        "ds_write": 0
      },
      "next_occ_step": {
        "waves": 5,
        "vgpr_budget": 51
      },
      "headline": "stall-bound (43% LGKMCNT-wait + 50% \"other\"), occ 4/SIMD; FlyDSL 30.9\u00b5s vs AIter-HIP 6.7\u00b5s (0.22\u00d7) \u2014 K=6 serial shuffle_xor argmax butterflies (ds_swizzle/ds_bpermute on LGKMCNT) are the ceiling, not memory.",
      "top_recommendation": "Replace shuffle_xor group reductions with DPP / v_permlane16 cross-lane ALU ops (ROCmKernelWiki technique-wave-reduce) to eliminate the per-step LGKMCNT drain \u2014 the 8-lane group fits entirely in a 16-lane DPP row.",
      "bound_type": "stall-bound",
      "status": "ok",
      "ck_candidate": "vLLM::topk_gating_softmax (reference: softmax + topk + renormalization)",
      "aiter_comparable": "None",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "LayerNorm",
      "test": "test_layernorm.py",
      "stem": "test_layernorm",
      "example": "layernorm",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/layernorm/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/layernorm",
      "op_category": "norm",
      "kernels": [
        "layernorm_kernel"
      ],
      "jit_kernel": "layernorm_kernel_0",
      "flydsl_us": 24.1,
      "baseline_us": 24.7,
      "baseline_name": "AIter",
      "speedup_vs_baseline": 1.025,
      "verdict": "comparable",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": 1490.17,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": 112,
      "waves": 16,
      "ins": 457,
      "mapped_pct": 99.8,
      "occupancy": "3",
      "arch_vgpr": "69",
      "stall_pct_total": 60.3,
      "top_stall_type": "LDS/SMEM-wait",
      "stall_breakdown": [
        {
          "type": "LDS/SMEM-wait",
          "stall": "26.8K",
          "pct": 58.2
        },
        {
          "type": "VMEM-wait",
          "stall": "12.8K",
          "pct": 27.9
        },
        {
          "type": "other",
          "stall": "4.0K",
          "pct": 8.6
        },
        {
          "type": "VMEM-load",
          "stall": "952",
          "pct": 2.1
        },
        {
          "type": "barrier",
          "stall": "824",
          "pct": 1.8
        },
        {
          "type": "VMEM-store",
          "stall": "644",
          "pct": 1.4
        }
      ],
      "top_source_lines": [
        {
          "stall": "22.8K",
          "pct_total": 49.6,
          "domtype": "LDS/SMEM-wait",
          "source": "thon_packages/flydsl/expr/rocdl/universal.py:144"
        },
        {
          "stall": "21.4K",
          "pct_total": 46.63,
          "domtype": "VMEM-wait",
          "source": "kspace/FlyDSL-lab/kernels/layernorm_kernel.py:50"
        },
        {
          "stall": "1.1K",
          "pct_total": 2.38,
          "domtype": "VMEM-load",
          "source": "space/FlyDSL-lab/kernels/layernorm_kernel.py:151"
        },
        {
          "stall": "644",
          "pct_total": 1.4,
          "domtype": "VMEM-store",
          "source": "space/FlyDSL-lab/kernels/layernorm_kernel.py:157"
        }
      ],
      "inst_mix": {
        "mfma": 0,
        "buffer_load": 12,
        "buffer_store": 4,
        "ds_read": 3,
        "ds_write": 2
      },
      "next_occ_step": {
        "waves": 4,
        "vgpr_budget": 64
      },
      "headline": "stall-bound (58% LDS/SMEM-wait), occ 3/SIMD (VGPR 72); FlyDSL 24.1\u00b5s \u2248 AIter 24.7\u00b5s (1.03\u00d7) \u2014 dependent ds_* cross-lane reduction tree (shuffle_xor) is the ceiling, not HBM (~1490 GB/s).",
      "top_recommendation": "Replace the 6-step shuffle_xor wave-reduction (ds_bpermute/ds_swizzle + lgkmcnt(0)) with gfx950 v_permlane16_b32 + DPP pure-VALU tree to move the 58% LDS-wait off the LDS crossbar (ROCmKernelWiki technique-wave-reduce).",
      "bound_type": "stall-bound",
      "status": "ok",
      "ck_candidate": "CK LayerNorm (ck_launch_and_time baseline if available)",
      "aiter_comparable": "aiter.ops.triton.norm.layer_norm",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "RMSNorm",
      "test": "test_rmsnorm.py",
      "stem": "test_rmsnorm",
      "example": "rmsnorm",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/rmsnorm/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/rmsnorm",
      "op_category": "norm",
      "kernels": [
        "rmsnorm_kernel",
        "rmsnorm_large_m_small_n_kernel"
      ],
      "jit_kernel": "rmsnorm_kernel_0",
      "flydsl_us": 25.1,
      "baseline_us": 22.4,
      "baseline_name": "AIter",
      "speedup_vs_baseline": 0.892,
      "verdict": "FlyDSL slower",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": 4729.73,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": 111,
      "waves": 64,
      "ins": 338,
      "mapped_pct": 99.7,
      "occupancy": "4",
      "arch_vgpr": "60",
      "stall_pct_total": 72.1,
      "top_stall_type": "VMEM-wait",
      "stall_breakdown": [
        {
          "type": "VMEM-wait",
          "stall": "98.4K",
          "pct": 41.1
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "48.1K",
          "pct": 20.1
        },
        {
          "type": "other",
          "stall": "37.2K",
          "pct": 15.6
        },
        {
          "type": "barrier",
          "stall": "32.5K",
          "pct": 13.6
        },
        {
          "type": "VMEM-load",
          "stall": "22.2K",
          "pct": 9.3
        },
        {
          "type": "VMEM-store",
          "stall": "796",
          "pct": 0.3
        },
        {
          "type": "LDS",
          "stall": "68",
          "pct": 0.0
        },
        {
          "type": "SMEM",
          "stall": "20",
          "pct": 0.0
        }
      ],
      "top_source_lines": [
        {
          "stall": "179.6K",
          "pct_total": 75.02,
          "domtype": "VMEM-wait",
          "source": "rkspace/FlyDSL-lab/kernels/rmsnorm_kernel.py:115"
        },
        {
          "stall": "32.8K",
          "pct_total": 13.7,
          "domtype": "LDS/SMEM-wait",
          "source": "thon_packages/flydsl/expr/rocdl/universal.py:144"
        },
        {
          "stall": "26.1K",
          "pct_total": 10.89,
          "domtype": "VMEM-load",
          "source": "orkspace/FlyDSL-lab/kernels/rmsnorm_kernel.py:59"
        },
        {
          "stall": "892",
          "pct_total": 0.37,
          "domtype": "VMEM-store",
          "source": "orkspace/FlyDSL-lab/kernels/rmsnorm_kernel.py:66"
        },
        {
          "stall": "24",
          "pct_total": 0.01,
          "domtype": "LDS",
          "source": "rkspace/FlyDSL-lab/kernels/rmsnorm_kernel.py:177"
        },
        {
          "stall": "8",
          "pct_total": 0.0,
          "domtype": "LDS",
          "source": "rkspace/FlyDSL-lab/kernels/rmsnorm_kernel.py:165"
        }
      ],
      "inst_mix": {
        "mfma": 0,
        "buffer_load": 8,
        "buffer_store": 4,
        "ds_read": 2,
        "ds_write": 2
      },
      "next_occ_step": {
        "waves": 5,
        "vgpr_budget": 51
      },
      "headline": "bandwidth-bound (41% VMEM-wait), occ 4/SIMD @60 VGPR; FlyDSL 25.1\u00b5s vs AIter 22.4\u00b5s (0.89\u00d7, slower) \u2014 single vmcnt(0) load-drain before the block reduction is the ceiling.",
      "top_recommendation": "Software-pipeline pass-1 loads with staged s_waitcnt vmcnt(N) per-tile gating instead of one vmcnt(0) full drain (ROCmKernelWiki technique-vectorized-loads + technique-lds-double-buffering) to attack the 41.1% VMEM-wait bucket.",
      "bound_type": "bandwidth-bound",
      "status": "ok",
      "ck_candidate": "none (RMSNorm is a reduction/normalization op, not GEMM; rocBLAS does not provide RMSNorm)",
      "aiter_comparable": "aiter.ops.triton.rmsnorm.rms_norm",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "Per-Token Quant",
      "test": "test_quant.py",
      "stem": "test_quant",
      "example": "quant",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/quant/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/quant",
      "op_category": "quant",
      "kernels": [
        "quant_kernel"
      ],
      "jit_kernel": "quant_kernel_0",
      "flydsl_us": 16.742,
      "baseline_us": 16.045,
      "baseline_name": "aiter per_token_quant_hip (aiter.ops.quant, fp16->int8 per-token quant)",
      "speedup_vs_baseline": 0.958,
      "verdict": "comparable",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": 5910.18,
      "extra_baselines": {},
      "baseline_source": "harness_flag",
      "baseline_notes": "Built-in baseline path: test_quant.py has a native aiter comparison gated behind FLYDSL_RUN_QUANT=1 (it imports aiter.ops.quant.per_token_quant_hip and prints a FlyDSL-vs-Reference Performance Comparison block in one run). This is the fastest, exactly-matched path and was used. GPU 5 = AMD Instinct MI350X (gfx950).\n\nThis is a MEMORY-BOUND elementwise per-token quant (fp16->int8), so GB/s is the meaningful metric, not TFLOPS (baseline_tflops/flydsl_tflops left null intentionally). The harness computes both impls' bandwidth from the same fixed traffic model total_bytes = M*N*2 (fp16 read) + M*N*1 (int8 write) + M*4 (f32 scales) = 100,679,680 B, so us values are directly comparable. baseline_us was derived from the printed aiter GB/s and that exact total_bytes (16.045 us = 100679680 / (6274.67e9) * 1e6).\n\nTwo confirming runs on GPU 5, both stable:\n  run1: FlyDSL 16.742 us / 6013.49 GB/s, aiter 16.045 us / 6274.67 GB/s, harness Speedup 0.96x\n  run2: FlyDSL ~16.79 us / 5993.93 GB/s, aiter ~16.15 us / 6233.75 GB/s, harness Speedup 0.96x\n\nVERDICT: FlyDSL does NOT win here. aiter per_token_quant_hip is ~4% faster (speedup_flydsl_vs_baseline = baseline_us/flydsl_us = 0.96, i.e. FlyDSL is slower). Both kernels hit ~6.0-6.3 TB/s, which is near the MI350X HBM3e roofline, so both are bandwidth-saturated and the gap is small. This matches the prior sweep_master.json entry (test_quant.py: bandwidth_gbs=5910.18, speedup_reported=0.95). Correctness was equivalent: max output diff 1.0 (int8 rounding boundary, expected) and max scale diff ~3.7e-9 for aiter / 0 for FlyDSL.\n\nNo CK kernel exists for this op (ck_baseline_candidate='none' in sweep); aiter per_token_quant_hip IS the strongest comparable non-FlyDSL implementation. No standalone script needed; nothing saved under results/baselines/test_quant/.",
      "calls": 23,
      "waves": 260,
      "ins": 302,
      "mapped_pct": 99.7,
      "occupancy": "5",
      "arch_vgpr": "44",
      "stall_pct_total": 77.5,
      "top_stall_type": "VMEM-wait",
      "stall_breakdown": [
        {
          "type": "VMEM-wait",
          "stall": "730.0K",
          "pct": 35.9
        },
        {
          "type": "VMEM-load",
          "stall": "425.0K",
          "pct": 20.9
        },
        {
          "type": "other",
          "stall": "352.1K",
          "pct": 17.3
        },
        {
          "type": "barrier",
          "stall": "265.2K",
          "pct": 13.1
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "208.1K",
          "pct": 10.2
        },
        {
          "type": "VMEM-store",
          "stall": "41.3K",
          "pct": 2.0
        },
        {
          "type": "MFMA/FMA",
          "stall": "8.5K",
          "pct": 0.4
        },
        {
          "type": "LDS",
          "stall": "644",
          "pct": 0.0
        }
      ],
      "top_source_lines": [
        {
          "stall": "1.35M",
          "pct_total": 66.5,
          "domtype": "VMEM-wait",
          "source": "kspace/FlyDSL-lab/tests/kernels/test_quant.py:99"
        },
        {
          "stall": "482.7K",
          "pct_total": 23.77,
          "domtype": "VMEM-load",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:178"
        },
        {
          "stall": "130.5K",
          "pct_total": 6.43,
          "domtype": "LDS/SMEM-wait",
          "source": "thon_packages/flydsl/expr/rocdl/universal.py:144"
        },
        {
          "stall": "36.4K",
          "pct_total": 1.79,
          "domtype": "VMEM-store",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:242"
        },
        {
          "stall": "12.6K",
          "pct_total": 0.62,
          "domtype": "other",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:128"
        },
        {
          "stall": "4.9K",
          "pct_total": 0.24,
          "domtype": "VMEM-store",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:219"
        },
        {
          "stall": "3.2K",
          "pct_total": 0.16,
          "domtype": "LDS/SMEM-wait",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:138"
        },
        {
          "stall": "3.0K",
          "pct_total": 0.15,
          "domtype": "other",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:133"
        },
        {
          "stall": "2.6K",
          "pct_total": 0.13,
          "domtype": "other",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:202"
        },
        {
          "stall": "2.1K",
          "pct_total": 0.1,
          "domtype": "other",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:212"
        },
        {
          "stall": "1.5K",
          "pct_total": 0.07,
          "domtype": "other",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:215"
        },
        {
          "stall": "704",
          "pct_total": 0.03,
          "domtype": "other",
          "source": "space/FlyDSL-lab/tests/kernels/test_quant.py:135"
        }
      ],
      "inst_mix": {
        "mfma": 0,
        "buffer_load": 4,
        "buffer_store": 5,
        "ds_read": 2,
        "ds_write": 2
      },
      "next_occ_step": {
        "waves": 6,
        "vgpr_budget": 42
      },
      "headline": "bandwidth-bound (56.8% VMEM stalls, 77.5% total), occ 5/SIMD; FlyDSL 16.74\u00b5s vs aiter 16.05\u00b5s (0.96\u00d7) \u2014 near the HBM3E roofline, the recoverable budget is the ~23% barrier+LDS-wait from a two-barrier in-kernel block reduction.",
      "top_recommendation": "Drop the two-barrier LDS block-max reduction (use BLOCK_THREADS=64 / RED_SLOTS==1, a barrier-free single-wave-per-row reduce) to recover the ~23% barrier+LDS-wait stall and close the 4% gap to aiter.",
      "bound_type": "bandwidth-bound",
      "status": "ok",
      "ck_candidate": "none",
      "aiter_comparable": "per_token_quant_hip",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "MoE Reduction",
      "test": "test_moe_reduce.py",
      "stem": "test_moe_reduce",
      "example": "moe_reduce",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/moe_reduce/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/moe_reduce",
      "op_category": "reduction",
      "kernels": [
        "moe_reduction_kernel"
      ],
      "jit_kernel": "moe_reduction_kernel_0",
      "flydsl_us": 382.7,
      "baseline_us": 382.6,
      "baseline_name": "torch.sum(dim=1) (== aiter.moe_sum, which falls through to at::sum_out for topk=8)",
      "speedup_vs_baseline": 1,
      "verdict": "comparable",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": 26.86,
      "extra_baselines": {
        "aiter.moe_sum_standalone_us": 412.7,
        "torch.sum_standalone_us": 391.4,
        "flydsl_us_run2": 385,
        "torch.sum_us_run2": 386.1,
        "flydsl_us_run3": 385.2,
        "torch.sum_us_run3": 383.6
      },
      "baseline_source": "harness_flag",
      "baseline_notes": "Kernel is bandwidth-bound, not GEMM (no TFLOPS). The FlyDSL harness (/sgl-workspace/FlyDSL-lab/tests/kernels/test_moe_reduce.py) has a BUILT-IN baseline: it times FlyDSL and torch.sum(dim=1) with the same run_perftest harness (warmup=10, iters=50) and prints a speedup line. Authoritative result (best of 3 runs, GPU 6 / MI350X): FlyDSL 382.7 us @ 5523.9 GB/s vs torch.sum 382.6 us @ 5524.5 GB/s => 1.00x. Three runs were tight: FlyDSL 382.7/385.0/385.2 us, torch.sum 382.6/386.1/383.6 us. Both saturate HBM at ~5.5 TB/s on MI350X, so neither can meaningfully win \u2014 this is the memory-bandwidth ceiling for a topk-8 reduction.\n\nSTRONGEST EXTERNAL COMPARABLE = aiter.moe_sum (module_moe_asm, the vLLM/AIter MoE reduce). KEY FINDING: aiter.moe_sum only ships specialized HIP kernels for topk in {2,4,5}; for ANY other topk (including the FlyDSL shape's topk=8) it falls through to `default: at::sum_out(output, input, 1)` in /sgl-workspace/aiter/csrc/kernels/topk_softmax_kernels.cu:817-855. So at topk=8 aiter.moe_sum IS torch.sum. I confirmed this: standalone /sgl-workspace/flydsl-prof/results/baselines/test_moe_reduce/bench_aiter_moe_sum.py shows aiter.moe_sum vs torch.sum max abs err = 0.0000 (bit-identical output). In that micro-bench loop aiter.moe_sum measured 412.7 us vs torch.sum 391.4 us \u2014 the ~20us gap is per-call Python/dispatch overhead, not a real kernel difference; the in-harness torch.sum (392 us standalone vs 383 us in run_perftest) confirms timing-harness variance. There is NO faster CK / hipBLASLt / rocBLAS / dedicated-ASM moe-reduce kernel for topk=8 in this AIter build; op_tests/ has no moe_reduce/moe_sum harness. Conclusion: FlyDSL matches the best available baseline (torch.sum == aiter.moe_sum) at the bandwidth ceiling; it does not beat it but is not beaten either (1.00x). aiter import requires PYTHONPATH build-fly/python_packages FIRST (else flydsl._mlir ModuleNotFoundError via aiter.ops.flydsl). Standalone script saved at /sgl-workspace/flydsl-prof/results/baselines/test_moe_reduce/bench_aiter_moe_sum.py",
      "calls": 26,
      "waves": 4112,
      "ins": 432,
      "mapped_pct": 99.8,
      "occupancy": "4",
      "arch_vgpr": "53",
      "stall_pct_total": 77.4,
      "top_stall_type": "VMEM-load",
      "stall_breakdown": [
        {
          "type": "VMEM-load",
          "stall": "32.23M",
          "pct": 70.5
        },
        {
          "type": "VMEM-wait",
          "stall": "9.43M",
          "pct": 20.6
        },
        {
          "type": "other",
          "stall": "2.53M",
          "pct": 5.5
        },
        {
          "type": "VMEM-store",
          "stall": "1.05M",
          "pct": 2.3
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "433.6K",
          "pct": 0.9
        },
        {
          "type": "SMEM",
          "stall": "21.9K",
          "pct": 0.0
        }
      ],
      "top_source_lines": [
        {
          "stall": "33.55M",
          "pct_total": 73.42,
          "domtype": "VMEM-load",
          "source": "space/FlyDSL-lab/kernels/moe_gemm_2stage.py:3177"
        },
        {
          "stall": "10.79M",
          "pct_total": 23.6,
          "domtype": "VMEM-wait",
          "source": "space/FlyDSL-lab/kernels/moe_gemm_2stage.py:3101"
        },
        {
          "stall": "1.05M",
          "pct_total": 2.3,
          "domtype": "VMEM-store",
          "source": "space/FlyDSL-lab/kernels/moe_gemm_2stage.py:3211"
        },
        {
          "stall": "162.6K",
          "pct_total": 0.36,
          "domtype": "LDS/SMEM-wait",
          "source": "ld-fly/python_packages/flydsl/expr/typing.py:896"
        },
        {
          "stall": "139.5K",
          "pct_total": 0.31,
          "domtype": "LDS/SMEM-wait",
          "source": "thon_packages/flydsl/expr/rocdl/universal.py:144"
        },
        {
          "stall": "4.8K",
          "pct_total": 0.01,
          "domtype": "SMEM",
          "source": "space/FlyDSL-lab/kernels/moe_gemm_2stage.py:3243"
        }
      ],
      "inst_mix": {
        "mfma": 0,
        "buffer_load": 72,
        "buffer_store": 9,
        "ds_read": 0,
        "ds_write": 0
      },
      "next_occ_step": {
        "waves": 5,
        "vgpr_budget": 51
      },
      "headline": "bandwidth-bound (91% VMEM load+wait), occ 4/SIMD @ 56 VGPR; FlyDSL 382.7\u00b5s \u2248 torch.sum/aiter.moe_sum 382.6\u00b5s (1.00\u00d7) at the ~5.5 TB/s HBM ceiling \u2014 already optimal, no win available.",
      "top_recommendation": "Add a non-temporal (streaming) hint to the once-read 128-bit loads (technique-vectorized-loads); width is already optimal so this is the only load-side knob, but expect <3% since HBM is saturated.",
      "bound_type": "bandwidth-bound",
      "status": "ok",
      "ck_candidate": "N/A (reduction, not GEMM; torch.sum is built-in baseline)",
      "aiter_comparable": "torch.sum(dim=1) on [1,6,5120] f16 tensor",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "Softmax",
      "test": "test_softmax.py",
      "stem": "test_softmax",
      "example": "softmax",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/softmax/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/softmax",
      "op_category": "reduction",
      "kernels": [
        "softmax_kernel"
      ],
      "jit_kernel": "softmax_kernel_0",
      "flydsl_us": 271.8,
      "baseline_us": 558.0,
      "baseline_name": "AIter",
      "speedup_vs_baseline": 2.053,
      "verdict": "FlyDSL faster",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": 5246.7,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": 112,
      "waves": 2056,
      "ins": 732,
      "mapped_pct": 99.9,
      "occupancy": "5",
      "arch_vgpr": "47",
      "stall_pct_total": 76.4,
      "top_stall_type": "VMEM-load",
      "stall_breakdown": [
        {
          "type": "VMEM-load",
          "stall": "14.60M",
          "pct": 45.9
        },
        {
          "type": "VMEM-store",
          "stall": "4.65M",
          "pct": 14.6
        },
        {
          "type": "barrier",
          "stall": "4.54M",
          "pct": 14.3
        },
        {
          "type": "VMEM-wait",
          "stall": "3.86M",
          "pct": 12.1
        },
        {
          "type": "other",
          "stall": "2.73M",
          "pct": 8.6
        },
        {
          "type": "LDS/SMEM-wait",
          "stall": "1.36M",
          "pct": 4.3
        },
        {
          "type": "SMEM",
          "stall": "29.0K",
          "pct": 0.1
        },
        {
          "type": "MFMA/FMA",
          "stall": "23.0K",
          "pct": 0.1
        }
      ],
      "top_source_lines": [
        {
          "stall": "14.75M",
          "pct_total": 46.37,
          "domtype": "VMEM-load",
          "source": "rkspace/FlyDSL-lab/kernels/softmax_kernel.py:186"
        },
        {
          "stall": "12.20M",
          "pct_total": 38.36,
          "domtype": "barrier",
          "source": "orkspace/FlyDSL-lab/kernels/softmax_kernel.py:41"
        },
        {
          "stall": "4.67M",
          "pct_total": 14.69,
          "domtype": "VMEM-store",
          "source": "rkspace/FlyDSL-lab/kernels/softmax_kernel.py:194"
        },
        {
          "stall": "151.6K",
          "pct_total": 0.48,
          "domtype": "LDS/SMEM-wait",
          "source": "thon_packages/flydsl/expr/rocdl/universal.py:144"
        },
        {
          "stall": "16.9K",
          "pct_total": 0.05,
          "domtype": "other",
          "source": "orkspace/FlyDSL-lab/kernels/softmax_kernel.py:96"
        },
        {
          "stall": "8.5K",
          "pct_total": 0.03,
          "domtype": "LDS",
          "source": "orkspace/FlyDSL-lab/kernels/softmax_kernel.py:99"
        },
        {
          "stall": "4.6K",
          "pct_total": 0.01,
          "domtype": "LDS",
          "source": "orkspace/FlyDSL-lab/kernels/softmax_kernel.py:84"
        },
        {
          "stall": "1.5K",
          "pct_total": 0.0,
          "domtype": "other",
          "source": "orkspace/FlyDSL-lab/kernels/softmax_kernel.py:90"
        }
      ],
      "inst_mix": {
        "mfma": 0,
        "buffer_load": 32,
        "buffer_store": 32,
        "ds_read": 4,
        "ds_write": 4
      },
      "next_occ_step": {
        "waves": 6,
        "vgpr_budget": 42
      },
      "headline": "stall-bound (76% stalled, 46% VMEM-load), occ 5/SIMD; FlyDSL 271.8\u00b5s vs AIter triton 558\u00b5s (2.05x win) at 32768x8192 bf16 \u2014 but the fast vectorized path is dead-coded (False-gated), so it runs scalar 16-bit loads, leaving the HBM roofline on the table.",
      "top_recommendation": "Delete the `False and` guard at softmax_kernel.py:104 to activate the pre-written BufferCopy128b fast path (8x bytes/instruction); directly attacks the 46% VMEM-load + 15% VMEM-store stall (technique-vectorized-loads). Effort: low.",
      "bound_type": "stall-bound",
      "status": "ok",
      "ck_candidate": "none (softmax is elementwise reduction; no direct CK/rocBLAS baseline. PyTorch softmax torch.nn.functional.softmax serves as reference.)",
      "aiter_comparable": "aiter.ops.triton.softmax",
      "capture_error": null,
      "has_bundle": true
    },
    {
      "name": "Fused RoPE + KV-Cache",
      "test": "test_fused_rope_cache.py",
      "stem": "test_fused_rope_cache",
      "example": "fused_rope_cache",
      "report_url": "https://github.com/jhinpan/flydsl-kernel-profiling/blob/main/examples/fused_rope_cache/REPORT.md",
      "bundle_url": "https://github.com/jhinpan/flydsl-kernel-profiling/tree/main/examples/fused_rope_cache",
      "op_category": "rope",
      "kernels": [
        "fused_qk_rope_reshape_and_cache"
      ],
      "jit_kernel": "fused_qk_rope_reshape_and_cache_0",
      "flydsl_us": 219.6,
      "baseline_us": 37.5,
      "baseline_name": "AIter",
      "speedup_vs_baseline": 0.171,
      "verdict": "FlyDSL slower",
      "tflops": null,
      "baseline_tflops": null,
      "tbps": null,
      "bandwidth_gbs": null,
      "extra_baselines": {},
      "baseline_source": null,
      "baseline_notes": null,
      "calls": 3761,
      "waves": 4,
      "ins": 208,
      "mapped_pct": 99.5,
      "occupancy": "8",
      "arch_vgpr": "12",
      "stall_pct_total": 69.5,
      "top_stall_type": "LDS/SMEM-wait",
      "stall_breakdown": [
        {
          "type": "LDS/SMEM-wait",
          "stall": "9.9K",
          "pct": 79.7
        },
        {
          "type": "VMEM-wait",
          "stall": "2.3K",
          "pct": 18.6
        },
        {
          "type": "VMEM-load",
          "stall": "128",
          "pct": 1.0
        },
        {
          "type": "VMEM-store",
          "stall": "64",
          "pct": 0.5
        },
        {
          "type": "other",
          "stall": "16",
          "pct": 0.1
        }
      ],
      "top_source_lines": [
        {
          "stall": "4.6K",
          "pct_total": 36.93,
          "domtype": "LDS/SMEM-wait",
          "source": "thon_packages/flydsl/expr/rocdl/universal.py:144"
        },
        {
          "stall": "2.8K",
          "pct_total": 22.2,
          "domtype": "LDS/SMEM-wait",
          "source": "lyDSL-lab/kernels/fused_rope_cache_kernel.py:182"
        },
        {
          "stall": "2.5K",
          "pct_total": 19.76,
          "domtype": "VMEM-wait",
          "source": "lyDSL-lab/kernels/fused_rope_cache_kernel.py:105"
        },
        {
          "stall": "2.3K",
          "pct_total": 18.34,
          "domtype": "LDS/SMEM-wait",
          "source": "lyDSL-lab/kernels/fused_rope_cache_kernel.py:214"
        },
        {
          "stall": "136",
          "pct_total": 1.1,
          "domtype": "VMEM-wait",
          "source": "lyDSL-lab/kernels/fused_rope_cache_kernel.py:429"
        },
        {
          "stall": "96",
          "pct_total": 0.77,
          "domtype": "VMEM-load",
          "source": "lyDSL-lab/kernels/fused_rope_cache_kernel.py:141"
        },
        {
          "stall": "64",
          "pct_total": 0.52,
          "domtype": "VMEM-store",
          "source": "lyDSL-lab/kernels/fused_rope_cache_kernel.py:148"
        },
        {
          "stall": "32",
          "pct_total": 0.26,
          "domtype": "VMEM-load",
          "source": "lyDSL-lab/kernels/fused_rope_cache_kernel.py:187"
        },
        {
          "stall": "16",
          "pct_total": 0.13,
          "domtype": "LDS/SMEM-wait",
          "source": "lyDSL-lab/kernels/fused_rope_cache_kernel.py:405"
        }
      ],
      "inst_mix": {
        "mfma": 0,
        "buffer_load": 7,
        "buffer_store": 6,
        "ds_read": 0,
        "ds_write": 0
      },
      "next_occ_step": null,
      "headline": "stall-bound (80% lgkmcnt s_waitcnt), occ 8/SIMD but 1 wave/block; FlyDSL 219.6\u00b5s vs AIter 37.5\u00b5s (0.17x) \u2014 serialized buffer-descriptor + position + ds_bpermute fence chain in a single 64-lane wave is the ceiling, not bandwidth.",
      "top_recommendation": "Pack multiple (head,token) pairs per workgroup so several wavefronts overlap each other's lgkmcnt/vmcnt fences (technique-occupancy-tuning) \u2014 the occupancy headroom exists, the single-wave-per-block kernel just never exposes it.",
      "bound_type": "stall-bound",
      "status": "ok",
      "ck_candidate": "false",
      "aiter_comparable": "true",
      "capture_error": null,
      "has_bundle": true
    }
  ],
  "deferred": [
    {
      "test": "test_allreduce.py",
      "reason": "multi-GPU + torch.distributed; needs 2+ ranks",
      "op_category": "comm"
    },
    {
      "test": "test_flydsl_shmem.py",
      "reason": "mori shmem, 2 PEs, no perf print",
      "op_category": "comm"
    }
  ]
};
