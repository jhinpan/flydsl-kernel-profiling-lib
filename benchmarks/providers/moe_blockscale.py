"""MoE Block-Scale (2-stage) provider adapters (FP8 block-scaled expert MoE).

op_type is "moe_blockscale"; the FlyDSL kernel file is
kernels/moe_blockscale_2stage.py. Like moe_gemm_2stage, FlyDSL has NO single
fused call: it exposes two per-stage grouped-GEMM builders --
compile_moe_blockscale_gemm1 (gate+up, silu(g)*u) and
compile_moe_blockscale_gemm2 (down + topk reduce). The difference vs moe_gemm is
the SCALE GRANULARITY: this kernel is FP8-only with per-(1x128) BLOCK scales
(ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128) instead of per-token scales.
To race FlyDSL against aiter fairly we model the FULL op here: the FlyDSL
provider composes stage1 -> block-requantize the [tokens,topk,inter] intermediate
-> stage2, and output() returns the final [tokens, model_dim] tensor. The fp32
reference is the standard block-scaled torch MoE (torch_moe_blockscale_ref in
tests/kernels/test_moe_blockscale.py).

Asymmetry to keep honest (recorded in flags/provider_detail):
  * FlyDSL is un-fused: the inter-stage block-requantize (per_group_quant_hip /
    pertoken_quant of the stage1 output, group_size=scale_block_k) is INHERENT to
    chaining the two grouped GEMMs and is timed as part of run(). Routing/sorting,
    weight block-quant + preshuffle, and the input block-quant are built ONCE per
    shape (outside the timed region) and shared across launches ->
    includes_allocation stays False for the FlyDSL path.
  * aiter.fmoe_fp8_blockscale_g1u1 is end-to-end fused (both block-scaled GEMMs +
    reduce in one launch); routing/sorting is built once outside the timed region
    but the fused kernel is NOT a per-kernel apples-to-apples match for FlyDSL's
    two grouped GEMMs -> includes_layout_conversion=True and provider_detail says
    so.

Reachable on this node (recipe-confirmed imports): flydsl (block-scale fp8 path)
+ aiter (fmoe_fp8_blockscale_g1u1 end-to-end) + pytorch (torch_moe_blockscale_ref
eager, doubles as a slow reference provider). The aiter CK per-stage kernels
(ck_moe_stage1_fwd/stage2_fwd, QuantType.per_1x128) are PER-STAGE and only used
in the per-stage REPORT.md flow -> honest stub here (this adapter models the full
op). aiter_triton has no block-scale MoE entrypoint -> stub. Only fp8 is in
scope: bf16/fp16/int8/int4/fp4 route to a different builder (moe_gemm_2stage /
mixed_moe_gemm_2stage); they are rejected in supports().

CORRECTNESS CAVEAT: fp8 round-trips through THREE block quantizations (input,
weights, inter-stage) so the gap vs the fp32 golden is large; the FlyDSL test
bar is rtol=atol=0.1 (err_ratio), and common.TOL["fp8"]=(0.15,0.15) reflects this
-- the Op's tolerance() override returns (0.2, 0.2) to match the multi-quant
accumulation. See provider_detail.
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# Block-scale MoE is FP8-only (the kernel hard-codes in_dtype="fp8"). The ledger
# dtype string for this op is "fp8"/"fp8_e4m3"; everything else is out of scope.
_OK_DTYPE = {"fp8", "fp8_e4m3"}

# Canonical block-scale tiles validated against the kernel's compile-time asserts
# (see kernels/moe_blockscale_2stage.py) and the example test
# (tests/kernels/test_moe_blockscale.py uses tile_m=16, tile_n=256, tile_k=128).
#   ScaleBlockN = ScaleBlockK = 128 are fixed; scale_block_k=128.
#   stage1 (K=model_dim, N=2*inter_dim):
#     model_dim % scale_block_k == 0, (2*inter_dim) % 128 == 0,
#     tile_k % scale_block_k == 0, (tile_k*1) % 64 == 0,
#     (tile_m*tile_k*1) % 256 == 0, (2*inter_dim) % tile_n == 0
#   stage2 (K=inter_dim, N=model_dim):
#     inter_dim % scale_block_k == 0, model_dim % 128 == 0,
#     tile_k % scale_block_k == 0, model_dim % tile_n == 0
_SCALE_BLOCK_K = 128
_TILE_M = 16
_TILE_N = 256
_TILE_K = 128
_WAVES_PER_EU = 2


def _tiles_ok(model_dim: int, inter_dim: int) -> tuple[bool, str | None]:
    sbk = _SCALE_BLOCK_K
    # --- fixed block-scale granularity preconditions ---
    if model_dim % sbk != 0:
        return False, f"model_dim={model_dim} % scale_block_k={sbk} != 0"
    if (2 * inter_dim) % 128 != 0:
        return False, f"2*inter_dim={2*inter_dim} % 128 (ScaleBlockN) != 0"
    if inter_dim % sbk != 0:
        return False, f"inter_dim={inter_dim} % scale_block_k={sbk} != 0 (stage2 K-blocks)"
    if model_dim % 128 != 0:
        return False, f"model_dim={model_dim} % 128 (stage2 ScaleBlockN) != 0"
    # --- tile preconditions (fp8 -> elem_bytes=1) ---
    if _TILE_K % sbk != 0:
        return False, f"tile_k={_TILE_K} % scale_block_k={sbk} != 0"
    if (_TILE_K * 1) % 64 != 0:
        return False, f"tile_k_bytes={_TILE_K} % 64 != 0"
    if (_TILE_M * _TILE_K * 1) % 256 != 0:
        return False, f"tile_m*tile_k*elem_bytes={_TILE_M*_TILE_K} % 256 != 0"
    if (2 * inter_dim) % _TILE_N != 0:
        return False, f"stage1 N=2*inter_dim={2*inter_dim} % tile_n={_TILE_N} != 0"
    if model_dim % _TILE_N != 0:
        return False, f"stage2 N=model_dim={model_dim} % tile_n={_TILE_N} != 0"
    return True, None


class FlyDSL(ProviderAdapter):
    """Full 2-stage block-scale MoE composed from compile_moe_blockscale_gemm1 +
    compile_moe_blockscale_gemm2.

    The timed region (run) launches stage1, block-requantizes the intermediate,
    then launches stage2 -- the minimal work to chain FlyDSL's two un-fused
    block-scaled grouped GEMMs. Compiled launchers, block-quantized + preshuffled
    weights, sorted routing buffers, block-quantized inputs, and output buffers
    are built ONCE per shape key (outside the timed region).
    """

    name = "flydsl"
    includes_allocation = False
    includes_jit = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "moe_blockscale":
            return False, "flydsl moe_blockscale adapter only implements moe_blockscale"
        if shape["dtype"] not in _OK_DTYPE:
            return False, (f"FlyDSL moe_blockscale_2stage is FP8-only (in_dtype hard-coded 'fp8'); "
                           f"{shape['dtype']} routes to a different builder "
                           f"(moe_gemm_2stage / mixed_moe_gemm_2stage)")
        a = shape["args"]
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        ok, why = _tiles_ok(model_dim, inter_dim)
        if not ok:
            return False, why
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _build(self, shape, inputs):
        import torch

        a = shape["args"]
        key = (int(a["Dim1"]), int(a["Dim2"]), int(a["E"]), int(a["TopK"]),
               int(inputs["M"]), shape["dtype"])
        if key in self._cache:
            return self._cache[key]

        common.bootstrap_env()
        import math

        import flydsl.compiler as flyc
        from flydsl.runtime.device import get_rocm_arch
        from kernels.moe_blockscale_2stage import (
            compile_moe_blockscale_gemm1,
            compile_moe_blockscale_gemm2,
        )
        from tests.kernels.test_moe_gemm import build_routing_buffers
        from tests.utils import pertoken_quant, shuffle_weight

        # gfx950 uses OCP fp8 (e4m3fn); MI300 uses the _fnuz spelling. The kernel
        # test keys this off the arch string.
        arch = str(get_rocm_arch())
        fp8_dtype = torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz

        try:
            from aiter.fused_moe import moe_sorting as _aiter_moe_sorting  # noqa: F401
            from aiter.ops.quant import per_group_quant_hip
            _has_aiter = True
        except Exception:
            per_group_quant_hip = None
            _has_aiter = False

        tokens = int(inputs["M"])
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        experts = int(a["E"])
        topk = int(a["TopK"])
        scale_blk_n = _SCALE_BLOCK_K
        scale_blk_k = _SCALE_BLOCK_K

        x_fp32 = inputs["x_fp32"]
        w1_fp32 = inputs["w1_fp32"]
        w2_fp32 = inputs["w2_fp32"]
        topk_ids = inputs["topk_ids"]
        topk_weights = inputs["topk_weights"]

        # --- routing buffers (built ONCE; aiter sort if available else torch) ---
        (sorted_ids, sorted_weights, sorted_expert_ids,
         num_valid_ids, _sorted_size, _blocks) = build_routing_buffers(
            topk_ids=topk_ids, topk_weights=topk_weights, experts=experts,
            model_dim=model_dim, tile_m=_TILE_M,
            moe_sort_mode="aiter" if _has_aiter else "torch")
        size_expert_ids = sorted_expert_ids.numel()

        # --- per-expert block-quantize weights (ONCE) ---
        def block_quant_expert(w_fp32, blk_n, blk_k):
            """Block-quantize a single expert weight [N, K] -> (fp8 [N,K], scale [flat])."""
            N_w, K_w = w_fp32.shape
            nbn, nbk = N_w // blk_n, K_w // blk_k
            tmp = (w_fp32.float()
                   .view(nbn, blk_n, nbk, blk_k)
                   .permute(0, 2, 1, 3)
                   .reshape(nbn * nbk, blk_n * blk_k))
            q, sc = pertoken_quant(tmp, quant_dtype=fp8_dtype)
            q = q.view(nbn, nbk, blk_n, blk_k).permute(0, 2, 1, 3).reshape(N_w, K_w)
            return q, sc.view(-1)

        w1_bq_list, w1_bscale_list = [], []
        w2_bq_list, w2_bscale_list = [], []
        for e_i in range(experts):
            q1, s1 = block_quant_expert(w1_fp32[e_i], scale_blk_n, scale_blk_k)
            w1_bq_list.append(q1)
            w1_bscale_list.append(s1)
            q2, s2 = block_quant_expert(w2_fp32[e_i], scale_blk_n, scale_blk_k)
            w2_bq_list.append(q2)
            w2_bscale_list.append(s2)
        w1_bq = torch.stack(w1_bq_list)
        w1_bscale = torch.stack(w1_bscale_list)
        w2_bq = torch.stack(w2_bq_list)
        w2_bscale = torch.stack(w2_bscale_list)

        # --- preshuffle block-quantized weights (ONCE) ---
        # The kernel addresses arg_w as a raw byte buffer (create_buffer_resource
        # + num_records_bytes computed from the integer dims), so it is layout-
        # agnostic about the *host* tensor shape -- it only needs the data_ptr.
        # Pass a 2-D [experts*N, K] view (matching the working moe_gemm provider)
        # instead of a flat 1-D view: for 256/384-expert DeepSeek/Kimi shapes the
        # flat element count (e.g. 384*4096*7168 ~= 1.1e10) exceeds 2^31 and
        # overflows FlyDSL's int32 per-dim shape codec in pack_layout_buffer;
        # the 2-D form keeps every packed dim well under 2^31.
        w1_bq_shuf = shuffle_weight(w1_bq, layout=(16, 16))
        w2_bq_shuf = shuffle_weight(w2_bq, layout=(16, 16))
        w1_shuf = w1_bq_shuf.reshape(experts * (2 * inter_dim), model_dim).contiguous()
        w2_shuf = w2_bq_shuf.reshape(experts * model_dim, inter_dim).contiguous()
        w1_scale_fly = w1_bscale.view(-1)
        w2_scale_fly = w2_bscale.view(-1)

        # --- block-quantize the input X (ONCE), transposed scale layout ---
        # x_q is the per-token fp8 of x (so all providers see identical x bits);
        # then re-block-quantize per (1x128) group for the kernel.
        x_q, _x_scale = pertoken_quant(x_fp32, quant_dtype=fp8_dtype)
        if _has_aiter:
            a1_bq, a1_scale_fly = per_group_quant_hip(
                x_q.to(torch.bfloat16), quant_dtype=fp8_dtype,
                group_size=scale_blk_k, transpose_scale=True)
            a1_scale_fly = a1_scale_fly.view(-1).contiguous()
        else:
            a1_bq, a1_bscale = pertoken_quant(
                x_q.float().view(-1, model_dim // scale_blk_k, scale_blk_k),
                quant_dtype=fp8_dtype)
            a1_bq = a1_bq.view(-1, model_dim)
            a1_bscale = a1_bscale.squeeze(-1)
            a1_scale_fly = a1_bscale.t().contiguous().view(-1)

        out_dtype_s = "f16"
        out_torch = torch.float16
        stream = torch.cuda.current_stream()

        # --- stage1 launcher (gate+up; doweight_stage1=False -> stage2 weights) ---
        exe1 = compile_moe_blockscale_gemm1(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
            tile_m=_TILE_M, tile_n=_TILE_N, tile_k=_TILE_K,
            doweight_stage1=False, scale_block_k=scale_blk_k,
            out_dtype=out_dtype_s, waves_per_eu=_WAVES_PER_EU)

        out1 = torch.zeros((tokens, topk, inter_dim), dtype=out_torch, device="cuda")

        def s1_args():
            return (out1.view(-1), a1_bq.view(-1), w1_shuf, a1_scale_fly, w1_scale_fly,
                    sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                    tokens, inter_dim, model_dim, int(size_expert_ids), stream)

        compiled1 = flyc.compile(exe1, *s1_args())

        # --- stage2 launcher (down + topk reduce; doweight_stage2=True; atomic) ---
        # stage2 f16/bf16 output requires the CShuffle epilogue (prior review).
        exe2 = compile_moe_blockscale_gemm2(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
            tile_m=_TILE_M, tile_n=_TILE_N, tile_k=_TILE_K,
            doweight_stage2=True, scale_block_k=scale_blk_k,
            out_dtype=out_dtype_s, accumulate=True, use_cshuffle_epilog=True,
            waves_per_eu=_WAVES_PER_EU)

        out2 = torch.zeros((tokens, model_dim), dtype=out_torch, device="cuda")

        # build a representative a2 (block-quant of out1) so flyc.compile can
        # specialize stage2 once; the real a2 is recomputed each run() launch.
        if _has_aiter:
            a2_bq0, a2_scale_fly0 = per_group_quant_hip(
                out1.to(torch.bfloat16).view(-1, inter_dim), quant_dtype=fp8_dtype,
                group_size=scale_blk_k, transpose_scale=True)
            a2_scale_fly0 = a2_scale_fly0.view(-1).contiguous()
        else:
            a2_bq0, a2_bscale0 = pertoken_quant(
                out1.float().view(-1, inter_dim // scale_blk_k, scale_blk_k),
                quant_dtype=fp8_dtype)
            a2_bq0 = a2_bq0.view(-1, inter_dim)
            a2_scale_fly0 = a2_bscale0.squeeze(-1).t().contiguous().view(-1)

        def s2_args(a2_flat, a2_scale):
            return (out2.view(-1), a2_flat, w2_shuf, a2_scale, w2_scale_fly,
                    sorted_ids, sorted_expert_ids, sorted_weights, num_valid_ids,
                    tokens, model_dim, inter_dim, int(size_expert_ids), stream)

        compiled2 = flyc.compile(exe2, *s2_args(a2_bq0.view(-1), a2_scale_fly0))

        ctx = {
            "compiled1": compiled1, "compiled2": compiled2,
            "exe1": exe1, "exe2": exe2,
            "out1": out1, "out2": out2,
            "a1_bq": a1_bq, "w1_shuf": w1_shuf, "w2_shuf": w2_shuf,
            "a1_scale_fly": a1_scale_fly, "w1_scale_fly": w1_scale_fly,
            "w2_scale_fly": w2_scale_fly,
            "sorted_ids": sorted_ids, "sorted_expert_ids": sorted_expert_ids,
            "sorted_weights": sorted_weights, "num_valid_ids": num_valid_ids,
            "size_expert_ids": int(size_expert_ids),
            "s1_args": s1_args, "s2_args": s2_args,
            "per_group_quant_hip": per_group_quant_hip, "pertoken_quant": pertoken_quant,
            "has_aiter": _has_aiter, "fp8_dtype": fp8_dtype,
            "tokens": tokens, "topk": topk, "inter_dim": inter_dim,
            "model_dim": model_dim, "scale_blk_k": scale_blk_k, "stream": stream,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"compose moe_blockscale_gemm1->block-requant->moe_blockscale_gemm2 "
            f"(fp8 per-1x{scale_blk_k} block scale, out={out_dtype_s}, "
            f"tile_m={_TILE_M} tile_n={_TILE_N} tile_k={_TILE_K}, "
            f"doweight in stage2, stage2=atomic-accumulate + cshuffle epilog); "
            f"un-fused: inter-stage block-requant is timed; "
            f"routing({'aiter' if _has_aiter else 'torch'}-sort)+weight block-quant"
            f"+preshuffle+input block-quant built once. "
            f"fp8 triple-block-quant -> wide tol")
        return ctx

    def run(self, shape, inputs):
        import torch

        ctx = self._build(shape, inputs)
        out1 = ctx["out1"]
        out2 = ctx["out2"]
        inter_dim = ctx["inter_dim"]
        scale_blk_k = ctx["scale_blk_k"]
        fp8 = ctx["fp8_dtype"]

        out1.zero_()
        # --- stage1: gate+up ---
        ctx["compiled1"](*ctx["s1_args"]())

        # --- inter-stage block-requantize (inherent to FlyDSL's un-fused MoE) ---
        if ctx["has_aiter"]:
            a2_bq, a2_scale_fly = ctx["per_group_quant_hip"](
                out1.to(torch.bfloat16).view(-1, inter_dim), quant_dtype=fp8,
                group_size=scale_blk_k, transpose_scale=True)
            a2_scale_fly = a2_scale_fly.view(-1).contiguous()
        else:
            a2_bq, a2_bscale = ctx["pertoken_quant"](
                out1.float().view(-1, inter_dim // scale_blk_k, scale_blk_k),
                quant_dtype=fp8)
            a2_bq = a2_bq.view(-1, inter_dim)
            a2_scale_fly = a2_bscale.squeeze(-1).t().contiguous().view(-1)

        # --- stage2: down + topk reduce (routed weight applied here; atomic) ---
        out2.zero_()  # atomic accumulate -> must start zeroed
        ctx["compiled2"](*ctx["s2_args"](a2_bq.view(-1), a2_scale_fly))
        return out2

    def output(self, shape, inputs):
        # run() already writes the final comparable [tokens, model_dim] tensor.
        return self.run(shape, inputs)


class Aiter(ProviderAdapter):
    """aiter.fmoe_fp8_blockscale_g1u1 -- END-TO-END fused block-scale MoE.

    Both block-scaled GEMMs + the topk reduce happen in one fused launch. Routing
    (aiter moe_sorting), weight block-quant + preshuffle, and the input block-quant
    are built ONCE per shape key (outside the timed region) and shared; only the
    fused kernel launch is timed. It is NOT a per-kernel match for FlyDSL's two
    grouped GEMMs -> includes_layout_conversion=True.
    """

    name = "aiter"
    includes_allocation = False
    includes_layout_conversion = True

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "moe_blockscale":
            return False, "aiter moe_blockscale adapter only implements moe_blockscale"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"aiter fmoe_fp8_blockscale_g1u1 is fp8 block-scale only, not {shape['dtype']}"
        try:
            import aiter  # noqa: F401
            from aiter.fused_moe import moe_sorting as _ms  # noqa: F401
            from aiter.ops.quant import per_group_quant_hip  # noqa: F401
        except Exception as e:
            return False, f"import aiter (fmoe_fp8_blockscale_g1u1) failed ({type(e).__name__}); launch via benchmarks/env.sh"
        if not hasattr(__import__("aiter"), "fmoe_fp8_blockscale_g1u1"):
            return False, "aiter has no fmoe_fp8_blockscale_g1u1 on this node"
        a = shape["args"]
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        ok, why = _tiles_ok(model_dim, inter_dim)
        return (ok, why) if not ok else (True, None)

    def _build(self, shape, inputs):
        import torch

        a = shape["args"]
        key = (int(a["Dim1"]), int(a["Dim2"]), int(a["E"]), int(a["TopK"]),
               int(inputs["M"]), shape["dtype"])
        if key in self._cache:
            return self._cache[key]

        common.bootstrap_env()
        import aiter
        from aiter.fused_moe import moe_sorting as aiter_moe_sorting
        from aiter.ops.quant import per_group_quant_hip
        from flydsl.runtime.device import get_rocm_arch
        from tests.utils import pertoken_quant, shuffle_weight

        arch = str(get_rocm_arch())
        fp8_dtype = torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz

        tokens = int(inputs["M"])
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        experts = int(a["E"])
        topk = int(a["TopK"])
        scale_blk_n = _SCALE_BLOCK_K
        scale_blk_k = _SCALE_BLOCK_K
        nblk_k_w1 = model_dim // scale_blk_k

        x_fp32 = inputs["x_fp32"]
        w1_fp32 = inputs["w1_fp32"]
        w2_fp32 = inputs["w2_fp32"]
        topk_ids = inputs["topk_ids"]
        topk_weights = inputs["topk_weights"]
        dtype = torch.bfloat16

        def block_quant_expert(w_fp32, blk_n, blk_k):
            N_w, K_w = w_fp32.shape
            nbn, nbk = N_w // blk_n, K_w // blk_k
            tmp = (w_fp32.float()
                   .view(nbn, blk_n, nbk, blk_k)
                   .permute(0, 2, 1, 3)
                   .reshape(nbn * nbk, blk_n * blk_k))
            q, sc = pertoken_quant(tmp, quant_dtype=fp8_dtype)
            q = q.view(nbn, nbk, blk_n, blk_k).permute(0, 2, 1, 3).reshape(N_w, K_w)
            return q, sc.view(-1)

        w1_bq_list, w1_bscale_list, w2_bq_list, w2_bscale_list = [], [], [], []
        for e_i in range(experts):
            q1, s1 = block_quant_expert(w1_fp32[e_i], scale_blk_n, scale_blk_k)
            w1_bq_list.append(q1)
            w1_bscale_list.append(s1)
            q2, s2 = block_quant_expert(w2_fp32[e_i], scale_blk_n, scale_blk_k)
            w2_bq_list.append(q2)
            w2_bscale_list.append(s2)
        w1_bq = torch.stack(w1_bq_list)
        w1_bscale = torch.stack(w1_bscale_list)
        w2_bq = torch.stack(w2_bq_list)
        w2_bscale = torch.stack(w2_bscale_list)
        w1_bq_shuf = shuffle_weight(w1_bq, layout=(16, 16))
        w2_bq_shuf = shuffle_weight(w2_bq, layout=(16, 16))

        x_q, _ = pertoken_quant(x_fp32, quant_dtype=fp8_dtype)
        a1_bq, a1_scale_fly = per_group_quant_hip(
            x_q.to(torch.bfloat16), quant_dtype=fp8_dtype,
            group_size=scale_blk_k, transpose_scale=True)
        # fmoe_fp8_blockscale_g1u1 expects a1_scale as [nblk_k_w1, tokens] contiguous.
        a1_scale_aiter = a1_scale_fly.view(nblk_k_w1, tokens)

        ck_block_m = 32
        sorted_ids_a, sorted_weights_a, sorted_expert_ids_a, num_valid_ids_a, out_aiter = \
            aiter_moe_sorting(topk_ids.to(torch.int32), topk_weights.float(),
                              experts, model_dim, dtype, ck_block_m)

        ctx = {
            "aiter": aiter, "out_aiter": out_aiter, "a1_bq": a1_bq,
            "w1_bq_shuf": w1_bq_shuf, "w2_bq_shuf": w2_bq_shuf,
            "sorted_ids_a": sorted_ids_a, "sorted_weights_a": sorted_weights_a,
            "sorted_expert_ids_a": sorted_expert_ids_a, "num_valid_ids_a": num_valid_ids_a,
            "topk": topk, "a1_scale_aiter": a1_scale_aiter,
            "w1_bscale": w1_bscale, "w2_bscale": w2_bscale,
            "scale_blk_n": scale_blk_n, "scale_blk_k": scale_blk_k,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"aiter.fmoe_fp8_blockscale_g1u1 (end-to-end fused block-scale MoE; "
            f"per-1x{scale_blk_k} scale, aiter moe_sorting; both GEMMs + reduce in one launch); "
            f"routing+block-quant+preshuffle built once")
        return ctx

    def run(self, shape, inputs):
        ctx = self._build(shape, inputs)
        out_aiter = ctx["out_aiter"]
        out_aiter.zero_()
        ctx["aiter"].fmoe_fp8_blockscale_g1u1(
            out_aiter,
            ctx["a1_bq"],
            ctx["w1_bq_shuf"],
            ctx["w2_bq_shuf"],
            ctx["sorted_ids_a"],
            ctx["sorted_weights_a"],
            ctx["sorted_expert_ids_a"],
            ctx["num_valid_ids_a"],
            ctx["topk"],
            ctx["a1_scale_aiter"],
            ctx["w1_bscale"],
            ctx["w2_bscale"],
            "",
            ctx["scale_blk_n"],
            ctx["scale_blk_k"],
            None,
        )
        return out_aiter

    def output(self, shape, inputs):
        return self.run(shape, inputs)


class PyTorch(ProviderAdapter):
    """Eager block-scale torch MoE (mirrors torch_moe_blockscale_ref). Slow; also
    the structure the fp32 golden is computed from."""

    name = "pytorch"
    provider_detail = "torch block-scale MoE (per-expert block-dequant -> silu(g)*u @ W2, topk reduce; fp32 compute)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "moe_blockscale":
            return False, "pytorch moe_blockscale adapter only implements moe_blockscale"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"pytorch moe_blockscale reference is fp8 block-scale only, not {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        # The Op's fp32 reference already implements the full block-scale MoE in
        # fp32. Re-using it here gives a torch "provider" row consistent with the
        # golden (it IS the golden), so vs-best stays honest.
        from benchmarks import ops as _ops

        op = _ops.get_op("moe_blockscale")
        return op.reference(shape, inputs)

    def output(self, shape, inputs):
        return self.run(shape, inputs)


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class AiterTriton(_Stub):
    name = "aiter_triton"
    _reason = ("no aiter Triton block-scale MoE entrypoint; aiter's block-scale "
               "MoE is the compiled fmoe_fp8_blockscale_g1u1 (exposed as 'aiter')")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = ("ck_moe_stage1_fwd/stage2_fwd (QuantType.per_1x128) are PER-STAGE "
               "(the REPORT.md head-to-head); this adapter models the full 2-stage "
               "op, and CK JIT .so load is best-effort")


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no Python-selectable ASM block-scale MoE path exercised here"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK block-scale MoE adapter (CK block-scale MoE is reached via aiter, see aiter_ck)"


class Triton(_Stub):
    name = "triton"
    _reason = "no standalone (non-aiter) Triton block-scale MoE kernel on this node"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon block-scale MoE kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is dense-GEMM only (no expert-grouped block-scale MoE op)"
