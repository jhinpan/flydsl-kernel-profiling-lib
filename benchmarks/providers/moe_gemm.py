"""MoE-GEMM provider adapters (full 2-stage expert-grouped MoE).

op_type is "moe_gemm" (the harness ledger op_type); the FlyDSL kernel file is
kernels/moe_gemm_2stage.py. FlyDSL has NO single fused MoE call: it exposes two
per-stage grouped-GEMM builders -- compile_moe_gemm1 (gate+up, silu(g)*u) and
compile_moe_gemm2 (down + topk reduce). To race it against aiter's end-to-end
fused_moe fairly we model the FULL op here: the FlyDSL provider composes
stage1 -> requantize the [tokens,topk,inter] intermediate -> stage2, and
output() returns the final [tokens, model_dim] tensor. The fp32 reference is the
standard torch MoE (silu(x@W1g^T)*(x@W1u^T) @ W2^T, routed-weighted, summed over
topk).

Asymmetry to keep honest (recorded in flags/provider_detail):
  * FlyDSL is un-fused: the inter-stage requantize (fp8 pertoken_quant of the
    stage1 output) is INHERENT to chaining the two grouped GEMMs and is timed as
    part of run(). Routing/sorting + weight preshuffle are built ONCE per shape
    (outside the timed region) and shared across launches -- includes_allocation
    stays False for the FlyDSL path.
  * aiter.fused_moe is end-to-end: routing, moe_sorting, both GEMMs, and the
    reduce all live in the timed region. That extra routing/sorting work makes it
    NOT a per-kernel apples-to-apples match for FlyDSL's two GEMMs ->
    includes_layout_conversion=True and provider_detail says so.

Reachable on this node (recipe-confirmed imports): flydsl (fp8 path) + aiter
(fused_moe end-to-end) + pytorch (torch_moe eager, doubles as a slow reference
provider). aiter_triton.e2e_moe imports but needs hand-wired pre-sorted buffers
and a config dict (long, untested signature) -> honest stub. aiter_ck per-stage
kernels are best-effort JIT and only used in the per-stage REPORT.md flow ->
stub here (this adapter models the full op, not per-stage). int4/int8/fp4 input
dtypes are out of this adapter's plain-fp8/bf16 scope (fp4/a8w4 route to a
different builder, mixed_moe_gemm_2stage).

CORRECTNESS CAVEAT: fp8 round-trips through TWO quantizations (inputs + inter-
stage) so the gap vs the fp32 golden is large; common.TOL["fp8"]=(0.15,0.15)
already reflects this. bf16 is tighter. See provider_detail.
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# in_dtype strings the in-scope moe_gemm_2stage builder supports that we wire
# here. fp8 is the most-exercised / profiled path; bf16/fp16 are the no-scale
# paths. int8/int4*/fp4/a8w4 are intentionally excluded (see module docstring).
_OK_DTYPE = {"fp8", "fp8_e4m3", "bf16", "bfloat16", "fp16", "f16"}

# Canonical tiles per (stage, dtype). Validated against the builder's tile
# constraints for the candidate shapes; supports() re-checks divisibility so an
# unsupported shape is skipped, not crashed.
#   stage1: inter%tile_n1==0, model%tile_k1==0, (tile_k1*eb)%64==0, (tile_m*tile_k1*eb)%256==0
#   stage2: model%tile_n2==0, inter%tile_k2==0, (tile_m*tile_k2)%256==0, ((tile_m*tile_k2)//256)%4==0
_TILE_M = 32
_TILE_N1 = 128
_TILE_K1 = 256   # fp8 eb=1 -> 256 bytes (%64==0); 32*256%256==0
_TILE_N2 = 256
_TILE_K2 = 64    # 32*64=2048 %256==0; 2048//256=8 %4==0


def _flydsl_in_dtype(dtype: str) -> str:
    """Map ledger dtype -> moe_gemm_2stage in_dtype string."""
    if dtype in ("fp8", "fp8_e4m3"):
        return "fp8"
    if dtype in ("bf16", "bfloat16"):
        return "bf16"
    return "fp16"  # fp16/f16


def _tiles_ok(model_dim: int, inter_dim: int, dtype: str) -> tuple[bool, str | None]:
    eb = 1 if dtype in ("fp8", "fp8_e4m3") else 2
    if inter_dim % _TILE_N1 != 0:
        return False, f"stage1 inter_dim={inter_dim} % tile_n1={_TILE_N1} != 0"
    if model_dim % _TILE_K1 != 0:
        return False, f"stage1 model_dim={model_dim} % tile_k1={_TILE_K1} != 0"
    if (_TILE_K1 * eb) % 64 != 0:
        return False, f"stage1 tile_k1*eb={_TILE_K1*eb} % 64 != 0"
    if (_TILE_M * _TILE_K1 * eb) % 256 != 0:
        return False, f"stage1 tile_m*tile_k1*eb={_TILE_M*_TILE_K1*eb} % 256 != 0"
    if model_dim % _TILE_N2 != 0:
        return False, f"stage2 model_dim={model_dim} % tile_n2={_TILE_N2} != 0"
    if inter_dim % _TILE_K2 != 0:
        return False, f"stage2 inter_dim={inter_dim} % tile_k2={_TILE_K2} != 0"
    if (_TILE_M * _TILE_K2) % 256 != 0:
        return False, f"stage2 (tile_m*tile_k2)={_TILE_M*_TILE_K2} % 256 != 0"
    if ((_TILE_M * _TILE_K2) // 256) % 4 != 0:
        return False, "stage2 ((tile_m*tile_k2)//256) % 4 != 0"
    return True, None


class FlyDSL(ProviderAdapter):
    """Full 2-stage MoE composed from compile_moe_gemm1 + compile_moe_gemm2.

    The timed region (run) launches stage1, requantizes the intermediate (fp8
    path), then launches stage2 -- the minimal work to chain FlyDSL's two un-fused
    grouped GEMMs. Launchers, preshuffled weights, sorted routing buffers, and
    output buffers are built ONCE per shape key (outside the timed region).
    """

    name = "flydsl"
    includes_allocation = False
    includes_jit = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "moe_gemm":
            return False, "flydsl moe_gemm adapter only implements moe_gemm"
        if shape["dtype"] not in _OK_DTYPE:
            return False, (f"FlyDSL moe_gemm_2stage in-scope path is fp8/bf16/fp16 only; "
                           f"{shape['dtype']} (block-scale/int8/int4/fp4/a8w4) routes to a "
                           f"different builder")
        a = shape["args"]
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        ok, why = _tiles_ok(model_dim, inter_dim, shape["dtype"])
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
        from kernels.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2
        import flydsl.compiler as flyc
        from tests.utils import pertoken_quant, shuffle_weight
        from tests.kernels.test_moe_gemm import build_routing_buffers

        tokens = int(inputs["M"])
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        experts = int(a["E"])
        topk = int(a["TopK"])
        in_dtype = _flydsl_in_dtype(shape["dtype"])
        is_fp8 = in_dtype == "fp8"
        # gfx950 OCP fp8 (NOT _fnuz). The harness common.torch_dtype maps fp8 to
        # the MI300 _fnuz spelling, so we pick the gfx950 dtype here explicitly.
        fp8_dtype = torch.float8_e4m3fn
        out_dtype_s = "bf16" if in_dtype == "bf16" else "f16"
        out_torch = torch.bfloat16 if out_dtype_s == "bf16" else torch.float16

        x_fp32 = inputs["x_fp32"]
        w1_fp32 = inputs["w1_fp32"]
        w2_fp32 = inputs["w2_fp32"]
        topk_ids = inputs["topk_ids"]
        topk_weights = inputs["topk_weights"]

        # --- routing buffers (built ONCE; torch sort for portability) ---
        (sorted_token_ids, sorted_weights, sorted_expert_ids,
         num_valid_ids, _sorted_size, blocks) = build_routing_buffers(
            topk_ids=topk_ids, topk_weights=topk_weights, experts=experts,
            model_dim=model_dim, tile_m=_TILE_M, moe_sort_mode="torch")
        sorted_weights_1d = sorted_weights.contiguous().view(-1)

        # --- quantize + preshuffle weights (ONCE) ---
        if is_fp8:
            x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=fp8_dtype)
            w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=fp8_dtype)
            w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=fp8_dtype)
            w1_sh = shuffle_weight(w1_q).view(experts * (2 * inter_dim), model_dim).contiguous()
            w2_sh = shuffle_weight(w2_q).view(experts * model_dim, inter_dim).contiguous()
            x_q = x_q.contiguous().view(tokens, model_dim)
            scale_x_1d = scale_x.view(-1).contiguous()
            scale_w1_1d = scale_w1.view(-1).contiguous()
            scale_w2_1d = scale_w2.view(-1).contiguous()
        else:
            cast = torch.bfloat16 if in_dtype == "bf16" else torch.float16
            x_q = x_fp32.to(cast).contiguous().view(tokens, model_dim)
            w1_q = w1_fp32.to(cast)
            w2_q = w2_fp32.to(cast)
            w1_sh = shuffle_weight(w1_q).view(experts * (2 * inter_dim), model_dim).contiguous()
            w2_sh = shuffle_weight(w2_q).view(experts * model_dim, inter_dim).contiguous()
            # f16/bf16 paths pass 0-sized scale tensors; the kernel ignores them.
            scale_x_1d = torch.empty((0,), device="cuda", dtype=torch.float32)
            scale_w1_1d = torch.empty((0,), device="cuda", dtype=torch.float32)
            scale_w2_1d = torch.empty((0,), device="cuda", dtype=torch.float32)

        # --- preallocated outputs ---
        out1 = torch.empty((tokens, topk, inter_dim), device="cuda", dtype=out_torch)
        out2 = torch.zeros((tokens, model_dim), device="cuda", dtype=out_torch)

        stream = torch.cuda.current_stream()

        # --- stage1 launcher (gate+up; doweight_stage1=False -> stage2 applies weight) ---
        launch1 = compile_moe_gemm1(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
            tile_m=_TILE_M, tile_n=_TILE_N1, tile_k=_TILE_K1,
            doweight_stage1=False, in_dtype=in_dtype, out_dtype=out_dtype_s,
            use_cshuffle_epilog=False)

        def s1_args(o, x, w, sx, sw, st, eids, swt):
            return (o, x, w, sx, sw, st, eids, swt, num_valid_ids,
                    tokens, inter_dim, model_dim, int(blocks), stream)

        compiled1 = flyc.compile(launch1, *s1_args(
            out1, x_q, w1_sh, scale_x_1d, scale_w1_1d,
            sorted_token_ids, sorted_expert_ids, sorted_weights_1d))

        # --- stage2 launcher (down + topk reduce; doweight_stage2=True; atomic) ---
        launch2 = compile_moe_gemm2(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
            tile_m=_TILE_M, tile_n=_TILE_N2, tile_k=_TILE_K2,
            doweight_stage2=True, in_dtype=in_dtype, out_dtype=out_dtype_s,
            # stage2 f16/bf16 output requires the CShuffle epilogue.
            accumulate=True, use_cshuffle_epilog=True)

        def s2_args(o, a2flat, w, sa2, sw2, st, eids, swt):
            return (o, a2flat, w, sa2, sw2, st, eids, swt, num_valid_ids,
                    tokens, model_dim, inter_dim, int(blocks), stream)

        # build a representative a2 (fp8 quant of out1) so flyc.compile can
        # specialize stage2 once; the real a2 is recomputed each run() launch.
        if is_fp8:
            a2_q0, a2_scale0 = pertoken_quant(out1.float(), quant_dtype=fp8_dtype)
            a2_flat0 = a2_q0.view(-1).contiguous()
            a2_scale0_1d = a2_scale0.view(-1).contiguous()
        else:
            a2_flat0 = out1.view(-1).contiguous()
            a2_scale0_1d = torch.empty((0,), device="cuda", dtype=torch.float32)

        compiled2 = flyc.compile(launch2, *s2_args(
            out2, a2_flat0, w2_sh, a2_scale0_1d, scale_w2_1d,
            sorted_token_ids, sorted_expert_ids, sorted_weights_1d))

        ctx = {
            "compiled1": compiled1, "compiled2": compiled2,
            "out1": out1, "out2": out2,
            "x_q": x_q, "w1_sh": w1_sh, "w2_sh": w2_sh,
            "scale_x_1d": scale_x_1d, "scale_w1_1d": scale_w1_1d, "scale_w2_1d": scale_w2_1d,
            "sorted_token_ids": sorted_token_ids, "sorted_expert_ids": sorted_expert_ids,
            "sorted_weights_1d": sorted_weights_1d,
            "s1_args": s1_args, "s2_args": s2_args,
            "is_fp8": is_fp8, "fp8_dtype": fp8_dtype,
            "pertoken_quant": pertoken_quant,
            "tokens": tokens, "topk": topk, "inter_dim": inter_dim, "model_dim": model_dim,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"compose moe_gemm1->requant->moe_gemm2 (in_dtype={in_dtype}, out={out_dtype_s}, "
            f"tile_m={_TILE_M} s1=({_TILE_N1},{_TILE_K1}) s2=({_TILE_N2},{_TILE_K2}), "
            f"doweight in stage2, stage2=atomic-accumulate); "
            f"un-fused: inter-stage {'fp8 ' if is_fp8 else ''}requant is timed; "
            f"routing(torch-sort)+preshuffle built once. "
            f"fp8 double-quant -> wide tol (TOL[fp8]=(0.15,0.15))")
        return ctx

    def run(self, shape, inputs):
        import torch

        ctx = self._build(shape, inputs)
        out2 = ctx["out2"]
        out2.zero_()  # stage2 uses atomic accumulate -> must start zeroed

        # --- stage1: gate+up ---
        ctx["compiled1"](*ctx["s1_args"](
            ctx["out1"], ctx["x_q"], ctx["w1_sh"], ctx["scale_x_1d"], ctx["scale_w1_1d"],
            ctx["sorted_token_ids"], ctx["sorted_expert_ids"], ctx["sorted_weights_1d"]))

        # --- inter-stage requantize (inherent to FlyDSL's un-fused MoE) ---
        if ctx["is_fp8"]:
            a2_q, a2_scale = ctx["pertoken_quant"](ctx["out1"].float(), quant_dtype=ctx["fp8_dtype"])
            a2_flat = a2_q.view(-1).contiguous()
            a2_scale_1d = a2_scale.view(-1).contiguous()
        else:
            a2_flat = ctx["out1"].view(-1).contiguous()
            a2_scale_1d = ctx["scale_x_1d"]  # 0-sized; ignored by kernel

        # --- stage2: down + topk reduce (routed weight applied here) ---
        ctx["compiled2"](*ctx["s2_args"](
            out2, a2_flat, ctx["w2_sh"], a2_scale_1d, ctx["scale_w2_1d"],
            ctx["sorted_token_ids"], ctx["sorted_expert_ids"], ctx["sorted_weights_1d"]))
        return out2

    def output(self, shape, inputs):
        # run() already writes the final comparable [tokens, model_dim] tensor.
        return self.run(shape, inputs)


class Aiter(ProviderAdapter):
    """aiter.fused_moe.fused_moe -- END-TO-END fused MoE (routing+sorting+both GEMMs).

    Not a per-kernel match for FlyDSL's two grouped GEMMs (it also does routing +
    moe_sorting in the timed region) -> includes_layout_conversion=True. fp8 path
    uses per-Token quant (QuantType.per_Token) + pertoken_quant scales; bf16/fp16
    use QuantType.No.
    """

    name = "aiter"
    includes_allocation = True
    includes_layout_conversion = True

    def supports(self, shape):
        if shape.get("op_type") != "moe_gemm":
            return False, "aiter moe_gemm adapter only implements moe_gemm"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"aiter fused_moe wired for fp8/bf16/fp16 here, not {shape['dtype']}"
        try:
            from aiter.fused_moe import fused_moe  # noqa: F401
            from aiter.ops.enum import ActivationType, QuantType  # noqa: F401
        except Exception as e:
            return False, f"import aiter.fused_moe failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.fused_moe import fused_moe
        from aiter.ops.enum import ActivationType, QuantType

        dtype = shape["dtype"]
        x_fp32 = inputs["x_fp32"]
        w1_fp32 = inputs["w1_fp32"]
        w2_fp32 = inputs["w2_fp32"]
        topk_weights = inputs["topk_weights"]
        topk_ids = inputs["topk_ids"].to(torch.int32)

        if dtype in ("fp8", "fp8_e4m3"):
            from tests.utils import pertoken_quant
            fp8 = torch.float8_e4m3fn
            x_q, a1_scale = pertoken_quant(x_fp32, quant_dtype=fp8)
            w1_q, w1_scale = pertoken_quant(w1_fp32, quant_dtype=fp8)
            w2_q, w2_scale = pertoken_quant(w2_fp32, quant_dtype=fp8)
            self.provider_detail = "aiter.fused_moe (end-to-end fused; QuantType.per_Token fp8; routing+sorting timed)"
            return fused_moe(
                x_q, w1_q, w2_q, topk_weights, topk_ids,
                activation=ActivationType.Silu, quant_type=QuantType.per_Token,
                doweight_stage1=False,
                w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale)
        cast = torch.bfloat16 if dtype in ("bf16", "bfloat16") else torch.float16
        self.provider_detail = f"aiter.fused_moe (end-to-end fused; QuantType.No {dtype}; routing+sorting timed)"
        return fused_moe(
            x_fp32.to(cast), w1_fp32.to(cast), w2_fp32.to(cast), topk_weights, topk_ids,
            activation=ActivationType.Silu, quant_type=QuantType.No)


class PyTorch(ProviderAdapter):
    """Eager torch MoE (aiter.fused_moe.torch_moe). Slow; also the fp32 golden."""

    name = "pytorch"
    provider_detail = "aiter.fused_moe.torch_moe (eager loop over experts; fp32 compute)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "moe_gemm":
            return False, "pytorch moe_gemm adapter only implements moe_gemm"
        try:
            from aiter.fused_moe import torch_moe  # noqa: F401
        except Exception as e:
            return False, f"import aiter.fused_moe.torch_moe failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.fused_moe import torch_moe

        dtype = shape["dtype"]
        cast = (torch.bfloat16 if dtype in ("bf16", "bfloat16")
                else torch.float16 if dtype in ("fp16", "f16") else torch.float16)
        return torch_moe(
            inputs["x_fp32"].to(cast), inputs["w1_fp32"].to(cast), inputs["w2_fp32"].to(cast),
            inputs["topk_weights"], inputs["topk_ids"].to(torch.int32))


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class AiterTriton(_Stub):
    name = "aiter_triton"
    _reason = ("aiter.ops.triton.moe.moe_op_e2e.e2e_moe imports but needs hand-wired "
               "pre-sorted buffers (moe_align_block_size) + preallocated Intermediate/C "
               "+ a config dict (long untested signature); not wired for the full-op model")


class Triton(_Stub):
    name = "triton"
    _reason = "no standalone (non-aiter) Triton MoE kernel on this node"


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = ("ck_moe_stage1_fwd/stage2_fwd are PER-STAGE (the REPORT.md head-to-head); "
               "this adapter models the full 2-stage op, and CK JIT .so load is best-effort")


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no Python-selectable ASM MoE-GEMM path exercised here"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK MoE adapter (CK MoE is reached via aiter, see aiter_ck)"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon MoE kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is dense-GEMM only (no expert-grouped MoE op)"
