"""TopK Gating Softmax provider adapters (fused softmax + top-K + renorm).

op_type is "topk_gating_softmax". The FlyDSL kernel
(kernels/topk_gating_softmax_kernel.py) fuses, per token row of a
[num_tokens, num_experts] gating tensor:

  1. softmax over the expert dim,
  2. iterative top-K argmax selection,
  3. optional renormalization (selected weights rescaled to sum to 1),

writing topk_weights (f32), topk_indices (i32) and token_expert_indices (i32),
each shaped [num_tokens, topk]. We always build with renormalize=True so the
weights sum to 1 over the K selected experts (the DeepSeek-V3 / classic-MoE
convention).

CORRECTNESS CONTRACT (mirrors tests/kernels/test_topk_gating_softmax.py):
indices may legitimately differ between implementations when several experts
share the K-th-largest probability (a common bf16/f16 quantization artifact), so
we do NOT compare raw indices. The comparable tensor is the SORTED-descending
topk_weights [num_tokens, topk]; the fp32 golden (ops.TopkGatingSoftmaxOp) is
softmax->topk->renorm->sort. Every provider's output() returns its own
sorted-descending topk_weights upcast to fp32.

Provider wiring on this node:
  * flydsl       -> kernels.topk_gating_softmax_kernel.build_topk_gating_softmax_module
                    (fused, f32/f16/bf16). Layout fast-path requires
                    num_experts // VPT to be a power of 2 <= WARP_SIZE for some
                    VPT in [16,8,4,2,1]; supports() re-checks (e.g. Kimi-K2's 384
                    experts has no valid layout and is rejected, not crashed).
  * pytorch      -> torch.softmax -> torch.topk -> renorm (also the fp32 golden).
  * aiter        -> aiter.topk_gating(score_func="softmax") compiled C++ kernel.
                    Its softmax path forces need_renorm=False (softmax weights are
                    already a distribution over ALL experts), so we renormalize the
                    gathered top-K weights ourselves in run() to match the renorm
                    reference -> includes_layout_conversion=True (post-renorm step).
  * aiter_triton -> torch.softmax -> aiter.ops.triton.topk.topk (the accelerated
                    Triton top-K) -> renorm. The top-K selection is the Triton
                    kernel; the softmax + renorm are torch wrappers.
  * triton       -> honest stub: no standalone fused topk-gating-softmax Triton
                    kernel on this node (the only Triton top-K is aiter's, exposed
                    as aiter_triton).
  * aiter_ck / aiter_asm / ck / gluon / hipblaslt -> honest stubs.
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# FlyDSL kernel: elem_bits==16 path covers f16/bf16, 32 covers f32.
_OK_DTYPE = {"bf16", "bfloat16", "fp16", "f16", "fp32", "f32"}

# WARP_SIZE on gfx94x / gfx95x (CDNA). The FlyDSL builder reads this from
# kernels.kernels_common.get_warp_size(); 64 is the CDNA value the candidate
# shapes are profiled against. supports() uses it only to PRE-reject shapes the
# layout picker can't serve (the real value is read at build time).
_WARP_SIZE = 64


def _pick_layout(num_experts: int, warp_size: int = _WARP_SIZE):
    """Pure-Python mirror of _pick_layout in topk_gating_softmax_kernel.py.

    Returns (VPT, THREADS_PER_TOKEN) or (None, None) if no valid layout exists.
    The kernel requires num_experts // VPT to be a power of 2 <= warp_size for
    the largest VPT in [16, 8, 4, 2, 1] that divides num_experts.
    """
    for vpt in (16, 8, 4, 2, 1):
        if num_experts % vpt != 0:
            continue
        tpt = num_experts // vpt
        if tpt > warp_size:
            continue
        if (tpt & (tpt - 1)) != 0:  # not a power of 2
            continue
        return vpt, tpt
    return None, None


def _NEK(shape):
    a = shape["args"]
    return int(a["num_tokens"]), int(a["num_experts"]), int(a["topk"])


def _sorted_weights(weights):
    """[N, topk] -> fp32, sorted descending per row (the comparable tensor)."""
    s, _ = weights.float().sort(dim=1, descending=True)
    return s


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "topk_gating_softmax":
            return False, "flydsl topk_gating_softmax adapter only implements topk_gating_softmax"
        if shape["dtype"] not in common.FLYDSL_DTYPE:
            return False, f"FlyDSL topk_gating_softmax has no {shape['dtype']} path (f32/f16/bf16 only)"
        num_tokens, num_experts, topk = _NEK(shape)
        if topk > num_experts:
            return False, f"topk={topk} > num_experts={num_experts}"
        vpt, tpt = _pick_layout(num_experts)
        if vpt is None:
            return False, (f"num_experts={num_experts} has no valid multi-token layout "
                           f"(needs num_experts//VPT a power of 2 <= {_WARP_SIZE} for "
                           f"VPT in [16,8,4,2,1]); e.g. 384 (Kimi-K2) is unsupported")
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _launcher(self, shape, inputs):
        import torch

        num_tokens, num_experts, topk = _NEK(shape)
        ds = common.FLYDSL_DTYPE[shape["dtype"]]
        key = (num_experts, topk, ds)
        if key not in self._cache:
            common.bootstrap_env()
            from kernels.topk_gating_softmax_kernel import build_topk_gating_softmax_module
            launch = build_topk_gating_softmax_module(
                num_experts=num_experts, topk=topk, dtype_str=ds, renormalize=True)
            vpt, tpt = _pick_layout(num_experts)
            self._cache[key] = launch
            self.provider_detail = (
                f"build_topk_gating_softmax_module(num_experts={num_experts},topk={topk},"
                f"{ds},renormalize=True); fused softmax+topk+renorm, layout VPT={vpt} "
                f"THREADS_PER_TOKEN={tpt}")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch

        launch = self._launcher(shape, inputs)
        num_tokens, num_experts, topk = _NEK(shape)
        gating = inputs["gating"]  # [num_tokens, num_experts] in kernel dtype
        weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
        indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
        tei = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
        launch(gating, weights, indices, tei, num_tokens, stream=torch.cuda.current_stream())
        # stash raw weights so output() can sort without re-launching
        self._last_weights = weights
        return weights

    def output(self, shape, inputs):
        return _sorted_weights(self.run(shape, inputs))


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = "torch.softmax(dim=1) -> torch.topk -> renorm (also the fp32 reference)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "topk_gating_softmax":
            return False, "pytorch topk_gating_softmax adapter only implements topk_gating_softmax"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"no torch softmax/topk path for {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        import torch

        num_tokens, num_experts, topk = _NEK(shape)
        gating = inputs["gating"].float()
        probs = torch.softmax(gating, dim=1)
        w, _idx = torch.topk(probs, topk, dim=1)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-20)
        return w

    def output(self, shape, inputs):
        return _sorted_weights(self.run(shape, inputs))


class Aiter(ProviderAdapter):
    """aiter.topk_gating(score_func="softmax") compiled C++ MoE-topk kernel.

    aiter's softmax path forces need_renorm=False (softmax already normalizes over
    ALL experts), so it returns the gathered top-K softmax probabilities. The
    FlyDSL kernel + our reference renormalize over the K selected experts, so we
    apply that renorm to aiter's output inside run() (a small torch post-step) ->
    includes_layout_conversion=True.
    """

    name = "aiter"
    includes_allocation = True
    includes_layout_conversion = True

    def supports(self, shape):
        if shape.get("op_type") != "topk_gating_softmax":
            return False, "aiter topk_gating_softmax adapter only implements topk_gating_softmax"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"aiter topk_gating wired for f32/f16/bf16, not {shape['dtype']}"
        try:
            from aiter.ops.topk import topk_gating  # noqa: F401
        except Exception as e:
            return False, f"import aiter.ops.topk.topk_gating failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.topk import topk_gating

        num_tokens, num_experts, topk = _NEK(shape)
        gating = inputs["gating"]
        topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
        topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
        topk_gating(
            topk_weights, topk_indices, gating,
            correction_bias=None, need_renorm=False, routed_scaling_factor=1.0,
            score_func="softmax")
        self.provider_detail = ("aiter.topk_gating(score_func='softmax', need_renorm forced False); "
                                "top-K softmax probs renormalized in adapter to match renorm reference")
        # renorm over the K selected experts to match the renorm reference
        return topk_weights / topk_weights.sum(dim=1, keepdim=True).clamp(min=1e-20)

    def output(self, shape, inputs):
        return _sorted_weights(self.run(shape, inputs))


class AiterTriton(ProviderAdapter):
    """torch.softmax -> aiter.ops.triton.topk.topk (Triton top-K) -> renorm.

    The accelerated kernel is the Triton top-K selection; softmax + renorm are
    torch wrappers (no fused Triton softmax+topk gating exists on this node).
    """

    name = "aiter_triton"
    includes_allocation = True
    includes_layout_conversion = True

    def supports(self, shape):
        if shape.get("op_type") != "topk_gating_softmax":
            return False, "aiter_triton topk_gating_softmax adapter only implements topk_gating_softmax"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"validated f32/f16/bf16, not {shape['dtype']}"
        try:
            from aiter.ops.triton.topk import topk as _triton_topk  # noqa: F401
        except Exception as e:
            return False, f"import aiter.ops.triton.topk failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.topk import topk as triton_topk

        num_tokens, num_experts, topk = _NEK(shape)
        probs = torch.softmax(inputs["gating"].float(), dim=1).contiguous()
        w, _idx = triton_topk(probs, topk, dim=-1, largest=True, sorted=True)
        self.provider_detail = ("torch.softmax(dim=1) -> aiter.ops.triton.topk.topk "
                                "(1-/2-stage Triton top-K) -> renorm")
        return w / w.sum(dim=1, keepdim=True).clamp(min=1e-20)

    def output(self, shape, inputs):
        return _sorted_weights(self.run(shape, inputs))


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Triton(_Stub):
    name = "triton"
    _reason = ("no standalone fused topk-gating-softmax Triton kernel on this node; "
               "the only Triton top-K is aiter.ops.triton.topk (exposed as aiter_triton)")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "aiter MoE-topk gating is a single C++/HIP kernel (use aiter); no separately-selectable CK topk-gating from Python"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = ("aiter's ASM topk paths (moe_fused_gate/biased_grouped_topk_hip) are "
               "grouped/biased routing, not plain softmax topk-gating-softmax")


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK topk-gating-softmax adapter on this node (CK topk is reached via aiter)"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon topk-gating-softmax kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is a GEMM library (no topk-gating-softmax op)"
