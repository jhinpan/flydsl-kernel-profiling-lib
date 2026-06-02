"""Fused RoPE + reshape-and-cache provider adapters (one class per provider).

This is the NeoX-only fused kernel from kernels/fused_rope_cache_kernel.py: a
SINGLE launch that rotates Q and K by position-indexed cos/sin (writing Q_out,
K_out) and scatters rotated K + raw V into a paged KV cache. It is NOT the
standalone `rope` op (aiter.ops.triton.rope), which has a different op_type.

baseline_matrix entrypoints select a class, e.g.
benchmarks.providers.fused_rope_cache:FlyDSL.

Reachable on this node: flydsl (the candidate), aiter_triton (the exact fused op
the FlyDSL test cross-checks against), pytorch (eager reference + perf anchor).
Everything else is an honest stub -- there is no separately-selectable compiled
aiter / CK / ASM / standalone-Triton / Gluon / hipBLASLt fused rope+cache op.

CONTRACT NOTES
--------------
* The Op (ops.FusedRopeCacheOp) builds ALL inputs ONCE in FlyDSL/reference layout
  (i32 positions+slots, 2-D cos/sin, flash KV caches, fp32 scales=ones) and shares
  them across providers.
* run() does ONLY the launch (writes into preallocated Q_out/K_out + caches); it
  never assembles the comparable tensor or syncs.
* output() calls run() then returns concat([Q_out, K_out]) upcast -- the single
  comparable tensor the runner checks against the fp32 reference.
* This is a tiny, latency-bound kernel (one 64-lane wave per (head,token)); the
  fair metric is the CUDA-graph kernel-only median from common.measure_both.
* aiter_triton needs a DIFFERENT layout (i64 pos/slots, 4-D cos/sin) and writes
  the caches in-place -> it converts inside run() and sets
  includes_layout_conversion=True, using its own private zeroed cache copies.
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# Builder accepts ONLY bf16/f16 (ValueError otherwise) -> no f32, no fp8 element.
_OK16 = {"bf16", "bfloat16", "fp16", "f16"}


def _qk_concat(q_out, k_out):
    """Single comparable tensor: rotated Q and K flattened+concatenated (fp32).

    Q_out is [T, QH, D], K_out is [T, KH, D]; their head counts differ so we
    flatten each to [T, -1] and concat along the last dim. The fp32 reference
    (ops.FusedRopeCacheOp.reference) returns the same assembly so shapes match.
    """
    import torch

    return torch.cat([q_out.reshape(q_out.shape[0], -1),
                      k_out.reshape(k_out.shape[0], -1)], dim=-1).float()


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "fused_rope_cache":
            return False, "flydsl fused_rope_cache adapter only implements fused_rope_cache"
        if shape["dtype"] not in _OK16:
            return False, f"FlyDSL fused_rope_cache builder takes bf16/f16 only, not {shape['dtype']}"
        if shape["args"].get("rotate_style") != "neox":
            return False, ("FlyDSL fused_rope_cache is NeoX-only "
                           f"(is_neox=True); rotate_style={shape['args'].get('rotate_style')!r} unsupported")
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _launcher(self, shape, inputs):
        a = shape["args"]
        D = int(a["head_dim"])
        QH = int(a["num_heads"])
        KH = int(a["num_kv_heads"])
        bs = int(a.get("block_size", 16))
        flash = bool(a.get("flash_layout", True))
        reuse = bool(a.get("reuse_freqs_front_part", True))
        pos_dtype = a.get("pos_dtype", "i32")
        ds = common.FLYDSL_DTYPE[shape["dtype"]]  # 'bf16' | 'f16'
        # T MUST be in the key: the cached Q_out/K_out/KV-cache buffers are sized to
        # this shape's seq_len (empty_like below). Omitting T would reuse a smaller-T
        # shape's buffers for a larger-T launch -> out-of-bounds write -> GPU fault.
        T = int(inputs["T"])
        key = (D, QH, KH, T, bs, flash, ds, False, reuse, pos_dtype)
        if key not in self._cache:
            common.bootstrap_env()
            from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
            launch = build_fused_rope_cache_module(
                head_dim=D, num_q_heads=QH, num_kv_heads=KH, block_size=bs,
                is_neox=True, flash_layout=flash, dtype_str=ds,
                apply_scale=False, reuse_freqs_front_part=reuse, pos_dtype=pos_dtype,
            )
            # Preallocate the writable outputs ONCE (excluded from the timed region).
            import torch
            q_out = torch.empty_like(inputs["Q"])
            k_out = torch.empty_like(inputs["K"])
            key_cache = torch.zeros_like(inputs["KeyCache"])
            value_cache = torch.zeros_like(inputs["ValueCache"])
            self._cache[key] = (launch, q_out, k_out, key_cache, value_cache)
            self.provider_detail = (
                f"build_fused_rope_cache_module(D={D},QH={QH},KH={KH},{ds},"
                f"flash={flash},reuse={reuse},pos={pos_dtype},apply_scale=False); "
                "path=single 64-lane wave per (head,token), ds_bpermute NeoX pair; "
                "latency/stall-bound (verified ~0.17x vs aiter_triton on gfx950)")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        launch, q_out, k_out, kc, vc = self._launcher(shape, inputs)
        # Pure launch: writes Q_out/K_out + caches in place. num_tokens is a plain
        # int (the grid Y dim, NOT inferred from Q). KScale/VScale are ALWAYS
        # required even with apply_scale=False (pass the fp32 ones tensors).
        launch(inputs["Q"], inputs["K"], inputs["V"], inputs["Positions"],
               inputs["CosCache"], inputs["SinCache"], inputs["SlotMapping"],
               kc, vc, q_out, k_out, int(inputs["T"]),
               inputs["KScale"], inputs["VScale"],
               stream=torch.cuda.current_stream())
        return q_out

    def output(self, shape, inputs):
        # run() launches and fills the cached q_out/k_out; assemble the comparable
        # tensor OUTSIDE the timed region.
        self.run(shape, inputs)
        _, q_out, k_out, _, _ = self._launcher(shape, inputs)
        return _qk_concat(q_out, k_out)


class AiterTriton(ProviderAdapter):
    name = "aiter_triton"
    # aiter wants i64 pos/slots + 4-D cos/sin (vs the shared i32 / 2-D form) and
    # writes the caches in-place -> we convert inside run().
    includes_allocation = False
    includes_layout_conversion = True

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "fused_rope_cache":
            return False, "aiter_triton fused_rope_cache adapter only implements fused_rope_cache"
        if shape["dtype"] not in _OK16:
            return False, f"aiter fused rope+cache validated for bf16/f16, not {shape['dtype']}"
        if shape["args"].get("rotate_style") != "neox":
            return False, ("aiter fused op called with is_neox=True here; "
                           f"rotate_style={shape['args'].get('rotate_style')!r} not benchmarked")
        try:
            from aiter.ops.triton.fusions.fused_kv_cache import (  # noqa: F401
                fused_qk_rope_reshape_and_cache,
            )
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def _prep(self, shape, inputs):
        """Build the aiter-layout views ONCE (layout conversion is excluded from
        timing per includes_layout_conversion; the launch itself reuses them)."""
        a = shape["args"]
        flash = bool(a.get("flash_layout", True))
        # T in the key: cached converted slots/pos/cos/sin + private KV-cache and
        # q_out/k_out are all T-sized; reusing across T -> OOB read/write -> GPU fault.
        key = (int(a["head_dim"]), int(a["num_heads"]), int(a["num_kv_heads"]), int(inputs["T"]), flash)
        if key not in self._cache:
            import torch
            slots_i64 = inputs["SlotMapping"].to(torch.int64)
            pos_i64 = inputs["Positions"].to(torch.int64)
            cos_4d = inputs["CosCache"].unsqueeze(1).unsqueeze(1)  # [max_pos,1,1,cols]
            sin_4d = inputs["SinCache"].unsqueeze(1).unsqueeze(1)
            # Private zeroed caches so in-place writes never leak across providers.
            kc = torch.zeros_like(inputs["KeyCache"])
            vc = torch.zeros_like(inputs["ValueCache"])
            q_out = torch.empty_like(inputs["Q"])
            k_out = torch.empty_like(inputs["K"])
            self._cache[key] = (slots_i64, pos_i64, cos_4d, sin_4d, kc, vc, q_out, k_out, flash)
            self.provider_detail = ("aiter.ops.triton.fusions.fused_kv_cache."
                                    "fused_qk_rope_reshape_and_cache (Triton); "
                                    "layout-converted i32->i64 pos/slots, 2-D->4-D cos/sin")
        return self._cache[key]

    def run(self, shape, inputs):
        from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache
        slots_i64, pos_i64, cos_4d, sin_4d, kc, vc, q_out, k_out, flash = self._prep(shape, inputs)
        fused_qk_rope_reshape_and_cache(
            inputs["Q"], inputs["K"], inputs["V"], kc, vc,
            slots_i64, pos_i64, cos_4d, sin_4d,
            inputs["KScale"], inputs["VScale"],
            is_neox=True, flash_layout=flash, apply_scale=False, offs=None,
            q_out=q_out, k_out=k_out, output_zeros=False)
        return q_out

    def output(self, shape, inputs):
        self.run(shape, inputs)
        _, _, _, _, _, _, q_out, k_out, _ = self._prep(shape, inputs)
        return _qk_concat(q_out, k_out)


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = "eager NeoX RoPE (cat/mul) + python slot-scatter loop; reference + anchor"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "fused_rope_cache":
            return False, "pytorch fused_rope_cache adapter only implements fused_rope_cache"
        if shape["args"].get("rotate_style") != "neox":
            return False, f"reference is NeoX-only; rotate_style={shape['args'].get('rotate_style')!r}"
        return True, None

    def run(self, shape, inputs):
        # The same NeoX rotation as ops.FusedRopeCacheOp.reference, run in native
        # dtype (to match HW rounding). No fused torch op exists; this is eager
        # cat/mul -> it is mostly a correctness anchor, not a perf target.
        import torch

        Q, K = inputs["Q"], inputs["K"]
        cos = inputs["CosCache"][inputs["Positions"].long()].unsqueeze(1).to(Q.dtype)
        sin = inputs["SinCache"][inputs["Positions"].long()].unsqueeze(1).to(Q.dtype)
        if bool(shape["args"].get("reuse_freqs_front_part", True)):
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        Dh = Q.shape[-1] // 2

        def rot(X):
            x1, x2 = X[..., :Dh], X[..., Dh:]
            return torch.cat([x1 * cos[..., :Dh] - x2 * sin[..., :Dh],
                              x2 * cos[..., Dh:] + x1 * sin[..., Dh:]], dim=-1)

        return _qk_concat(rot(Q), rot(K))


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Aiter(_Stub):
    name = "aiter"
    _reason = ("no separately-reachable compiled (C++/CK/ASM) fused rope+reshape-and-cache "
               "op on this node; the only fused-rope-cache is the Triton one (use aiter_triton)")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "no separately-selectable CK fused rope+cache from Python"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no ASM fused rope+cache path in AITER"


class Triton(_Stub):
    name = "triton"
    _reason = "no standalone (non-aiter) Triton fused rope+reshape-and-cache kernel on this node"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone Composable Kernel fused rope+cache adapter on this node"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon fused rope+cache kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is GEMM-only (no rope/cache op)"
