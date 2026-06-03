"""Per-token (per-row) symmetric int8 quant provider adapters.

For each row of x (M,N): scale = max(|x|)/127, q = round(x/scale) clamped to
int8, scale=1.0 when the row is all-zero. Output is (q:int8[M,N], scale:fp32[M]).

The op is judged on the DEQUANTIZED tensor (q.float() * scale) vs the fp32
golden, so a kernel that rounds slightly differently (+-1 ULP on q) still
passes within tolerance, while comparing raw int8 + scales separately would be
brittle across providers. Every provider returns the same comparable tensor via
output() = dequant in fp32.

  * flydsl       -> kernels/test_quant.py build_quant_module(N) -> (launch, cfg);
                    launch(Input, Output, Scales, M). The kernel loads the row as
                    f16 (BufferCopy128b, 8xf16) so the SHARED input must be f16;
                    Output int8[M,N], Scales fp32[M]. Built+cached by N (grid is
                    (M,1,1); the same launcher serves any M for a given N).
  * pytorch      -> reference per-token quant in torch (also the dequant golden).
  * aiter        -> aiter.ops.quant.per_token_quant_hip(x, quant_dtype=int8)
                    (compiled HIP dynamic per-token scaled quant).
  * aiter_triton -> aiter.ops.quant.per_token_quant_triton(x, quant_dtype=int8).
  * triton       -> self-contained inline Triton per-token int8 quant.
  * aiter_ck/aiter_asm/ck/gluon/hipblaslt -> honest stubs.

FlyDSL is f16-only here (the kernel's load atom is f16); the ledger dtype for
this op is fp16 so FlyDSL gets a valid layout and every provider quantizes the
exact same f16 bits.
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# the FlyDSL kernel loads rows as f16 (BufferCopy128b, 8xf16=128b); only f16 is a
# valid input layout for it. The shared ledger dtype for this op is fp16.
_OK_FLYDSL = {"fp16", "f16", "float16"}
# baselines (torch/aiter/triton) additionally accept bf16/f32 inputs.
_OK_INPUT = {"fp16", "f16", "float16", "bf16", "bfloat16", "fp32", "f32", "float32"}

# the FlyDSL kernel tiles the row in chunks of BLOCK_THREADS*VEC_WIDTH; the
# partial-tile guard only fires on the last tile, but the f16 BufferCopy128b
# load requires N divisible by VEC_WIDTH (8) so a whole 8xf16 vector is in-bounds
# per lane. Match the test's own shapes (all N % 8 == 0).
_VEC_WIDTH = 8


def _MN(shape):
    a = shape["args"]
    return int(a["M"]), int(a["N"])


def _torch_quant_dtype():
    """torch int8 -- the quant target. Kept as a helper so providers agree."""
    import torch
    return torch.int8


def _ref_per_token_int8(x):
    """Per-token symmetric int8 quant in fp32 math. Returns (q_int8, scale_f32[M]).

    Mirrors aiter.pertoken_quant / the FlyDSL kernel: amax/127, zero-row -> 1.0,
    round-to-nearest, clamp to [-127,127]. q is computed in fp32 then cast."""
    import torch
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)        # [M,1]
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # zero-row -> 1.0
    q = torch.clamp(torch.round(xf / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(-1).contiguous()


def _dequant(q, scale):
    """q:int8[M,N], scale:fp32[M] (or [M,1]) -> fp32 dequant [M,N]."""
    s = scale.float()
    if s.dim() == 1:
        s = s.unsqueeze(-1)
    return q.float() * s


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "quant":
            return False, "flydsl quant adapter only implements quant (per-token int8)"
        if shape["dtype"] not in _OK_FLYDSL:
            return False, (f"FlyDSL per-token quant loads rows as f16 "
                           f"(BufferCopy128b 8xf16); no {shape['dtype']} input path")
        M, N = _MN(shape)
        if N % _VEC_WIDTH != 0:
            return False, (f"N={N} not divisible by VEC_WIDTH={_VEC_WIDTH} "
                           "(f16 BufferCopy128b needs whole 8xf16 vectors in-bounds)")
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _launcher(self, shape, inputs):
        import torch

        M, N = inputs["M"], inputs["N"]
        # the kernel is parametrized by N only (grid=(M,1,1)); cache by N and
        # reuse for any M. Output/Scales buffers depend on M too -> key on (M,N).
        key = (M, N)
        if key not in self._cache:
            import os
            common.bootstrap_env()
            # test_quant.py does pytest.skip(allow_module_level=True) at IMPORT
            # time unless FLYDSL_RUN_QUANT is truthy -> set it before importing.
            os.environ.setdefault("FLYDSL_RUN_QUANT", "1")
            from tests.kernels.test_quant import build_quant_module
            launch, config = build_quant_module(N)
            out = torch.empty((M, N), device="cuda", dtype=torch.int8)
            scales = torch.empty((M,), device="cuda", dtype=torch.float32)
            self._cache[key] = (launch, out, scales)
            self.provider_detail = (
                f"build_quant_module(N={N}); launch(Input,Output,Scales,M={M}); "
                f"f16->int8 per-token; cfg={config}")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        launch, out, scales = self._launcher(shape, inputs)
        # Input is the shared f16 x; output int8[M,N], scales fp32[M].
        # Pass current stream so CUDA-graph capture lands the kernel on the
        # capture stream (launcher defaults to stream 0 -> empty graph otherwise).
        launch(inputs["x"], out, scales, inputs["M"], stream=torch.cuda.current_stream())
        return out, scales

    def output(self, shape, inputs):
        q, scales = self.run(shape, inputs)
        return _dequant(q, scales)


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = "torch per-token int8 quant (amax/127, round, clamp); also the dequant reference"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "quant":
            return False, "pytorch quant adapter only implements quant"
        if shape["dtype"] not in _OK_INPUT:
            return False, f"no torch per-token quant path for {shape['dtype']}"
        return True, None

    def run(self, shape, inputs):
        return _ref_per_token_int8(inputs["x"])

    def output(self, shape, inputs):
        q, scale = self.run(shape, inputs)
        return _dequant(q, scale)


class Aiter(ProviderAdapter):
    name = "aiter"
    provider_detail = "aiter.ops.quant.per_token_quant_hip(x, quant_dtype=int8) (compiled HIP)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "quant":
            return False, "aiter quant adapter only implements quant"
        if shape["dtype"] not in _OK_INPUT:
            return False, f"aiter per_token_quant_hip validated f16/bf16/f32, not {shape['dtype']}"
        try:
            from aiter.ops.quant import per_token_quant_hip  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.quant import per_token_quant_hip
        with torch.no_grad():
            return per_token_quant_hip(inputs["x"], quant_dtype=torch.int8)

    def output(self, shape, inputs):
        q, scale = self.run(shape, inputs)  # q:int8[M,N], scale:fp32[M,1]
        return _dequant(q, scale)


class AiterTriton(ProviderAdapter):
    name = "aiter_triton"
    provider_detail = "aiter.ops.quant.per_token_quant_triton(x, quant_dtype=int8) (Triton)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "quant":
            return False, "aiter_triton quant adapter only implements quant"
        if shape["dtype"] not in _OK_INPUT:
            return False, f"aiter per_token_quant_triton validated f16/bf16/f32, not {shape['dtype']}"
        try:
            from aiter.ops.quant import per_token_quant_triton  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.quant import per_token_quant_triton
        with torch.no_grad():
            return per_token_quant_triton(inputs["x"], quant_dtype=torch.int8)

    def output(self, shape, inputs):
        q, scale = self.run(shape, inputs)  # q:int8[M,N], scale:fp32[M,1]
        return _dequant(q, scale)


class Triton(ProviderAdapter):
    name = "triton"
    provider_detail = "standalone inline Triton per-token int8 quant"
    includes_allocation = True
    _kernel = None

    def supports(self, shape):
        if shape.get("op_type") != "quant":
            return False, "triton quant adapter only implements quant"
        if shape["dtype"] not in _OK_INPUT:
            return False, f"validated f16/bf16/f32, not {shape['dtype']}"
        try:
            import triton  # noqa: F401
        except Exception as e:
            return False, f"import triton failed ({type(e).__name__})"
        return True, None

    @classmethod
    def _get_kernel(cls):
        if cls._kernel is None:
            import triton
            import triton.language as tl

            @triton.jit
            def _quant(x_ptr, q_ptr, s_ptr, n_rows, n_cols, stride, BLOCK: tl.constexpr):
                row = tl.program_id(0)
                if row < n_rows:
                    offs = tl.arange(0, BLOCK)
                    mask = offs < n_cols
                    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
                    amax = tl.max(tl.abs(x), axis=0)
                    scale = amax / 127.0
                    scale = tl.where(scale == 0.0, 1.0, scale)
                    tl.store(s_ptr + row, scale)
                    # round-half-to-even via floor(v + 0.5) is close enough for
                    # the dequant comparison; clamp to int8 symmetric range.
                    q = tl.math.round(x / scale)
                    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
                    tl.store(q_ptr + row * stride + offs, q.to(tl.int8), mask=mask)

            cls._kernel = _quant
        return cls._kernel

    def run(self, shape, inputs):
        import torch
        import triton
        x = inputs["x"]
        M, N = inputs["M"], inputs["N"]
        q = torch.empty((M, N), device="cuda", dtype=torch.int8)
        scale = torch.empty((M,), device="cuda", dtype=torch.float32)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
        self._get_kernel()[(M,)](x, q, scale, M, N, x.stride(0), BLOCK=BLOCK, num_warps=num_warps)
        return q, scale

    def output(self, shape, inputs):
        q, scale = self.run(shape, inputs)
        return _dequant(q, scale)


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "no separately-selectable CK per-token quant from Python (HIP path is exposed as 'aiter')"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no Python-selectable ASM per-token quant path in AITER"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK per-token quant adapter on this node (use 'aiter' for the compiled HIP path)"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon per-token quant kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is a GEMM library (no per-token quant op)"
