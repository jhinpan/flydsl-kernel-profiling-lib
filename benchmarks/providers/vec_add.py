"""Vector-add provider adapters (C = A + B over a 1-D [n] fp32 buffer).

This is FlyDSL's pure-bandwidth sanity kernel (3*n*4 bytes moved, ~zero flops):
read A, read B, write C. The interesting metric is effective GB/s vs HBM peak.

FlyDSL has NO reusable kernels/ module for this op -- the @flyc.kernel
vecAddKernel + @flyc.jit vecAdd launcher are defined INSIDE
tests/kernels/test_vec_add.py. The adapter first tries to import that launcher;
if the test module can't be imported (its module body calls pytest.skip at
import time when CUDA is unavailable, and it depends on tests.test_common), the
adapter REPLICATES the tiny kernel+launcher inline so it is self-contained. The
call convention mirrors the test exactly:

    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)
    vecAdd(tA, B, C, n, const_n=n, block_dim=256, vec_width=4, stream=stream)

Constraints baked into the kernel (BufferCopy128b == 4 fp32 = 128b, 256-thread
block): vec_width is fixed at 4, block_dim at 256, and n MUST be a multiple of
block_dim*vec_width = 1024 (the kernel does no tail handling -- grid_x is a
ceil-div but the last partial tile would read/write OOB). supports() rejects
non-multiples and non-fp32 dtypes. fp32 only.

Baselines per the recipe: pytorch = torch.add(A,B) (also the fp32 reference).
aiter_triton/triton get a self-contained inline Triton vector-add (no standalone
sglang/aiter "vector add" op exists). Everything else is an honest _Stub.
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# FlyDSL vecAddKernel is fp32-only (rmem tensors + addf are Float32, BufferCopy128b).
_OKF32 = {"fp32", "f32", "float32"}

# Fixed by the kernel: 256-thread block, 128-bit buffer copy = 4 fp32 lanes.
_BLOCK_DIM = 256
_VEC_WIDTH = 4
_TILE_ELEMS = _BLOCK_DIM * _VEC_WIDTH  # 1024; n must be a multiple of this


def _n(shape) -> int:
    return int(shape["args"]["n"])


def _shape_ok(shape) -> tuple[bool, str | None]:
    n = _n(shape)
    if n <= 0:
        return False, f"n must be positive (got {n})"
    if n % _TILE_ELEMS != 0:
        return False, (f"n={n} not a multiple of block_dim*vec_width={_TILE_ELEMS}; "
                       "vecAddKernel has no tail handling (partial tile reads/writes OOB)")
    return True, None


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "vec_add":
            return False, "flydsl vec_add adapter only implements vec_add"
        if shape["dtype"] not in _OKF32:
            return False, f"FlyDSL vecAddKernel is fp32-only (BufferCopy128b/addf Float32), not {shape['dtype']}"
        ok, why = _shape_ok(shape)
        if not ok:
            return False, why
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    @staticmethod
    def _resolve_launcher():
        """Return the @flyc.jit vecAdd launcher. Prefer the one defined in the
        test module; if that module can't import (it pytest.skips at import
        time / pulls tests.test_common), replicate the tiny kernel inline."""
        try:
            from tests.kernels.test_vec_add import vecAdd  # type: ignore
            return vecAdd, "tests.kernels.test_vec_add:vecAdd"
        except Exception:
            pass

        import flydsl.compiler as flyc
        import flydsl.expr as fx

        @flyc.kernel
        def vecAddKernel(
            A: fx.Tensor,
            B: fx.Tensor,
            C: fx.Tensor,
            block_dim: fx.Constexpr[int],
            vec_width: fx.Constexpr[int],
        ):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x

            tile_elems = block_dim * vec_width

            A = fx.rocdl.make_buffer_tensor(A)
            B = fx.rocdl.make_buffer_tensor(B)
            C = fx.rocdl.make_buffer_tensor(C)

            tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
            tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
            tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))

            tA = fx.slice(tA, (None, bid))
            tB = fx.slice(tB, (None, bid))
            tC = fx.slice(tC, (None, bid))

            tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
            tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
            tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

            copyAtom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

            rA = fx.make_rmem_tensor(vec_width, fx.Float32)
            rB = fx.make_rmem_tensor(vec_width, fx.Float32)
            rC = fx.make_rmem_tensor(vec_width, fx.Float32)

            fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
            fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

            vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
            fx.memref_store_vec(vC, rC)

            fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

        @flyc.jit
        def vecAdd(
            A: fx.Tensor,
            B: fx.Tensor,
            C,
            n: fx.Int32,
            const_n: fx.Constexpr[int],
            block_dim: fx.Constexpr[int],
            vec_width: fx.Constexpr[int],
            stream: fx.Stream = fx.Stream(None),
        ):
            tile_elems = block_dim * vec_width
            grid_x = (n + tile_elems - 1) // tile_elems
            vecAddKernel(A, B, C, block_dim, vec_width).launch(
                grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)

        return vecAdd, "inline-replicated vecAdd (test module not importable)"

    def _launcher(self, shape, inputs):
        import torch

        import flydsl.compiler as flyc

        n = inputs["n"]
        key = n
        if key not in self._cache:
            common.bootstrap_env()
            vecAdd, src = self._resolve_launcher()
            # wrap A as a static buffer-descriptor memref (matches the test).
            tA = flyc.from_dlpack(inputs["a"]).mark_layout_dynamic(leading_dim=0, divisibility=_VEC_WIDTH)
            out = torch.empty_like(inputs["a"])
            self._cache[key] = (vecAdd, tA, out)
            self.provider_detail = (f"vecAdd(n={n}, block_dim={_BLOCK_DIM}, vec_width={_VEC_WIDTH}); "
                                    f"BufferCopy128b f32; src={src}")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch

        vecAdd, tA, out = self._launcher(shape, inputs)
        n = inputs["n"]
        vecAdd(tA, inputs["b"], out, n, n, _BLOCK_DIM, _VEC_WIDTH,
               stream=torch.cuda.current_stream())
        return out


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = "torch.add(A, B)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "vec_add":
            return False, "pytorch vec_add adapter only implements vec_add"
        if shape["dtype"] not in _OKF32:
            return False, f"this vec_add ledger is fp32; {shape['dtype']} not supported"
        return True, None

    def run(self, shape, inputs):
        import torch
        return torch.add(inputs["a"], inputs["b"])


class AiterTriton(ProviderAdapter):
    name = "aiter_triton"
    provider_detail = "self-contained inline Triton vector-add (no standalone aiter vec_add op)"
    includes_allocation = True
    _kernel = None

    def supports(self, shape):
        if shape.get("op_type") != "vec_add":
            return False, "aiter_triton vec_add adapter only implements vec_add"
        if shape["dtype"] not in _OKF32:
            return False, f"inline Triton vec_add is fp32; {shape['dtype']} not supported"
        try:
            import triton  # noqa: F401
        except Exception as e:
            return False, f"import triton failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    @classmethod
    def _get_kernel(cls):
        if cls._kernel is None:
            import triton
            import triton.language as tl

            @triton.jit
            def _add(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                a = tl.load(a_ptr + offs, mask=mask)
                b = tl.load(b_ptr + offs, mask=mask)
                tl.store(o_ptr + offs, a + b, mask=mask)

            cls._kernel = _add
        return cls._kernel

    def run(self, shape, inputs):
        import torch
        import triton
        a, b = inputs["a"], inputs["b"]
        n = inputs["n"]
        out = torch.empty_like(a)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        self._get_kernel()[grid](a, b, out, n, BLOCK=BLOCK)
        return out


class Triton(AiterTriton):
    name = "triton"
    provider_detail = "self-contained inline Triton vector-add (standalone)"


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Aiter(_Stub):
    name = "aiter"
    _reason = "no standalone compiled aiter vector-add op (elementwise add is fused into other ops)"


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "no separately-selectable CK vector-add from Python"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no Python-selectable ASM vector-add path in AITER"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK vector-add adapter on this node"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon vector-add kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is a GEMM library (no elementwise vector-add op)"
