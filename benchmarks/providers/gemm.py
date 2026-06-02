"""GEMM provider adapters for the FlyDSL hgemm_splitk kernel (C = A @ B^T).

Layout contract (verified against kernels/hgemm_splitk.py + tests/kernels/
test_hgemm_splitk.py): A is (M,K) row-major, B is (N,K) row-major (B stored as
the transpose, one row per output column), C is (M,N). Every provider takes the
SAME a,b -- no layout conversion anywhere (includes_layout_conversion=False).

  * flydsl       -> kernels.hgemm_splitk.hgemm_splitk_ (split-K MFMA, f16/bf16)
  * pytorch      -> torch.matmul(a, b.T)  (also the fp32 reference source)
  * hipblaslt    -> torch.matmul(a, b.T)  (bf16/fp16 torch mm routes through
                    hipBLASLt on ROCm; the backend is not Python-selectable, so
                    this is the same call as pytorch, labeled honestly)
  * aiter        -> aiter.ops.gemm_op_a16w16.gemm_a16w16_asm (compiled ASM/CK)
  * aiter_triton -> aiter.ops.triton.gemm_a16w16.gemm_a16w16 (pure Triton)
  * triton       -> honest stub (triton.ops removed in triton 3.6.0; the only
                    Triton GEMM on this node is aiter's, exposed as aiter_triton)
  * ck/gluon/aiter_ck/aiter_asm -> honest stubs

FlyDSL is f16/bf16 only (NotImplementedError otherwise) -> the int8/fp4/fp8 rows
in the shared gemm ledger are rejected in supports().
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

_OK16 = {"bf16", "bfloat16", "fp16", "f16"}

# WMMA m16n16k* on the supported archs -> the MFMA output atom is 16x16.
_WMMA = 16
# split-K semaphore length bound from kernels/hgemm_splitk.py.
_SPLIT_K_SEMAPHORE_MAX_LEN = 256

# Known-good base configs, ordered: validated gfx950 params from the example
# test first, then conservative general tiles. The adapter picks the FIRST whose
# pure-shape divisibility preconditions hold for (M,N,K); the rest of the
# kernel's asserts (LDS capacity, vec alignment) surface at compile time inside
# run() if a shape slips past the precheck. STAGES/B_TO_LDS left to defaults
# except where a validated tuple set them. Tuple order matches the test's
# (TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS).
_CONFIGS = [
    # --- validated gfx950 configs from tests/kernels/test_hgemm_splitk.py ---
    dict(TILE_M=128, TILE_N=128, TILE_K=64, STAGES=4, SPLIT_K=1, BLOCK_M_WARPS=4, BLOCK_N_WARPS=4, BLOCK_K_WARPS=1),
    dict(TILE_M=32, TILE_N=64, TILE_K=64, STAGES=5, SPLIT_K=16, BLOCK_M_WARPS=2, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1),
    dict(TILE_M=16, TILE_N=64, TILE_K=128, STAGES=4, SPLIT_K=1, BLOCK_M_WARPS=1, BLOCK_N_WARPS=1, BLOCK_K_WARPS=2),
    dict(TILE_M=32, TILE_N=64, TILE_K=256, STAGES=3, SPLIT_K=16, BLOCK_M_WARPS=1, BLOCK_N_WARPS=4, BLOCK_K_WARPS=1),
    dict(TILE_M=16, TILE_N=64, TILE_K=64, STAGES=5, SPLIT_K=3, BLOCK_M_WARPS=1, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1),
    dict(TILE_M=16, TILE_N=64, TILE_K=128, STAGES=5, SPLIT_K=2, BLOCK_M_WARPS=1, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1),
    # --- conservative general tiles (SPLIT_K=1, STAGES=2, small N tiles) ---
    dict(TILE_M=64, TILE_N=64, TILE_K=64, STAGES=2, SPLIT_K=1, BLOCK_M_WARPS=2, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1),
    dict(TILE_M=32, TILE_N=64, TILE_K=64, STAGES=2, SPLIT_K=1, BLOCK_M_WARPS=2, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1),
    dict(TILE_M=16, TILE_N=64, TILE_K=64, STAGES=2, SPLIT_K=1, BLOCK_M_WARPS=1, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1),
    dict(TILE_M=64, TILE_N=128, TILE_K=64, STAGES=2, SPLIT_K=1, BLOCK_M_WARPS=2, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1),
]


def _config_ok(M, N, K, cfg) -> bool:
    """Pure-Python check of the SHAPE-dependent preconditions in
    compile_hgemm_kernel (no GPU). Conservative: a True here means the obvious
    divisibility asserts pass; the LDS-capacity / vec-alignment asserts still
    run on-device."""
    tm, tn, tk = cfg["TILE_M"], cfg["TILE_N"], cfg["TILE_K"]
    sk, st = cfg["SPLIT_K"], cfg["STAGES"]
    mw, nw, kw = cfg["BLOCK_M_WARPS"], cfg["BLOCK_N_WARPS"], cfg["BLOCK_K_WARPS"]
    if mw * nw * kw > 16:
        return False
    if tm * tn * tk > 256 * 256 * 64:
        return False
    if st < 2:
        return False
    if N % tn != 0 or N < tn:
        return False
    if K % sk != 0:
        return False
    ks = K // sk
    if ks % tk != 0 or ks // tk < st:  # BLOCK_K_LOOPS >= STAGES, BLOCK_K == TILE_K
        return False
    if tk < 32:
        return False
    # TILE_M/TILE_N must tile cleanly into warp atoms (WMMA 16x16).
    if tm % (mw * _WMMA) != 0 or tn % (nw * _WMMA) != 0:
        return False
    # split-K semaphore length: ceil(M/TILE_M) * (N//TILE_N) must fit.
    if sk > 1:
        bm = (M + tm - 1) // tm
        bn = N // tn
        if bm * bn > _SPLIT_K_SEMAPHORE_MAX_LEN:
            return False
    return True


def _pick_config(M, N, K):
    for cfg in _CONFIGS:
        if _config_ok(M, N, K, cfg):
            return cfg
    return None


class FlyDSL(ProviderAdapter):
    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "gemm":
            return False, "flydsl gemm adapter only implements gemm (hgemm_splitk)"
        if shape["dtype"] not in _OK16:
            return False, (f"FlyDSL hgemm_splitk is f16/bf16 only "
                           f"(NotImplementedError for {shape['dtype']})")
        M, N, K = _MNK(shape)
        if _pick_config(M, N, K) is None:
            return False, (f"no valid hgemm_splitk tile config for M={M},N={N},K={K} "
                           "(shape fails kernel divisibility asserts: "
                           "N%TILE_N, K%SPLIT_K, (K/SPLIT_K)%TILE_K, BLOCK_K_LOOPS>=STAGES)")
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _launcher(self, shape, inputs):
        import torch

        M, N, K = inputs["M"], inputs["N"], inputs["K"]
        ds = common.FLYDSL_DTYPE[shape["dtype"]]
        key = (M, N, K, ds)
        if key not in self._cache:
            common.bootstrap_env()
            from kernels.hgemm_splitk import hgemm_splitk_
            cfg = dict(_pick_config(M, N, K))  # checked non-None in supports()
            c = torch.empty((M, N), device="cuda", dtype=inputs["dtype"])
            self._cache[key] = (hgemm_splitk_, c, cfg)
            self.provider_detail = (f"hgemm_splitk_(M={M},N={N},K={K},{ds}); "
                                    f"split-K MFMA; kwargs={cfg}")
        return self._cache[key]

    def run(self, shape, inputs):
        import torch
        launch, c, cfg = self._launcher(shape, inputs)
        launch(c, inputs["a"], inputs["b"], None, cfg, torch.cuda.current_stream())
        return c


class PyTorch(ProviderAdapter):
    name = "pytorch"
    provider_detail = "torch.matmul(a, b.T)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "gemm":
            return False, "pytorch gemm adapter only implements gemm"
        if shape["dtype"] not in _OK16:
            return False, f"this gemm adapter compares f16/bf16; {shape['dtype']} not supported"
        return True, None

    def run(self, shape, inputs):
        import torch
        return torch.matmul(inputs["a"], inputs["b"].transpose(-1, -2))


class HipBLASLt(ProviderAdapter):
    name = "hipblaslt"
    provider_detail = ("torch.matmul(a, b.T) (bf16/fp16 mm dispatches to hipBLASLt on "
                       "ROCm; backend not Python-selectable -> same call as pytorch)")
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "gemm":
            return False, "hipblaslt gemm adapter only implements gemm"
        if shape["dtype"] not in _OK16:
            return False, f"hipBLASLt gemm path here is f16/bf16; {shape['dtype']} not supported"
        return True, None

    def run(self, shape, inputs):
        import torch
        return torch.matmul(inputs["a"], inputs["b"].transpose(-1, -2))


class Aiter(ProviderAdapter):
    name = "aiter"
    includes_allocation = False  # we preallocate the out tensor outside the timed region
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "gemm":
            return False, "aiter gemm adapter only implements gemm"
        if shape["dtype"] not in _OK16:
            return False, f"aiter gemm_a16w16_asm validated for f16/bf16, not {shape['dtype']}"
        try:
            from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def _out(self, shape, inputs):
        import torch
        M, N = inputs["M"], inputs["N"]
        key = (M, N, inputs["dtype"])
        if key not in self._cache:
            self._cache[key] = torch.empty((M, N), device="cuda", dtype=inputs["dtype"])
        return self._cache[key]

    def run(self, shape, inputs):
        from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm
        out = self._out(shape, inputs)
        self.provider_detail = ("aiter.ops.gemm_op_a16w16.gemm_a16w16_asm "
                                "(compiled ASM/CK a16w16; backend opaque from Python)")
        gemm_a16w16_asm(inputs["a"], inputs["b"], out, bias=None, splitK=None, bpreshuffle=False)
        return out


class AiterTriton(ProviderAdapter):
    name = "aiter_triton"
    provider_detail = "aiter.ops.triton.gemm_a16w16.gemm_a16w16 (Triton; Y = X @ W^T)"
    includes_allocation = True
    includes_layout_conversion = False

    def supports(self, shape):
        if shape.get("op_type") != "gemm":
            return False, "aiter_triton gemm adapter only implements gemm"
        if shape["dtype"] not in _OK16:
            return False, f"aiter triton gemm_a16w16 validated for f16/bf16, not {shape['dtype']}"
        try:
            from aiter.ops.triton.gemm_a16w16 import gemm_a16w16  # noqa: F401
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def run(self, shape, inputs):
        import torch
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
        with torch.no_grad():
            return gemm_a16w16(inputs["a"], inputs["b"], bias=None, dtype=inputs["dtype"])


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Triton(_Stub):
    name = "triton"
    _reason = ("no standalone non-aiter Triton GEMM on this node; triton.ops "
               "(triton.ops.matmul) was removed in triton 3.6.0 -- the only Triton "
               "GEMM is aiter.ops.triton.gemm_a16w16 (see the aiter_triton provider)")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "CK gemm backend not separately selectable from Python; folded into 'aiter'"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no separately-selectable ASM gemm entrypoint; gemm_a16w16_asm is exposed as 'aiter'"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK gemm adapter on this node (use 'aiter' for the compiled CK/ASM path)"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon gemm kernel on this node"


def _MNK(shape):
    a = shape["args"]
    return int(a["M"]), int(a["N"]), int(a["K"])
