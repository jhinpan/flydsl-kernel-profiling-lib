"""MLA-decode (fp8 Q / fp8 KV, nhead=128, page_size=1) provider adapters.

The FlyDSL kernel is kernels/mla_fwd_decode.py::flydsl_mla_fwd_decode, which for
num_heads==128 + fp8 Q/KV dispatches to
kernels/mla_fwd_decode_m16x8_fp8_fp8.launch_mla_fwd_decode_m16x8_fp8_fp8 (a
split-K MFMA m16x8 decode kernel that writes per-split partials, then a separate
mla_reduce_v1 combines splits). MLA dims (DeepSeek-V3/R1): KV_LORA_RANK=512,
QK_ROPE=64, QK_HEAD_DIM=576, V_HEAD_DIM=512, NHEAD=128, page_size=1.

Layout/call convention copied verbatim from tests/kernels/test_mla_decode.py
(run_single + get_mla_metadata_v1 + torch_mla_extend reference):
  query     [num_seqs, 128, 576] fp8
  kv_buffer [num_page, 1, 1, 576] fp8
  final_out [num_seqs, 128, 512] bf16
The split partials/lse buffers and the aiter-generated work metadata
(get_mla_metadata_info_v1 / get_mla_metadata_v1) are built ONCE per shape key,
outside the timed region, and shared by the FlyDSL + aiter providers (those
metadata kernels are a tiny one-shot setup, not the op under test).

This op is decode-only stage1 (the FlyDSL kernel) PLUS the mla_reduce_v1 split
combine -- both are required to produce the comparable [num_seqs,128,512]
output, so run() times stage1+reduce together (the test times stage1 alone but
runs reduce once for correctness; here we keep them together so every provider's
run() yields the same final tensor). Honest: includes_layout_conversion stays
False (all providers take the same shared inputs).

Providers wired:
  * flydsl       -> flydsl_mla_fwd_decode (stage1) + aiter.mla_reduce_v1 (combine)
  * aiter        -> aiter.hk_mla_decode_fwd (stage1, work-map driven) + mla_reduce_v1
  * aiter_asm    -> aiter.mla_decode_stage1_asm_fwd (ASM stage1) + mla_reduce_v1
  * pytorch      -> torch_mla_extend (the fp32 golden; eager paged-MLA reference)
  * triton/aiter_triton/aiter_ck/ck/gluon/hipblaslt -> honest stubs
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# Model constants (DeepSeek-V3 / R1), from tests/kernels/test_mla_decode.py +
# kernels/mla_fwd_decode_m16x8_fp8_fp8.py.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                      # 512
NHEAD = 128
NHEAD_KV = 1
PAGE_SIZE = 1

# The FlyDSL fp8 path AND the aiter MLA-decode ops only exist for fp8 Q/KV.
_OK_DTYPE = {"fp8", "fp8_e4m3"}


def _bc(shape):
    a = shape["args"]
    return int(a["batch"]), int(a["ctx_len"]), int(a.get("decode_qlen", 1))


def _aiter_fp8():
    """gfx950/MI350X OCP fp8 dtype, via aiter's own arch-aware source of truth
    (aiter.dtypes.fp8 == get_dtype_fp8()). Falls back to torch.float8_e4m3fn."""
    try:
        from aiter import dtypes
        return dtypes.fp8
    except Exception:
        import torch
        return torch.float8_e4m3fn


def _build_metadata(batch_size, ctx_len, decode_qlen, q_fp8, kv_buffer_fp8,
                    kv_indices, max_split_per_batch=32):
    """Replicate run_single()'s aiter metadata + buffer allocation EXACTLY.

    Returns a dict of everything the stage1 kernels + mla_reduce_v1 need. Built
    once per shape key (outside the timed region) and shared across providers.
    """
    import torch
    from aiter import dtypes
    from aiter.ops.attention import (
        get_mla_metadata_info_v1,
        get_mla_metadata_v1,
    )

    fp8 = dtypes.fp8
    out_dtype = torch.bfloat16
    page_size = PAGE_SIZE
    nhead = NHEAD
    nhead_kv = NHEAD_KV

    num_page = kv_buffer_fp8.shape[0]

    seq_lens_kv = torch.full((batch_size,), ctx_len, dtype=torch.int, device="cuda")
    kv_block_nums = torch.full((batch_size,), (ctx_len + page_size - 1) // page_size,
                               dtype=torch.int, device="cuda")
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int, device="cuda")
    if ctx_len % page_size != 0:
        kv_last_page_lens.fill_(ctx_len % page_size)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda")
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)

    seq_lens_qo = torch.full((batch_size,), decode_qlen, dtype=torch.int, device="cuda")
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda")
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = int(qo_indptr[-1].item())
    max_seqlen_qo = decode_qlen

    sm_scale = 1.0 / (QK_HEAD_DIM ** 0.5)

    # limit splits the same way the test does
    gpu = torch.cuda.current_device()
    cu_num = torch.cuda.get_device_properties(gpu).multi_processor_count
    max_split_per_batch = min((cu_num + batch_size - 1) // batch_size, max_split_per_batch)

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        batch_size, max_seqlen_qo, nhead, fp8, fp8,
        is_sparse=False, fast_mode=True,
        num_kv_splits=max_split_per_batch, intra_batch_mode=False,
    )

    work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device="cuda")
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device="cuda")
    work_info_set = torch.empty(work_info_set_size, dtype=work_info_set_type, device="cuda")
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device="cuda")
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device="cuda")
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        nhead // nhead_kv, nhead_kv, False,
        work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=fp8, dtype_kv=fp8,
    )

    num_partial = reduce_partial_map.size(0)
    logits = torch.empty((num_partial * max_seqlen_qo, 1, nhead, V_HEAD_DIM),
                         dtype=torch.float32, device="cuda")
    attn_lse = torch.empty((num_partial * max_seqlen_qo, 1, nhead, 1),
                           dtype=torch.float32, device="cuda")

    return {
        "fp8": fp8, "out_dtype": out_dtype, "page_size": page_size,
        "nhead": nhead, "nhead_kv": nhead_kv, "num_page": num_page,
        "kv_indptr": kv_indptr, "qo_indptr": qo_indptr,
        "kv_last_page_lens": kv_last_page_lens, "kv_indices": kv_indices,
        "total_q": total_q, "max_seqlen_qo": max_seqlen_qo, "sm_scale": sm_scale,
        "max_split_per_batch": max_split_per_batch,
        "work_meta_data": work_meta_data, "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr, "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "logits": logits, "attn_lse": attn_lse,
    }


class FlyDSL(ProviderAdapter):
    """flydsl_mla_fwd_decode (m16x8 fp8 stage1) + aiter.mla_reduce_v1 combine."""

    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "mla_decode":
            return False, "flydsl mla_decode adapter only implements mla_decode"
        if shape["dtype"] not in _OK_DTYPE:
            return False, (f"FlyDSL mla_fwd_decode_m16x8_fp8_fp8 is fp8 Q/KV only "
                           f"(num_heads==128 fp8 path); {shape['dtype']} unsupported")
        # mla_reduce_v1 (the split combine) comes from aiter
        try:
            from aiter.ops.attention import mla_reduce_v1  # noqa: F401
        except Exception as e:
            return False, f"aiter.ops.attention import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _ctx(self, shape, inputs):
        import torch

        batch_size, ctx_len, decode_qlen = _bc(shape)
        key = (batch_size, ctx_len, decode_qlen)
        if key in self._cache:
            return self._cache[key]

        common.bootstrap_env()
        from kernels.mla_fwd_decode import flydsl_mla_fwd_decode
        from aiter.ops.attention import mla_reduce_v1

        fp8 = _aiter_fp8()
        q_fp8 = inputs["q_fp32"].to(fp8).contiguous()
        kv_buffer_fp8 = inputs["kv_fp32"].to(fp8).contiguous()
        kv_indices = inputs["kv_indices"]

        md = _build_metadata(batch_size, ctx_len, decode_qlen, q_fp8,
                             kv_buffer_fp8, kv_indices)
        total_q = md["total_q"]
        nhead = md["nhead"]
        out_dtype = md["out_dtype"]
        num_page = md["num_page"]

        out_asm = torch.empty((total_q, nhead, V_HEAD_DIM), dtype=out_dtype, device="cuda")
        kv_view = kv_buffer_fp8.view(num_page, md["page_size"], md["nhead_kv"], QK_HEAD_DIM)

        ctx = {
            "flydsl_mla_fwd_decode": flydsl_mla_fwd_decode,
            "mla_reduce_v1": mla_reduce_v1,
            "q_fp8": q_fp8, "kv_view": kv_view, "out_asm": out_asm, "md": md,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"flydsl_mla_fwd_decode (m16x8 fp8/fp8, nhead=128, page_size=1) stage1 "
            f"-> aiter.mla_reduce_v1 split-combine; b={batch_size},ctx={ctx_len}; "
            f"max_split={md['max_split_per_batch']}; metadata built once (excluded)")
        return ctx

    def run(self, shape, inputs):
        ctx = self._ctx(shape, inputs)
        md = ctx["md"]
        ctx["flydsl_mla_fwd_decode"](
            ctx["q_fp8"], ctx["kv_view"], md["kv_indices"],
            md["work_indptr"], md["work_info_set"],
            ctx["out_asm"], md["logits"], md["attn_lse"], md["sm_scale"],
        )
        ctx["mla_reduce_v1"](
            md["logits"], md["attn_lse"], md["reduce_indptr"],
            md["reduce_final_map"], md["reduce_partial_map"],
            md["max_seqlen_qo"], ctx["out_asm"], None,
        )
        return ctx["out_asm"]

    def output(self, shape, inputs):
        return self.run(shape, inputs)


class Aiter(ProviderAdapter):
    """aiter.hk_mla_decode_fwd (work-map stage1) + aiter.mla_reduce_v1 combine."""

    name = "aiter"
    includes_allocation = False
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "mla_decode":
            return False, "aiter mla_decode adapter only implements mla_decode"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"aiter hk_mla_decode_fwd is fp8 Q/KV only, not {shape['dtype']}"
        try:
            from aiter.ops.attention import hk_mla_decode_fwd, mla_reduce_v1  # noqa: F401
        except Exception as e:
            return False, f"aiter.ops.attention import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def _ctx(self, shape, inputs):
        import torch
        from aiter.ops.attention import hk_mla_decode_fwd, mla_reduce_v1

        batch_size, ctx_len, decode_qlen = _bc(shape)
        key = (batch_size, ctx_len, decode_qlen)
        if key in self._cache:
            return self._cache[key]

        fp8 = _aiter_fp8()
        q_fp8 = inputs["q_fp32"].to(fp8).contiguous()
        kv_buffer_fp8 = inputs["kv_fp32"].to(fp8).contiguous()
        kv_indices = inputs["kv_indices"]

        md = _build_metadata(batch_size, ctx_len, decode_qlen, q_fp8,
                             kv_buffer_fp8, kv_indices)
        total_q = md["total_q"]
        nhead = md["nhead"]
        out_dtype = md["out_dtype"]
        num_page = md["num_page"]

        out = torch.empty((total_q, nhead, V_HEAD_DIM), dtype=out_dtype, device="cuda")
        hk_logits = torch.empty_like(md["logits"])
        hk_lse = torch.empty_like(md["attn_lse"])
        kv_view = kv_buffer_fp8.view(num_page, md["page_size"], md["nhead_kv"], QK_HEAD_DIM)

        ctx = {
            "hk_mla_decode_fwd": hk_mla_decode_fwd, "mla_reduce_v1": mla_reduce_v1,
            "q_fp8": q_fp8, "kv_view": kv_view, "out": out,
            "hk_logits": hk_logits, "hk_lse": hk_lse, "md": md,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"aiter.hk_mla_decode_fwd (work-map stage1, fp8/fp8) -> aiter.mla_reduce_v1; "
            f"b={batch_size},ctx={ctx_len}; metadata built once (excluded)")
        return ctx

    def run(self, shape, inputs):
        ctx = self._ctx(shape, inputs)
        md = ctx["md"]
        ctx["hk_mla_decode_fwd"](
            ctx["q_fp8"], ctx["kv_view"],
            md["qo_indptr"], md["kv_indptr"], md["kv_indices"],
            md["kv_last_page_lens"], md["work_indptr"], md["work_info_set"],
            md["max_seqlen_qo"], md["sm_scale"],
            ctx["hk_logits"], ctx["hk_lse"], ctx["out"],
        )
        ctx["mla_reduce_v1"](
            ctx["hk_logits"], ctx["hk_lse"], md["reduce_indptr"],
            md["reduce_final_map"], md["reduce_partial_map"],
            md["max_seqlen_qo"], ctx["out"], None,
        )
        return ctx["out"]

    def output(self, shape, inputs):
        return self.run(shape, inputs)


class AiterASM(ProviderAdapter):
    """aiter.mla_decode_stage1_asm_fwd (ASM stage1) + aiter.mla_reduce_v1 combine."""

    name = "aiter_asm"
    includes_allocation = False
    includes_layout_conversion = False

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "mla_decode":
            return False, "aiter_asm mla_decode adapter only implements mla_decode"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"aiter mla_decode_stage1_asm_fwd is fp8 Q/KV only, not {shape['dtype']}"
        try:
            from aiter.ops.attention import mla_decode_stage1_asm_fwd, mla_reduce_v1  # noqa: F401
        except Exception as e:
            return False, f"aiter.ops.attention import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        return True, None

    def _ctx(self, shape, inputs):
        import torch
        from aiter.ops.attention import mla_decode_stage1_asm_fwd, mla_reduce_v1

        batch_size, ctx_len, decode_qlen = _bc(shape)
        key = (batch_size, ctx_len, decode_qlen)
        if key in self._cache:
            return self._cache[key]

        fp8 = _aiter_fp8()
        q_fp8 = inputs["q_fp32"].to(fp8).contiguous()
        kv_buffer_fp8 = inputs["kv_fp32"].to(fp8).contiguous()
        kv_indices = inputs["kv_indices"]

        md = _build_metadata(batch_size, ctx_len, decode_qlen, q_fp8,
                             kv_buffer_fp8, kv_indices)
        total_q = md["total_q"]
        nhead = md["nhead"]
        out_dtype = md["out_dtype"]
        num_page = md["num_page"]

        out = torch.empty((total_q, nhead, V_HEAD_DIM), dtype=out_dtype, device="cuda")
        asm_logits = torch.empty_like(md["logits"])
        asm_lse = torch.empty_like(md["attn_lse"])
        q_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
        kv_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
        kv_view = kv_buffer_fp8.view(num_page, md["page_size"], md["nhead_kv"], QK_HEAD_DIM)

        ctx = {
            "mla_decode_stage1_asm_fwd": mla_decode_stage1_asm_fwd,
            "mla_reduce_v1": mla_reduce_v1,
            "q_fp8": q_fp8, "kv_view": kv_view, "out": out,
            "asm_logits": asm_logits, "asm_lse": asm_lse,
            "q_scale": q_scale, "kv_scale": kv_scale, "md": md,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"aiter.mla_decode_stage1_asm_fwd (ASM stage1, fp8/fp8) -> aiter.mla_reduce_v1; "
            f"b={batch_size},ctx={ctx_len}; q_scale=kv_scale=1; metadata built once (excluded)")
        return ctx

    def run(self, shape, inputs):
        ctx = self._ctx(shape, inputs)
        md = ctx["md"]
        ctx["mla_decode_stage1_asm_fwd"](
            ctx["q_fp8"], ctx["kv_view"],
            md["qo_indptr"], md["kv_indptr"], md["kv_indices"],
            md["kv_last_page_lens"], None,
            md["work_meta_data"], md["work_indptr"], md["work_info_set"],
            md["max_seqlen_qo"], md["page_size"], md["nhead_kv"], md["sm_scale"],
            ctx["asm_logits"], ctx["asm_lse"], ctx["out"],
            None, ctx["q_scale"], ctx["kv_scale"],
        )
        ctx["mla_reduce_v1"](
            ctx["asm_logits"], ctx["asm_lse"], md["reduce_indptr"],
            md["reduce_final_map"], md["reduce_partial_map"],
            md["max_seqlen_qo"], ctx["out"], None,
        )
        return ctx["out"]

    def output(self, shape, inputs):
        return self.run(shape, inputs)


class PyTorch(ProviderAdapter):
    """Eager paged-MLA reference (test's torch_mla_extend). Also the fp32 golden.

    Slow (einsum per sequence), but it is the only non-aiter, non-FlyDSL path and
    doubles as a sanity provider. Operates on the fp8-cast Q/KV upcast to float,
    matching ops.MlaDecodeOp.reference exactly."""

    name = "pytorch"
    provider_detail = "torch_mla_extend (eager paged MLA; fp8 data upcast to float)"
    includes_allocation = True

    def supports(self, shape):
        if shape.get("op_type") != "mla_decode":
            return False, "pytorch mla_decode adapter only implements mla_decode"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"this mla_decode reference is the fp8 path; {shape['dtype']} unsupported"
        return True, None

    def run(self, shape, inputs):
        from benchmarks import ops as _ops
        op = _ops.get_op("mla_decode")
        return op.reference(shape, inputs)

    def output(self, shape, inputs):
        return self.run(shape, inputs)


class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class AiterTriton(_Stub):
    name = "aiter_triton"
    _reason = ("no Triton MLA-decode stage1 in aiter.ops.triton wired for this fp8 "
               "nhead=128 page_size=1 work-map contract (decode MLA is ASM/HK only here)")


class Triton(_Stub):
    name = "triton"
    _reason = "no standalone (non-aiter) Triton MLA-decode kernel on this node"


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = ("CK MLA-decode backend not separately selectable from Python; the "
               "compiled stage1 paths are exposed as 'aiter' (HK) / 'aiter_asm' (ASM)")


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK MLA-decode adapter (reach CK via aiter, see aiter_ck)"


class Gluon(_Stub):
    name = "gluon"
    _reason = "no Gluon MLA-decode kernel on this node"


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is a dense-GEMM library (no attention / MLA-decode op)"
