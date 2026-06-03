"""Paged-Attn Decode (PS) provider adapters -- op_type "pa_decode".

FlyDSL kernel: kernels/pa_decode_fp8.py, persistent-scheduling (PS) paged decode.
Primary regression harness: FlyDSL-lab/tests/kernels/test_pa.py
(run_pa_decode_ps_test + torch_mha_extend). This adapter copies that test's input
construction and torch reference verbatim so the comparison is apples-to-apples.

Shape contract (recipe-fixed, matches the PS-only test harness):
  * query        [B*qlen, num_q_heads, head_size] bf16 (PS launcher casts q->fp8
                 internally with a unit query scale; query is NOT pre-quantized)
  * key_cache    paged fp8 (torch.uint8 packed), layout
                 [num_blocks, num_kv_heads, head_size//16, block_size, 16]
  * value_cache  paged fp8, [num_blocks, num_kv_heads, head_size, block_size];
                 with trans_v=True it is re-laid-out to a 5-D shuffled form
  * context_lengths / kv_page_indices / kv_indptr / block_tables  int32
  * key_scale/value_scale: per_token -> [num_blocks, num_kv_heads, block_size, 1]
                           per_tensor -> [1]  (PS launcher detects mode by ndim)
Configs (recipe): num_heads (8,1)/(16,1), head_size 128, ctx 1027, batch 3/81,
block_size 1024, qlen 1..4, quant per_token/per_tensor, compute fp8, gfx950.

Building the paged KV cache + quantization + PS metadata is done ONCE per shape
key OUTSIDE the timed region (includes_layout_conversion=True on the real
providers); run() only launches the kernel(s).

Providers:
  * flydsl  -> kernels.pa_decode_fp8.pa_decode_ps_launch (PS split-reduce path)
  * gluon   -> torch.ops.aiter.pa_decode_gluon(..., ps=True)   (recipe baseline)
  * pytorch -> torch_mha_extend (the fp32 reference itself, also the golden source)
  * aiter / aiter_triton / aiter_ck / aiter_asm / ck / triton / hipblaslt -> honest
    stubs (the only Python-selectable AITER PA-decode-PS path on this node is the
    Gluon op, exposed as the `gluon` provider).
"""

from __future__ import annotations

from benchmarks import common
from benchmarks.providers.base import ProviderAdapter

# This PS harness is fp8-compute / bf16-query only (matches test_pa.py).
_OK_DTYPE = {"fp8", "fp8_e4m3"}
_CONTEXT_PARTITION_SIZE = 256  # CONTEXT_PARTITION_SIZE_OPTIONS in test_pa.py
_UNIFORM_RANGE = (-1, 1)


def _dims(shape):
    a = shape["args"]
    return (int(a["num_q_heads"]), int(a["num_kv_heads"]), int(a["head_size"]),
            int(a["context_length"]), int(a["batch"]), int(a["query_length"]),
            int(a["block_size"]))


def _quant_mode(shape) -> str:
    return str(shape["args"].get("quant_mode", "per_token"))


def _sliding_window(shape) -> int:
    return int(shape["args"].get("sliding_window", 0))


class FlyDSL(ProviderAdapter):
    """kernels/pa_decode_fp8.py PS decode (get_pa_metadata + pa_decode_ps_launch).

    The PS metadata (worklist/reduce maps + partial buffers), the output buffer,
    and the per-token/per-tensor scale tensors are built ONCE per shape key
    (outside run()). run() launches the PS main kernel + pa_reduce_v1. The paged
    fp8 cache build + quantization happen in the shared make_inputs (untimed) ->
    includes_layout_conversion=True.
    """

    name = "flydsl"
    includes_allocation = False
    includes_jit = False
    includes_layout_conversion = True  # paged cache build + KV quant done untimed

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "pa_decode":
            return False, "flydsl pa_decode adapter only implements pa_decode"
        if shape["dtype"] not in _OK_DTYPE:
            return False, (f"FlyDSL PA-decode-PS is fp8-compute only "
                           f"(bf16 query, fp8 paged KV); {shape['dtype']} unsupported")
        nq, nkv, hs, _ctx, _b, _ql, bsz = _dims(shape)
        if nq % nkv != 0:
            return False, f"num_q_heads={nq} must be divisible by num_kv_heads={nkv}"
        if hs != 128:
            return False, f"PS harness validated head_size=128, not {hs}"
        if bsz != 1024:
            return False, f"PS harness validated block_size=1024, not {bsz}"
        ok, why = common.flydsl_runtime_ok()
        return (ok, why) if not ok else (True, None)

    def _ctx(self, shape, inputs):
        import torch

        nq, nkv, hs, ctx, b, ql, bsz = _dims(shape)
        key = (nq, nkv, hs, ctx, b, ql, bsz, _quant_mode(shape), _sliding_window(shape))
        if key in self._cache:
            return self._cache[key]

        common.bootstrap_env()
        from kernels.pa_decode_fp8 import (
            get_pa_metadata as flydsl_get_pa_metadata,
            get_sw_ps_max_context_partition_num,
            pa_decode_ps_launch as flydsl_ps_launch,
        )

        query = inputs["query"]
        quantized_keys = inputs["quantized_keys"]
        quantized_values = inputs["quantized_values_flydsl"]  # trans_v shuffled
        context_lengths = inputs["context_lengths"]
        kv_page_indices = inputs["kv_page_indices"]
        kv_indptr = inputs["kv_indptr"]
        block_tables = inputs["block_tables"]
        softmax_scale = inputs["softmax_scale"]
        # PS detects per-token (ndim>1) vs per-tensor (shape [1]) from scale ndim.
        ps_key_scale = inputs["key_scale_original"]
        ps_value_scale = inputs["value_scale_original"]
        sliding_window = _sliding_window(shape)

        # Build PS metadata ONCE (untimed). Pass the bf16 query (PS casts to fp8
        # internally with a unit query scale; matches the Gluon query path).
        ps_metadata = flydsl_get_pa_metadata(
            query, quantized_keys, context_lengths, kv_indptr, nq, nkv)
        max_context_partition_num = get_sw_ps_max_context_partition_num(
            sliding_window, _CONTEXT_PARTITION_SIZE, ql)

        out = torch.empty_like(inputs["reference_output"])

        ctx = {
            "launch": flydsl_ps_launch,
            "out": out,
            "query": query,
            "quantized_keys": quantized_keys,
            "quantized_values": quantized_values,
            "context_lengths": context_lengths,
            "kv_page_indices": kv_page_indices,
            "kv_indptr": kv_indptr,
            "block_tables": block_tables,
            "softmax_scale": softmax_scale,
            "ps_key_scale": ps_key_scale,
            "ps_value_scale": ps_value_scale,
            "ps_metadata": ps_metadata,
            "sliding_window": sliding_window,
            "max_context_partition_num": max_context_partition_num,
        }
        self._cache[key] = ctx
        self.provider_detail = (
            f"pa_decode_ps_launch (PS split-reduce); quant={_quant_mode(shape)}, "
            f"trans_v=True, qlen={ql}, heads=({nq},{nkv}), ctx={ctx_str(shape)}; "
            f"metadata(get_pa_metadata)+partials built once; paged fp8 cache+KV-quant "
            f"untimed (includes_layout_conversion)")
        return ctx

    def run(self, shape, inputs):
        c = self._ctx(shape, inputs)
        import torch

        c["launch"](
            c["out"],
            c["query"],
            c["quantized_keys"],
            c["quantized_values"],
            c["context_lengths"],
            c["kv_page_indices"],
            c["kv_indptr"],
            c["softmax_scale"],
            key_scale=c["ps_key_scale"],
            value_scale=c["ps_value_scale"],
            sliding_window=c["sliding_window"],
            metadata=c["ps_metadata"],
            block_tables=c["block_tables"],
            max_context_partition_num=c["max_context_partition_num"],
            stream=torch.cuda.current_stream(),
        )
        return c["out"]


def ctx_str(shape):
    _nq, _nkv, _hs, ctx, _b, _ql, _bsz = _dims(shape)
    return str(ctx)


class Gluon(ProviderAdapter):
    """AITER Gluon PA-decode-PS: torch.ops.aiter.pa_decode_gluon(..., ps=True).

    The recipe's named baseline. Uses the SAME quantized paged cache the FlyDSL
    path consumes; the PS intermediate buffers (exp_sums/max_logits/temporary_
    output) are sized + allocated ONCE per shape key (untimed). query_scale=None
    (bf16 query path), key/value scale = the per-token/per-tensor "original"
    (un-flattened) scale tensors that the Gluon op expects.
    """

    name = "gluon"
    includes_allocation = False
    includes_layout_conversion = True

    def __init__(self, op_type):
        super().__init__(op_type)
        self._cache = {}

    def supports(self, shape):
        if shape.get("op_type") != "pa_decode":
            return False, "gluon pa_decode adapter only implements pa_decode"
        if shape["dtype"] not in _OK_DTYPE:
            return False, f"Gluon PA-decode-PS fp8 path here; {shape['dtype']} unsupported"
        try:
            import aiter  # noqa: F401
            import triton  # noqa: F401
            from aiter.ops.triton.gluon.pa_decode_gluon import (  # noqa: F401
                get_recommended_splits,
            )
        except Exception as e:
            return False, f"import failed ({type(e).__name__}); launch via benchmarks/env.sh"
        if not hasattr(__import__("torch").ops.aiter, "pa_decode_gluon"):
            return False, "torch.ops.aiter.pa_decode_gluon not registered on this node"
        return True, None

    def _ctx(self, shape, inputs):
        import torch
        import triton
        import aiter
        from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits

        nq, nkv, hs, ctx, b, ql, bsz = _dims(shape)
        key = (nq, nkv, hs, ctx, b, ql, bsz, _quant_mode(shape), _sliding_window(shape))
        if key in self._cache:
            return self._cache[key]

        sliding_window = _sliding_window(shape)
        # max_context_partition_num: SW path uses the FlyDSL helper, else the
        # Gluon recommender (mirrors get_gluon_partition_count in test_pa.py).
        if sliding_window > 0:
            common.bootstrap_env()
            from kernels.pa_decode_fp8 import get_sw_ps_max_context_partition_num
            max_cpn = get_sw_ps_max_context_partition_num(
                sliding_window, _CONTEXT_PARTITION_SIZE, ql)
        else:
            split_kv_blocks = triton.cdiv(bsz, _CONTEXT_PARTITION_SIZE)
            max_cpn = get_recommended_splits(b, nkv, split_kv_blocks)

        eqgs = ql * (nq // nkv)
        ref = inputs["reference_output"]
        intermediate_shape = (b, nkv, max_cpn, eqgs)
        exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device="cuda")
        max_logits = torch.empty(intermediate_shape, dtype=torch.float32, device="cuda")
        temporary_output = torch.empty(*intermediate_shape, hs, dtype=ref.dtype, device="cuda")
        out = torch.empty_like(ref)

        ctx_d = {
            "out": out,
            "query": inputs["query"],
            "quantized_keys": inputs["quantized_keys"],
            "quantized_values": inputs["quantized_values_flydsl"],
            "context_lengths": inputs["context_lengths"],
            "block_tables": inputs["block_tables"],
            "softmax_scale": inputs["softmax_scale"],
            "query_length": ql,
            "max_cpn": max_cpn,
            "compute_type": aiter.dtypes.fp8,
            "key_scale_original": inputs["key_scale_original"],
            "value_scale_original": inputs["value_scale_original"],
            "exp_sums": exp_sums,
            "max_logits": max_logits,
            "temporary_output": temporary_output,
            "sliding_window": sliding_window,
        }
        self._cache[key] = ctx_d
        self.provider_detail = (
            f"torch.ops.aiter.pa_decode_gluon(ps=True); quant={_quant_mode(shape)}, "
            f"trans_v=True, max_context_partition_num={max_cpn}, "
            f"context_partition_size={_CONTEXT_PARTITION_SIZE}; intermediate buffers "
            f"built once (untimed)")
        return ctx_d

    def run(self, shape, inputs):
        import torch
        c = self._ctx(shape, inputs)
        torch.ops.aiter.pa_decode_gluon(
            c["out"],
            c["query"],
            c["quantized_keys"],
            c["quantized_values"],
            c["context_lengths"],
            c["block_tables"],
            c["softmax_scale"],
            c["query_length"],
            c["max_cpn"],
            _CONTEXT_PARTITION_SIZE,
            c["compute_type"],
            None,  # query_scale (bf16 query path)
            c["key_scale_original"],
            c["value_scale_original"],
            exp_sums=c["exp_sums"],
            max_logits=c["max_logits"],
            temporary_output=c["temporary_output"],
            alibi_slopes=None,
            sinks=None,
            sliding_window=c["sliding_window"],
            ps=True,
        )
        return c["out"]


class PyTorch(ProviderAdapter):
    """torch_mha_extend reference (the fp32 golden source, per test_pa.py).

    Slow eager gather+masked-attention over the paged cache; also the correctness
    reference. Runs in the SAME bits as the FlyDSL/Gluon paths (shared make_inputs).
    """

    name = "pytorch"
    provider_detail = "torch_mha_extend (eager paged masked attention; fp32 reference)"
    includes_allocation = True
    includes_layout_conversion = True

    def supports(self, shape):
        if shape.get("op_type") != "pa_decode":
            return False, "pytorch pa_decode adapter only implements pa_decode"
        return True, None

    def run(self, shape, inputs):
        # Recompute the reference (in the shared output dtype) from the SAME
        # paged cache the kernels consume. Uses the reference-layout (un-shuffled)
        # value cache + flat per-token/per-tensor scales.
        return _torch_mha_extend(
            inputs["query"],
            inputs["quantized_keys"],
            inputs["quantized_values_ref"],
            inputs["block_tables"],
            inputs["context_lengths"],
            inputs["query_output_indptr"],
            inputs["key_scale_flat"],
            inputs["value_scale_flat"],
            sliding_window=_sliding_window(shape),
        ).to(inputs["query"].dtype)


# --------------------------------------------------------------------------- #
# torch reference (copied verbatim from FlyDSL-lab/tests/kernels/test_pa.py:
# reference_masked_attention + torch_mha_extend) so the Op and the pytorch
# provider share one source of truth.
# --------------------------------------------------------------------------- #
def _reference_masked_attention(query, key, value, softmax_scale, output_dtype,
                                is_causal=True, sliding_window=0):
    import torch

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    s_q = query.shape[0]
    s_k = key.shape[0]
    key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=1)

    attention_weights = torch.einsum("qhd,khd->hqk", query, key) * softmax_scale

    if is_causal:
        query_len = query.shape[0]
        key_len = key.shape[0]
        attention_bias = torch.zeros(query_len, key_len, dtype=torch.float32, device=query.device)
        causal_mask = torch.ones(query_len, key_len, dtype=torch.bool, device=query.device).tril(
            diagonal=key_len - query_len)
        attention_bias.masked_fill_(causal_mask.logical_not(), float(-3.4e38))
        attention_weights += attention_bias

    if s_q == s_k:
        query_positions = torch.arange(s_q, device=query.device)
        key_positions = torch.arange(s_k, device=query.device)
    else:
        query_positions = torch.arange(s_k - s_q, s_k, device=query.device)
        key_positions = torch.arange(s_k, device=query.device)

    pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)

    window_mask = torch.ones_like(attention_weights, dtype=torch.bool)
    if sliding_window > 0:
        sliding_window_mask = pos_diff >= sliding_window + 1
        window_mask &= sliding_window_mask
        attention_weights.masked_fill_(window_mask, float("-inf"))

    attention_weights = torch.softmax(attention_weights, dim=-1)
    output = torch.einsum("hqk,khd->qhd", attention_weights, value)
    return output.to(output_dtype)


def _torch_mha_extend(query, key_cache, value_cache, block_tables, context_lengths,
                      query_output_indptr, key_scale=None, value_scale=None,
                      sliding_window=0):
    import torch

    num_blocks, num_heads, head_size, block_size = value_cache.shape
    softmax_scale = 1.0 / (head_size ** 0.5)

    output_dtype = query.dtype
    kv_dtype = key_cache.dtype

    queries_split = torch.tensor_split(query, query_output_indptr.tolist()[1:])
    key_cache_flat = key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_size)
    value_cache_flat = value_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_size)

    batch_size = query_output_indptr.shape[0] - 1
    outputs = []

    for batch_idx in range(batch_size):
        current_query = queries_split[batch_idx]
        current_block_table = block_tables[batch_idx]
        current_context_length = context_lengths[batch_idx].item()

        token_indices = (
            current_block_table.repeat_interleave(block_size)[:current_context_length] * block_size
            + torch.arange(current_context_length, device=current_block_table.device) % block_size
        )

        gathered_keys = key_cache_flat.view(torch.int8)[token_indices].view(kv_dtype).to(torch.float)
        if key_scale is not None:
            gathered_keys *= key_scale[:, token_indices].t().unsqueeze(-1)

        gathered_values = value_cache_flat.view(torch.int8)[token_indices].view(kv_dtype).to(torch.float)
        if value_scale is not None:
            gathered_values *= value_scale[:, token_indices].t().unsqueeze(-1)

        attention_output = _reference_masked_attention(
            current_query, gathered_keys, gathered_values, softmax_scale,
            output_dtype, is_causal=True, sliding_window=sliding_window)
        outputs.append(attention_output)

    return torch.cat(outputs)


# --------------------------------------------------------------------------- #
# Stubs for every other provider name in the enum.
# --------------------------------------------------------------------------- #
class _Stub(ProviderAdapter):
    _reason = "not available"

    def supports(self, shape):
        return False, self._reason


class Aiter(_Stub):
    name = "aiter"
    _reason = ("no compiled (CK/ASM) AITER paged-attn-decode-PS op selectable from Python; "
               "the only AITER PA-decode-PS path on this node is the Gluon Triton op "
               "(torch.ops.aiter.pa_decode_gluon) -- see the `gluon` provider")


class AiterTriton(_Stub):
    name = "aiter_triton"
    _reason = ("AITER's Triton paged-decode is the Gluon kernel (uses triton.experimental.gluon); "
               "exposed as the `gluon` provider, not a plain Triton aiter op")


class AiterCK(_Stub):
    name = "aiter_ck"
    _reason = "no separately-selectable CK paged-attn-decode-PS from Python on this node"


class AiterASM(_Stub):
    name = "aiter_asm"
    _reason = "no Python-selectable ASM paged-attn-decode-PS path in AITER"


class CK(_Stub):
    name = "ck"
    _reason = "no standalone CK paged-attn-decode adapter on this node"


class Triton(_Stub):
    name = "triton"
    _reason = ("no standalone (non-aiter) Triton paged-attn-decode-PS kernel; the only "
               "Triton PA-decode-PS is aiter's Gluon op (see `gluon`)")


class HipBLASLt(_Stub):
    name = "hipblaslt"
    _reason = "hipBLASLt is a GEMM library (no paged-attention op)"
