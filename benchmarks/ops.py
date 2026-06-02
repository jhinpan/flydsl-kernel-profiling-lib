"""Per-op canonical inputs, fp32 reference, and roofline model.

This is the single source of truth so that (a) every provider receives the
EXACT same input tensors for a shape (fair comparison), (b) correctness is
judged against one fp32 golden, and (c) effective GB/s / TFLOP/s use one
bytes/flops model. Providers only implement supports()+run(); they never build
their own inputs.

Adding a new op = add one Op subclass + register it.
"""

from __future__ import annotations

from benchmarks import common


def _elem_bytes(dtype: str) -> float:
    # quantized dtypes need real byte widths or effective_gbps is wrong for
    # int8/fp4/mixed GEMM+MoE shapes. fp4 = 0.5 B, a8w4 ~ 0.75 B.
    return {"fp32": 4, "f32": 4, "float32": 4,
            "fp16": 2, "f16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
            "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "int8": 1, "i8": 1,
            "fp4": 0.5, "f4": 0.5, "mixed_a8w4": 0.75}.get(dtype, 2.0)


def _safe_torch_dtype(name: str):
    """torch dtype for a ledger dtype, or None for quant dtypes torch can't map
    directly (int8/fp4/mixed) -- providers handle those per-dtype themselves."""
    try:
        return common.torch_dtype(name)
    except KeyError:
        return None


class Op:
    op_type = "base"

    def make_inputs(self, shape: dict, seed: int) -> dict:
        raise NotImplementedError

    def reference(self, shape: dict, inputs: dict):
        """fp32 golden output tensor."""
        raise NotImplementedError

    def bytes_moved(self, shape: dict) -> int:
        return 0

    def flops(self, shape: dict) -> int:
        return 0

    def args_summary(self, shape: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in shape.get("args", {}).items())

    def tolerance(self, shape: dict) -> tuple[float, float]:
        """(rtol, atol) for correctness vs the fp32 reference. Default = dtype-based;
        ops that accumulate (GEMM/MoE over K) override with a looser tol."""
        return common.tol_for(shape.get("dtype", "bf16"))

    def effective(self, shape: dict, median_us: float | None) -> dict:
        if not median_us or median_us <= 0:
            return {"effective_gbps": None, "effective_tflops": None}
        sec = median_us * 1e-6
        b = self.bytes_moved(shape)
        f = self.flops(shape)
        return {
            "effective_gbps": (b / sec / 1e9) if b else None,
            "effective_tflops": (f / sec / 1e12) if f else None,
        }


class RmsNormOp(Op):
    op_type = "rmsnorm"

    def _MN(self, shape: dict) -> tuple[int, int]:
        a = shape["args"]
        return int(a["M"]), int(a["N"])

    def _eps(self, shape: dict) -> float:
        return float(shape.get("args", {}).get("eps", 1e-5))

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        M, N = self._MN(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32 then cast -> identical bits for every provider
        x = torch.randn((M, N), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        w = torch.rand((N,), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        return {"x": x, "weight": w, "eps": self._eps(shape), "M": M, "N": N, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        x, w, eps = inputs["x"], inputs["weight"], inputs["eps"]
        N = inputs["N"]
        return torch.nn.functional.rms_norm(x.float(), (N,), w.float(), eps=eps)

    def bytes_moved(self, shape: dict) -> int:
        M, N = self._MN(shape)
        e = _elem_bytes(shape["dtype"])
        # read x (M*N) + read weight (N) + write out (M*N)
        return (2 * M * N + N) * e

    def flops(self, shape: dict) -> int:
        M, N = self._MN(shape)
        # x^2 (1) + add to acc (1) + rsqrt(~) + mul rstd (1) + mul weight (1) ~ 4*M*N
        return 4 * M * N

    def args_summary(self, shape: dict) -> str:
        M, N = self._MN(shape)
        return f"M={M},N={N}"


class LayerNormOp(Op):
    op_type = "layernorm"

    def _MN(self, shape):
        a = shape["args"]
        return int(a["M"]), int(a["N"])

    def _eps(self, shape):
        return float(shape.get("args", {}).get("eps", 1e-5))

    def make_inputs(self, shape, seed):
        import torch

        M, N = self._MN(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        x = torch.randn((M, N), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        gamma = torch.rand((N,), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        beta = torch.rand((N,), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        return {"x": x, "gamma": gamma, "beta": beta, "eps": self._eps(shape), "M": M, "N": N, "dtype": td}

    def reference(self, shape, inputs):
        import torch

        return torch.nn.functional.layer_norm(inputs["x"].float(), (inputs["N"],),
                                              inputs["gamma"].float(), inputs["beta"].float(), eps=inputs["eps"])

    def bytes_moved(self, shape):
        M, N = self._MN(shape)
        return (2 * M * N + 2 * N) * _elem_bytes(shape["dtype"])

    def flops(self, shape):
        M, N = self._MN(shape)
        return 6 * M * N

    def args_summary(self, shape):
        M, N = self._MN(shape)
        return f"M={M},N={N}"


class SoftmaxOp(Op):
    op_type = "softmax"

    def _MN(self, shape):
        a = shape["args"]
        return int(a["M"]), int(a["N"])

    def make_inputs(self, shape, seed):
        import torch

        M, N = self._MN(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        x = torch.randn((M, N), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        return {"x": x, "M": M, "N": N, "dtype": td}

    def reference(self, shape, inputs):
        import torch

        return torch.softmax(inputs["x"].float(), dim=-1)

    def bytes_moved(self, shape):
        M, N = self._MN(shape)
        return 2 * M * N * _elem_bytes(shape["dtype"])

    def flops(self, shape):
        M, N = self._MN(shape)
        return 5 * M * N

    def args_summary(self, shape):
        M, N = self._MN(shape)
        return f"M={M},N={N}"


class GemmOp(Op):
    op_type = "gemm"

    def _MNK(self, shape: dict) -> tuple[int, int, int]:
        a = shape["args"]
        return int(a["M"]), int(a["N"]), int(a["K"])

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        M, N, K = self._MNK(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32, uniform(-1,1) (matches the example test), cast -> identical
        # bits for every provider. a:(M,K), b:(N,K) stored transposed; C = A @ B^T.
        a = torch.empty((M, K), device="cuda", dtype=torch.float32).uniform_(-1, 1, generator=g).to(td).contiguous()
        b = torch.empty((N, K), device="cuda", dtype=torch.float32).uniform_(-1, 1, generator=g).to(td).contiguous()
        return {"a": a, "b": b, "M": M, "N": N, "K": K, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        # fp32 golden, exactly the example's run_torch_acc: C = A @ B^T
        return torch.mm(inputs["a"].float(), inputs["b"].float().T)

    def bytes_moved(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        e = _elem_bytes(shape["dtype"])
        # read A (M*K) + read B (N*K) + write C (M*N); single-write GEMM convention
        # (SPLIT_K>1 adds zero-init + atomic partials, not counted -- matches example).
        return (M * K + N * K + M * N) * e

    def flops(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        return 2 * M * N * K

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # GEMM accumulates over K (cancellation near zero); match FlyDSL's own
        # hgemm_splitk test bar, looser for fp8.
        return (0.2, 0.2) if shape.get("dtype", "bf16").startswith("fp8") else (0.1, 0.1)

    def args_summary(self, shape: dict) -> str:
        M, N, K = self._MNK(shape)
        return f"M={M},N={N},K={K}"

class FusedRopeCacheOp(Op):
    op_type = "fused_rope_cache"

    def _dims(self, shape):
        a = shape["args"]
        return (int(a["head_dim"]), int(a["num_heads"]), int(a["num_kv_heads"]),
                int(a["seq_len"]))

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        a = shape["args"]
        D, QH, KH, T = self._dims(shape)
        bs = int(a.get("block_size", 16))
        reuse = bool(a.get("reuse_freqs_front_part", True))
        max_pos = int(a.get("max_pos", 8192))
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        dev = "cuda"

        cols = (D // 2) if reuse else D
        num_blocks = max(32, (T + bs - 1) // bs + 4)

        # All tensors built ONCE in FlyDSL/reference layout (i32 pos/slot, 2-D
        # cos/sin, flash KV caches, fp32 scales=ones) and shared across providers.
        # aiter_triton converts to its layout (i64, 4-D) inside its own run().
        Q = torch.randn((T, QH, D), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        K = torch.randn((T, KH, D), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        V = torch.randn((T, KH, D), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        CosCache = torch.randn((max_pos, cols), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        SinCache = torch.randn((max_pos, cols), device=dev, dtype=torch.float32, generator=g).to(td).contiguous()
        Positions = torch.randint(0, max_pos, (T,), device=dev, dtype=torch.int32, generator=g)
        SlotMapping = torch.arange(T, device=dev, dtype=torch.int32)
        KeyCache = torch.zeros((num_blocks, bs, KH, D), device=dev, dtype=td)
        ValueCache = torch.zeros((num_blocks, bs, KH, D), device=dev, dtype=td)
        # KScale/VScale ALWAYS required by the launcher even with apply_scale=False.
        KScale = torch.ones(1, device=dev, dtype=torch.float32)
        VScale = torch.ones(1, device=dev, dtype=torch.float32)

        return {"Q": Q, "K": K, "V": V, "CosCache": CosCache, "SinCache": SinCache,
                "Positions": Positions, "SlotMapping": SlotMapping,
                "KeyCache": KeyCache, "ValueCache": ValueCache,
                "KScale": KScale, "VScale": VScale,
                "T": T, "D": D, "QH": QH, "KH": KH, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        # NeoX RoPE in native dtype (matches HW rounding), then upcast. Comparable
        # tensor = concat([q_out, k_out] flattened to [T, -1]); providers' output()
        # assemble the same way.
        Q, K = inputs["Q"], inputs["K"]
        reuse = bool(shape["args"].get("reuse_freqs_front_part", True))
        cos = inputs["CosCache"][inputs["Positions"].long()].unsqueeze(1).to(Q.dtype)
        sin = inputs["SinCache"][inputs["Positions"].long()].unsqueeze(1).to(Q.dtype)
        if reuse:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        Dh = Q.shape[-1] // 2

        def rot(X):
            x1, x2 = X[..., :Dh], X[..., Dh:]
            return torch.cat([x1 * cos[..., :Dh] - x2 * sin[..., :Dh],
                              x2 * cos[..., Dh:] + x1 * sin[..., Dh:]], dim=-1)

        q_out, k_out = rot(Q), rot(K)
        return torch.cat([q_out.reshape(q_out.shape[0], -1),
                          k_out.reshape(k_out.shape[0], -1)], dim=-1).float()

    def bytes_moved(self, shape: dict) -> int:
        D, QH, KH, T = self._dims(shape)
        reuse = bool(shape["args"].get("reuse_freqs_front_part", True))
        cols = (D // 2) if reuse else D
        e = _elem_bytes(shape["dtype"])
        # Q read+write (2*QH*D) + K read/K_out/KeyCache (3*KH*D) + V read/ValueCache
        # (2*KH*D) + cos+sin reads (2*cols), per token *e; plus i32 pos+slot (8B/tok).
        return e * T * (2 * QH * D + 5 * KH * D + 2 * cols) + 8 * T

    def flops(self, shape: dict) -> int:
        D, QH, KH, T = self._dims(shape)
        # NeoX rotation: 2 mul + 1 add per rotated element; ~negligible (single wave).
        return 3 * T * (QH + KH) * D

    def args_summary(self, shape: dict) -> str:
        D, QH, KH, T = self._dims(shape)
        return f"D={D},QH={QH},KH={KH},T={T}"

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # RoPE is a rotation: outputs contain near-zero elements (q1*cos - q2*sin
        # can cancel), where relative tol is meaningless. Use an atol-dominated
        # bound so a correct kernel with near-zero outputs isn't false-failed
        # (e.g. aiter_triton f16, which otherwise gets excluded and inflates vs-best).
        return (2e-2, 2e-2) if shape.get("dtype", "bf16") in ("bf16", "bfloat16") else (1e-2, 1e-2)


class MoeGemmOp(Op):
    """Full 2-stage MoE (gate+up silu, then down + topk reduce).

    Ledger args: M=tokens, E=experts, Dim1=model_dim, Dim2=2*inter_dim
    (inter_dim=Dim2//2), TopK=topk. Inputs are built ONCE in fp32 and shared
    across providers; the per-provider quantize/cast/preshuffle happens inside
    run(). The fp32 reference is the standard torch MoE and matches both the
    FlyDSL composed output (moe_gemm1 -> requant -> moe_gemm2, routed weight in
    stage2) and aiter.fused_moe (doweight_stage1=False).
    """

    op_type = "moe_gemm"

    def _dims(self, shape: dict):
        a = shape["args"]
        tokens = int(a["M"])
        experts = int(a["E"])
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        topk = int(a["TopK"])
        return tokens, experts, model_dim, inter_dim, topk

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        g = common.make_generator(seed)
        # build in fp32 once -> every provider gets identical bits, then quantizes/casts itself
        x_fp32 = (torch.randn((tokens, model_dim), device="cuda", dtype=torch.float32, generator=g) * 0.2).contiguous()
        w1_fp32 = (torch.randn((experts, 2 * inter_dim, model_dim), device="cuda", dtype=torch.float32, generator=g) * 0.2).contiguous()
        w2_fp32 = (torch.randn((experts, model_dim, inter_dim), device="cuda", dtype=torch.float32, generator=g)
                   * (0.2 / (inter_dim ** 0.5))).contiguous()
        score = torch.rand((tokens, experts), device="cuda", dtype=torch.float32, generator=g)
        topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32).contiguous()
        return {
            "x_fp32": x_fp32, "w1_fp32": w1_fp32, "w2_fp32": w2_fp32,
            "topk_ids": topk_ids.to(torch.int32).contiguous(), "topk_weights": topk_weights,
            "M": tokens, "E": experts, "model_dim": model_dim, "inter_dim": inter_dim, "topk": topk,
            # inputs are fp32 + dtype-agnostic; providers quantize per-dtype themselves.
            # map only dtypes torch knows so make_inputs never KeyErrors on int8/fp4/mixed
            # (those become per-provider unsupported, not a shared shape-level failure).
            "dtype_str": shape["dtype"],
            "dtype": _safe_torch_dtype(shape["dtype"]),
        }

    def reference(self, shape: dict, inputs: dict):
        import torch
        import torch.nn.functional as F

        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        x = inputs["x_fp32"]
        w1 = inputs["w1_fp32"]
        w2 = inputs["w2_fp32"]
        topk_ids = inputs["topk_ids"].long()
        topk_w = inputs["topk_weights"]
        out = torch.zeros((tokens, model_dim), device="cuda", dtype=torch.float32)
        for e in range(experts):
            idx = (topk_ids == e).nonzero(as_tuple=False)  # [num, 2] (token, slot)
            if idx.numel() == 0:
                continue
            t_idx = idx[:, 0]
            s_idx = idx[:, 1]
            y2 = F.linear(x[t_idx, :], w1[e])          # [num, 2*inter_dim]
            gate = y2[:, :inter_dim]
            up = y2[:, inter_dim:]
            h = F.silu(gate) * up                      # [num, inter_dim]
            y = F.linear(h, w2[e])                     # [num, model_dim]
            y = y * topk_w[t_idx, s_idx].unsqueeze(-1)  # routed weight (applied in stage2)
            out.index_add_(0, t_idx, y)
        return out

    def bytes_moved(self, shape: dict) -> int:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        eb = _elem_bytes(shape["dtype"])
        e_active = min(experts, tokens * topk)  # weights touched = active experts only
        x_bytes = tokens * model_dim * eb
        w1_bytes = e_active * (2 * inter_dim) * model_dim * eb
        w2_bytes = e_active * model_dim * inter_dim * eb
        out_bytes = tokens * model_dim * 2  # final [tokens, model_dim] f16/bf16
        return x_bytes + w1_bytes + w2_bytes + out_bytes

    def flops(self, shape: dict) -> int:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        # stage1 2*M*T*(2I)*K + stage2 2*M*T*K*I = 6*M*T*K*I (silu negligible)
        return 6 * tokens * topk * model_dim * inter_dim

    def args_summary(self, shape: dict) -> str:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        return f"tokens={tokens},E={experts},model_dim={model_dim},inter_dim={inter_dim},topk={topk}"


# add to _REGISTRY dict:


_REGISTRY: dict[str, Op] = {
    RmsNormOp.op_type: RmsNormOp(),
    LayerNormOp.op_type: LayerNormOp(),
    SoftmaxOp.op_type: SoftmaxOp(),
    GemmOp.op_type: GemmOp(),
    FusedRopeCacheOp.op_type: FusedRopeCacheOp(),
    MoeGemmOp.op_type: MoeGemmOp(),
}


def get_op(op_type: str) -> Op | None:
    return _REGISTRY.get(op_type)


def register(op: Op) -> None:
    _REGISTRY[op.op_type] = op



# === campaign-expansion ops (auto-merged 2026-06-02) ===


class VecAddOp(Op):
    op_type = "vec_add"

    def _n(self, shape: dict) -> int:
        return int(shape["args"]["n"])

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        n = self._n(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32 then cast -> identical bits for every provider (this op is
        # fp32-only so the cast is a no-op, but keep the canonical convention).
        a = torch.randn((n,), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        b = torch.randn((n,), device="cuda", dtype=torch.float32, generator=g).to(td).contiguous()
        return {"a": a, "b": b, "n": n, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        # fp32 golden C = A + B
        return torch.add(inputs["a"].float(), inputs["b"].float())

    def bytes_moved(self, shape: dict) -> int:
        n = self._n(shape)
        # read A (n) + read B (n) + write C (n); this is a pure-bandwidth kernel
        return 3 * n * _elem_bytes(shape["dtype"])

    def flops(self, shape: dict) -> int:
        n = self._n(shape)
        # one add per element (negligible vs the bandwidth)
        return n

    def args_summary(self, shape: dict) -> str:
        return f"n={self._n(shape)}"


class QuantOp(Op):
    """Per-token (per-row) symmetric int8 quant of x (M,N) -> (q:int8, scale:fp32[M]).

    Inputs are built in fp32 then cast to the ledger dtype so every provider
    quantizes the EXACT same bits. The kernel judged here is the FlyDSL f16->int8
    path, so the ledger dtype is fp16. The fp32 golden is the *dequantized* row
    (q.float()*scale) of the cast input -- i.e. the best symmetric-int8 round-trip
    of the f16 input -- so providers are compared on dequant error, robust to
    +-1 ULP rounding differences in q (FlyDSL truncates via fptosi, torch rounds).
    Args M,N.
    """

    op_type = "quant"

    def _MN(self, shape: dict) -> tuple[int, int]:
        a = shape["args"]
        return int(a["M"]), int(a["N"])

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        M, N = self._MN(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32 (uniform(-5,5) like the FlyDSL test) then cast -> identical
        # bits for every provider; the cast x is what every provider quantizes.
        x = torch.empty((M, N), device="cuda", dtype=torch.float32).uniform_(-5.0, 5.0, generator=g).to(td).contiguous()
        return {"x": x, "M": M, "N": N, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        # fp32 dequant golden: per-token amax/127 scale (zero-row -> 1.0), round,
        # clamp to [-127,127], then dequant = q*scale. Computed from the cast input.
        x = inputs["x"].float()
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = amax / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(x / scale), -127, 127)
        return q * scale

    def bytes_moved(self, shape: dict) -> int:
        M, N = self._MN(shape)
        # read x (M*N * in_dtype) + write q int8 (M*N * 1) + write scales fp32 (M*4)
        return int(M * N * _elem_bytes(shape["dtype"]) + M * N * 1 + M * 4)

    def flops(self, shape: dict) -> int:
        M, N = self._MN(shape)
        # pass1 abs+max (~2N), pass2 mul-by-inv-scale + round-to-int8 (~2N) per row
        return 4 * M * N

    def args_summary(self, shape: dict) -> str:
        M, N = self._MN(shape)
        return f"M={M},N={N}"

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # int8 quant is lossy by construction: dequant differs from the fp32 input
        # by up to ~0.5*scale per element, and FlyDSL truncates (fptosi) where the
        # torch golden rounds -> up to ~1*scale (~amax/127) of dequant gap. With
        # amax<=5 here that is ~0.04; use an atol-dominated bound with headroom.
        return (5e-2, 6e-2)


class TopkGatingSoftmaxOp(Op):
    """Fused softmax + top-K expert selection + renormalization for MoE gating.

    Ledger args: num_tokens, num_experts, topk. The gating logits are built ONCE
    in fp32 (uniform(-2, 2), matching tests/kernels/test_topk_gating_softmax.py)
    then cast to the row dtype, so every provider sees identical bytes -- this
    matters because the K-th expert can flip at bf16/f16 precision boundaries.

    The comparable output is the SORTED-descending topk_weights [num_tokens, topk]
    (renormalized to sum to 1 over the K selected experts). Indices are NOT
    compared: implementations may legitimately disagree on which expert to take
    when several share the K-th-largest probability. The fp32 reference is
    softmax(dim=1) -> topk -> renorm -> sort.
    """

    op_type = "topk_gating_softmax"

    def _NEK(self, shape: dict) -> tuple[int, int, int]:
        a = shape["args"]
        return int(a["num_tokens"]), int(a["num_experts"]), int(a["topk"])

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        num_tokens, num_experts, topk = self._NEK(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32, uniform(-2, 2) (matches the example test), then cast so
        # the reference sees the exact bytes the kernel sees.
        gating_fp32 = ((torch.rand((num_tokens, num_experts), device="cuda",
                                   dtype=torch.float32, generator=g) * 4.0) - 2.0).contiguous()
        gating = gating_fp32.to(td).contiguous()
        return {"gating": gating, "gating_fp32": gating.float(),
                "num_tokens": num_tokens, "num_experts": num_experts, "topk": topk,
                "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        num_tokens, num_experts, topk = self._NEK(shape)
        # reference sees the quantized-then-upcast bytes (gating_fp32 = cast.float())
        probs = torch.softmax(inputs["gating_fp32"], dim=1)
        w, _idx = torch.topk(probs, topk, dim=1)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-20)
        s, _ = w.float().sort(dim=1, descending=True)
        return s

    def bytes_moved(self, shape: dict) -> int:
        num_tokens, num_experts, topk = self._NEK(shape)
        e = _elem_bytes(shape["dtype"])
        # read gating [N, E] + write topk_weights f32 + topk_indices i32 +
        # token_expert_indices i32 (each [N, topk], 4 B/elem)
        return int(num_tokens * num_experts * e + num_tokens * topk * (4 + 4 + 4))

    def flops(self, shape: dict) -> int:
        num_tokens, num_experts, topk = self._NEK(shape)
        # softmax ~5 ops/elem over E + iterative top-K (~topk passes over E) + renorm
        return int(num_tokens * (5 * num_experts + topk * num_experts + 2 * topk))

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # The kernel computes softmax/topk in fp32 internally; the only error is
        # the input quantization (matched by gating_fp32) plus exp2 vs exp. The
        # example test uses atol=2e-2 for bf16/f16 sorted-weight comparison; f32
        # is exact to ~1e-5. Use atol-dominated bounds (renormalized weights are
        # O(1/topk), small, so relative tol is harsh).
        dt = shape.get("dtype", "bf16")
        if dt in ("fp32", "f32", "float32"):
            return (1e-4, 1e-5)
        return (2e-2, 2e-2)

    def args_summary(self, shape: dict) -> str:
        num_tokens, num_experts, topk = self._NEK(shape)
        return f"num_tokens={num_tokens},num_experts={num_experts},topk={topk}"


class MoeReduceOp(Op):
    """MoE stage-2 reduction: sum X [tokens, topk, model_dim] over the topk dim.

    Ledger args: tokens, topk, model_dim, optional use_mask (default False).
    Bandwidth-bound. Inputs are built ONCE in fp32 then cast so every provider
    gets identical bits; the fp32 reference is the masked sum over topk. The
    mask (when use_mask) is a deterministic uint8 [tokens, topk]; providers that
    don't take a mask just receive the empty-(0,topk) sentinel.
    """

    op_type = "moe_reduce"

    def _dims(self, shape: dict) -> tuple[int, int, int]:
        a = shape["args"]
        return int(a["tokens"]), int(a["topk"]), int(a["model_dim"])

    def _use_mask(self, shape: dict) -> bool:
        return bool(shape.get("args", {}).get("use_mask", False))

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        tokens, topk, model_dim = self._dims(shape)
        use_mask = self._use_mask(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32 then cast -> identical bits for every provider
        x = torch.randn((tokens, topk, model_dim), device="cuda",
                        dtype=torch.float32, generator=g).to(td).contiguous()
        if use_mask:
            mask = torch.randint(0, 2, (tokens, topk), device="cuda",
                                 dtype=torch.uint8, generator=g).contiguous()
        else:
            # FlyDSL launcher always takes a mask arg; pass the empty sentinel the
            # test uses so the kernel's use_mask=False path ignores it.
            mask = torch.empty((0, topk), device="cuda", dtype=torch.uint8)
        return {"x": x, "mask": mask, "use_mask": use_mask,
                "tokens": tokens, "topk": topk, "model_dim": model_dim, "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch

        x = inputs["x"].float()
        if inputs["use_mask"]:
            x = x * inputs["mask"].to(torch.bool).unsqueeze(-1)
        return torch.sum(x, dim=1)

    def bytes_moved(self, shape: dict) -> int:
        tokens, topk, model_dim = self._dims(shape)
        e = _elem_bytes(shape["dtype"])
        # read X (tokens*topk*model_dim) + write Y (tokens*model_dim); the uint8
        # mask (tokens*topk) is negligible vs the model_dim-wide payload.
        rw = (tokens * topk * model_dim + tokens * model_dim) * e
        return rw + (tokens * topk if self._use_mask(shape) else 0)

    def flops(self, shape: dict) -> int:
        tokens, topk, model_dim = self._dims(shape)
        # (topk-1) adds per output element; pure reduction, ~negligible vs bytes.
        return tokens * model_dim * max(topk - 1, 0)

    def args_summary(self, shape: dict) -> str:
        tokens, topk, model_dim = self._dims(shape)
        m = ",mask" if self._use_mask(shape) else ""
        return f"tokens={tokens},topk={topk},model_dim={model_dim}{m}"

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # Accumulates topk terms in f32 then truncates to the io dtype; the test
        # harness uses (1e-2, 1e-2). Keep f32 tight, f16/bf16 at the test bar.
        d = shape.get("dtype", "bf16")
        if d in ("fp32", "f32", "float32"):
            return (1e-5, 1e-6)
        return (1e-2, 1e-2)


class PreshuffleGemmOp(Op):
    """Preshuffle a8w8 (fp8) GEMM: C = (A*scale_a) @ (B*scale_b)^T.

    Inputs are built ONCE in fp32, then per-token-quantized to fp8 (e4m3) so
    every provider gets identical quantized bits (the FlyDSL kernel and the aiter
    a8w8-bpreshuffle path both read the SAME a_q/b_q + fp32 scales; B is
    preshuffled per-provider inside its own build cache, outside the timed run).
    The fp32 reference DEQUANTIZES those quantized bits -- C = (a_q*scale_a) @
    (b_q*scale_b)^T -- exactly matching the example test's run_torch, so the
    golden reflects the quantization the kernels actually see.
    """

    op_type = "preshuffle_gemm"

    def _MNK(self, shape: dict) -> tuple[int, int, int]:
        a = shape["args"]
        return int(a["M"]), int(a["N"]), int(a["K"])

    def _fp8_dtype(self):
        import torch
        # gfx950 OCP fp8 (e4m3fn). The kernel/test use float8_e4m3fn on gfx95*;
        # common.torch_dtype maps "fp8" to the MI300 _fnuz spelling, so pick the
        # gfx950 dtype here explicitly to match compile_preshuffle_gemm_a8.
        return torch.float8_e4m3fn

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch
        from tests.utils import pertoken_quant

        M, N, K = self._MNK(shape)
        g = common.make_generator(seed)
        fp8 = self._fp8_dtype()
        # build in fp32 (uniform[0,1), matches the example test's torch.rand),
        # then per-token quantize ONCE -> identical bits for every provider.
        a_fp32 = torch.rand((M, K), device="cuda", dtype=torch.float32, generator=g).contiguous()
        b_fp32 = torch.rand((N, K), device="cuda", dtype=torch.float32, generator=g).contiguous()
        a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=fp8)
        b_q, scale_b = pertoken_quant(b_fp32, quant_dtype=fp8)
        a_q = a_q.contiguous()
        b_q = b_q.contiguous()
        scale_a = scale_a.contiguous().view(-1).to(torch.float32)
        scale_b = scale_b.contiguous().view(-1).to(torch.float32)
        return {
            "a_q": a_q, "b_q": b_q, "scale_a": scale_a, "scale_b": scale_b,
            "M": M, "N": N, "K": K,
            # inputs are fp8 + dtype-agnostic; the dtype string drives provider scope.
            "dtype_str": shape["dtype"],
            "dtype": _safe_torch_dtype(shape["dtype"]),
        }

    def reference(self, shape: dict, inputs: dict):
        import torch
        # fp32 golden, exactly the example's run_torch: dequant of the quantized
        # bits. C = (a_q*scale_a) @ (b_q*scale_b)^T.
        a_f32 = inputs["a_q"].to(torch.float32) * inputs["scale_a"].view(-1, 1)
        b_f32 = inputs["b_q"].to(torch.float32) * inputs["scale_b"].view(-1, 1)
        return torch.mm(a_f32, b_f32.T)

    def bytes_moved(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        # matches the test's roofline: A (fp8, M*K*1) + B (fp8, N*K*1) +
        # C (bf16 out, M*N*2) + per-token scales ((M+N)*4).
        size_a = M * K * 1
        size_b = N * K * 1
        size_c = M * N * 2
        return size_a + size_b + size_c + (M + N) * 4

    def flops(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        return 2 * M * N * K

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # fp8 a8w8 GEMM: per-token quant on A and B + accumulation over K + bf16
        # output rounding. The example test passes at rtol=atol=0.1; use the
        # GemmOp fp8 bar (0.2,0.2) to stay robust to the extra bf16-output round.
        return (0.2, 0.2)

    def args_summary(self, shape: dict) -> str:
        M, N, K = self._MNK(shape)
        return f"M={M},N={N},K={K}"


class BlockScalePreshuffleGemmOp(Op):
    """FP8 A8W8 block-scaled GEMM: C[M,N] = dequant(x[M,K]) @ dequant(weight[N,K])^T.

    Per-block scales with BLOCK_SHAPE=(block_n=128, block_k=128), ScaleBlockM=1:
      x_scale : [M, scale_k] fp32 (per token, per K-block)
      w_scale : [scale_n, scale_k] fp32 (per N-block, per K-block)
      scale_k = ceil(K/128), scale_n = ceil(N/128)

    Inputs are built ONCE in fp16 (matching the test's rand/10 magnitude) then
    cast to fp8 so every provider gets identical bits; the UN-preshuffled weight
    and raw 2-D scales are stored, and each provider does its own preshuffle /
    scale relayout inside run() (FlyDSL: shuffle_weight(16,16) + x_scale
    transpose-flatten; aiter: aiter shuffle + 2-D transposed x_scale). The fp32
    golden is run_torch_blockscale (per-block dequant -> F.linear in fp32),
    exactly the test's reference.

    Ledger dtype is "fp8" (the input operand dtype); the output dtype is carried
    in args.out_dtype ("bf16" | "fp16", default "bf16").
    """

    op_type = "blockscale_preshuffle_gemm"

    _BLOCK_N = 128
    _BLOCK_K = 128

    def _MNK(self, shape: dict) -> tuple[int, int, int]:
        a = shape["args"]
        return int(a["M"]), int(a["N"]), int(a["K"])

    def _scale_dims(self, shape: dict) -> tuple[int, int]:
        M, N, K = self._MNK(shape)
        scale_k = (K + self._BLOCK_K - 1) // self._BLOCK_K
        scale_n = (N + self._BLOCK_N - 1) // self._BLOCK_N
        return scale_n, scale_k

    @staticmethod
    def _fp8_dtype():
        import torch
        # gfx950/MI350 uses OCP fp8 (e4m3fn); MI300 uses the _fnuz spelling. The
        # benchmark node is gfx950 -> e4m3fn (matches the test's arch pick).
        try:
            from flydsl.runtime.device import get_rocm_arch
            arch = str(get_rocm_arch())
        except Exception:
            arch = common.arch() or ""
        return torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        M, N, K = self._MNK(shape)
        scale_n, scale_k = self._scale_dims(shape)
        g = common.make_generator(seed)
        fp8 = self._fp8_dtype()
        dev = "cuda"
        # build in fp16 / 10 then cast to fp8 -> identical bits for every provider
        x_fp8 = (torch.rand((M, K), dtype=torch.float16, device=dev, generator=g) / 10).to(fp8).contiguous()
        weight_fp8 = (torch.rand((N, K), dtype=torch.float16, device=dev, generator=g) / 10).to(fp8).contiguous()
        x_scale = torch.rand((M, scale_k), dtype=torch.float32, device=dev, generator=g).contiguous()
        w_scale = torch.rand((scale_n, scale_k), dtype=torch.float32, device=dev, generator=g).contiguous()
        return {
            "x_fp8": x_fp8, "weight_fp8": weight_fp8,
            "x_scale": x_scale, "w_scale": w_scale,
            "M": M, "N": N, "K": K, "scale_n": scale_n, "scale_k": scale_k,
            "out_dtype": str(shape.get("args", {}).get("out_dtype", "bf16")),
            "fp8_dtype": fp8,
        }

    @staticmethod
    def _torch_blockscale(x, weight, x_scale, w_scale, dtype):
        """Block-scaled dequant + F.linear; mirrors the test's run_torch_blockscale.
        Returns the result cast to `dtype` (use torch.float32 for the golden)."""
        import torch
        import torch.nn.functional as F

        block_shape_n, block_shape_k = (128, 128)
        m, k = x.shape
        n = weight.shape[0]
        scale_n = (n + block_shape_n - 1) // block_shape_n
        scale_k = (k + block_shape_k - 1) // block_shape_k

        x_f32 = x.to(x_scale.dtype).view(m, k // block_shape_k, block_shape_k) * x_scale.unsqueeze(-1)
        x_f32 = x_f32.view(m, k)

        w_scale_expanded = (
            w_scale.view(-1, 1)
            .repeat(1, block_shape_n * block_shape_k)
            .view(scale_n, scale_k, block_shape_n, block_shape_k)
            .permute(0, 2, 1, 3)
            .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
        )
        w_scale_expanded = w_scale_expanded[:n, :k]
        weight_f32 = weight.to(w_scale_expanded.dtype) * w_scale_expanded

        out = F.linear(x_f32.to(torch.float32), weight_f32.to(torch.float32))
        return out.to(dtype)

    def reference(self, shape: dict, inputs: dict):
        import torch
        # fp32 golden, exactly the test's run_torch_blockscale(dtype=torch.float32)
        return self._torch_blockscale(
            inputs["x_fp8"], inputs["weight_fp8"], inputs["x_scale"], inputs["w_scale"],
            dtype=torch.float32)

    def bytes_moved(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        scale_n, scale_k = self._scale_dims(shape)
        # read x (M*K, fp8=1B) + read weight (N*K, fp8=1B) + write C (M*N, out 2B)
        # + read both scale tensors (fp32=4B). Matches the test's accounting.
        return (M * K) + (N * K) + (M * N * 2) + (M * scale_k + scale_n * scale_k) * 4

    def flops(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        return 2 * M * N * K

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # Block scaling restores most of fp8's dynamic range, so the kernel
        # output (upcast to fp32) tracks the fp32 golden tightly. Match the
        # test's validated bar: verify_output(rtol=1e-2, atol=0.01).
        return (1e-2, 1e-2)

    def args_summary(self, shape: dict) -> str:
        M, N, K = self._MNK(shape)
        out = str(shape.get("args", {}).get("out_dtype", "bf16"))
        return f"M={M},N={N},K={K},out={out}"


class Fp8GemmRowscaleOp(Op):
    """FP8 row-scale GEMM: C = (A_fp8 * scale_a) @ (B_fp8 * scale_b)^T -> bf16.

    Mirrors tests/kernels/test_fp8_gemm_rowscale.py. Inputs are built ONCE in
    fp32 (uniform[0,1), matching the test's torch.rand), per-row quantized to
    fp8_e4m3fn (CDNA4 OCP fp8 -- the test uses torch.float8_e4m3fn explicitly,
    NOT the _fnuz spelling common.torch_dtype maps to), and the fp32 row scales
    are shared so every provider gets identical bits. The fp32 golden is the
    DEQUANT matmul of the quantized bits (a_q.float()*scale_a) @ (b_q.float()*
    scale_b)^T -- i.e. it reflects the fp8 round-trip exactly like the test's
    _run_torch, so correctness is judged against the achievable result not the
    pre-quant fp32. K must be % 128 == 0 (kernel BLOCK_K) and N % BLOCK_N for the
    FlyDSL path; those are enforced per-provider in supports(), not here.

    Ledger args: M, N, K (required); optional use_8w (bool) and b_preshuffled
    (bool) select the FlyDSL kernel variant -- they do not change the inputs or
    the reference, only how the flydsl adapter compiles.
    """

    op_type = "fp8_gemm_rowscale"

    def _MNK(self, shape: dict) -> tuple[int, int, int]:
        a = shape["args"]
        return int(a["M"]), int(a["N"]), int(a["K"])

    @staticmethod
    def _pertoken_quant_fp8(x):
        # Per-row symmetric quant to fp8_e4m3fn, matching tests.utils.pertoken_quant:
        # scale = max(|row|)/fp8_max; y = (x/scale).to(fp8); returns (y, scale_fp32).
        import torch

        fp8 = torch.float8_e4m3fn
        fp8_max = float(torch.finfo(fp8).max)
        x = x.to(torch.float32)
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = amax / fp8_max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        y = (x / scale).to(fp8)
        return y, scale.to(torch.float32)

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        M, N, K = self._MNK(shape)
        g = common.make_generator(seed)
        # fp32 uniform[0,1) (matches the test's torch.rand), then per-row fp8 quant.
        a_fp32 = torch.rand((M, K), device="cuda", dtype=torch.float32, generator=g)
        b_fp32 = torch.rand((N, K), device="cuda", dtype=torch.float32, generator=g)
        a_q, scale_a = self._pertoken_quant_fp8(a_fp32)
        b_q, scale_b = self._pertoken_quant_fp8(b_fp32)
        a_q = a_q.contiguous()
        b_q = b_q.contiguous()
        scale_a = scale_a.squeeze().contiguous()  # (M,)
        scale_b = scale_b.squeeze().contiguous()  # (N,)
        return {"a": a_q, "b": b_q, "scale_a": scale_a, "scale_b": scale_b,
                "M": M, "N": N, "K": K, "dtype": torch.bfloat16}

    def reference(self, shape: dict, inputs: dict):
        import torch

        M, N = inputs["M"], inputs["N"]
        a_f32 = inputs["a"].to(torch.float32) * inputs["scale_a"].view(M, 1)
        b_f32 = inputs["b"].to(torch.float32) * inputs["scale_b"].view(N, 1)
        return torch.mm(a_f32, b_f32.T)

    def bytes_moved(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        # read A (M*K fp8=1B) + read B (N*K fp8=1B) + write C (M*N bf16=2B)
        # + read scales ((M+N)*4B). Matches the test's bytes_moved accounting.
        return M * K + N * K + M * N * 2 + (M + N) * 4

    def flops(self, shape: dict) -> int:
        M, N, K = self._MNK(shape)
        return 2 * M * N * K

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # fp8 GEMM accumulates over K with quantized inputs; the kernel/baselines
        # all bar themselves at rtol=atol=0.1 (verify_output in the test). Hold
        # that bar rather than TOL[fp8]=(0.15,0.15) since the golden already
        # reflects the fp8 round-trip.
        return (0.1, 0.1)

    def args_summary(self, shape: dict) -> str:
        M, N, K = self._MNK(shape)
        a = shape.get("args", {})
        extra = ""
        if a.get("use_8w"):
            extra += ",8wave"
        if a.get("b_preshuffled"):
            extra += ",preshuffle_b"
        return f"M={M},N={N},K={K}{extra}"


class MoeBlockscaleOp(Op):
    """Full 2-stage FP8 block-scale MoE (gate+up silu, then down + topk reduce).

    Same routed-MoE shape as MoeGemmOp, but the kernel is FP8-ONLY with per-(1x128)
    BLOCK scales (ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128). Ledger args:
    M=tokens, E=experts, Dim1=model_dim, Dim2=2*inter_dim (inter_dim=Dim2//2),
    TopK=topk. Inputs are built ONCE in fp32 and shared across providers; the
    per-provider block-quantize/cast/preshuffle happens inside run(). The fp32
    reference mirrors torch_moe_blockscale_ref (tests/kernels/test_moe_blockscale.py):
    block-quantize x and per-expert W1/W2 to fp8, block-dequantize back to fp32,
    then run the standard MoE (silu(x@W1g^T)*(x@W1u^T) @ W2^T, routed-weighted,
    summed over topk). This matches the FlyDSL composed output and aiter's fused
    fmoe_fp8_blockscale_g1u1 (doweight in stage2).
    """

    op_type = "moe_blockscale"
    SCALE_BLK = 128  # ScaleBlockN == ScaleBlockK == scale_block_k

    def _dims(self, shape: dict):
        a = shape["args"]
        tokens = int(a["M"])
        experts = int(a["E"])
        model_dim = int(a["Dim1"])
        inter_dim = int(a["Dim2"]) // 2
        topk = int(a["TopK"])
        return tokens, experts, model_dim, inter_dim, topk

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        g = common.make_generator(seed)
        s = 0.2
        # build in fp32 once -> every provider gets identical bits, then block-quantizes itself
        x_fp32 = (torch.randn((tokens, model_dim), device="cuda", dtype=torch.float32, generator=g) * s).contiguous()
        w1_fp32 = (torch.randn((experts, 2 * inter_dim, model_dim), device="cuda", dtype=torch.float32, generator=g) * s).contiguous()
        w2_fp32 = (torch.randn((experts, model_dim, inter_dim), device="cuda", dtype=torch.float32, generator=g)
                   * (s / (inter_dim ** 0.5))).contiguous()
        score = torch.rand((tokens, experts), device="cuda", dtype=torch.float32, generator=g)
        topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32).contiguous()
        return {
            "x_fp32": x_fp32, "w1_fp32": w1_fp32, "w2_fp32": w2_fp32,
            "topk_ids": topk_ids.to(torch.int32).contiguous(), "topk_weights": topk_weights,
            "M": tokens, "E": experts, "model_dim": model_dim, "inter_dim": inter_dim, "topk": topk,
            # inputs are fp32 + dtype-agnostic; providers block-quantize per-dtype themselves.
            "dtype_str": shape["dtype"],
            "dtype": _safe_torch_dtype(shape["dtype"]),
        }

    def _fp8_dtype(self):
        import torch

        # gfx950 OCP fp8; fall back to the MI300 _fnuz spelling if e4m3fn is absent.
        return getattr(torch, "float8_e4m3fn", None) or torch.float8_e4m3fnuz

    def _block_quant_dequant_w_expert(self, w_e_fp32, blk_n, blk_k):
        """(1x128) block quant->dequant of a SINGLE expert weight [N, K] -> fp32 [N, K].

        Operates one expert at a time so the fp32 dequant working set never holds
        all E experts at once (the 256/384-expert DeepSeek/Kimi weights are ~45 GB
        in fp32; materializing the grouped blocks + q + deq for every expert
        simultaneously OOMs even with ~300 GB free)."""
        import torch

        N_w, K_w = w_e_fp32.shape
        fp8 = self._fp8_dtype()
        dtype_max = torch.finfo(fp8).max
        nbn, nbk = N_w // blk_n, K_w // blk_k
        blocks = (w_e_fp32.float()
                  .view(nbn, blk_n, nbk, blk_k)
                  .permute(0, 2, 1, 3)
                  .reshape(nbn * nbk, blk_n * blk_k))
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = amax / dtype_max
        q = (blocks / scale).to(fp8).float()
        return (q * scale).view(nbn, nbk, blk_n, blk_k).permute(0, 2, 1, 3).reshape(N_w, K_w)

    def _block_quant_dequant_a(self, a_fp32, blk_k):
        """Per-row (1x128 along K) block quant->dequant of an activation [R, K] -> fp32 [R, K]."""
        import torch

        R, K = a_fp32.shape
        fp8 = self._fp8_dtype()
        dtype_max = torch.finfo(fp8).max
        nbk = K // blk_k
        blocks = a_fp32.float().view(R, nbk, blk_k)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = amax / dtype_max
        q = (blocks / scale).to(fp8).float()
        return (q * scale).view(R, K)

    def reference(self, shape: dict, inputs: dict):
        import torch
        import torch.nn.functional as F

        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        blk = self.SCALE_BLK
        # block quant->dequant inputs in fp32 (matches the kernel's fp8 round-trip)
        x = self._block_quant_dequant_a(inputs["x_fp32"], blk)
        w1_fp32 = inputs["w1_fp32"]
        w2_fp32 = inputs["w2_fp32"]
        topk_ids = inputs["topk_ids"].long()
        topk_w = inputs["topk_weights"]
        out = torch.zeros((tokens, model_dim), device="cuda", dtype=torch.float32)
        for e in range(experts):
            idx = (topk_ids == e).nonzero(as_tuple=False)  # [num, 2] (token, slot)
            if idx.numel() == 0:
                continue
            t_idx = idx[:, 0]
            s_idx = idx[:, 1]
            # Dequantize this expert's weights only (freed each iteration) so the
            # reference never holds all-E fp32 dequant copies at once.
            w1_e = self._block_quant_dequant_w_expert(w1_fp32[e], blk, blk)
            y2 = F.linear(x[t_idx, :], w1_e)           # [num, 2*inter_dim]
            del w1_e
            gate = y2[:, :inter_dim]
            up = y2[:, inter_dim:]
            # inter-stage block re-quant of the intermediate (inherent to the 2-stage pipeline)
            h = self._block_quant_dequant_a(F.silu(gate) * up, blk)  # [num, inter_dim]
            w2_e = self._block_quant_dequant_w_expert(w2_fp32[e], blk, blk)
            y = F.linear(h, w2_e)                      # [num, model_dim]
            del w2_e
            y = y * topk_w[t_idx, s_idx].unsqueeze(-1)  # routed weight (applied in stage2)
            out.index_add_(0, t_idx, y)
        return out

    def bytes_moved(self, shape: dict) -> int:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        eb = _elem_bytes(shape["dtype"])  # fp8 -> 1 B
        e_active = min(experts, tokens * topk)  # weights touched = active experts only
        x_bytes = tokens * model_dim * eb
        w1_bytes = e_active * (2 * inter_dim) * model_dim * eb
        w2_bytes = e_active * model_dim * inter_dim * eb
        out_bytes = tokens * model_dim * 2  # final [tokens, model_dim] f16/bf16
        return x_bytes + w1_bytes + w2_bytes + out_bytes

    def flops(self, shape: dict) -> int:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        # stage1 2*M*T*(2I)*K + stage2 2*M*T*K*I = 6*M*T*K*I (silu negligible)
        return 6 * tokens * topk * model_dim * inter_dim

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # fp8 round-trips through THREE block quantizations (input, weights,
        # inter-stage). The FlyDSL test bar is rtol=atol=0.1 (err_ratio); use a
        # slightly wider (0.2,0.2) absolute bound for the elementwise allclose.
        return (0.2, 0.2)

    def args_summary(self, shape: dict) -> str:
        tokens, experts, model_dim, inter_dim, topk = self._dims(shape)
        return f"tokens={tokens},E={experts},model_dim={model_dim},inter_dim={inter_dim},topk={topk}"


class FlashAttnOp(Op):
    """Flash multi-head attention (BSHD layout), causal or non-causal.

    Ledger args: batch, seq_len, num_heads, head_dim, causal(bool). Inputs are
    built ONCE in fp32 with uniform(-1,1) (matching the FlyDSL test's
    UNIFORM_RANGE) then cast -> every provider gets identical bits. The fp32
    golden is torch SDPA on the [B,H,S,D] transpose (transpose back to BSHD),
    exactly pytorch_ref_attention from tests/kernels/test_flash_attn_func.py.
    Comparable tensor for every provider is the BSHD output.
    """

    op_type = "flash_attn"

    def _BSHD(self, shape):
        a = shape["args"]
        return (int(a["batch"]), int(a["seq_len"]), int(a["num_heads"]), int(a["head_dim"]))

    def _causal(self, shape) -> bool:
        return bool(shape.get("args", {}).get("causal", True))

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        B, S, H, D = self._BSHD(shape)
        td = common.torch_dtype(shape["dtype"])
        g = common.make_generator(seed)
        # build in fp32 uniform(-1,1) then cast -> identical bits for every provider
        def _qkv():
            return (torch.empty((B, S, H, D), device="cuda", dtype=torch.float32)
                    .uniform_(-1, 1, generator=g).to(td).contiguous())
        q, k, v = _qkv(), _qkv(), _qkv()
        return {"q": q, "k": k, "v": v,
                "B": B, "S": S, "H": H, "D": D,
                "causal": self._causal(shape), "dtype": td}

    def reference(self, shape: dict, inputs: dict):
        import torch
        import torch.nn.functional as F

        D = inputs["D"]
        sm_scale = 1.0 / (D ** 0.5)
        # SDPA wants [B,H,S,D]; transpose in, run fp32 golden, transpose back to BSHD
        q = inputs["q"].float().transpose(1, 2)
        k = inputs["k"].float().transpose(1, 2)
        v = inputs["v"].float().transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=inputs["causal"], scale=sm_scale)
        return out.transpose(1, 2).contiguous()

    def bytes_moved(self, shape: dict) -> int:
        B, S, H, D = self._BSHD(shape)
        e = _elem_bytes(shape["dtype"])
        # read Q,K,V (3*B*S*H*D) + write O (B*S*H*D); the S*S score matrix lives in
        # registers/LDS (flash attention never materializes it) -> not counted.
        return 4 * B * S * H * D * e

    def flops(self, shape: dict) -> int:
        B, S, H, D = self._BSHD(shape)
        # QK^T (2*S*S*D) + PV (2*S*S*D) per (batch,head); causal halves the work.
        s_eff = (S * (S + 1) / 2.0) if self._causal(shape) else float(S * S)
        return int(4.0 * B * H * D * s_eff)

    def args_summary(self, shape: dict) -> str:
        B, S, H, D = self._BSHD(shape)
        c = "causal" if self._causal(shape) else "noncausal"
        return f"B={B},S={S},H={H},D={D},{c}"

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # attention accumulates softmax-weighted PV over the full key axis (long
        # reductions, exp); use a GEMM-like loose bar for f16/bf16 (matches the
        # FlyDSL test's max_err<1e-2 spirit but on the upcast-fp32 allclose path).
        return (0.05, 0.05)


class PagedAttnDecodeOp(Op):
    """Paged-Attn Decode (PS), fp8 paged KV cache (kernels/pa_decode_fp8.py).

    Inputs mirror FlyDSL-lab/tests/kernels/test_pa.py::run_pa_decode_ps_test: a
    bf16 qkv tensor (query kept bf16 -- the PS launcher casts to fp8 internally),
    an fp8 paged KV cache built in the kernel's 5-D key / 4-D value layout, and a
    per-token or per-tensor symmetric KV quantization. All providers receive the
    SAME quantized cache + scales (identical bits). The heavy paged-cache build +
    quantization happen here (untimed); providers set includes_layout_conversion.

    The fp32 golden is the test's torch_mha_extend (gather paged KV -> dequant ->
    masked causal attention), shared with the pytorch provider via
    benchmarks.providers.pa_decode._torch_mha_extend.

    Ledger args: num_q_heads, num_kv_heads, head_size(=128), context_length,
    batch, query_length(1..4), block_size(=1024), quant_mode(per_token/per_tensor),
    sliding_window(default 0). dtype is "fp8".
    """

    op_type = "pa_decode"

    def _dims(self, shape):
        a = shape["args"]
        return (int(a["num_q_heads"]), int(a["num_kv_heads"]), int(a["head_size"]),
                int(a["context_length"]), int(a["batch"]), int(a["query_length"]),
                int(a["block_size"]))

    def _quant_mode(self, shape) -> str:
        return str(shape.get("args", {}).get("quant_mode", "per_token"))

    def _sliding_window(self, shape) -> int:
        return int(shape.get("args", {}).get("sliding_window", 0))

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import random
        import torch
        import triton
        from aiter import dtypes as aiter_dtypes, per_tensor_quant, pertoken_quant  # noqa: F401

        nq, nkv, hs, ctx, b, ql, bsz = self._dims(shape)
        quant_mode = self._quant_mode(shape)
        device = torch.device("cuda")
        fp8 = aiter_dtypes.fp8

        # deterministic init -- mirror test_pa.py setup_seed for the paged cache
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        data_type = torch.bfloat16  # compute_type fp8 -> bf16 query/output
        softmax_scale = 1.0 / (hs ** 0.5)
        total_queries = b * ql
        query_output_indptr = torch.arange(
            0, (b + 1) * ql, ql, dtype=torch.int32, device=device)

        # qkv in one tensor (matches the test), uniform(-1,1) query
        qkv = torch.randn(total_queries, nq + 2 * nkv, hs, dtype=data_type, device=device)
        query, _key, _value = torch.split(qkv, [nq, nkv, nkv], dim=1)
        query = query.contiguous()
        query.uniform_(-1, 1)

        context_lengths = torch.full((b,), ctx, dtype=torch.int32, device=device)
        max_context_length = max(16384, ctx)
        max_blocks_per_sequence = triton.cdiv(max_context_length, bsz)
        total_blocks = max_blocks_per_sequence * b
        blocks_per_sequence = triton.cdiv(ctx, bsz)
        block_tables_list = []
        for _ in range(b):
            block_tables_list.append(
                [random.randint(0, total_blocks - 1) for _ in range(blocks_per_sequence)])
        block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

        # --- paged KV cache (float, kernel layout) -> quantized to fp8 below ---
        # Mirrors test_pa.py::create_kv_cache: with cache_dtype="auto" the cache is
        # built in the *model* dtype (here bf16) and filled with uniform_(-1, 1),
        # then quantize_kv_cache_* casts it down to fp8. Allocating uint8 here is the
        # bug -- uniform_ is not implemented for Byte and the quant helpers expect a
        # float cache.
        elements_per_vector = 16  # 16 // itemsize(1) for the fp8 packed layout
        key_cache = torch.empty(
            (total_blocks, nkv, hs // elements_per_vector, bsz, elements_per_vector),
            dtype=data_type, device=device)
        value_cache = torch.empty((total_blocks, nkv, hs, bsz), dtype=data_type, device=device)
        key_cache.uniform_(-1, 1)
        value_cache.uniform_(-1, 1)

        # --- symmetric KV quantization (per_token or per_tensor) ---
        if quant_mode == "per_token":
            (quantized_keys, key_scale_flat, quantized_values, value_scale_flat,
             key_scale_original, value_scale_original) = _quantize_kv_symmetric(
                key_cache, value_cache, fp8)
        else:
            (quantized_keys, key_scale_flat, quantized_values, value_scale_flat,
             key_scale_original, value_scale_original) = _quantize_kv_per_tensor(
                key_cache, value_cache, fp8)

        # reference (un-shuffled value) layout + golden output BEFORE trans_v shuffle
        from benchmarks.providers.pa_decode import _torch_mha_extend
        reference_output = _torch_mha_extend(
            query, quantized_keys, quantized_values, block_tables, context_lengths,
            query_output_indptr, key_scale_flat, value_scale_flat,
            sliding_window=self._sliding_window(shape)).to(data_type)

        quantized_values_ref = quantized_values
        quantized_values_flydsl = _shuffle_value_cache_layout(quantized_values)  # trans_v=True

        # PS page data (kv_page_indices, kv_indptr) from block tables
        actual_blocks = (context_lengths + bsz - 1) // bsz
        kv_indptr = torch.zeros(b + 1, dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
        page_indices = []
        for bi, nb in enumerate(actual_blocks.tolist()):
            page_indices.extend(block_tables_list[bi][:nb])
        kv_page_indices = torch.tensor(page_indices, dtype=torch.int32, device=device)

        return {
            "query": query,
            "quantized_keys": quantized_keys,
            "quantized_values_ref": quantized_values_ref,
            "quantized_values_flydsl": quantized_values_flydsl,
            "key_scale_flat": key_scale_flat,
            "value_scale_flat": value_scale_flat,
            "key_scale_original": key_scale_original,
            "value_scale_original": value_scale_original,
            "context_lengths": context_lengths,
            "block_tables": block_tables,
            "kv_page_indices": kv_page_indices,
            "kv_indptr": kv_indptr,
            "query_output_indptr": query_output_indptr,
            "softmax_scale": softmax_scale,
            "reference_output": reference_output,
            "dtype": data_type,
        }

    def reference(self, shape: dict, inputs: dict):
        # fp32 golden = torch_mha_extend over the SAME paged cache (reference V layout)
        from benchmarks.providers.pa_decode import _torch_mha_extend
        return _torch_mha_extend(
            inputs["query"], inputs["quantized_keys"], inputs["quantized_values_ref"],
            inputs["block_tables"], inputs["context_lengths"], inputs["query_output_indptr"],
            inputs["key_scale_flat"], inputs["value_scale_flat"],
            sliding_window=self._sliding_window(shape)).float()

    def bytes_moved(self, shape: dict) -> int:
        nq, nkv, hs, ctx, b, ql, bsz = self._dims(shape)
        # dominant traffic: read paged K + V over the context for every sequence
        # (fp8 = 1 B/elem), plus per-token scales (fp32) and the bf16 output.
        kv_bytes = b * ctx * nkv * hs * 1 * 2  # K + V, 1 byte each
        scale_bytes = b * ctx * nkv * 4 * 2    # per-token K/V scales fp32
        q_bytes = b * ql * nq * hs * 2         # bf16 query read
        out_bytes = b * ql * nq * hs * 2       # bf16 output write
        return kv_bytes + scale_bytes + q_bytes + out_bytes

    def flops(self, shape: dict) -> int:
        nq, nkv, hs, ctx, b, ql, bsz = self._dims(shape)
        # QK^T (2*ql*ctx*hs) + softmax@V (2*ql*ctx*hs) per query head, per sequence
        return 2 * 2 * b * ql * nq * ctx * hs

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # fp8 paged KV decode round-trips through quantization; the PS test bar is
        # 5e-3 (per_token, no varlen) widening to 5e-2 with varlen/sliding window.
        # Use the looser of the two so a correct kernel is never false-failed.
        return (5e-2, 5e-2)

    def args_summary(self, shape: dict) -> str:
        nq, nkv, hs, ctx, b, ql, bsz = self._dims(shape)
        return (f"hq={nq},hkv={nkv},d={hs},ctx={ctx},b={b},ql={ql},"
                f"blk={bsz},quant={self._quant_mode(shape)}")


def _quantize_kv_symmetric(key_cache, value_cache, quant_dtype):
    """Per-token symmetric KV quant (test_pa.py::quantize_kv_cache_symmetric)."""
    import torch
    from aiter import pertoken_quant

    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    total_tokens = num_blocks * block_size
    key_cache_reshaped = key_cache.permute(0, 1, 3, 2, 4).reshape(
        num_blocks, num_heads, block_size, -1).contiguous()
    value_cache_reshaped = value_cache.permute(0, 1, 3, 2).reshape(
        num_blocks, num_heads, block_size, -1).contiguous()
    quantized_keys, key_scales_original = pertoken_quant(key_cache_reshaped, quant_dtype=quant_dtype)
    quantized_values, value_scales_original = pertoken_quant(value_cache_reshaped, quant_dtype=quant_dtype)
    elements_per_vector = 16 // quant_dtype.itemsize
    quantized_keys = (quantized_keys.view(
        num_blocks, num_heads, block_size, head_dim // elements_per_vector, elements_per_vector)
        .permute(0, 1, 3, 2, 4).contiguous())
    quantized_values = (quantized_values.view(
        num_blocks, num_heads, block_size, head_dim).permute(0, 1, 3, 2).contiguous())
    key_scales_flat = key_scales_original.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    value_scales_flat = value_scales_original.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    return (quantized_keys, key_scales_flat, quantized_values, value_scales_flat,
            key_scales_original, value_scales_original)


def _quantize_kv_per_tensor(key_cache, value_cache, quant_dtype):
    """Per-tensor symmetric KV quant (test_pa.py::quantize_kv_cache_per_tensor)."""
    import torch
    from aiter import per_tensor_quant

    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    elements_per_vector = 16 // quant_dtype.itemsize
    key_cache_reshaped = key_cache.permute(0, 1, 3, 2, 4).reshape(
        num_blocks, num_heads, block_size, -1).contiguous()
    key_cache_reshaped = (key_cache_reshaped.view(
        num_blocks, num_heads, block_size, head_dim // elements_per_vector, elements_per_vector)
        .permute(0, 1, 3, 2, 4).contiguous())
    quantized_keys, key_scales_original = per_tensor_quant(key_cache_reshaped, quant_dtype=quant_dtype)
    quantized_values, value_scales_original = per_tensor_quant(value_cache, quant_dtype=quant_dtype)
    key_scales_flat = key_scales_original.expand(num_heads, num_blocks * block_size)
    value_scales_flat = value_scales_original.expand(num_heads, num_blocks * block_size)
    return (quantized_keys, key_scales_flat, quantized_values, value_scales_flat,
            key_scales_original, value_scales_original)


def _shuffle_value_cache_layout(value_cache):
    """trans_v V-cache relayout (test_pa.py::shuffle_value_cache_layout)."""
    elements_per_vector = 16 // value_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache.shape
    value_cache_reshaped = value_cache.view(
        num_blocks, num_kv_heads, head_size, block_size // elements_per_vector, elements_per_vector)
    return value_cache_reshaped.permute(0, 1, 3, 2, 4).contiguous()


class MlaDecodeOp(Op):
    """MLA decode (fp8 Q / fp8 KV, nhead=128, page_size=1) -- DeepSeek-V3/R1 dims.

    Args ledger: batch=num_seqs, ctx_len=KV length per seq, decode_qlen=1.
    Fixed dims (the FlyDSL fp8 path requires them): KV_LORA_RANK=512, QK_ROPE=64,
    QK_HEAD_DIM=576, V_HEAD_DIM=512, NHEAD=128, NHEAD_KV=1, PAGE_SIZE=1.

    Inputs are built ONCE in fp32 (q [total_q,128,576], kv_buffer
    [num_page,1,1,576]) plus the integer page-permutation kv_indices, and shared
    across providers; each provider casts q/kv to fp8 itself (identical bits) and
    builds the aiter work-map metadata in its own (cached, untimed) setup. The
    fp32 golden is the test's torch_mla_extend over the fp8-cast values, so it
    matches what every fp8 kernel actually computes (the reference uses the SAME
    fp8 rounding, then upcasts).
    """

    op_type = "mla_decode"

    # MLA constants (must match providers/mla_decode.py + the FlyDSL fp8 kernel).
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    QK_HEAD_DIM = 576       # KV_LORA_RANK + QK_ROPE_HEAD_DIM
    V_HEAD_DIM = 512        # == KV_LORA_RANK
    NHEAD = 128
    NHEAD_KV = 1
    PAGE_SIZE = 1

    def _dims(self, shape):
        a = shape["args"]
        return int(a["batch"]), int(a["ctx_len"]), int(a.get("decode_qlen", 1))

    def make_inputs(self, shape: dict, seed: int) -> dict:
        import torch

        batch, ctx_len, decode_qlen = self._dims(shape)
        page_size = self.PAGE_SIZE
        g = common.make_generator(seed)

        # num_page == sum over batch of ceil(ctx_len/page_size) (page_size=1 ->
        # batch*ctx_len), matching run_single's kv_indptr[-1].
        kv_block_nums = (ctx_len + page_size - 1) // page_size
        num_page = batch * kv_block_nums
        total_q = batch * decode_qlen

        # build q / kv in fp32 once -> every provider casts to fp8 from identical
        # bits. randn matches the test's torch.randn(..., bf16) distribution
        # (we keep fp32 here; the cast-to-fp8 happens per provider).
        q_fp32 = torch.randn((total_q, self.NHEAD, self.QK_HEAD_DIM),
                             device="cuda", dtype=torch.float32, generator=g).contiguous()
        kv_fp32 = torch.randn((num_page, page_size, self.NHEAD_KV, self.QK_HEAD_DIM),
                              device="cuda", dtype=torch.float32, generator=g).contiguous()
        # page permutation (paged-KV indirection). int32 like the test.
        kv_indices = torch.randperm(num_page, device="cuda", generator=g).to(torch.int32)

        return {
            "q_fp32": q_fp32, "kv_fp32": kv_fp32, "kv_indices": kv_indices,
            "batch": batch, "ctx_len": ctx_len, "decode_qlen": decode_qlen,
            "num_page": num_page, "total_q": total_q, "page_size": page_size,
            "dtype": _safe_torch_dtype(shape["dtype"]), "dtype_str": shape["dtype"],
        }

    def reference(self, shape: dict, inputs: dict):
        import torch

        # torch_mla_extend over the fp8-cast Q/KV (the test's golden). We replicate
        # the metadata layout for decode (qo_indptr/kv_indptr/kv_last_page_lens)
        # purely in torch so the reference needs no aiter import.
        from benchmarks.providers.mla_decode import _aiter_fp8, KV_LORA_RANK
        batch, ctx_len, decode_qlen = self._dims(shape)
        page_size = self.PAGE_SIZE
        sm_scale = 1.0 / (self.QK_HEAD_DIM ** 0.5)

        fp8 = _aiter_fp8()
        q = inputs["q_fp32"].to(fp8)
        kvc_cache = inputs["kv_fp32"].to(fp8).view(
            inputs["num_page"], page_size, self.NHEAD_KV, self.QK_HEAD_DIM)
        kv_indices = inputs["kv_indices"]

        kv_block_nums = torch.full((batch,), (ctx_len + page_size - 1) // page_size,
                                   dtype=torch.int, device="cuda")
        kv_last_page_lens = torch.ones(batch, dtype=torch.int, device="cuda")
        if ctx_len % page_size != 0:
            kv_last_page_lens.fill_(ctx_len % page_size)
        kv_indptr = torch.zeros(batch + 1, dtype=torch.int, device="cuda")
        kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
        qo_indptr = torch.zeros(batch + 1, dtype=torch.int, device="cuda")
        qo_indptr[1:] = torch.cumsum(
            torch.full((batch,), decode_qlen, dtype=torch.int, device="cuda"), dim=0)

        # cast fp8 -> float (matches torch_mla_extend's is_fp8 branch)
        qf = q.to(torch.float)
        kvf = kvc_cache.to(torch.float)
        qs = torch.tensor_split(qf, qo_indptr.tolist()[1:])
        kvc = torch.index_select(kvf, 0, kv_indices.long())
        kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])

        os_list = []
        for i in range(batch):
            cur_num_page = kvs[i].shape[0]
            real_kv_seq_len = (cur_num_page - 1) * page_size + int(kv_last_page_lens[i].item())
            kvc_i = kvs[i].flatten(0, 1)[:real_kv_seq_len]
            q_i = qs[i]
            k_i = kvc_i
            v_i = kvc_i[:, :, :KV_LORA_RANK]
            attn = torch.einsum("qhd,khd->hqk", q_i.float(), k_i.float()) * sm_scale
            m = attn.max(-1).values
            attn_exp = torch.exp(attn - m.unsqueeze(-1))
            denom = attn_exp.sum(-1)
            o_i = torch.einsum("hqk,khd->qhd", attn_exp.float(), v_i.float())
            o_i = o_i / denom.transpose(0, 1).unsqueeze(-1)
            os_list.append(o_i)
        o = torch.concat(os_list)  # [total_q, nhead, V_HEAD_DIM]
        return o.float()

    def bytes_moved(self, shape: dict) -> int:
        batch, ctx_len, decode_qlen = self._dims(shape)
        total_kv = batch * ctx_len
        total_q = batch * decode_qlen
        # matches the test's bw model: KV read (fp8=1B) + Q read (fp8=1B) +
        # output write (bf16=2B). nhead_kv=1 for KV, nhead=128 for Q/out.
        return (total_kv * self.NHEAD_KV * self.QK_HEAD_DIM * 1
                + total_q * self.NHEAD * self.QK_HEAD_DIM * 1
                + total_q * self.NHEAD * self.V_HEAD_DIM * 2)

    def flops(self, shape: dict) -> int:
        batch, ctx_len, decode_qlen = self._dims(shape)
        total_kv = batch * ctx_len
        # test's flops model: decode_qlen * total_kv * nhead * (QK+V) * 2
        return decode_qlen * total_kv * self.NHEAD * (self.QK_HEAD_DIM + self.V_HEAD_DIM) * 2

    def args_summary(self, shape: dict) -> str:
        batch, ctx_len, decode_qlen = self._dims(shape)
        return f"batch={batch},ctx_len={ctx_len},decode_qlen={decode_qlen}"

    def tolerance(self, shape: dict) -> tuple[float, float]:
        # fp8 Q/KV double-rounded + softmax + V accumulate over a long context;
        # the test bars correctness on cosine-diff<3e-2, not allclose. Against the
        # fp32 (fp8-cast) golden an absolute/relative bound this wide is the
        # honest fp8 attention tolerance (cf. common.TOL["fp8"]=(0.15,0.15)).
        return (1.5e-1, 1.5e-1)


# register campaign-expansion ops
register(VecAddOp())
register(QuantOp())
register(TopkGatingSoftmaxOp())
register(MoeReduceOp())
register(PreshuffleGemmOp())
register(BlockScalePreshuffleGemmOp())
register(Fp8GemmRowscaleOp())
register(MoeBlockscaleOp())
register(FlashAttnOp())
register(PagedAttnDecodeOp())
register(MlaDecodeOp())
