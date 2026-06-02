"""Rule-based bottleneck classification for slow/hot FlyDSL shapes.

The classified metric is the KERNEL-ONLY (CUDA-graph) speedup vs the best
baseline -- so this judges the kernel itself, not host launch overhead (that
eager-only effect is reported separately by the summary). Categories are the
spec set. Heuristic triage; corroborate hot regressions with a profiler trace.
"""

from __future__ import annotations

CATEGORIES = ["tuning_gap", "implementation_gap", "algorithm_gap", "flydsl_codegen_gap",
              "launch_or_roofline_limited", "measurement_issue", "baseline_unfair_or_unmatched",
              "ok"]

_16BIT = ("bf16", "f16", "fp16", "bfloat16", "float16")
# MI350X (gfx950) has ~256 CUs; FlyDSL's grid=(M,1,1) is genuinely under-occupied
# only when M is well below the CU count. M~256 is roughly full occupancy.
_UNDEROCCUPIED_M = 64


def classify(s: dict, profiler_evidence: dict | None = None) -> dict:
    args = s["shape"].get("args", {})
    N = args.get("N")
    M = args.get("M")
    dtype = s["shape"].get("dtype", "")
    sp = s.get("speedup_vs_best")          # KERNEL-ONLY speedup vs best baseline
    kern_us = s.get("flydsl_us")
    stable = s.get("flydsl_stable")
    aligned = (N is not None and N >= 2048 and N % 2048 == 0 and dtype in _16BIT)

    if s.get("flydsl_correct") is False:
        return _r("baseline_unfair_or_unmatched",
                  "FlyDSL output failed correctness vs fp32 reference", "fix kernel/dtype before trusting timing")
    if sp is None:
        return _r("measurement_issue", "no comparable baseline measured", "ensure >=1 correct baseline runs")
    if sp >= 0.95:
        return _r("ok", f"kernel-only vs-best {sp:.2f}x (>= parity)", "none")

    # genuine kernel-only regression below parity.
    # Alignment first: a non-2048-multiple N drops FlyDSL to the generic scalar
    # path, a PER-BLOCK efficiency loss that applies at any M (profiler-confirmed:
    # at M=1 both FlyDSL and triton run 1 block, yet FlyDSL is ~3x slower).
    if not aligned and N is not None:
        extra = ""
        if M is not None and M <= _UNDEROCCUPIED_M:
            extra = (f" Compounded at small M={M}: grid=(M,1,1) launches one workgroup per row, so only ~{M} "
                     "of the ~256 CUs are used (under-occupied).")
        return _r("implementation_gap",
                  (f"N={N} misses the fast-vectorized path (needs N>=2048 & N%2048==0 & 16-bit) -> generic "
                   f"scalar path; per-block efficiency loss (kernel-only vs-best {sp:.2f}x).{extra}"),
                  "vectorize the generic/tail path (widen loads, handle remainder); for small M also split work across N so >1 workgroup runs")
    if M is not None and M <= _UNDEROCCUPIED_M:
        return _r("implementation_gap",
                  (f"aligned N={N} but small M={M}: FlyDSL launches grid=(M,1,1) -> one workgroup per row, so only "
                   f"~{M} of the ~256 CUs are used (under-occupied; kernel-only vs-best {sp:.2f}x)."),
                  "parallelize across N (split-N / persistent blocks) for small M so occupancy is not capped at M")
    if stable is False:
        return _r("measurement_issue", f"unstable timing (p90/p10>1.2); vs-best {sp:.2f}x may be noise",
                  "re-measure with rocprofv3 kernel duration")
    if profiler_evidence:
        return _r("flydsl_codegen_gap",
                  f"aligned large-M shape still slow (vs-best {sp:.2f}x); profiler: {profiler_evidence}",
                  "inspect generated ISA for waits/spills/poor vectorization")
    return _r("tuning_gap",
              (f"aligned large-M shape but vs-best {sp:.2f}x; fixed FlyDSL schedule vs tuned baseline, "
               "no structural cause evident"),
              "add a per-shape tuned schedule (block size, vector width, waves); capture rocprofv3 to confirm")


def classify_eager(s: dict) -> dict | None:
    """Separate verdict on the EAGER launcher-overhead dimension (not kernel-only)."""
    kern_us = s.get("flydsl_us")
    host_ovh = s.get("flydsl_host_overhead_us")
    if kern_us and host_ovh and host_ovh > 3 * kern_us and host_ovh > 10:
        return _r("launch_or_roofline_limited",
                  (f"eager call adds {host_ovh:.0f}us host launch overhead (kernel {kern_us:.1f}us) -- the "
                   "@flyc.jit launcher rebuilds its cache-key per call; dominates short/decode shapes in eager mode "
                   "(mitigated when serving captures decode in a CUDA/hipgraph, as SGLang does)"),
                  "add a fast-path launch cache / persistent launch handle in the FlyDSL launcher (host-side, not a kernel change)")
    return None


def _r(c, ev, fix):
    return {"classification": c, "evidence": ev, "likely_fix": fix}
