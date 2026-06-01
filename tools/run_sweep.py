#!/usr/bin/env python3
"""FlyDSL kernel timing-sweep driver (recipe-driven, GPU-pinned, resumable).

Runs ONE kernel test harness from recipes.json, captures full stdout/stderr,
and parses a generic set of perf metrics (FlyDSL/AIter/CK/asm/Torch us, TFLOPS,
TB/s, GB/s, speedup). Emits a result JSON. Designed to be fanned out one-per-GPU.

Usage:
  run_sweep.py --test test_rmsnorm.py --gpu 0 [--timeout 600] [--outdir DIR] [--recipes PATH]
"""
from __future__ import annotations
import argparse, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

ROOT = "/sgl-workspace/FlyDSL-lab"
PROF = "/sgl-workspace/flydsl-prof"

def split_inline_env(invocation: str):
    """Separate leading KEY=VAL env assignments from the actual command."""
    toks = shlex.split(invocation)
    env = {}
    i = 0
    while i < len(toks) and re.match(r'^[A-Z_][A-Z0-9_]*=', toks[i]):
        k, v = toks[i].split('=', 1)
        env[k] = v
        i += 1
    cmd = toks[i:]
    # ensure python prefix when the command begins with a .py path
    if cmd and cmd[0] not in ('python', 'python3') and cmd[0].endswith('.py'):
        cmd = ['python'] + cmd
    # resolve a bare/relative .py path against the repo's tests/kernels dir
    for j, tok in enumerate(cmd):
        if tok.endswith('.py') and not tok.startswith('-'):
            if not os.path.exists(os.path.join(ROOT, tok)):
                cand = os.path.join('tests', 'kernels', os.path.basename(tok))
                if os.path.exists(os.path.join(ROOT, cand)):
                    cmd[j] = cand
            break
    return env, cmd

# ---- generic perf parsers -------------------------------------------------
# Each returns dict of {label: us} / scalar; we keep everything we can find.
US_PATTERNS = [
    # "[Perf] FlyDSL softmax gpu: 24.7 us" / "[Perf] AIter softmax gpu: 75.7 us"
    (re.compile(r'\[Perf\]\s+(FlyDSL|AIter|AITER|Aiter|Reference|Torch|torch)\b[^:]*:\s*([\d.]+)\s*us', re.I), 'perf_us'),
    # "FlyDSL MoE stage1[fp8]: 12.3 us" / "[aiter] stage1: 9.9 us" / "ck-stage1: 8.0 us"
    (re.compile(r'\b(flydsl[-\w]*|aiter[-\w]*|ck[-\w]*|\[aiter\][-\w ]*)\s*:?\s*([\d.]+)\s*us', re.I), 'labeled_us'),
    # "Torch(us): 41.0, Speedup: 1.23" handled below
]
FLOAT = r'([-+]?\d+(?:\.\d+)?)'

def parse_metrics(text: str) -> dict:
    m = {}
    def grabf(pat, grp=1):
        r = re.search(pat, text, re.I)
        return float(r.group(grp)) if r else None
    # bandwidth / throughput / scalars
    m['bandwidth_gbs'] = grabf(r'Bandwidth:\s*'+FLOAT+r'\s*GB/s')
    m['tbps']          = grabf(r'\bTB/s[=:]\s*'+FLOAT) or grabf(r'\bBW:\s*'+FLOAT+r'\s*TB/s')
    m['tflops']        = grabf(FLOAT+r'\s*TFLOPS') or grabf(r'TFLOPS[=:]\s*'+FLOAT)
    m['kernel_avg_ms'] = grabf(r'Kernel avg time:\s*'+FLOAT+r'\s*ms')
    m['speedup']       = grabf(r'Speedup:\s*'+FLOAT+r'x?') or grabf(r'speedup[=:]\s*'+FLOAT+r'x?')
    us = {}
    def put(key, pat, grp=1):
        if key in us: return
        r = re.search(pat, text, re.I)
        if r:
            try: us[key] = float(r.group(grp))
            except Exception: pass
    # --- explicit "[Perf] FlyDSL/AIter/Ref ... : N us" ---
    put('flydsl', r'\[Perf\]\s*FlyDSL[^:]*:\s*'+FLOAT+r'\s*us')
    put('aiter',  r'\[Perf\]\s*(?:AIter|AITER|Aiter)[^:]*:\s*'+FLOAT+r'\s*us')
    put('reference', r'\[Perf\]\s*(?:Reference|Torch|torch)[^:]*:\s*'+FLOAT+r'\s*us')
    # --- rope style: "FlyDSL=215.9us AITER=38.7us" ---
    put('flydsl', r'FlyDSL=\s*'+FLOAT+r'\s*us')
    put('aiter',  r'(?:AITER|AIter|Aiter)=\s*'+FLOAT+r'\s*us')
    # --- [flyc] Throughput: N us ---
    put('flydsl', r'\[flyc\]\s*Throughput:\s*'+FLOAT+r'\s*us')
    # --- hgemm Torch(us): N ---
    put('torch', r'Torch\(us\):\s*'+FLOAT)
    # --- moe_blockscale text + table ---
    put('flydsl_total', r'flydsl:.*?total\s+'+FLOAT+r'\s*us')        # "= total 52.2us"
    put('aiter',        r'aiter fused:\s*'+FLOAT+r'\s*us')
    put('ck_s1',        r'ck stage1:\s*'+FLOAT+r'\s*us')
    put('ck_s2',        r'ck stage2:\s*'+FLOAT+r'\s*us')
    put('flydsl_s1',    r'flydsl-s1\s*\|\s*'+FLOAT)
    put('flydsl_s2',    r'flydsl-s2\s*\|\s*'+FLOAT)
    put('flydsl_total', r'flydsl-total\s*\|\s*'+FLOAT)
    put('aiter',        r'aiter-fused\s*\|\s*'+FLOAT)
    put('ck_s1',        r'ck-stage1\s*\|\s*'+FLOAT)
    put('ck_s2',        r'ck-stage2\s*\|\s*'+FLOAT)
    # --- moe_gemm two-stage: "FlyDSL MoE stage1[..]: N us", "[aiter] stage1: N us" ---
    put('flydsl_s1', r'FlyDSL MoE stage1[^:]*:\s*'+FLOAT+r'\s*us')
    put('flydsl_s2', r'FlyDSL MoE stage2[^:]*:\s*'+FLOAT+r'\s*us')
    put('aiter_s1',  r'\[aiter\]\s*stage1:\s*'+FLOAT+r'\s*us')
    put('aiter_s2',  r'\[aiter\]\s*stage2:\s*'+FLOAT+r'\s*us')
    # --- aiter gemm bench: "ck us", "asm us", "hipmm ... us" ---
    put('ck',   r'\bck us[^:]*[:=]?\s*'+FLOAT)
    put('asm',  r'\basm us[^:]*[:=]?\s*'+FLOAT)
    put('hipblaslt', r'hipmm[^:]*us[^:]*[:=]?\s*'+FLOAT)
    # --- pa / mla ---
    put('flydsl_ps', r'\bus_flydsl_ps\b[^\d]*'+FLOAT)
    put('gluon',     r'\bus_gluon\b[^\d]*'+FLOAT)
    put('flydsl_p50', r'us_p50=\s*'+FLOAT)
    # --- preshuffle v2: "v2 us / old us" ---
    put('v2',  r'\bv2 us[^\d]*'+FLOAT)
    put('old', r'\bold us[^\d]*'+FLOAT)
    # --- flash_attn table: first "| PASS | <err> <cos> | <us> <tflops>" row ---
    fr = re.search(r'\|\s*PASS\s*\|\s*[\d.eE+-]+\s+[\d.]+\s*\|\s*'+FLOAT+r'\s+'+FLOAT, text)
    if fr:
        us.setdefault('flydsl', float(fr.group(1)))
        if m['tflops'] is None:
            try: m['tflops'] = float(fr.group(2))
            except Exception: pass
    m['us'] = us
    # derived totals
    if 'flydsl_total' not in us and ('flydsl_s1' in us and 'flydsl_s2' in us):
        us['flydsl_total'] = round(us['flydsl_s1'] + us['flydsl_s2'], 2)
    if 'aiter_total' not in us and ('aiter_s1' in us and 'aiter_s2' in us):
        us['aiter_total'] = round(us['aiter_s1'] + us['aiter_s2'], 2)
    if 'ck_total' not in us and ('ck_s1' in us and 'ck_s2' in us):
        us['ck_total'] = round(us['ck_s1'] + us['ck_s2'], 2)
    m['flydsl_us'] = us.get('flydsl') or us.get('flydsl_total') or us.get('flydsl_ps') or us.get('flydsl_p50') or us.get('v2')
    m['aiter_us']  = us.get('aiter') or us.get('aiter_total')
    m['ck_us']     = us.get('ck') or us.get('ck_total')
    m['asm_us']    = us.get('asm')
    m['hipblaslt_us'] = us.get('hipblaslt')
    m['torch_us']  = us.get('torch') or us.get('reference') or us.get('gluon')
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True, help='test basename, e.g. test_rmsnorm.py')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--recipes', default=f"{PROF}/recipes.json")
    ap.add_argument('--outdir', default=f"{PROF}/results/sweep")
    args = ap.parse_args()

    recipes = json.load(open(args.recipes))
    rec = next((r for r in recipes if r['test_file'].split('/')[-1] == args.test), None)
    if rec is None:
        print(f"NO RECIPE for {args.test}", file=sys.stderr); sys.exit(2)

    os.makedirs(args.outdir, exist_ok=True)
    stem = args.test.replace('.py','')
    logp = Path(args.outdir)/f"{stem}.log"
    resp = Path(args.outdir)/f"{stem}.json"

    inline_env, cmd = split_inline_env(rec['invocation'])
    env = dict(os.environ)
    env['PYTHONPATH'] = f"{ROOT}/build-fly/python_packages:{ROOT}"
    env.update(inline_env)
    for k, v in (rec.get('shape_env') or {}).items():
        # only set simple scalar env vars (skip descriptive ones with spaces/commas-as-doc)
        if re.match(r'^[A-Za-z0-9_,.;:\- ]+$', str(v)) and not v.strip().endswith('format'):
            env.setdefault(k, str(v))
    # ROCDSL_COMPARE_AITER=1 also gates the FlyDSL gpu-us print in several harnesses
    env.setdefault('ROCDSL_COMPARE_AITER', '1')
    env['HIP_VISIBLE_DEVICES'] = str(args.gpu)   # override pin
    env['FLYDSL_RUNTIME_ENABLE_CACHE'] = '1'

    t0 = time.time()
    rc = None; timed_out = False
    with open(logp, 'w') as lf:
        lf.write(f"# CMD: {' '.join(cmd)}\n# ENV: HIP={args.gpu} PYTHONPATH set; extra={inline_env}\n\n")
        lf.flush()
        try:
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=args.timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    dt = time.time() - t0
    text = open(logp, errors='replace').read()
    metrics = parse_metrics(text)
    ok = (rc == 0) and not timed_out and (metrics.get('flydsl_us') is not None or metrics.get('tflops') is not None or metrics.get('bandwidth_gbs') is not None)
    passed = ('PASS' in text or 'Passed' in text or 'cos' in text.lower()) and 'FAIL' not in text.upper().replace('FAILURES=0','')

    result = {
        'test': args.test, 'kernels': rec.get('kernels'), 'op_category': rec.get('op_category'),
        'gpu': args.gpu, 'returncode': rc, 'timed_out': timed_out, 'wall_s': round(dt,1),
        'ok': bool(ok), 'correctness_seen': bool(passed),
        'metrics': metrics, 'cmd': ' '.join(cmd),
        'ck_baseline_candidate': rec.get('ck_baseline_candidate'),
        'aiter_comparable': rec.get('aiter_comparable'),
        'log': str(logp),
    }
    json.dump(result, open(resp,'w'), indent=2)
    print(json.dumps({k: result[k] for k in ('test','ok','returncode','timed_out','wall_s')}))
    print("  flydsl_us=%s aiter_us=%s ck_us=%s asm_us=%s tflops=%s tbps=%s gbs=%s speedup=%s" % (
        metrics.get('flydsl_us'), metrics.get('aiter_us'), metrics.get('ck_us'), metrics.get('asm_us'),
        metrics.get('tflops'), metrics.get('tbps'), metrics.get('bandwidth_gbs'), metrics.get('speedup')))

if __name__ == '__main__':
    main()
