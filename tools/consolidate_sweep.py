#!/usr/bin/env python3
"""Consolidate per-kernel sweep JSONs into sweep_master.json (normalized + provenance)."""
import json, glob, os, re

PROF = "/sgl-workspace/flydsl-prof"
PROV = {
    "flydsl_version": "0.1.9.dev594",
    "flydsl_commit": "18c5a7ed",
    "flydsl_branch": "docs/update-compile-pipeline",
    "gpu": "AMD Instinct MI350X (gfx950, CDNA4)",
    "gpu_count": 8,
    "rocm": "7.2.0",
    "rocprofv3": "1.1.0",
    "capture_date": "2026-06-01",
}

# manual patches for harnesses whose perf lives in a side CSV or that fail
PATCH = {
    "test_pa.py": {  # timing from run_pa_decode_ps_test*.csv
        "flydsl_us": 169.5, "baseline_us": 80.6, "baseline_name": "AIter Gluon (pa_decode_gluon)",
        "ok": True, "note": "timing read from harness CSV (us_flydsl_ps vs us_gluon); shape b=3 c=1027 nq=8 nkv=1 d=128 fp8 per_token",
    },
    "test_fp8_gemm_rowscale.py": {
        "ok": False, "status": "compile_fail",
        "note": "flyc.compile(): missing _reusable_slot_spec on fast-dispatch path (fp8_gemm_4wave, M=4096 N=4096 K=4096 tile256). Real finding: this kernel path fails to compile in this build.",
    },
}

def best_baseline(m):
    """pick the strongest external baseline us + label."""
    us = m.get('us', {})
    order = [('ck_total','CK'), ('ck','CK'), ('aiter','AIter'), ('aiter_total','AIter'),
             ('hipblaslt','hipBLASLt'), ('asm','CK-asm'), ('torch','PyTorch'), ('reference','reference'), ('gluon','AIter Gluon')]
    for k, label in order:
        if us.get(k): return us[k], label
    if m.get('aiter_us'): return m['aiter_us'], 'AIter'
    if m.get('torch_us'): return m['torch_us'], 'PyTorch'
    return None, None

def main():
    rows = []
    for f in sorted(glob.glob(f"{PROF}/results/sweep/*.json")):
        try: d = json.load(open(f))
        except Exception: continue
        if 'test' not in d or 'metrics' not in d: continue
        m = d['metrics']
        bl_us, bl_name = best_baseline(m)
        rec = {
            'test': d['test'], 'kernels': d.get('kernels'), 'op_category': d.get('op_category'),
            'flydsl_us': m.get('flydsl_us'),
            'baseline_us': bl_us, 'baseline_name': bl_name,
            'tflops': m.get('tflops'), 'tbps': m.get('tbps'), 'bandwidth_gbs': m.get('bandwidth_gbs'),
            'speedup_reported': m.get('speedup'),
            'us_detail': m.get('us'),
            'ok': d.get('ok'), 'correctness_seen': d.get('correctness_seen'),
            'wall_s': d.get('wall_s'),
            'ck_baseline_candidate': d.get('ck_baseline_candidate'),
            'aiter_comparable': d.get('aiter_comparable'),
            'status': 'ok' if d.get('ok') else 'parse_or_run_issue',
        }
        rows.append(rec)
    # apply patches
    by = {r['test']: r for r in rows}
    for t, patch in PATCH.items():
        r = by.get(t) or {'test': t, 'op_category': 'gemm' if 'gemm' in t else 'attention'}
        r.update(patch)
        if t not in by: rows.append(r)
    # derived speedup vs baseline (>1 => FlyDSL faster)
    for r in rows:
        f, b = r.get('flydsl_us'), r.get('baseline_us')
        r['speedup_vs_baseline'] = round(b / f, 3) if (f and b) else None
        # recompute status post-patch
        if r.get('status') != 'compile_fail':
            r['status'] = 'ok' if r.get('ok') else 'parse_or_run_issue'
        if r['speedup_vs_baseline'] is not None:
            r['verdict'] = 'FlyDSL faster' if r['speedup_vs_baseline'] >= 1.05 else ('comparable' if r['speedup_vs_baseline'] >= 0.95 else 'FlyDSL slower')
        else:
            r['verdict'] = None
    rows.sort(key=lambda r: (r['op_category'] or 'z', r['test']))
    out = {'provenance': PROV, 'kernels': rows,
           'deferred': [
               {'test':'test_allreduce.py','reason':'multi-GPU + torch.distributed; needs 2+ ranks','op_category':'comm'},
               {'test':'test_flydsl_shmem.py','reason':'mori shmem, 2 PEs, no perf print','op_category':'comm'},
           ]}
    json.dump(out, open(f"{PROF}/sweep_master.json",'w'), indent=2)
    # print table
    print(f"{'test':34s} {'cat':6s} {'fly_us':>8s} {'base_us':>8s} {'base':10s} {'spd':>5s} {'tflops':>7s} {'verdict':14s}")
    print('-'*100)
    for r in rows:
        def s(v,f="%.1f"): return (f%v) if isinstance(v,(int,float)) else '-'
        print(f"{r['test'][:33]:34s} {(r['op_category'] or '')[:6]:6s} {s(r.get('flydsl_us')):>8s} {s(r.get('baseline_us')):>8s} {str(r.get('baseline_name') or '-')[:10]:10s} {s(r.get('speedup_vs_baseline'),'%.2f'):>5s} {s(r.get('tflops')):>7s} {str(r.get('verdict') or r.get('status'))[:14]:14s}")
    print(f"\n{len(rows)} kernels consolidated -> sweep_master.json")

if __name__ == '__main__':
    main()
