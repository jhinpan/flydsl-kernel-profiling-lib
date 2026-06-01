#!/usr/bin/env python3
"""Merge sweep_master.json + analysis_manifest.json (+ baselines.json if present)
   into the dashboard dataset docs/data/kernels.json."""
import json, os, re

PROF = "/sgl-workspace/flydsl-prof"
DOCS = "/sgl-workspace/flydsl-kernel-profiling/docs"

PRETTY = {
    'test_softmax':'Softmax', 'test_rmsnorm':'RMSNorm', 'test_layernorm':'LayerNorm',
    'test_topk_gating_softmax':'TopK Gating Softmax', 'test_quant':'Per-Token Quant',
    'test_fused_rope_cache':'Fused RoPE + KV-Cache', 'test_pa':'Paged-Attn Decode (PS)',
    'test_mla_decode':'MLA Decode (fp8)', 'test_flash_attn_func':'Flash Attention',
    'test_hgemm_splitk':'HGEMM Split-K', 'test_preshuffle_gemm':'Preshuffle GEMM',
    'bench_preshuffle_gemm_v2':'Preshuffle GEMM v2', 'test_blockscale_preshuffle_gemm':'Block-Scale Preshuffle GEMM',
    'test_fp8_gemm_rowscale':'FP8 Row-Scale GEMM', 'test_moe_gemm':'MoE GEMM (2-stage)',
    'test_moe_blockscale':'MoE Block-Scale (2-stage)', 'test_moe_reduce':'MoE Reduction',
    'test_vec_add':'Vector Add',
}

def fnum(v):
    return v if isinstance(v,(int,float)) else None

def main():
    sweep = json.load(open(f"{PROF}/sweep_master.json"))
    manifest = json.load(open(f"{PROF}/analysis_manifest.json"))
    mbyt = {m['test']: m for m in manifest['kernels']}
    baselines = {}
    bp = f"{PROF}/baselines.json"
    if os.path.exists(bp):
        for b in json.load(open(bp)):
            baselines[b['test']] = b
    heads = {}
    hp = f"{PROF}/headlines.json"
    if os.path.exists(hp):
        for h in json.load(open(hp)):
            heads[h['stem']] = h

    kernels = []
    for r in sweep['kernels']:
        test = r['test']; stem = test.replace('.py','')
        man = mbyt.get(test, {})
        cap = man.get('capture', {}); hot = man.get('hotspot', {})
        bl = baselines.get(test, {})
        # prefer richer baseline from the baseline workflow if present
        baseline_us = fnum(bl.get('baseline_us')) or fnum(r.get('baseline_us'))
        baseline_name = bl.get('baseline_impl') or r.get('baseline_name')
        flydsl_us = fnum(bl.get('flydsl_us')) or fnum(r.get('flydsl_us'))
        speedup = fnum(bl.get('speedup_flydsl_vs_baseline')) or fnum(r.get('speedup_vs_baseline'))
        if speedup is None and baseline_us and flydsl_us:
            speedup = round(baseline_us/flydsl_us, 3)
        verdict = r.get('verdict')
        if speedup is not None:
            verdict = 'FlyDSL faster' if speedup>=1.05 else ('comparable' if speedup>=0.95 else 'FlyDSL slower')
        example = stem.replace('test_','').replace('bench_','')
        rec = {
            'name': PRETTY.get(stem, stem), 'test': test, 'stem': stem,
            'example': example,
            'report_url': f"../examples/{example}/REPORT.md",
            'bundle_url': f"../examples/{example}/",
            'op_category': r.get('op_category'), 'kernels': r.get('kernels'),
            'jit_kernel': cap.get('kernel'),
            'flydsl_us': flydsl_us, 'baseline_us': baseline_us, 'baseline_name': baseline_name,
            'speedup_vs_baseline': speedup, 'verdict': verdict,
            'tflops': fnum(r.get('tflops')) or fnum(bl.get('flydsl_tflops')),
            'baseline_tflops': fnum(bl.get('baseline_tflops')),
            'tbps': fnum(r.get('tbps')), 'bandwidth_gbs': fnum(r.get('bandwidth_gbs')),
            'extra_baselines': bl.get('extra_baselines') or {},
            'baseline_source': bl.get('source'),
            'baseline_notes': bl.get('notes'),
            # ATT
            'calls': cap.get('calls'), 'waves': cap.get('waves'), 'ins': cap.get('ins'),
            'mapped_pct': cap.get('mapped_pct'), 'occupancy': cap.get('occupancy'),
            'arch_vgpr': cap.get('arch_vgpr'), 'stall_pct_total': fnum(_f(cap.get('stall_pct_total'))),
            'top_stall_type': cap.get('top_stall_type'),
            'stall_breakdown': hot.get('stall_breakdown', []),
            'top_source_lines': [_clean_src(s) for s in hot.get('top_source_lines', [])],
            'inst_mix': hot.get('inst_mix', {}),
            'next_occ_step': hot.get('next_occ_step'),
            'headline': (heads.get(stem) or {}).get('headline'),
            'top_recommendation': (heads.get(stem) or {}).get('top_recommendation'),
            'bound_type': (heads.get(stem) or {}).get('bound_type'),
            'status': r.get('status'),
            'ck_candidate': r.get('ck_baseline_candidate'),
            'aiter_comparable': r.get('aiter_comparable'),
            'capture_error': man.get('capture_error'),
            'has_bundle': bool(cap.get('ins')),
        }
        kernels.append(rec)

    # headroom ranking: kernels where FlyDSL is slower, ranked by how much
    headroom = sorted([k for k in kernels if k['speedup_vs_baseline'] and k['speedup_vs_baseline']<0.95],
                      key=lambda k:k['speedup_vs_baseline'])
    wins = sorted([k for k in kernels if k['speedup_vs_baseline'] and k['speedup_vs_baseline']>=1.05],
                  key=lambda k:-k['speedup_vs_baseline'])
    cats = {}
    for k in kernels:
        cats.setdefault(k['op_category'] or 'other', []).append(k['stem'])

    out = {
        'provenance': sweep['provenance'],
        'generated': '2026-06-01',
        'summary': {
            'total': len(kernels),
            'with_att': sum(1 for k in kernels if k['has_bundle']),
            'with_baseline': sum(1 for k in kernels if k['baseline_us']),
            'wins': len(wins), 'headroom': len(headroom),
            'categories': {c: len(v) for c,v in cats.items()},
        },
        'headroom_ranking': [k['stem'] for k in headroom],
        'win_ranking': [k['stem'] for k in wins],
        'kernels': kernels,
        'deferred': sweep.get('deferred', []),
    }
    os.makedirs(f"{DOCS}/data", exist_ok=True)
    json.dump(out, open(f"{DOCS}/data/kernels.json",'w'), indent=2)
    with open(f"{DOCS}/data/data.js",'w') as f:
        f.write("window.KERNEL_DATA = ")
        json.dump(out, f, indent=2)
        f.write(";\n")
    print(f"exported {len(kernels)} kernels -> {DOCS}/data/kernels.json (+ data.js)")
    print(f"  with_att={out['summary']['with_att']} with_baseline={out['summary']['with_baseline']} wins={out['summary']['wins']} headroom={out['summary']['headroom']}")
    print(f"  headroom: {out['headroom_ranking']}")
    print(f"  wins: {out['win_ranking']}")

def _f(v):
    try: return float(v)
    except Exception: return None

def _clean_src(s):
    src = s.get('source','')
    # keep file:line tail (analyzer truncates the left)
    m = re.search(r'([\w./\-]+\.py:\d+)$', src)
    s = dict(s)
    s['source'] = m.group(1) if m else src
    return s

if __name__ == '__main__':
    main()
