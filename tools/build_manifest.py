#!/usr/bin/env python3
"""Build analysis_manifest.json: per-kernel paths + sweep record + capture summary
   for the report-writing phase. Also parses top hotspot source lines from hotspot_*.txt."""
import json, glob, os, re

PROF = "/sgl-workspace/flydsl-prof"
ROOT = "/sgl-workspace/FlyDSL-lab"

def parse_hotspot(txt_path):
    if not os.path.exists(txt_path): return {}
    t = open(txt_path, errors='replace').read()
    out = {}
    # stall breakdown rows: "VMEM-wait  100.9K  [..]  38.0%"
    stalls = []
    for m in re.finditer(r'^\s*([A-Za-z/\-]+)\s+([\d.]+[KM]?)\s+\[[^\]]*\]\s+([\d.]+)%', t, re.M):
        stalls.append({'type': m.group(1), 'stall': m.group(2), 'pct': float(m.group(3))})
    out['stall_breakdown'] = stalls[:8]
    # top source lines table
    srcs = []
    for m in re.finditer(r'^\s*\d+\s+([\d.]+[KM]?)\s+([\d.]+)%\s+\[[^\]]*\]\s+[\d.]+%\s+(\S+)\s+(\S+)', t, re.M):
        srcs.append({'stall': m.group(1), 'pct_total': float(m.group(2)), 'domtype': m.group(3), 'source': m.group(4)})
    out['top_source_lines'] = srcs[:12]
    # instruction mix
    mix = re.search(r'Instruction mix:\s*\n\s*(.*?)\n\s*(?:ds_read.*)?\n?═', t)
    mm = re.search(r'MFMA:\s*(\d+),\s*buffer_load:\s*(\d+),\s*buffer_store:\s*(\d+)', t)
    if mm: out['inst_mix'] = {'mfma': int(mm.group(1)), 'buffer_load': int(mm.group(2)), 'buffer_store': int(mm.group(3))}
    dm = re.search(r'ds_read:\s*(\d+),\s*ds_write:\s*(\d+)', t)
    if dm: out.setdefault('inst_mix', {}).update({'ds_read': int(dm.group(1)), 'ds_write': int(dm.group(2))})
    occ = re.search(r'-> (\d+) waves requires max\(arch,accum\) <= (\d+)', t)
    if occ: out['next_occ_step'] = {'waves': int(occ.group(1)), 'vgpr_budget': int(occ.group(2))}
    return out

def main():
    sweep = json.load(open(f"{PROF}/sweep_master.json"))
    srec = {r['test']: r for r in sweep['kernels']}
    manifest = {'provenance': sweep['provenance'], 'kernels': []}
    for capf in sorted(glob.glob(f"{PROF}/results/att/*/capture.json")):
        cap = json.load(open(capf))
        stem = os.path.basename(os.path.dirname(capf))
        test = stem if stem.endswith('.py') else stem + '.py'
        # resolve kernel source files
        rec = srec.get(test, {})
        ksrcs = []
        for k in (cap.get('kernels') or rec.get('kernels') or []):
            p = f"{ROOT}/kernels/{k}.py"
            if os.path.exists(p): ksrcs.append(p)
        hp = f"{PROF}/results/att/{stem}/hotspot_big.txt"
        entry = {
            'test': test, 'stem': stem,
            'jit_kernel': cap.get('kernel'),
            'op_category': rec.get('op_category'),
            'capture': {k: cap.get(k) for k in ['kernel','calls','iter_range','waves','ins','mapped','mapped_pct',
                        'arch','arch_vgpr','accum_vgpr','occupancy','stall_pct_total','top_stall_type','capture_s']},
            'capture_error': cap.get('error'),
            'sweep': {k: rec.get(k) for k in ['flydsl_us','baseline_us','baseline_name','speedup_vs_baseline',
                      'verdict','tflops','tbps','bandwidth_gbs','status','ck_baseline_candidate','aiter_comparable']},
            'hotspot': parse_hotspot(hp),
            'paths': {
                'capture_json': capf,
                'hotspot_txt': hp if os.path.exists(hp) else None,
                'bundle_dir': f"{PROF}/results/att/{stem}",
                'kernel_sources': ksrcs,
                'test_harness': f"{ROOT}/tests/kernels/{test}",
            },
        }
        manifest['kernels'].append(entry)
    json.dump(manifest, open(f"{PROF}/analysis_manifest.json",'w'), indent=2)
    ok = [e for e in manifest['kernels'] if not e['capture_error']]
    print(f"manifest: {len(manifest['kernels'])} kernels, {len(ok)} with valid ATT")
    for e in manifest['kernels']:
        c = e['capture']; s = e['sweep']
        top = (e['hotspot'].get('top_source_lines') or [{}])[0].get('source','-')
        print(f"  {e['stem'][:30]:31s} {str(c.get('top_stall_type') or '-'):14s} occ={c.get('occupancy')} ins={c.get('ins')} spd={s.get('speedup_vs_baseline')} verdict={s.get('verdict')}  hot1={top[:48]}")

if __name__ == '__main__':
    main()
