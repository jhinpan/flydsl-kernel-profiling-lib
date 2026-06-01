#!/usr/bin/env python3
"""FlyDSL ATT capture driver — one kernel, one GPU, per AGENTS.md.

Steps: fresh debug cache -> rocprofv3 --stats discovery -> pick FlyDSL kernel ->
write input_trace.yaml -> rocprofv3 ATT capture -> clean empty shells ->
verify source mapping -> hotspot_analyzer.py -> emit summary JSON + copy support files.

Usage:
  att_capture.py --test test_rmsnorm.py --gpu 0 --outdir DIR [--tag big]
                 [--iter-range "[6,[8-8]]"] [--buffer 0x6000000] [--cmd-override "..."]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, shlex, shutil, subprocess, sys, time
from pathlib import Path

ROOT = "/sgl-workspace/FlyDSL-lab"
PROF = "/sgl-workspace/flydsl-prof"
ANALYZER = f"{PROF}/drivers/hotspot_analyzer.py"
PY = "/opt/venv/bin/python"
ROCPROF = "rocprofv3"

NOISE = re.compile(r'at::native|rocclr|rocprim|Cijk|hipblas|rocblas|elementwise|reduce_kernel|vectorized|__amd|memset|memcpy|fill_|cast_|to_copy|triton_|sort|scan|cub::|::detail', re.I)

def split_inline_env(invocation: str):
    toks = shlex.split(invocation)
    env = {}; i = 0
    while i < len(toks) and re.match(r'^[A-Z_][A-Z0-9_]*=', toks[i]):
        k, v = toks[i].split('=', 1); env[k] = v; i += 1
    cmd = toks[i:]
    if cmd and cmd[0] not in ('python','python3') and cmd[0].endswith('.py'):
        cmd = ['python'] + cmd
    for j, tok in enumerate(cmd):
        if tok.endswith('.py') and not tok.startswith('-'):
            if not os.path.exists(os.path.join(ROOT, tok)):
                cand = os.path.join('tests','kernels', os.path.basename(tok))
                if os.path.exists(os.path.join(ROOT, cand)): cmd[j] = cand
            break
    return env, cmd

def build_env(rec, gpu, cache_dir):
    inline, cmd = split_inline_env(rec['invocation'])
    env = dict(os.environ)
    env['PYTHONPATH'] = f"{ROOT}/build-fly/python_packages:{ROOT}"
    env.update(inline)
    for k, v in (rec.get('shape_env') or {}).items():
        if re.match(r'^[A-Za-z0-9_,.;:\- ]+$', str(v)) and not str(v).strip().endswith('format'):
            env.setdefault(k, str(v))
    env['HIP_VISIBLE_DEVICES'] = str(gpu)
    env['ROCDSL_COMPARE_AITER'] = '0'          # keep capture clean: FlyDSL kernel only
    env['FLYDSL_DEBUG_ENABLE_DEBUG_INFO'] = '1' # DWARF line tables for source mapping
    env['FLYDSL_RUNTIME_CACHE_DIR'] = cache_dir # fresh debug cache (cold)
    env['FLYDSL_RUNTIME_ENABLE_CACHE'] = '1'
    return env, cmd

def discover_kernel(rec, env, cmd, outdir):
    disc = f"{outdir}/discover"
    p = subprocess.run([ROCPROF,'--stats','--kernel-trace','-f','csv','-o',disc,'--',*cmd],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    stats = f"{disc}_kernel_stats.csv"
    cands = []
    if os.path.exists(stats):
        with open(stats) as f:
            for row in csv.DictReader(f):
                name = row.get('Name') or row.get('KernelName') or ''
                if not name or NOISE.search(name): continue
                # numeric columns vary; grab calls + total/avg duration if present
                def num(*keys):
                    for k in keys:
                        if k in row and row[k] not in (None,''):
                            try: return float(row[k])
                            except: pass
                    return 0.0
                calls = int(num('Calls','TotalCalls') or 0)
                totdur = num('TotalDurationNs','TotalDuration','DurationNs')
                avgdur = num('AverageNs','AvgNs','Average')
                cands.append({'name':name,'calls':calls,'total_ns':totdur,'avg_ns':avgdur})
    hint = (rec.get('kernel_name_hint') or '').strip()
    hint_tok = re.split(r'[ (]', hint)[0] if hint else ''
    # prefer hint match, else longest total duration
    chosen = None
    if hint_tok:
        for c in cands:
            if hint_tok and hint_tok in c['name']: chosen = c; break
    for k in rec.get('kernels',[]):
        if chosen: break
        for c in cands:
            if k.replace('_kernel','') in c['name']: chosen = c; break
    if not chosen and cands:
        chosen = sorted(cands, key=lambda c:(c['total_ns'], c['calls']), reverse=True)[0]
    return chosen, cands, p.returncode

def auto_iter_range(calls):
    c = max(calls, 1)
    if c >= 14: skip = 6; m = 8
    elif c >= 6: skip = max(2, c//2 - 1); m = skip + 1
    elif c >= 3: skip = 1; m = 2
    else: skip = 0; m = max(1, c-1)
    return f"[{skip}, [{m}-{m}]]"

YAML = """GlobalParameters:
  KeepBuildTmp: True
  AsmDebug: True
jobs:
    -
        kernel_include_regex: "{regex}"
        kernel_iteration_range: "{iters}"
        output_file: out
        output_directory: {odir}
        output_format: [json, csv]
        truncate_kernels: true
        sys_trace: false
        advanced_thread_trace: true
        att_target_cu: {tcu}
        att_shader_engine_mask: "0xf"
        att_simd_select: "0xf"
        att_buffer_size: "{buf}"
pmc:
  - SQ_INSTS_VALU
  - SQ_INSTS_MFMA
  - SQ_INSTS_VMEM
  - SQ_WAVES
  - SQ_WAIT_INST_LDS
  - SQ_LDS_BANK_CONFLICT
  - GRBM_GUI_ACTIVE
"""

def inspect_dispatch(d):
    waves = len(glob.glob(f"{d}/se*_sm*_*.json"))
    ins = 0; mapped = 0
    cj = os.path.join(d, 'code.json')
    if os.path.exists(cj):
        try:
            code = json.load(open(cj)).get('code') or []
            ins = len(code)
            mapped = sum(1 for r in code if len(r) > 3 and r[3])
        except Exception: pass
    return waves, ins, mapped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--tag', default='big')
    ap.add_argument('--iter-range', default=None)
    ap.add_argument('--buffer', default='0x6000000')
    ap.add_argument('--target-cu', type=int, default=1)
    ap.add_argument('--recipes', default=f"{PROF}/recipes.json")
    ap.add_argument('--cmd-override', default=None, help='override invocation (e.g. larger shape for big)')
    ap.add_argument('--timeout', type=int, default=1200)
    args = ap.parse_args()

    recipes = json.load(open(args.recipes))
    rec = next((r for r in recipes if r['test_file'].split('/')[-1]==args.test), None)
    if not rec: print(f"NO RECIPE {args.test}", file=sys.stderr); sys.exit(2)
    if args.cmd_override: rec = {**rec, 'invocation': args.cmd_override}

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    cache = f"{outdir}/.flydsl_trace_cache"
    shutil.rmtree(cache, ignore_errors=True)
    env, cmd = build_env(rec, args.gpu, cache)
    result = {'test':args.test,'tag':args.tag,'gpu':args.gpu,'cmd':' '.join(cmd),'outdir':outdir}

    # Step: discovery
    chosen, cands, drc = discover_kernel(rec, env, cmd, outdir)
    if not chosen:
        result.update(error='no kernel discovered', discover_rc=drc, candidates=cands[:10])
        json.dump(result, open(f"{outdir}/capture.json",'w'), indent=2)
        print(json.dumps({'test':args.test,'error':'no_kernel'})); sys.exit(3)
    regex = re.escape(chosen['name']) if not re.search(r'[.*+?\[]', chosen['name']) else chosen['name']
    iters = args.iter_range or auto_iter_range(chosen['calls'])
    result.update(kernel=chosen['name'], calls=chosen['calls'], avg_ns=chosen.get('avg_ns'),
                  total_ns=chosen.get('total_ns'), iter_range=iters, buffer=args.buffer)

    # Step: write yaml + capture
    attdir = f"{outdir}/att"
    shutil.rmtree(attdir, ignore_errors=True); os.makedirs(attdir, exist_ok=True)
    yml = f"{outdir}/input_trace.yaml"
    open(yml,'w').write(YAML.format(regex=regex, iters=iters, odir=attdir, buf=args.buffer, tcu=args.target_cu))
    t0 = time.time()
    cap = subprocess.run([ROCPROF,'-i',yml,'--',*cmd], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=args.timeout)
    result['capture_rc'] = cap.returncode
    result['capture_s'] = round(time.time()-t0,1)
    open(f"{outdir}/capture_stdout.log",'w').write((cap.stdout or '')+"\n--STDERR--\n"+(cap.stderr or ''))

    # Step: inspect + clean empty shells
    disps = sorted(glob.glob(f"{attdir}/ui_output_agent_*"))
    kept = None; best = (-1,-1,-1)
    inv = []
    for d in disps:
        w,i,m = inspect_dispatch(d)
        inv.append({'dir':os.path.basename(d),'waves':w,'ins':i,'mapped':m})
        if (w,i,m) > best and w>0 and i>0: best=(w,i,m); kept=d
    # delete non-kept empty shells
    for d in disps:
        if d != kept:
            w,i,_ = inspect_dispatch(d)
            if w==0 or i==0: shutil.rmtree(d, ignore_errors=True)
    result['dispatch_inventory'] = inv
    if not kept:
        result.update(error='no valid ATT dispatch (all empty)')
        json.dump(result, open(f"{outdir}/capture.json",'w'), indent=2)
        print(json.dumps({'test':args.test,'kernel':chosen['name'],'error':'empty_att','iters':iters}))
        sys.exit(4)
    w,i,m = best
    result.update(kept_dir=os.path.basename(kept), waves=w, ins=i, mapped=m,
                  mapped_pct=round(100*m/max(i,1),1))

    # Step: hotspot analysis
    an = subprocess.run([PY, ANALYZER, kept, '--topk','15','--mode','both'],
                        capture_output=True, text=True, timeout=300)
    open(f"{outdir}/hotspot_{args.tag}.txt",'w').write(an.stdout + "\n" + an.stderr)
    txt = an.stdout
    def grab(pat):
        r = re.search(pat, txt); return r.group(1) if r else None
    result['arch'] = grab(r'Architecture:\s*(\S+)')
    result['arch_vgpr'] = grab(r'arch_vgpr:\s*~?(\d+)')
    result['accum_vgpr'] = grab(r'accum_vgpr:\s*(\d+)')
    result['occupancy'] = grab(r'occupancy:\s*(\d+)\s*waves')
    result['stall_pct_total'] = grab(r'Total stalls:.*\(([\d.]+)% of total')
    # top stall type (first row of breakdown)
    mt = re.search(r'Stall Breakdown by Type.*?\n(?:.*\n)*?\s+(\S+)\s+[\d.]+[KM]?\s+\[', txt)
    result['top_stall_type'] = mt.group(1) if mt else None

    # Step: copy support files
    for f in ['out_results.json','out_agent_info.csv']:
        src = f"{attdir}/{f}"
        if os.path.exists(src): shutil.copy(src, f"{outdir}/{args.tag}_{f}")
    for f in glob.glob(f"{outdir}/discover_*.csv"): pass  # already in outdir

    json.dump(result, open(f"{outdir}/capture.json",'w'), indent=2)
    print(json.dumps({'test':args.test,'kernel':chosen['name'],'calls':chosen['calls'],
                      'iters':iters,'kept':result.get('kept_dir'),'waves':w,'ins':i,
                      'mapped_pct':result.get('mapped_pct'),'arch':result.get('arch'),
                      'occ':result.get('occupancy'),'top_stall':result.get('top_stall_type'),
                      'capture_s':result['capture_s']}))

if __name__ == '__main__':
    main()
