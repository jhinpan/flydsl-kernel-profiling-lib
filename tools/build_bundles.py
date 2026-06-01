#!/usr/bin/env python3
"""Assemble examples/<kernel>/ bundles in flydsl-kernel-profiling per AGENTS.md output structure.
   Pulls trace + counters + discover CSVs + source + REPORT.md from results/att/<stem>/.
   Skips kernels whose example already exists with richer content (flash_attn_func)."""
import json, glob, os, shutil, re

PROF="/sgl-workspace/flydsl-prof"
ROOT="/sgl-workspace/FlyDSL-lab"
REPO="/sgl-workspace/flydsl-kernel-profiling"
ANALYZER=f"{PROF}/drivers/hotspot_analyzer.py"
SKIP={'flash_attn_func'}  # keep the existing richer 2-shape example

PRETTY=json.load(open(f"{PROF}/analysis_manifest.json"))  # for kernels list
RECIPES={r['test_file'].split('/')[-1]:r for r in json.load(open(f"{PROF}/recipes.json"))}
HEAD={}
if os.path.exists(f"{PROF}/headlines.json"):
    HEAD={h['stem']:h for h in json.load(open(f"{PROF}/headlines.json"))}

def cp(src,dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst),exist_ok=True); shutil.copy(src,dst); return True
    return False

def main():
    man={m['stem']:m for m in PRETTY['kernels']}
    built=[]
    for stem,m in man.items():
        cap=m['capture']
        if m.get('capture_error') or not cap.get('ins'): continue
        ex=stem.replace('test_','').replace('bench_','')
        if ex in SKIP:
            print(f"skip {ex} (existing example preserved)"); continue
        bd=f"{PROF}/results/att/{stem}"
        EX=f"{REPO}/examples/{ex}"
        for d in ['att_viewer/big','compute_viewer','source']:
            os.makedirs(f"{EX}/{d}",exist_ok=True)
        # trace (kept dispatch)
        kept=cap.get('kernel') and m['capture'].get('kept_dir') if False else None
        capj=json.load(open(f"{bd}/capture.json"))
        keptdir=capj.get('kept_dir')
        if keptdir and os.path.isdir(f"{bd}/att/{keptdir}"):
            dst=f"{EX}/att_viewer/big/{keptdir}"
            shutil.rmtree(dst,ignore_errors=True); shutil.copytree(f"{bd}/att/{keptdir}",dst)
        # counters
        cp(f"{bd}/big_out_results.json",f"{EX}/compute_viewer/big_results.json")
        cp(f"{bd}/big_out_agent_info.csv",f"{EX}/compute_viewer/big_agent_info.csv")
        for c in glob.glob(f"{bd}/discover_*.csv"): cp(c,f"{EX}/compute_viewer/{os.path.basename(c)}")
        # source
        for k in (m.get('kernels') or RECIPES.get(stem+'.py',{}).get('kernels') or []):
            cp(f"{ROOT}/kernels/{k}.py",f"{EX}/source/{k}.py")
        cp(m['paths']['test_harness'],f"{EX}/source/{os.path.basename(m['paths']['test_harness'])}")
        cp(f"{bd}/input_trace.yaml",f"{EX}/source/input_trace.yaml")
        cp(f"{bd}/hotspot_big.txt",f"{EX}/source/hotspot_output.txt")
        cp(ANALYZER,f"{EX}/source/hotspot_analyzer.py")
        # REPORT.md
        if not cp(f"{bd}/REPORT.md",f"{EX}/REPORT.md"):
            pass
        # README.md
        write_readme(EX,ex,stem,m,capj)
        built.append(ex)
    print(f"\nbuilt {len(built)} example bundles: {built}")

def write_readme(EX,ex,stem,m,capj):
    cap=m['capture']; sw=m['sweep']
    h=HEAD.get(stem,{})
    headline=h.get('headline') or _auto_headline(m)
    name=os.path.basename(EX)
    keptdir=capj.get('kept_dir','ui_output_agent_*')
    spd=sw.get('speedup_vs_baseline'); base=sw.get('baseline_name')
    md=f"""# {name} — FlyDSL kernel ATT bundle

**{headline}**

- **Kernel (JIT):** `{cap.get('kernel')}`  ·  **arch:** gfx950 / MI350X (CDNA4)
- **FlyDSL:** 0.1.9.dev594 @ 18c5a7ed  ·  **ROCm** 7.2.0  ·  **rocprofv3** 1.1.0  ·  captured 2026-06-01
- **Latency:** {sw.get('flydsl_us')} µs (FlyDSL){f" vs {base} {sw.get('baseline_us')} µs → {spd}× " if sw.get('baseline_us') else ""}
- **ATT:** {cap.get('ins')} ISA instructions, {cap.get('mapped_pct')}% source-mapped · {cap.get('waves')} waves sampled · occ {cap.get('occupancy')} waves/SIMD · top stall **{cap.get('top_stall_type')}** ({_p(cap.get('stall_pct_total'))} of cycles)

See **[REPORT.md](REPORT.md)** for the full instruction-level analysis and ranked optimization plan.

## Layout
```
{name}/
├── REPORT.md                         analysis writeup + ranked optimizations
├── att_viewer/big/{keptdir}/   ATT trace (load in AMD ATT Viewer)
├── compute_viewer/                   big_results.json (PMC counters), agent_info, discover_*.csv
└── source/                           kernel .py, test harness, input_trace.yaml, hotspot_analyzer.py, hotspot_output.txt
```

## Re-open the trace (no GPU)
```bash
cd att_viewer/big/{keptdir}/..
python3 -m http.server 8080   # open http://<host>:8080 → ATT Viewer
python3 ../../source/hotspot_analyzer.py {keptdir} --topk 15 --mode both
```

## Re-capture (needs MI350X)
```bash
python /sgl-workspace/flydsl-prof/drivers/att_capture.py --test {stem}.py --gpu 0 --outdir out --tag big
```
"""
    open(f"{EX}/README.md","w").write(md)

def _p(v):
    try: return f"{round(float(v))}%"
    except: return "—"
def _auto_headline(m):
    cap=m['capture']; sw=m['sweep']
    v=sw.get('verdict') or ''
    return f"{v or 'profiled'}: {cap.get('top_stall_type')}-bound ({_p(cap.get('stall_pct_total'))} stalled), occ {cap.get('occupancy')}/SIMD."

if __name__=='__main__':
    main()
