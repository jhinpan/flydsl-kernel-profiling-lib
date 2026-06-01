#!/usr/bin/env python3
"""Run run_sweep.py over many kernels with a free-GPU pool (no GPU collisions).

Each kernel gets a dedicated GPU for its lifetime; a GPU is returned to the pool
only when its job exits. Up to NGPU jobs run concurrently.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

PROF = "/sgl-workspace/flydsl-prof"
DRIVER = f"{PROF}/drivers/run_sweep.py"
PY = "/opt/venv/bin/python"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', required=True, help='comma-separated test basenames')
    ap.add_argument('--ngpu', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--outdir', default=f"{PROF}/results/sweep")
    ap.add_argument('--att', action='store_true', help='run att_capture.py instead of run_sweep.py')
    args = ap.parse_args()

    tests = [t.strip() for t in args.tests.split(',') if t.strip()]
    free = list(range(args.ngpu))
    running = {}   # pid_obj -> (test, gpu, t0)
    queue = list(tests)
    done = []
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    statusf = Path(args.outdir)/"pool.status"
    statusf.write_text("")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(statusf, 'a') as f: f.write(line+"\n")

    log(f"pool start: {len(tests)} tests, {args.ngpu} GPUs")
    while queue or running:
        # launch while GPUs free and work queued
        while queue and free:
            t = queue.pop(0); g = free.pop(0)
            stem = t.replace('.py','')
            if args.att:
                drv = f"{PROF}/drivers/att_capture.py"
                od = f"{args.outdir}/{stem}"
                cmd = [PY, drv, '--test', t, '--gpu', str(g), '--outdir', od,
                       '--tag', 'big', '--timeout', str(args.timeout)]
            else:
                cmd = [PY, DRIVER, '--test', t, '--gpu', str(g),
                       '--timeout', str(args.timeout), '--outdir', args.outdir]
            lf = open(Path(args.outdir)/f"{stem}.pool.out", 'w')
            Path(args.outdir).mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            running[p] = (t, g, time.time(), lf)
            log(f"launch {t} on gpu{g} (pid {p.pid})")
        # reap finished
        time.sleep(2)
        for p in list(running):
            if p.poll() is not None:
                t, g, t0, lf = running.pop(p)
                lf.close()
                free.append(g)
                dt = round(time.time()-t0, 1)
                stem = t.replace('.py','')
                if args.att:
                    rj = Path(args.outdir)/stem/"capture.json"; detail=''
                    if rj.exists():
                        try:
                            d=json.load(open(rj))
                            detail=f"kernel={d.get('kernel')} waves={d.get('waves')} ins={d.get('ins')} mapped%={d.get('mapped_pct')} stall={d.get('top_stall_type')} err={d.get('error')}"
                        except Exception: detail='(parse err)'
                    log(f"DONE {t} gpu{g} rc={p.returncode} {dt}s {detail}")
                else:
                    rj = Path(args.outdir)/f"{stem}.json"
                    ok = '?'; flydsl=None
                    if rj.exists():
                        try:
                            d = json.load(open(rj)); ok = d.get('ok'); flydsl = d.get('metrics',{}).get('flydsl_us')
                        except Exception: pass
                    log(f"DONE {t} gpu{g} rc={p.returncode} {dt}s ok={ok} flydsl_us={flydsl}")
                done.append(t)
    log(f"pool complete: {len(done)} tests finished")

if __name__ == '__main__':
    main()
