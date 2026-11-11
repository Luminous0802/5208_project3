#!/usr/bin/env python3
"""Local CSV analysis/plotting only. Never compiles, submits or runs a solver."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import platform
from lanczos_common import VARIANTS,summarize,validate_rows,write_csv

def plot(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=sorted({r['model'] for r in summary})
    times,ta=plt.subplots(len(models),2,figsize=(12,3.6*len(models)),squeeze=False,layout='constrained')
    accuracy,aa=plt.subplots(len(models),2,figsize=(12,3.6*len(models)),squeeze=False,layout='constrained')
    for i,model in enumerate(models):
        for variant in VARIANTS:
            rows=sorted([r for r in summary if r['model']==model and r['variant']==variant['name']],key=lambda r:int(r['budget']))
            if not rows:continue
            for j,key in enumerate(['solve_s','end_to_end_s']):
                median=[r[key+'_median'] for r in rows]
                ta[i,j].errorbar([int(r['budget']) for r in rows],median,
                    yerr=[[v-r[key+'_min'] for v,r in zip(median,rows)],[r[key+'_max']-v for v,r in zip(median,rows)]],fmt='o-',capsize=2,label=variant['name'])
            residual=[max(r['residual_median'],1e-17) for r in rows]
            # Requested budgets are used because actual early-breakdown step
            # counts are explicitly retained as sets in the summary CSV.
            aa[i,0].plot([int(r['budget']) for r in rows],residual,'o-',label=variant['name'])
            aa[i,1].plot([r['solve_s_median'] for r in rows],residual,'o-',label=variant['name'])
        for j in range(2):
            ta[i,j].set_xscale('log');ta[i,j].set_yscale('log');ta[i,j].set_xlabel('Requested Lanczos steps')
            ta[i,j].set_ylabel(model+(': solve time (s)' if j==0 else ': application time after MPI init (s)'))
            aa[i,j].set_xscale('log');aa[i,j].set_yscale('log');aa[i,j].set_ylabel(model+': direct residual')
            aa[i,j].axhline(1e-8,ls=':',c='gray')
        aa[i,0].set_xlabel('Requested Lanczos steps');aa[i,1].set_xlabel('Complete solve time (s)')
        for ax in [*ta[i],*aa[i]]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    title=f"Vanda job {summary[0]['job_id']} | FP64 lazy Lanczos | one physical GPU | {summary[0]['stage']}"
    times.suptitle(title);accuracy.suptitle(title)
    out.mkdir(parents=True,exist_ok=True)
    for fig,name in [(times,'timings'),(accuracy,'accuracy')]:
        for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=180)
        plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('runs',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--plot-dir',type=Path,required=True);a=ap.parse_args()
    source=a.runs.expanduser().resolve();before=hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='') as f:rows=list(csv.DictReader(f))
    validate_rows(rows)
    if a.out.exists() or (a.plot_dir.exists() and any(a.plot_dir.iterdir())):raise FileExistsError('Choose fresh summary and plot paths')
    summary=summarize(rows);write_csv(a.out,summary);plot(summary,a.plot_dir)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=before:raise RuntimeError('Input changed during analysis')
    record=dict(analysis_host=platform.node(),measurement_site='Vanda',job_id=rows[0]['job_id'],input_sha256=before,input_rows=len(rows),
                completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),note='Existing CSV only; no solver, graph generation or scheduler command was executed')
    a.out.with_suffix('.analysis.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'GPU Lanczos analysis: {len(rows)} rows, {len(summary)} configurations')
if __name__=='__main__':main()
