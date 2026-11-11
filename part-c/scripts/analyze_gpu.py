#!/usr/bin/env python3
import argparse,csv,hashlib,pathlib,statistics
COMMON=['measurement_id','job_id','hostname','site','hardware_tag','build_tag','source_sha256','graph_id','graph_sha256','model','seed','N','E','K','precision']
KEYS=[*COMMON,'backend','mapping','block_size','gpu_name','ranks','threads','partition','schedule','chunk','cpu_cores']
TIMES=['iteration_s','gpu_timeline_s','h2d_s','d2h_s','transfer_inclusive_s','end_to_end_s','validation_L1']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+');ap.add_argument('--out',required=True);ap.add_argument('--plot-dir');a=ap.parse_args()
    groups={}
    for file in a.runs:
        for row in csv.DictReader(open(file)):
            # Old files remain readable, but distinct legacy files are never silently paired.
            row.setdefault('measurement_id','legacy:'+hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest())
            for k in ['job_id','hostname']:row.setdefault(k,'')
            for k in ['ranks','threads']:row.setdefault(k,'0' if row['backend']=='cuda' else '1')
            for k in ['partition','schedule']:row.setdefault(k,'none')
            row.setdefault('chunk','0');row.setdefault('cpu_cores',str(int(row['ranks'])*int(row['threads'])))
            groups.setdefault(tuple(row[k] for k in KEYS),[]).append(row)
    result=[]
    for key,rows in groups.items():
        s=dict(zip(KEYS,key));s['samples']=len(rows)
        for field in TIMES:
            values=[float(r[field]) for r in rows if r.get(field,'')!='']
            for label,fn in [('median',statistics.median),('min',min),('max',max)]:s[field+'_'+label]=fn(values) if values else ''
        result.append(s)
    def match(s):return tuple(s[k] for k in COMMON)
    baselines={match(s):s for s in result if s['backend']=='cpu_serial'}
    hybrids={}
    for s in result:
        if s['backend']=='cpu_hybrid' and (match(s) not in hybrids or s['iteration_s_median']<hybrids[match(s)]['iteration_s_median']):
            hybrids[match(s)]=s
    for s in result:
        b=baselines.get(match(s));s['loop_speedup_vs_cpu']='';s['transfer_inclusive_speedup_vs_cpu']=''
        s['loop_speedup_vs_best_tested_hybrid']='';s['transfer_inclusive_speedup_vs_best_tested_hybrid']='';s['hybrid_baseline']=''
        s['comparison_note']='no matching CPU baseline'
        if b and s['backend']=='cuda':
            s['loop_speedup_vs_cpu']=b['iteration_s_median']/s['iteration_s_median']
            s['transfer_inclusive_speedup_vs_cpu']=b['iteration_s_median']/s['transfer_inclusive_s_median']
            s['comparison_note']='same-graph FP64 fixed-K CPU; transfer-inclusive excludes allocation, warmup and graph parsing'
        h=hybrids.get(match(s))
        if h and s['backend']=='cuda':
            s['loop_speedup_vs_best_tested_hybrid']=h['iteration_s_median']/s['iteration_s_median']
            s['transfer_inclusive_speedup_vs_best_tested_hybrid']=h['iteration_s_median']/s['transfer_inclusive_s_median']
            s['hybrid_baseline']=f"p{h['ranks']}t{h['threads']} {h['partition']} {h['schedule']} chunk{h['chunk']}"
            s['comparison_note']+='; hybrid baseline is fastest median among TESTED layouts in this measurement, not a global optimum'
    path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    columns=[*KEYS,'samples',*[field+'_'+label for field in TIMES for label in ['median','min','max']],
      'loop_speedup_vs_cpu','transfer_inclusive_speedup_vs_cpu','loop_speedup_vs_best_tested_hybrid',
      'transfer_inclusive_speedup_vs_best_tested_hybrid','hybrid_baseline','comparison_note']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(result)
    if not a.plot_dir:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=pathlib.Path(a.plot_dir);out.mkdir(parents=True,exist_ok=True)
    for gid in sorted({r['graph_id'] for r in result}):
        rows=sorted([r for r in result if r['graph_id']==gid],key=lambda r:(r['backend'],r['mapping'],int(r['block_size'])))
        if len({match(r) for r in rows})!=1:raise ValueError('plot one compatible hardware/build/graph group at a time')
        labels=['CPU 1 core' if r['backend']=='cpu_serial' else
                f"CPU {r['ranks']}x{r['threads']} {r['partition']} {r['schedule']}" if r['backend']=='cpu_hybrid' else
                f"GPU {r['mapping']} b{r['block_size']}" for r in rows]
        y=[r['iteration_s_median'] for r in rows]
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        axes[0].bar(range(len(rows)),y,yerr=[[v-r['iteration_s_min'] for v,r in zip(y,rows)],[r['iteration_s_max']-v for v,r in zip(y,rows)]],capsize=3)
        axes[0].set_ylabel('Host-observed iteration loop (s)');axes[0].set_xticks(range(len(rows)),labels,rotation=30,ha='right')
        gpu=[r for r in rows if r['backend']=='cuda'];x=list(range(len(gpu)))
        axes[1].bar([v-.18 for v in x],[float(r['loop_speedup_vs_cpu']) if r['loop_speedup_vs_cpu'] else float('nan') for r in gpu],.36,label='loop')
        axes[1].bar([v+.18 for v in x],[float(r['transfer_inclusive_speedup_vs_cpu']) if r['transfer_inclusive_speedup_vs_cpu'] else float('nan') for r in gpu],.36,label='H2D + loop + D2H')
        if any(r['loop_speedup_vs_best_tested_hybrid']!='' for r in gpu):
            axes[1].plot(x,[float(r['loop_speedup_vs_best_tested_hybrid']) if r['loop_speedup_vs_best_tested_hybrid'] else float('nan') for r in gpu],
                         'o--',color='black',label='loop vs best tested hybrid')
        axes[1].set_xticks(x,[f"{r['mapping']} b{r['block_size']}" for r in gpu],rotation=30,ha='right');axes[1].set_ylabel('Speedup vs matched CPU loop');axes[1].legend()
        for ax in axes:ax.grid(axis='y',alpha=.2)
        fig.suptitle(f"{gid} | {rows[0]['site']} | FP64, fixed K={rows[0]['K']}")
        fig.savefig(out/(gid+'.png'),dpi=180);fig.savefig(out/(gid+'.pdf'));plt.close(fig)
if __name__=='__main__':main()
