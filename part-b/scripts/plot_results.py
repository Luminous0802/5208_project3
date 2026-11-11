#!/usr/bin/env python3
"""Optional matplotlib plotter; summaries/measurements need only the Python standard library."""
import argparse,csv,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('summary');ap.add_argument('--graph-id',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    rows=[r for r in csv.DictReader(open(a.summary)) if r['graph_id']==a.graph_id]
    if not rows:raise ValueError('graph ID not found')
    if len({(r['site'],r['graph_sha256'],r['K'],r['source_sha256']) for r in rows})!=1:
        raise ValueError('choose one site, graph hash, K and source snapshot before plotting')
    rows.sort(key=lambda r:(r['version'],int(r['ranks']),int(r['threads']),r['schedule'],int(r['chunk'])))
    labels=[f"{r['version']} {r['ranks']}x{r['threads']}\n{r['schedule']} c={r['chunk']}" for r in rows]
    y=[float(r['iteration_s_median']) for r in rows]
    err=[[v-float(r['iteration_s_min']) for v,r in zip(y,rows)],[float(r['iteration_s_max'])-v for v,r in zip(y,rows)]]
    x=list(range(len(rows)));fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    axes[0].bar(x,y,yerr=err,capsize=3,color='#376B9F');axes[0].set_ylabel('Iteration loop time (s)');axes[0].set_title('Median and min-max range')
    for i,key in [(1,'speedup_serial'),(2,'efficiency')]:
        values=[float(r[key]) if r[key] else float('nan') for r in rows]
        axes[i].bar(x,values,color='#498779');axes[i].set_title('Speedup vs matched serial' if i==1 else 'Parallel efficiency')
    axes[1].plot(x,[int(r['cores']) for r in rows],'o--',color='#B65F3C',label='ideal');axes[1].legend()
    axes[2].axhline(1,color='#B65F3C',linestyle='--')
    for ax in axes:ax.set_xticks(x,labels,rotation=55,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f"{a.graph_id} | {rows[0]['site']} | K={rows[0]['K']}")
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'runtime_speedup_efficiency.png',dpi=180);fig.savefig(out/'runtime_speedup_efficiency.pdf');plt.close(fig)
if __name__=='__main__':main()
