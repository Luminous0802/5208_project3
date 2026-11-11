#!/usr/bin/env python3
"""Small serial numerical experiment: two-pass reorth vs uncorrected three-term recurrence."""
import argparse,csv,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--no-plot',action='store_true');a=ap.parse_args()
    out=ROOT/'results/local_lanczos';out.mkdir(parents=True,exist_ok=True)
    work=ROOT/'work/lanczos_diagnostic';work.mkdir(parents=True,exist_ok=True);graph=work/'uniform_n256_e2048.txt'
    subprocess.run([str(ROOT/'bin/make_graph'),'uniform','256','2048','20261029',str(graph)],check=True)
    rows=[]
    for seed in [20261029,20261030,20261031]:
        for reorth in ['full','none']:
            history=out/f'{reorth}_s{seed}.history.csv'
            command=[str(ROOT/'bin/lanczos'),str(graph),'50','--seed',str(seed),'--reorth',reorth,'--orthogonality','1','--history',str(history)]
            p=subprocess.run(command,capture_output=True,text=True,check=True)
            (out/f'{reorth}_s{seed}.stdout').write_text(p.stdout)
            m=dict(line.split(': ',1) for line in p.stdout.splitlines() if ': ' in line)
            rows.append(dict(seed=seed,reorth=reorth,steps=m['steps'],lambda_max_ritz=m['lambda_max_ritz'],
              ritz_residual=m['ritz_residual'],orthogonality_max=m['orthogonality_max']))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'provenance.json').write_text(json.dumps(dict(site='local',N=256,input_E=2048,graph_seed=20261029,steps=50,
       matrix='simple undirected union, S=D^-1/2 A D^-1/2',timing_claim='No cluster runtime claim; numerical diagnostics only'),indent=2)+'\n')
    if a.no_plot:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for reorth in ['full','none']:
        history=list(csv.DictReader((out/f'{reorth}_s20261029.history.csv').open()))
        axes[0].semilogy([int(r['step']) for r in history],[max(float(r['residual_estimate']),1e-18) for r in history],label=reorth)
    axes[0].set(xlabel='Lanczos step',ylabel='Projected residual estimate',title='Local numerical diagnostic; same graph / start');axes[0].legend();axes[0].grid(alpha=.2)
    labels=[f"{r['reorth']}\nseed {r['seed']}" for r in rows];axes[1].bar(range(len(rows)),[max(float(r['orthogonality_max']),1e-18) for r in rows]);axes[1].set_yscale('log');axes[1].set_xticks(range(len(rows)),labels,rotation=35,ha='right');axes[1].set_ylabel('max |Q^T Q - I|');axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Lanczos stability check (local, not cluster performance)')
    fig.savefig(ROOT/'figures/lanczos_local_diagnostics.png',dpi=180);fig.savefig(ROOT/'figures/lanczos_local_diagnostics.pdf');plt.close(fig)
if __name__=='__main__':main()
