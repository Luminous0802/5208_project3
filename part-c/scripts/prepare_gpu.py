#!/usr/bin/env python3
"""Exact N/E graph matrix for CPU/GPU comparison; planning does not imply execution."""
import argparse,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='work/gpu_manifest.json');ap.add_argument('--graph-dir',default='work/gpu_graphs')
    ap.add_argument('--sizes',default='100000,500000,1000000');ap.add_argument('--structure-N',type=int,default=1000000)
    ap.add_argument('--seeds',default='20261029');ap.add_argument('--generate',action='store_true');a=ap.parse_args();graphs={}
    def add(model,n,degree,seed,options):
        suffix='_shuffled' if '--shuffle' in options else ''
        gid=f'{model}_n{n}_d{degree}_s{seed}{suffix}'
        graphs[gid]=dict(graph_id=gid,path=str(pathlib.Path(a.graph_dir)/(gid+'.txt')),model=model,N=n,E=n*degree,seed=seed,options=options)
    for seed in map(int,a.seeds.split(',')):
        for n in map(int,a.sizes.split(',')):add('uniform',n,8,seed,[])
        for d in [4,8,16]:add('uniform',a.structure_N,d,seed,[])
        add('hub',a.structure_N,8,seed,[])
        add('community',a.structure_N,8,seed,['--communities','4','--mix','0.1'])
        add('community',a.structure_N,8,seed,['--communities','4','--mix','0.1','--shuffle','1'])
    if a.generate:
        for g in graphs.values():
            path=ROOT/g['path'];path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError(f'refusing to overwrite {path}; use a fresh --graph-dir')
            subprocess.run([str(ROOT/'bin/make_graph'),g['model'],str(g['N']),str(g['E']),str(g['seed']),str(path),*g['options']],check=True)
    out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(list(graphs.values()),indent=2)+'\n')
    print(f'{len(graphs)} graph configurations: {out}; '+('generated' if a.generate else 'PLAN ONLY'))
if __name__=='__main__':main()
