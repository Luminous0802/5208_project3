#!/usr/bin/env python3
"""Write an explicit graph manifest; --generate creates graphs, SHA256, and metadata."""
import argparse, hashlib, json, pathlib, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='work/experiments');ap.add_argument('--N',type=int,default=500000)
    ap.add_argument('--seeds',default='20261029,20261030,20261031')
    ap.add_argument('--sizes',default='100000,500000,1000000,2000000')
    ap.add_argument('--degrees',default='4,8,16,32');ap.add_argument('--generate',action='store_true')
    ap.add_argument('--experiments',default='E1,E2,E3,E4');a=ap.parse_args()
    folder=(ROOT/a.out).resolve();folder.mkdir(parents=True,exist_ok=True)
    selected=set(a.experiments.split(',')); graphs={}
    def add(n,degree,seed,model,experiment,mix=None,shuffle=False):
        gid=f'{model}_n{n}_d{degree}_s{seed}'+(f'_mix{mix}_'+('shuffled' if shuffle else 'ordered') if mix is not None else ('_shuffled' if shuffle else ''))
        if gid not in graphs:
            options=[]
            if model=='hub' and shuffle:options=['--shuffle','1']
            if model=='community':options=['--communities','4','--mix',str(mix),'--shuffle',str(int(shuffle))]
            graphs[gid]=dict(graph_id=gid,path=str(folder/(gid+'.txt')),N=n,E=n*degree,seed=seed,model=model,
                options=options,experiments=[],mix=mix,shuffled=shuffle)
        if experiment not in graphs[gid]['experiments']:graphs[gid]['experiments'].append(experiment)
    for seed in map(int,a.seeds.split(',')):
        if 'E1' in selected:
            for degree in map(int,a.degrees.split(',')):add(a.N,degree,seed,'uniform','E1')
        if 'E2' in selected:
            for model in ['uniform','hub']:add(a.N,8,seed,model,'E2')
            add(a.N,8,seed,'hub','E2',shuffle=True)
        if 'E3' in selected:
            for mix in [0.01,0.1,0.5]:
                for shuffle in [False,True]:add(a.N,8,seed,'community','E3',mix,shuffle)
        if 'E4' in selected:
            for n in map(int,a.sizes.split(',')):add(n,8,seed,'uniform','E4')
    if a.generate:
        for g in graphs.values():
            cmd=[str(ROOT/'bin/make_graph'),g['model'],str(g['N']),str(g['E']),str(g['seed']),g['path'],*g['options']]
            subprocess.run(cmd,check=True);g['sha256']=digest(g['path'])
            g['generation']=json.loads(pathlib.Path(g['path']+'.generation.json').read_text())
    manifest=folder/'manifest.json';manifest.write_text(json.dumps(list(graphs.values()),indent=2)+'\n')
    print(f'{len(graphs)} graph configurations: {manifest}')
    if not a.generate:print('Plan only; rerun with --generate to materialize graphs. No timing data created.')
if __name__=='__main__':main()
