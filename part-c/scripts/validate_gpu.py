#!/usr/bin/env python3
"""Independent PageRank oracle. --reference-only never claims a CUDA validation pass."""
import argparse,csv,hashlib,json,math,pathlib,random,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
def parse(s):return dict(line.split(': ',1) for line in s.splitlines() if ': ' in line)
def oracle(n,edges,k,tol):
    degree=[0]*n
    for u,v in edges:degree[u]+=1
    incoming=[[] for _ in range(n)]
    for u,v in edges:incoming[v].append(u)
    r=[1/n]*n
    for it in range(k):
        mass=math.fsum(r[u] for u in range(n) if degree[u]==0)
        nxt=[.15/n+.85*(math.fsum(r[u]/degree[u] for u in incoming[v])+mass/n) for v in range(n)]
        diff=math.fsum(abs(x-y) for x,y in zip(r,nxt));r=nxt
        if tol and diff<tol:break
    return r,it+1

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-only',action='store_true');ap.add_argument('--bin-dir',default=str(ROOT/'bin'));ap.add_argument('--out',default='results/gpu_validation');a=ap.parse_args()
    bins=pathlib.Path(a.bin_dir);out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True);work=ROOT/'work/gpu_validation';work.mkdir(parents=True,exist_ok=True)
    if not a.reference_only and not (bins/'pagerank_cuda').is_file():raise FileNotFoundError('CUDA executable missing; GPU validation has NOT run. Use --reference-only only for CPU checks.')
    random.seed(20261029)
    cases={'tiny':(2,[(0,1)]),'single':(1,[]),'zero':(7,[]),'self':(4,[(0,1),(0,2),(1,2),(2,2)]),
      'cycle':(11,[(i,(i+1)%11) for i in range(11)]),'in_star':(257,[(i,0) for i in range(1,257)]),
      'out_star':(97,[(0,i) for i in range(1,97)]),'random':(37,random.sample([(u,v) for u in range(37) for v in range(37)],201))}
    rows=[];device=None
    for name,(n,edges) in cases.items():
        graph=work/(name+'.txt');graph.write_text(f'{n} {len(edges)}\n'+''.join(f'{u} {v}\n' for u,v in edges))
        for k,tol in [(1,0),(50,0),(300,1e-8)]:
            ref,steps=oracle(n,edges,k,tol)
            configs=[('cpu_serial','thread',256)]
            if not a.reference_only:configs += [('cuda',mapping,block) for mapping in ['thread','warp'] for block in [64,128,256]]
            for backend,mapping,block in configs:
                dump=work/'rank.txt';exe=bins/('pagerank_reference' if backend=='cpu_serial' else 'pagerank_cuda')
                cmd=[str(exe),str(graph),str(k),*([str(tol)] if tol else []),'--dump',str(dump)]
                if backend=='cuda':cmd+=['--mapping',mapping,'--block-size',str(block),'--validate']
                p=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
                if p.returncode:raise RuntimeError(f'{cmd}\n{p.stderr}')
                m=parse(p.stdout);assert m['backend']==backend and m['precision']=='fp64'
                got=[float(line.split()[1]) for line in dump.read_text().splitlines()]
                error=math.fsum(abs(x-y) for x,y in zip(got,ref))
                assert len(got)==n and all(math.isfinite(x) and x>=0 for x in got) and abs(math.fsum(got)-1)<1e-10
                assert error<1e-8 and int(m['iterations'])==steps,(name,backend,error,m)
                if backend=='cuda':device=m['gpu_name'];assert 0<=float(m['validation_L1'])<1e-8
                rows.append(dict(graph=name,backend=backend,mapping=mapping,block=block,K=k,tol=tol,iterations=steps,L1=error))
        print('PASS',name,flush=True)
    # Invalid input must be rejected, not silently repaired.
    for name,text in [('duplicate','3 2\n0 1\n0 1\n'),('short','3 2\n0 1\n'),('id','3 1\n0 3\n')]:
        graph=work/(name+'.txt');graph.write_text(text)
        p=subprocess.run([str(bins/'pagerank_reference'),str(graph),'50'],capture_output=True,text=True);assert p.returncode!=0
    with (out/'comparisons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='CPU_PASS_GPU_NOT_RUN' if a.reference_only else 'GPU_PASS',comparisons=len(rows),gpu_name=device,
      max_L1=max(r['L1'] for r in rows),precision='fp64',gpu_executed=not a.reference_only,
      cuda_source_sha256=hashlib.sha256((ROOT/'src/pagerank_cuda.cu').read_bytes()).hexdigest(),
      note='CPU-only checks are not CUDA compilation or NVIDIA GPU execution evidence')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
