#!/usr/bin/env python3
"""Independent dense PageRank oracle; standard library only. Never a speed benchmark."""
import argparse, csv, hashlib, json, math, os, pathlib, random, shlex, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run(cmd, env=None, ok=True):
    p = subprocess.run([str(x) for x in cmd], text=True, capture_output=True, env=env, timeout=90)
    if ok and p.returncode:
        raise RuntimeError(f'{cmd}\n{p.stdout}\n{p.stderr}')
    if not ok and not p.returncode:
        raise AssertionError(f'Accepted invalid input: {cmd}')
    return p.stdout

def metrics(s):
    return dict(line.split(': ', 1) for line in s.splitlines() if ': ' in line)

def oracle(n, edges, k, tol=0):
    # Dense transition matrix, including dangling columns: independent of incoming CSR.
    degree = [0]*n
    for u, v in edges: degree[u] += 1
    matrix = [[(1-0.85)/n for _ in range(n)] for _ in range(n)]
    for u in range(n):
        if degree[u] == 0:
            for v in range(n): matrix[v][u] += 0.85/n
    for u, v in edges: matrix[v][u] += 0.85/degree[u]
    r = [1/n]*n
    for it in range(k):
        nxt = [math.fsum(matrix[v][u]*r[u] for u in range(n)) for v in range(n)]
        residual = math.fsum(abs(x-y) for x,y in zip(nxt,r)); r=nxt
        if tol and residual < tol: break
    return r, it+1

def read_graph(path):
    a = list(map(int, path.read_text().split())); n,e=a[:2]
    edges=list(zip(a[2::2],a[3::2])); assert len(edges)==e
    return n,edges

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='results/local_validation'); a=ap.parse_args()
    out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True)
    scratch=ROOT/'work/validation'; scratch.mkdir(parents=True,exist_ok=True)
    launch=shlex.split(os.environ.get('MPIEXEC','mpiexec'))+shlex.split(os.environ.get('MPIEXEC_ARGS',''))
    env=os.environ.copy(); env.update(OMP_DYNAMIC='FALSE',OMP_PROC_BIND='FALSE')
    rows=[]
    random.seed(730)
    cases={
      'tiny':(2,[(0,1)]), 'cycle':(7,[(i,(i+1)%7) for i in range(7)]),
      'chain_isolated':(9,[(i,i+1) for i in range(6)]), 'empty_edges':(5,[]),
      'self_loop':(4,[(0,1),(0,2),(1,2),(2,2)]),
      'star_in':(13,[(i,0) for i in range(1,13)]),
      'star_out':(13,[(0,i) for i in range(1,13)]),
      'random':(19,random.sample([(u,v) for u in range(19) for v in range(19)],78)),
      'single':(1,[])
    }
    start=time.monotonic()
    for name,(n,edges) in cases.items():
        path=scratch/(name+'.txt'); path.write_text(f'{n} {len(edges)}\n'+''.join(f'{u} {v}\n' for u,v in edges))
        for k,tol in [(1,0),(50,0),(300,1e-8)]:
            ref,iterations=oracle(n,edges,k,tol); args=[path,k]+([str(tol)] if tol else [])
            # A prints ten decimal places, so the vector comparison accounts for text rounding.
            for version,cmd in [('V0',[ROOT/'bin/pagerank',*args]),('V1',[*launch,'-n','2',ROOT/'bin/pagerank_mpi',*args])]:
                m=metrics(run(cmd,env)); got=[float(m[f'rank {v}']) for v in range(n)]
                error=math.fsum(abs(x-y) for x,y in zip(got,ref))
                assert error<1e-8 and int(m['iterations'])==iterations,(name,version,error)
                rows.append(dict(graph=name,version=version,ranks=1 if version=='V0' else 2,threads=1,schedule='none',K=k,tol=tol,L1=error))
            # Covers empty partitions (P>N), non-divisible ranges, thread reductions, schedules.
            configs=[(1,1,'static'),(2,2,'static'),(2,4,'dynamic'),(4,2,'dynamic')]
            for part in ['vertex','edge']:
                for ranks,threads,schedule in configs:
                    env['OMP_NUM_THREADS']=str(threads); dump=scratch/'rank.txt'; profile=scratch/'rank.csv'
                    m=metrics(run([*launch,'-n',ranks,ROOT/'bin/pagerank_hybrid',*args,
                        '--partition',part,'--schedule',schedule,*(['--chunk','7'] if schedule=='dynamic' else []),'--dump',dump,'--profile',profile],env))
                    got=[float(line.split()[1]) for line in dump.read_text().splitlines()]
                    error=math.fsum(abs(x-y) for x,y in zip(got,ref))
                    assert error<1e-8 and len(got)==n and all(math.isfinite(x) and x>=0 for x in got)
                    assert abs(math.fsum(got)-1)<1e-12 and int(m['iterations'])==iterations
                    assert int(m['threads'])==threads
                    per=list(csv.DictReader(profile.open()))
                    assert len(per)==ranks and sum(int(float(r['vertices'])) for r in per)==n
                    assert sum(int(float(r['edges'])) for r in per)==len(edges)
                    rows.append(dict(graph=name,version='V2' if part=='vertex' else 'V3',ranks=ranks,threads=threads,schedule=schedule,K=k,tol=tol,L1=error))
        print(f'PASS {name}',flush=True)
    # Exact edge count, deterministic generation, isomorphic community relabeling, stats invariants.
    for model in ['uniform','hub','community']:
        p1=scratch/f'{model}.txt'; p2=scratch/f'{model}_again.txt'
        cmd=[ROOT/'bin/make_graph',model,120,700,20261029]
        run([*cmd,p1]);run([*cmd,p2]);assert p1.read_bytes()==p2.read_bytes()
        n,edges=read_graph(p1);assert n==120 and len(edges)==700 and len(set(edges))==700
        assert all(0<=u<n and 0<=v<n and u!=v for u,v in edges)
        for part in ['vertex','edge']:
            stats=json.loads(run([ROOT/'bin/graph_stats',p1,4,part]))
            assert stats['N']==n and stats['E']==len(edges)
            assert sum(r['edges'] for r in stats['per_rank'])==700
            assert sum(r['vertices'] for r in stats['per_rank'])==120
            assert 0<=stats['cross_partition_edge_fraction']<=1
        # Generated graph larger than A's printed-rank limit: dump every rank and compare oracle.
        ref,_=oracle(n,edges,50)
        for part in ['vertex','edge']:
            env['OMP_NUM_THREADS']='2'; dump=scratch/'generated-rank.txt'
            m=metrics(run([*launch,'-n',3,ROOT/'bin/pagerank_hybrid',p1,50,'--partition',part,
                '--schedule','dynamic','--chunk',7,'--dump',dump],env))
            got=[float(line.split()[1]) for line in dump.read_text().splitlines()]
            error=math.fsum(abs(x-y) for x,y in zip(got,ref));assert error<1e-8
            rows.append(dict(graph=model,version='V2' if part=='vertex' else 'V3',ranks=3,threads=2,schedule='dynamic',K=50,tol=0,L1=error))
        if model=='community':
            shuffled=scratch/'community_shuffled.txt';run([*cmd,shuffled,'--shuffle',1])
            mapping=list(csv.DictReader(open(str(shuffled)+'.labels.tsv'),delimiter='\t'))
            perm={int(r['old_id']):int(r['new_id']) for r in mapping}
            _,other=read_graph(shuffled)
            assert {(perm[u],perm[v]) for u,v in edges}==set(other)
    # Known topology metrics including empty-edge behavior and indivisible in-hub.
    empty=json.loads(run([ROOT/'bin/graph_stats',scratch/'empty_edges.txt',8,'edge']))
    assert empty['weak_components']==5 and empty['dangling_fraction']==1 and empty['edge_imbalance']==0
    star=json.loads(run([ROOT/'bin/graph_stats',scratch/'star_in.txt',4,'edge']))
    assert star['edge_imbalance']==4 and star['largest_weak_component_fraction']==1
    for name,text in [('duplicate','3 2\n0 1\n0 1\n'),('short','3 2\n0 1\n'),('bad_id','3 1\n0 3\n'),('trailing','3 1\n0 1\n1 2\n')]:
        bad=scratch/(name+'.txt');bad.write_text(text)
        run([*launch,'-n',2,ROOT/'bin/pagerank_hybrid',bad,50],env,ok=False)
    for options in [['0'],['50','nan'],['50','--partition','bad'],['50','--chunk','0']]:
        run([*launch,'-n',2,ROOT/'bin/pagerank_hybrid',scratch/'tiny.txt',*options],env,ok=False)
    with (out/'comparisons.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='PASS',comparisons=len(rows),max_L1=max(r['L1'] for r in rows),
        tolerance=1e-8,elapsed_test_seconds=time.monotonic()-start,platform=os.uname().sysname,
        machine=os.uname().machine,mpi_launcher=launch,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*'))},
        binary_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'bin').iterdir()) if p.is_file()},
        note='Local correctness only; not Atlas/Vanda runtime evidence')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
