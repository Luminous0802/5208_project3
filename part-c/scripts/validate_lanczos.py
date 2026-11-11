#!/usr/bin/env python3
"""No numerical packages: dense residuals, analytic eigenvalues, and recurrence checks."""
import argparse,csv,hashlib,json,math,os,pathlib,random,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]

def metrics(s):return dict(line.split(': ',1) for line in s.splitlines() if ': ' in line)
def run(cmd,env,valid=True):
    p=subprocess.run(list(map(str,cmd)),capture_output=True,text=True,env=env,timeout=90)
    if valid and p.returncode:raise RuntimeError(str(cmd)+'\n'+p.stderr)
    if not valid and p.returncode==0:raise AssertionError('accepted invalid input')
    return metrics(p.stdout)
def matrix(n,edges):
    undirected={tuple(sorted((u,v))) for u,v in edges};a=[[0.0]*n for _ in range(n)]
    for u,v in undirected:a[u][v]=1;a[v][u]=1
    degree=[sum(row) for row in a]
    return [[a[i][j]/math.sqrt(degree[i]*degree[j]) if a[i][j] else 0.0 for j in range(n)] for i in range(n)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--serial-only',action='store_true');a=ap.parse_args()
    work=ROOT/'work/lanczos_validation';work.mkdir(parents=True,exist_ok=True)
    out=ROOT/'results/local_validation';out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OMP_DYNAMIC='FALSE',OMP_PROC_BIND='FALSE')
    launcher=shlex.split(os.getenv('MPIEXEC','mpiexec'))+shlex.split(os.getenv('MPIEXEC_ARGS',''))
    random.seed(20261029)
    cases={
      'single_zero':(1,[]), 'zero':(5,[]), 'tiny':(2,[(0,1)]),
      'path':(12,[(i,i+1) for i in range(11)]),
      'cycle':(11,[(i,(i+1)%11) for i in range(11)]),
      'star':(13,[(0,i) for i in range(1,13)]),
      'complete':(8,[(u,v) for u in range(8) for v in range(u+1,8)]),
      'disconnected_isolated':(8,[(0,1),(1,2),(3,4),(5,5)]),
      'reciprocal_self':(4,[(0,1),(1,0),(1,2),(2,2)]),
      'random':(19,random.sample([(u,v) for u in range(19) for v in range(u+1,19)],48))}
    configs=[('serial',1,1)] if a.serial_only else [('serial',1,1),('mpi',1,1),('mpi',2,2),('mpi',4,2)]
    rows=[]
    for name,(n,edges) in cases.items():
        path=work/(name+'.txt');path.write_text(f'{n} {len(edges)}\n'+''.join(f'{u} {v}\n' for u,v in edges))
        dense=matrix(n,edges);expected=1.0 if edges else 0.0
        for mode,p,t in configs:
            for requested in [1,n]:
                env['OMP_NUM_THREADS']=str(t);prefix=work/f'{name}_{mode}_{p}_{t}_{requested}'
                command=[ROOT/'bin'/('lanczos' if mode=='serial' else 'lanczos_mpi'),path,requested,
                    '--orthogonality',1,'--dump',prefix,'--history',str(prefix)+'.history.csv']
                if mode=='mpi':command=[*launcher,'-n',p,*command]
                m=run(command,env);v=[float(line.split()[1]) for line in pathlib.Path(str(prefix)+'.vector.txt').read_text().splitlines()]
                lam=float(m['lambda_max_ritz']);sv=[math.fsum(dense[i][j]*v[j] for j in range(n)) for i in range(n)]
                residual=math.sqrt(math.fsum((sv[i]-lam*v[i])**2 for i in range(n)))
                assert abs(math.fsum(x*x for x in v)-1)<1e-11
                assert abs(residual-float(m['ritz_residual']))<1e-11,(name,residual,m)
                assert abs(math.fsum(v[i]*sv[i] for i in range(n))-lam)<1e-11
                assert float(m['orthogonality_max'])<1e-11
                assert -1-1e-11<=lam<=1+1e-11
                if requested==n:assert abs(lam-expected)<1e-10 and residual<1e-10,(name,m)
                tri=list(csv.DictReader(open(str(prefix)+'.tridiagonal.csv')))
                hist=list(csv.DictReader(open(str(prefix)+'.history.csv')))
                assert len(tri)==len(hist)==int(m['steps'])<=min(n,requested)
                assert abs(float(hist[-1]['largest_ritz'])-lam)<1e-12
                assert int(m['ranks'])==p and int(m['threads'])==t
                rows.append(dict(graph=name,mode=mode,ranks=p,threads=t,requested=requested,steps=int(m['steps']),
                    lambda_ritz=lam,dense_residual=residual,orthogonality=float(m['orthogonality_max'])))
        print('PASS',name,flush=True)
    # Seed changes the initial Krylov direction but must preserve converged results.
    for seed in [0,1,99]:
        m=run([ROOT/'bin/lanczos',work/'path.txt',12,'--seed',seed],env)
        assert abs(float(m['lambda_max_ritz'])-1)<1e-10
    # Also exercise the low-cost three-term option; no orthogonality guarantee is assumed.
    m=run([ROOT/'bin/lanczos',work/'cycle.txt',5,'--reorth','none','--orthogonality',1],env)
    assert math.isfinite(float(m['ritz_residual']))
    bad=work/'bad.txt';bad.write_text('3 2\n0 1\n0 1\n')
    for options in [[bad,5],[work/'tiny.txt',0],[work/'tiny.txt',5,'--reorth','invalid']]:
        run([ROOT/'bin/lanczos',*options],env,valid=False)
    with (out/'lanczos_comparisons.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    full=[r for r in rows if r['requested']==cases[r['graph']][0]]
    summary=dict(status='PASS',comparisons=len(rows),serial_only=a.serial_only,
       max_full_run_dense_residual=max(r['dense_residual'] for r in full),
       max_orthogonality=max(r['orthogonality'] for r in rows),
       reference='Independent dense S*v, Rayleigh quotient and analytic lambda_max=1 (or 0 for zero matrix); no BLAS/LAPACK',
       source_sha256=hashlib.sha256((ROOT/'src/lanczos.cpp').read_bytes()).hexdigest(),
       note='Local correctness evidence only; not Atlas/Vanda performance')
    (out/'lanczos_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
