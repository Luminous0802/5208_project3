#!/usr/bin/env python3
"""Both solvers must act on the identical lazy operator, with independent dense checks."""
import argparse,csv,hashlib,json,math,os,pathlib,platform,shlex,subprocess,uuid
from validate_lanczos import matrix
from process_logging import run_logged
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--serial-only',action='store_true')
    ap.add_argument('--site',choices=['local','Atlas','Vanda']);ap.add_argument('--bin-dir',default='bin')
    ap.add_argument('--out',default='results/eigen_pair_validation');ap.add_argument('--work-dir')
    ap.add_argument('--timeout',type=int,default=90);a=ap.parse_args()
    if a.timeout<=0:ap.error('Positive timeout required')
    job=os.getenv('PBS_JOBID') or os.getenv('SLURM_JOB_ID') or 'none'
    if a.site is None:
        if job!='none':ap.error('Specify the actual --site for a scheduler job')
        a.site='local'
    if a.site!='local' and job=='none':ap.error('Cluster validation requires a scheduler allocation')
    if a.site=='local' and job!='none':ap.error('Use the actual cluster site inside a scheduler job')
    bins=(ROOT/a.bin_dir).resolve();out=(ROOT/a.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Use a fresh empty --out directory')
    work=(ROOT/a.work_dir).resolve() if a.work_dir else out/'inputs'
    work.mkdir(parents=True,exist_ok=True);rows=[]
    raw=out/'raw';raw.mkdir()
    identity=dict(site=a.site,hostname=platform.node(),job_id=job,measurement_id=uuid.uuid4().hex)
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
      ['src/lanczos.cpp','src/power_symmetric.cpp','src/symmetric.hpp','src/graph.hpp']}
    names=['power_symmetric','lanczos']+([] if a.serial_only else ['power_symmetric_mpi','lanczos_mpi'])
    binary_hashes={name:hashlib.sha256((bins/name).read_bytes()).hexdigest() for name in names}
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='FALSE')
    launcher=shlex.split(os.getenv('MPIEXEC','mpiexec'))+shlex.split(os.getenv('MPIEXEC_ARGS',''))
    cases={'zero':(3,[]),'single':(1,[]),'tiny':(2,[(0,1)]),'path':(12,[(i,i+1) for i in range(11)]),
      'cycle':(10,[(i,(i+1)%10) for i in range(10)]),'self_isolated':(5,[(0,0),(1,2),(2,3)])}
    for name,(n,edges) in cases.items():
        graph=work/(name+'.txt');graph.write_text(f'{n} {len(edges)}\n'+''.join(f'{u} {v}\n' for u,v in edges))
        s=matrix(n,edges);b=[[0.5*(s[i][j]+(1 if i==j else 0)) for j in range(n)] for i in range(n)]
        first={}
        for algorithm in ['power','lanczos']:
            for parallel in ([False] if a.serial_only else [False,True]):
                for budget in [1,1000 if algorithm=='power' else n]:
                    stem=work/f'{name}_{algorithm}_{parallel}_{budget}'
                    exe=('power_symmetric' if algorithm=='power' else 'lanczos')+('_mpi' if parallel else '')
                    cmd=[str(bins/exe),str(graph),str(budget),'--operator','lazy','--seed','20261029','--dump',str(stem)]
                    if algorithm=='power' and budget>1:cmd+=['--tol','1e-10']
                    if parallel:cmd=[*launcher,'-n','4',*cmd]
                    label=stem.name
                    print(f'VALIDATION_START {label}',flush=True)
                    p=run_logged(cmd,raw/label,env=env,timeout=a.timeout)
                    m=dict(line.split(': ',1) for line in p.stdout.splitlines() if ': ' in line)
                    assert m['operator']=='lazy'
                    assert int(m['ranks'])==(4 if parallel else 1) and int(m['threads'])==(2 if parallel else 1)
                    v=[float(line.split()[1]) for line in pathlib.Path(str(stem)+'.vector.txt').read_text().splitlines()]
                    bv=[math.fsum(b[i][j]*v[j] for j in range(n)) for i in range(n)];lam=float(m['lambda_max_ritz'])
                    residual=math.sqrt(math.fsum((bv[i]-lam*v[i])**2 for i in range(n)))
                    assert abs(residual-float(m['ritz_residual']))<1e-10 and abs(math.fsum(x*x for x in v)-1)<1e-10
                    if budget>1 or n==1:
                        expected=1 if edges else .5
                        assert abs(lam-expected)<1e-8 and residual<1e-8,(name,algorithm,m)
                    if budget==1 and not parallel:first[algorithm]=(lam,residual)
                    rows.append(dict(identity,graph=name,algorithm=algorithm,parallel=parallel,budget=budget,steps=m['steps'],lambda_ritz=lam,dense_residual=residual))
                    print(f'VALIDATION_PASS {label} residual={residual:.6g}',flush=True)
        assert abs(first['power'][0]-first['lanczos'][0])<1e-12 and abs(first['power'][1]-first['lanczos'][1])<1e-12
    with (out/'comparisons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(identity,status='PASS',comparisons=len(rows),serial_only=a.serial_only,operator='B=(I+S)/2',checks=['independent dense residual','same initial q and one-step Ritz pair','lazy shift resolves bipartite +/-1 ambiguity','zero matrix and empty MPI partitions'])
    summary.update(source_sha256=sources,binary_sha256=binary_hashes,MPIEXEC=launcher,configuration=vars(a))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
