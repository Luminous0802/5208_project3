#!/usr/bin/env python3
"""Synthetic UNIT TEST fixtures only: not hardware measurements and never plotted."""
import csv,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    work=ROOT/'work/gpu_analysis_unit';work.mkdir(parents=True,exist_ok=True)
    base=dict(site='UNIT_FIXTURE_NOT_A_SITE',hardware_tag='TEST_ONLY',build_tag='test',source_sha256='test-source',graph_id='fixture',
      graph_sha256='same-test-graph',model='unit',seed='0',N='4',E='4',K='50',precision='fp64',mapping='none',block_size='0',gpu_name='',
      measurement_id='UNIT_MEASUREMENT',job_id='UNIT_JOB',hostname='UNIT_HOST',
      iteration_s='2',gpu_timeline_s='',h2d_s='',d2h_s='',transfer_inclusive_s='',end_to_end_s='3',validation_L1='')
    rows=[]
    for i in range(2):
        rows.append(dict(base,backend='cpu_serial'))
        rows.append(dict(base,backend='cuda',mapping='thread',block_size='256',gpu_name='TEST_FIXTURE_NOT_HARDWARE',iteration_s='1',
          gpu_timeline_s='0.9',h2d_s='0.5',d2h_s='0.5',transfer_inclusive_s='2',validation_L1='0'))
    def check(items,label):
        path=work/(label+'.csv');summary=work/(label+'_summary.csv')
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_gpu.py'),str(path),'--out',str(summary)],check=True)
        return list(csv.DictReader(summary.open()))
    results=check(rows,'matched')
    gpu=next(r for r in results if r['backend']=='cuda')
    assert float(gpu['loop_speedup_vs_cpu'])==2 and float(gpu['transfer_inclusive_speedup_vs_cpu'])==1 and gpu['samples']=='2'
    for field,value in [('precision','fp32'),('site','OTHER_UNIT_SITE'),('graph_sha256','other-test-graph'),('source_sha256','other-source'),
                        ('measurement_id','OTHER_UNIT_BATCH'),('job_id','OTHER_UNIT_JOB'),('hostname','OTHER_UNIT_HOST')]:
        items=[dict(r,**{field:value}) if r['backend']=='cuda' else dict(r) for r in rows]
        result=check(items,'mismatch_'+field)
        assert all(r['loop_speedup_vs_cpu']==r['transfer_inclusive_speedup_vs_cpu']=='' for r in result if r['backend']=='cuda')
    # Two different hybrid layouts must remain distinct; choose the lowest tested median only.
    items=[dict(r,ranks='0' if r['backend']=='cuda' else '1',threads='0' if r['backend']=='cuda' else '1',
                partition='none',schedule='none',chunk='0',cpu_cores='0' if r['backend']=='cuda' else '1') for r in rows]
    for ranks,threads,part,seconds in [(2,6,'vertex',.8),(6,2,'edge',.5)]:
        items.append(dict(items[0],backend='cpu_hybrid',ranks=str(ranks),threads=str(threads),partition=part,schedule='static',
                          cpu_cores='12',iteration_s=str(seconds),validation_L1='0'))
    result=check(items,'hybrid')
    assert len([r for r in result if r['backend']=='cpu_hybrid'])==2
    gpu=next(r for r in result if r['backend']=='cuda')
    assert float(gpu['loop_speedup_vs_best_tested_hybrid'])==.5 and float(gpu['transfer_inclusive_speedup_vs_best_tested_hybrid'])==.25
    assert gpu['hybrid_baseline']=='p6t2 edge static chunk0'
    print('PASS GPU summary unit tests using synthetic test fixtures only; no GPU execution claimed.')
if __name__=='__main__':main()
