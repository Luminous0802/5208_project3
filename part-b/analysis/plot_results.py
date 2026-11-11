#!/usr/bin/env python3
"""Reproduce Part B tables and publication-style figures from official Vanda CSVs.
Run from anywhere: python analysis/plot_results.py
Requires pandas, numpy, matplotlib. Does not run PageRank or MPI.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'; TAB=ROOT/'tables'
FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
IDS={'B1':'b1_openmp_1515568','B2':'b2_hybrid_1515643','B3':'b3_graph_1515702','B4':'b4_large_1515718'}
EXPECTED={'B1':35,'B2':50,'B3':80,'B4':180}
NAVY='#142D46'; BLUE='#2876B7'; TEAL='#008F82'; GOLD='#C88B2D'; RED='#BA5662'; GRAY='#68798A'; PALE='#E9EFF4'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.titleweight':'bold','axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9BAAB7','axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':'#42576A','ytick.color':'#42576A','grid.color':'#DDE5EC','grid.alpha':0.75,'legend.frameon':False,'savefig.facecolor':'white','figure.facecolor':'white','figure.dpi':125})

def load():
    data={}
    qa=[]
    for k,folder in IDS.items():
        df=pd.read_csv(ROOT/'results'/folder/'runs.csv')
        assert len(df)==EXPECTED[k],f'{k}: expected {EXPECTED[k]}, got {len(df)}'
        assert df.iteration_s.notna().all() and np.isfinite(df.iteration_s).all() and (df.iteration_s>0).all()
        assert (df.iterations==50).all() and ((df.rank_sum-1).abs()<1e-8).all()
        groups=df.groupby(['graph_id','version','ranks','threads','schedule','chunk'],dropna=False)
        assert groups.size().eq(5).all(),f'{k}: each configuration should have 5 repeats'
        assert (df.K==50).all()
        qa.append({'experiment':k,'records':len(df),'configurations':len(groups),'max_abs_rank_sum_error':float((df.rank_sum-1).abs().max()),'min_iteration_s':df.iteration_s.min(),'max_iteration_s':df.iteration_s.max()})
        data[k]=df
    pd.DataFrame(qa).to_csv(TAB/'data_quality.csv',index=False)
    return data

def aggregate(df):
    keys=['graph_id','seed','model','version','partition','schedule','chunk','ranks','threads','cores']
    return df.groupby(keys,dropna=False).agg(time=('iteration_s','median'),time_min=('iteration_s','min'),time_max=('iteration_s','max'),compute=('compute_s','median'),reduce=('allreduce_s','median'),gather=('allgatherv_s','median'),imbalance=('edge_imbalance','median'),load=('load_s','median'),partition_time=('partition_s','median'),end_to_end=('end_to_end_s','median'),n=('iteration_s','size')).reset_index()

def style(ax,grid='y'):
    ax.grid(axis=grid,linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0,pad=7)

def log_time(ax):
    ax.set_yscale('log')
    ax.set_ylabel('50-iteration time (s) · log scale')
    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())

def save(fig,slug):
    fig.savefig(FIG/f'{slug}.png',dpi=260,bbox_inches='tight',pad_inches=.20)
    fig.savefig(FIG/f'{slug}.pdf',bbox_inches='tight',pad_inches=.20)
    plt.close(fig)

def title(ax,head,sub):
    ax.set_title(head,loc='left',pad=22,color=NAVY)
    ax.text(0,1.015,sub,transform=ax.transAxes,ha='left',va='bottom',fontsize=8.5,color=GRAY)

def main():
    d=load();s={k:aggregate(v) for k,v in d.items()}
    for k,t in s.items():t.to_csv(TAB/f'{k.lower()}_configuration_summary.csv',index=False)
    # 01: OpenMP scaling with observed repeat range
    b1=s['B1']; z=b1[(b1.version=='V2')&(b1.schedule=='static')].sort_values('threads').copy();base=z.loc[z.threads==1,'time'].iloc[0]
    z['speedup']=base/z.time;z['efficiency']=z.speedup/z.threads;z.to_csv(TAB/'openmp_scaling.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.1,4.8));ax.plot(z.threads,z.threads,'--',lw=1.8,color='#B7C4CF',label='Ideal linear')
    ax.plot(z.threads,z.speedup,'o-',lw=2.5,ms=7,color=BLUE,label='Measured')
    for row in z.itertuples():ax.annotate(f'{row.speedup:.2f}×',(row.threads,row.speedup),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8,color=BLUE)
    ax.set_xscale('log',base=2);ax.set_yscale('log',base=2);ax.set_xticks(z.threads,z.threads.astype(int).astype(str));ax.set_yticks([1,2,4,8,12],['1','2','4','8','12']);ax.set_ylim(.85,18);ax.set_xlabel('OpenMP threads');ax.set_ylabel('Speedup vs V2 single thread');title(ax,'OpenMP scaling','B1 · Uniform · 100K vertices / 800K edges · 5 repeats');style(ax);ax.legend(loc='upper left');save(fig,'01_openmp_speedup')
    fig,ax=plt.subplots(figsize=(8.1,4.3));ax.plot(z.threads,100*z.efficiency,'o-',color=TEAL,lw=2.5,ms=7);ax.axhline(100,color='#B7C4CF',ls='--',lw=1.5);ax.set_xscale('log',base=2);ax.set_xticks(z.threads,z.threads.astype(int).astype(str));ax.set_ylim(0,115);ax.set_ylabel('Parallel efficiency (%)');ax.set_xlabel('OpenMP threads');title(ax,'Efficiency declines with thread count','B1 · Speedup / threads · within-implementation baseline');style(ax);save(fig,'02_openmp_efficiency')
    # 03 B2 fixed-core
    b2=s['B2']; sel=b2[(b2.version=='V1')|((b2.version=='V2')&(b2.schedule=='static'))].sort_values('ranks',ascending=False).copy();sel['layout']=sel.apply(lambda r:f'{int(r.ranks)} MPI × {int(r.threads)} OMP',axis=1)
    fig,ax=plt.subplots(figsize=(8.5,4.7));xx=np.arange(len(sel));cols=[NAVY]+[BLUE]*(len(sel)-1)
    ax.plot(xx,sel.time,'o-',color=BLUE,lw=2.5,ms=8);ax.scatter(xx[0],sel.time.iloc[0],s=100,color=NAVY,zorder=4);ax.errorbar(xx,sel.time,yerr=[sel.time-sel.time_min,sel.time_max-sel.time],fmt='none',ecolor=GRAY,capsize=4,lw=1.2)
    for i,v in enumerate(sel.time):ax.text(i,v*1.045,f'{v:.4f}s',ha='center',fontsize=8.5)
    ax.set_xticks(xx,sel.layout);log_time(ax);ax.set_ylim(.016,.055);title(ax,'Fixed-core MPI / OpenMP trade-off','B2 · 12 CPU cores · Uniform 100K / 800K · median and observed min–max');style(ax);save(fig,'03_hybrid_layout')
    # 04 phase timings, separate nonadditive lines
    fig,ax=plt.subplots(figsize=(8.5,4.6));
    for key,lab,col in [('compute','Compute max',BLUE),('reduce','Allreduce max',GOLD),('gather','Allgatherv max',TEAL)]:ax.plot(xx,sel[key],marker='o',lw=2.2,ms=6,label=lab,color=col)
    ax.set_xticks(xx,sel.layout);log_time(ax);title(ax,'Compute and collective timings','B2 · Per-rank maxima, shown separately; NOT additive');style(ax);ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.17));fig.subplots_adjust(bottom=.23);save(fig,'04_mpi_phase_timings')
    # 05 structure paired comparison
    b3=s['B3'];st=b3[b3.schedule=='static'];order=['b3_uniform','b3_hub','b3_community_ordered','b3_community_shuffled'];labels=['Uniform','Hub','Community\nordered','Community\nshuffled'];wide=st.pivot(index='graph_id',columns='version',values='time').reindex(order)
    fig,ax=plt.subplots(figsize=(9,4.9));xx=np.arange(4);w=.32
    ax.bar(xx-w/2,wide.V2,w,color=BLUE,label='V2 · Vertex-balanced');ax.bar(xx+w/2,wide.V3,w,color=TEAL,label='V3 · Edge-balanced');ax.set_xticks(xx,labels);log_time(ax);title(ax,'Graph structure changes the value of balancing','B3 · 500K / 4M · 6 MPI × 2 OMP · static · 5 repeats');style(ax);ax.legend(loc='upper right');save(fig,'05_graph_structure')
    # 06 speedup heatmap
    wideall=b3.pivot(index=['graph_id','schedule'],columns='version',values='time').reset_index();wideall['speedup']=wideall.V2/wideall.V3
    arr=np.array([[wideall[(wideall.graph_id==g)&(wideall.schedule==sched)].speedup.iloc[0] for sched in ['static','dynamic']] for g in order]);fig,ax=plt.subplots(figsize=(7.5,4.6));im=ax.imshow(arr,cmap='BrBG',vmin=.8,vmax=2.8,aspect='auto');ax.set_yticks(range(4),labels);ax.set_xticks([0,1],['Static','Dynamic']);
    for i in range(4):
        for j in range(2):ax.text(j,i,f'{arr[i,j]:.2f}×',ha='center',va='center',fontsize=11,fontweight='bold',color='white' if arr[i,j]>2 else NAVY)
    title(ax,'Where edge balancing helps','B3 · V2 iteration time / V3 iteration time · >1 favors V3');fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label='V3 speedup (×)');save(fig,'06_partition_speedup_matrix')
    # 07 schedule: direct static/dynamic comparison
    fig,ax=plt.subplots(figsize=(8.8,4.7));subset=b3[b3.graph_id.isin(['b3_uniform','b3_hub'])].copy();lab=[];ss=[];dd=[]
    for g in ['b3_uniform','b3_hub']:
        for v in ['V2','V3']:
            t=subset[(subset.graph_id==g)&(subset.version==v)];lab.append(('Uniform' if g.endswith('uniform') else 'Hub')+' · '+v);ss.append(t[t.schedule=='static'].time.iloc[0]);dd.append(t[t.schedule=='dynamic'].time.iloc[0])
    xx=np.arange(len(lab));ax.bar(xx-.18,ss,.36,color=BLUE,label='Static');ax.bar(xx+.18,dd,.36,color=GOLD,label='Dynamic (chunk 64)');ax.set_xticks(xx,lab);log_time(ax);title(ax,'Scheduling helps only under specific workloads','B3 · Same 6 MPI × 2 OMP allocation · 500K / 4M');style(ax);ax.legend();save(fig,'07_scheduling_comparison')
    # 08 community ordering
    com=st[st.graph_id.str.contains('community')].copy();fig,ax=plt.subplots(figsize=(7.7,4.5));xx=np.arange(2);w=.32
    for offset,v,col in [(-w/2,'V2',BLUE),(w/2,'V3',TEAL)]:
        q=com[com.version==v].set_index('graph_id');vals=[q.loc['b3_community_ordered','time'],q.loc['b3_community_shuffled','time']];ax.bar(xx+offset,vals,w,color=col,label=v)
    ax.set_xticks(xx,['Ordered vertex IDs','Shuffled vertex IDs']);log_time(ax);ax.set_ylim(.08,.16);title(ax,'Vertex ordering and memory access','B3 · Community graph · same topology after relabeling · static');style(ax);ax.legend();save(fig,'08_community_ordering')
    # 09/10 B4 per seed paired speedups and imbalance
    b4=s['B4'];paired=b4.pivot(index=['graph_id','seed','model','ranks','threads'],columns='version',values='time').reset_index();paired['speedup']=paired.V2/paired.V3;paired.to_csv(TAB/'b4_paired_speedups.csv',index=False)
    summary=paired.groupby(['model','ranks','threads']).agg(median=('speedup','median'),minimum=('speedup','min'),maximum=('speedup','max')).reset_index();summary.to_csv(TAB/'b4_speedup_ranges.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.6,5));layout=[2,3,6];
    for model,col,offset in [('hub',RED,-.09),('uniform',TEAL,.09)]:
        group=summary[summary.model==model].set_index('ranks').loc[layout];ax.errorbar(np.array(layout)+offset,group['median'],yerr=[group['median']-group['minimum'],group['maximum']-group['median']],fmt='o-',color=col,capsize=4,ms=8,lw=2.4,label=model.title())
        for seed,sub in paired[paired.model==model].groupby('seed'):
            sub=sub.set_index('ranks').loc[layout];ax.scatter(np.array(layout)+offset,sub.speedup,color=col,alpha=.23,s=25)
    ax.axhline(1,color='#AAB7C3',ls='--');ax.set_xticks(layout,['2 MPI × 6 OMP','3 MPI × 4 OMP','6 MPI × 2 OMP']);ax.set_ylabel('V3 speedup over V2 (×)');title(ax,'Edge balancing is robust on Hub graphs','B4 · 1M / 8M · 3 seeds · medians with seed range and individual seeds');style(ax);ax.legend();save(fig,'09_seed_robustness')
    imb=b4.pivot(index=['model','seed','ranks'],columns='version',values='imbalance').reset_index();agg=imb.groupby(['model','ranks'])[['V2','V3']].median().reset_index();fig,ax=plt.subplots(figsize=(8.7,4.7));labels=[];v2=[];v3=[]
    for model in ['uniform','hub']:
        for ranks in [2,3,6]:
            row=agg[(agg.model==model)&(agg.ranks==ranks)].iloc[0];labels.append(f'{model.title()}\n{ranks} MPI');v2.append(row.V2);v3.append(row.V3)
    xx=np.arange(6);ax.bar(xx-.18,v2,.36,color=BLUE,label='V2 Vertex');ax.bar(xx+.18,v3,.36,color=TEAL,label='V3 Edge');ax.axhline(1,ls='--',color='#AAB7C3');ax.set_xticks(xx,labels);ax.set_ylabel('Max rank edges / average rank edges');title(ax,'MPI edge imbalance explains the Hub advantage','B4 · Median across 3 graph seeds · lower is better');style(ax);ax.legend();save(fig,'10_partition_imbalance')
    # 11: hub max compute vs total iteration
    h=b4[(b4.model=='hub')&(b4.ranks==6)].groupby('version')[['time','compute','reduce','gather']].median().loc[['V2','V3']];fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(2);w=.2
    for i,(col,label,color) in enumerate([('time','Iteration',NAVY),('compute','Compute max',BLUE),('gather','Allgatherv max',TEAL),('reduce','Allreduce max',GOLD)]):ax.bar(x+(i-1.5)*w,h[col],w,color=color,label=label)
    ax.set_xticks(x,['V2 Vertex','V3 Edge']);log_time(ax);title(ax,'Hub graph: compute bottleneck shrinks','B4 · 6 MPI × 2 OMP · median across seeds · phases NOT additive');style(ax);ax.legend(ncol=2);save(fig,'11_hub_bottleneck')
    # 12 end-to-end vs iteration B3
    fig,ax=plt.subplots(figsize=(8.5,4.6));q=b3[(b3.schedule=='static')&(b3.version=='V3')].set_index('graph_id').loc[order];xx=np.arange(4);w=.32
    ax.bar(xx-w/2,q.time,w,color=BLUE,label='Iteration only');ax.bar(xx+w/2,q.end_to_end,w,color=TEAL,label='End-to-end');ax.set_xticks(xx,['Uniform','Hub','Community\nordered','Community\nshuffled']);log_time(ax);title(ax,'Iteration time is not end-to-end runtime','B3 · V3 static · includes graph load and partition setup');style(ax);ax.legend();save(fig,'12_iteration_vs_end_to_end')
    print('Validated 345 official records; generated 12 PNG + 12 PDF figures and summary tables.')
if __name__=='__main__':main()
