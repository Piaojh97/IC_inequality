"""Reproduce figure panels and tables using only the public aggregate data."""
from pathlib import Path
import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

BLUE,ORANGE='#6384B3','#F08743'
COLORS=['#355E95','#6384B3','#A9BCD7','#F5C6A3','#F08743','#B65323']
AGES=['19–23','24–30','31–40','41–50','51–60','61–79']
RAW=['watched_search1','activeness','attention_span1','top10avg']
RAW_LABELS=['Searches per hour','Interactions per hour','Interested-video duration (s)','Top-decile duration (s)']
SPECS=['Age','+ Gender','+ Economic status','+ Interests','+ All']
OUT=ROOT/'outputs/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
MAP=LinearSegmentedColormap.from_list('risk',[ORANGE,'#FFFFFF',BLUE])


def read(name):return pd.read_csv(ROOT/'data'/name)
def stat(name):return read('statistical_results/'+name+'.csv')
def canvas(n=1,m=1,w=3.5,h=2.7):
    fig,ax=plt.subplots(n,m,figsize=(w*m,h*n),squeeze=False,layout='constrained')
    return fig,ax.ravel()
def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.png'),dpi=140)
    plt.close(fig)
def tag(ax,text):ax.set_title(text,loc='left',weight='bold')
def errors(ax,d,x=None,y='mean',color=BLUE,label=None):
    if x is None:x=np.arange(len(d))
    ax.errorbar(x,d[y],yerr=np.vstack([d[y]-d.lo,d.hi-d[y]]),fmt='o-',
                color=color,label=label,capsize=2,ms=3,lw=1)
def bars(ax,d,y='risk',labels=None,scale=1):
    x=np.arange(len(d));color=[COLORS[i%6] for i in x]
    ax.bar(x,d[y]*scale,color=color,width=.65)
    if {'lo','hi'}.issubset(d.columns):
        ax.errorbar(x,d[y]*scale,yerr=scale*np.vstack([d[y]-d.lo,d.hi-d[y]]),fmt='none',color='#444444',capsize=2,lw=.7)
    if labels is not None:ax.set_xticks(x,labels,rotation=30,ha='right')
def heat(ax,d,x='exploitation',y='exploration',z='risk',percent=True,title=None,limits=None,clabel=None):
    a=d.pivot(index=y,columns=x,values=z).sort_index().sort_index(axis=1)
    values=a.to_numpy()*(100 if percent else 1)
    kw={} if limits is None else dict(vmin=limits[0],vmax=limits[1])
    im=ax.imshow(values,origin='lower',aspect='auto',cmap=MAP if z=='risk' else 'Blues',**kw)
    ix=np.unique(np.linspace(0,len(a.columns)-1,min(4,len(a.columns))).astype(int))
    iy=np.unique(np.linspace(0,len(a.index)-1,min(4,len(a.index))).astype(int))
    ax.set_xticks(ix,a.columns[ix]);ax.set_yticks(iy,a.index[iy])
    ax.set(xlabel=x.replace('_',' ').capitalize(),ylabel=y.replace('_',' ').capitalize())
    if title:ax.set_title(title)
    ax.figure.colorbar(im,ax=ax,shrink=.75,label=clabel or ('Risk (%)' if z=='risk' else ('Users (%)' if percent else z)))
    return im
def box(ax,d,metric,ylabel):
    d=d[d.metric.eq(metric)].set_index('age').reindex(AGES)
    records=[dict(label=age,whislo=r.q05,q1=r.q25,med=r['median'],q3=r.q75,whishi=r.q95,fliers=[]) for age,r in d.iterrows()]
    artists=ax.bxp(records,showfliers=False,patch_artist=True)
    for p,c in zip(artists['boxes'],COLORS):p.set_facecolor(c);p.set_alpha(.75)
    ax.set(ylabel=ylabel,xlabel='Age (years)');ax.tick_params(axis='x',labelrotation=35)
def cdf(ax,d,group='group',xlabel='',scale_x=1):
    for i,(name,g) in enumerate(d.groupby(group,sort=False)):
        ax.plot(g.x*scale_x,g.cdf*100,lw=1.2,color=COLORS[i%6],label=str(name))
    ax.set(xlabel=xlabel,ylabel='Cumulative users (%)',ylim=(0,100));ax.legend(frameon=False,fontsize=6)


def main_1():
    d=read('empirical/min_entropy_histogram.csv');fig,axs=canvas();a=axs[0]
    a.plot(d.centre,d.density,color=BLUE);a.fill_between(d.centre,d.density,color=BLUE,alpha=.2)
    a.set(xlabel='Minimum relative entropy',ylabel='Density',xlim=(-.1,2));save(fig,'Figure1a')
    d=read('empirical/threshold_depth.csv').sort_values('one_minus_alpha');fig,axs=canvas();a=axs[0]
    a.plot(d.one_minus_alpha,d.criterion,'o-',ms=3,color=BLUE,label='Minimum-entropy criterion')
    a.plot(d.one_minus_alpha,d.conditional_mean_average,'s--',ms=3,color=ORANGE,label='Mean follow-up relative entropy')
    a.axvline(.9,c='.7',lw=4,alpha=.4);a.set(xlabel='1 − α',ylabel='Relative entropy');a.legend(frameon=False,fontsize=6);save(fig,'Figure1b')
    d=read('empirical/persistence_by_status.csv');fig,axs=canvas();a=axs[0]
    bars(a,d,y='mean',labels=['Deep IC','Other'],scale=100);a.set_ylabel('Time in ICs (%)');save(fig,'Figure1c')
    risks=read('empirical/age_risks.csv')
    for alpha,panel in [(.1,'d'),(.5,'f'),(.005,'g')]:
        d=risks[np.isclose(risks.alpha,alpha)].set_index('age').reindex(AGES)
        fig,axs=canvas();bars(axs[0],d,labels=AGES,scale=100)
        axs[0].set(ylabel='Deep-IC risk (%)',xlabel='Age (years)',title=f'α = {alpha:g}');save(fig,'Figure1'+panel)
    d=read('empirical/thresholds.csv').sort_values('one_minus_alpha');fig,axs=canvas();a=axs[0]
    a.plot(d.one_minus_alpha,d.disparity,'o-',color=BLUE,ms=3);a.set(xlabel='1 − α',ylabel='Age disparity');save(fig,'Figure1e')
    curves=read('statistical_results/regression_curves.csv')
    for outcome,panel,label in [('risk','h','Predicted deep-IC risk'),('depth','i','Predicted minimum relative entropy')]:
        fig,axs=canvas();a=axs[0]
        for i,(spec,g) in enumerate(curves[curves.outcome.eq(outcome)].groupby('specification')):
            a.plot(g.age_axis,g.estimate,color=COLORS[i],label=SPECS[i])
            a.fill_between(g.age_axis,g.lo,g.hi,color=COLORS[i],alpha=.10)
        a.set(xlabel='Age axis in the original plot',ylabel=label);a.legend(frameon=False,fontsize=6);save(fig,'Figure1'+panel)
    d=stat('regression_all_specifications');fig,axs=canvas(1,2)
    for a,outcome in zip(axs,['risk','depth']):
        g=d[d.outcome.eq(outcome)&d.term.eq('age_conti')].sort_values('specification').copy()
        if outcome=='risk':g[['estimate','lo','hi']]=np.exp(g[['estimate','lo','hi']])
        errors(a,g,y='estimate');a.set_xticks(range(5),SPECS,rotation=35,ha='right')
        a.set_ylabel('Age odds ratio' if outcome=='risk' else 'Age coefficient')
    save(fig,'Figure1j')


def main_2():
    d=read('empirical/behaviour_by_age_status.csv')
    for panel,metric,label in zip('abcd',RAW,RAW_LABELS):
        fig,axs=canvas();a=axs[0]
        for state,colour,name,offset in [(1,BLUE,'Deep IC',-.18),(0,ORANGE,'Other',.18)]:
            g=d[d.metric.eq(metric)&d.deep.eq(state)].set_index('age').reindex(AGES)
            x=np.arange(6)+offset;a.bar(x,g['mean'],width=.34,color=colour,label=name)
            a.errorbar(x,g['mean'],yerr=np.vstack([g['mean']-g.lo,g.hi-g['mean']]),fmt='none',color='.3',lw=.6,capsize=2)
        a.set_xticks(range(6),AGES,rotation=30,ha='right');a.set_ylabel(label);a.legend(frameon=False,fontsize=6);save(fig,'Figure2'+panel)
    d=read('model/inputs/age_behaviour_distribution.csv');fig,axs=canvas(2,3,w=2.7,h=2.5)
    for a,age in zip(axs,AGES):heat(a,d[d.age.eq(age)],z='proportion',title=age,limits=(0,100*d.proportion.max()))
    save(fig,'Figure2e')
    # Panels f,g are author-drawn schematics, not numerical plots.
    shutil.copy2(ROOT/'reference/figures/Figure_2.pdf',OUT/'Figure2_original_layout_and_schematics.pdf')


def main_3():
    d=stat('calibration_cell_support')
    for z,panel,title in [('empirical_risk','a','Empirical'),('simulated_risk','b','Simulated')]:
        fig,axs=canvas();g=d.rename(columns={z:'risk'})
        heat(axs[0],g,title=title,limits=(100*d[['empirical_risk','simulated_risk']].min().min(),100*d[['empirical_risk','simulated_risk']].max().max()));save(fig,'Figure3'+panel)
    fig,axs=canvas();a=axs[0]
    a.scatter(d.empirical_risk,d.simulated_risk,color=BLUE,s=12)
    a.plot([.1,.6],[.1,.6],'--',c='.6',lw=.8)
    r2=1-((d.empirical_risk-d.simulated_risk)**2).sum()/((d.empirical_risk-d.empirical_risk.mean())**2).sum()
    a.set(xlabel='Empirical risk',ylabel='Simulated risk',title=f'R² = {r2:.3f}');save(fig,'Figure3c')
    d=read('model/outputs/population_status.csv')
    for metric,panel,label,scale in [('timeInDEEP','d','Time in ICs (%)',100),('narrowing','e','Depth, 1 − mean relative entropy',1)]:
        fig,axs=canvas();g=d[d.metric.eq(metric)].set_index('deep').reindex([1,0])
        bars(axs[0],g,y='mean',labels=['Deep IC','Other'],scale=scale);axs[0].set_ylabel(label);save(fig,'Figure3'+panel)
    fig,axs=canvas();d=read('model/outputs/population_risks.csv').set_index('age').reindex(AGES)
    bars(axs[0],d,labels=AGES,scale=100);axs[0].set_ylabel('Simulated deep-IC risk (%)');save(fig,'Figure3f')


def main_4():
    d=read('model/outputs/phase_grid.csv')
    for panel,mask,x,y in [('a',d.beta.eq(6),'exploitation','exploration'),
                          ('b',d.exploitation.eq(9),'beta','exploration'),
                          ('c',d.exploration.eq(3),'beta','exploitation')]:
        fig,axs=canvas();heat(axs[0],d[mask],x=x,y=y,limits=(100*d.risk.min(),100*d.risk.max()));save(fig,'Figure4'+panel)
    curves=read('model/outputs/intervention_curves.csv')
    base=read('model/outputs/population_risks.csv')
    young=base.loc[base.age.eq(AGES[0]),'risk'].iloc[0]
    for parameter,panel in zip(['beta','alpha','sigma'],'def'):
        fig,axs=canvas();a=axs[0];g=curves[curves.parameter.eq(parameter)].sort_values('delta')
        errors(a,g,x=g.delta,y='risk');a.axhline(young,c=ORANGE,ls='--',label='Youngest baseline')
        a.set(xlabel='Normalized adjustment δ',ylabel='Oldest-group deep-IC risk');a.legend(frameon=False,fontsize=6);save(fig,'Figure4'+panel)


def s1():
    d=stat('note1_cohort_profile');fig,axs=canvas(2,3,w=2.7,h=2.4)
    labels=['Age','Gender','Handset price','City tier','Community type']
    for a,(dim,g),label in zip(axs,d.groupby('dimension',sort=False),labels):
        bars(a,g,y='percent',labels=g.category);a.set(ylabel='Users (%)',title=label)
    axs[-1].axis('off');save(fig,'FigureS1')


def s2():
    topics=stat('note1_topic_profile');fig,axs=canvas(2,2,w=5,h=3.3)
    for a,metric,title in [(axs[0],'candidate_topic_probability','Videos'),(axs[1],'mean_first_hour_share','First-hour viewing')]:
        a.bar(range(35),100*topics[metric],color=BLUE if a==axs[0] else ORANGE)
        a.set_xticks(range(35),topics.topic_name,rotation=90,fontsize=6);a.set(ylabel='Share (%)',title=title)
    d=read('empirical/distribution_cdfs.csv');d=d[d.group.eq('All')]
    cdf(axs[2],d,group='metric',xlabel='Content share')
    d=stat('note1_content_age_means');age_column=d.columns[0]
    for metric in ['entertainment_ratio','knowledge_ratio','news_ratio']:
        axs[3].plot(range(6),100*d[metric],marker='o',ms=3,label=metric.replace('_ratio',''))
    axs[3].set_xticks(range(6),AGES,rotation=30);axs[3].set_ylabel('Mean content share (%)');axs[3].legend(frameon=False,fontsize=6)
    save(fig,'FigureS2')


def s3():
    d=stat('fig_s1_hourly_digitized');fig,axs=canvas(1,3)
    for a,metric,label in zip(axs,['viewing_entropy','watch_time_per_video','exposure_coverage'],['Viewing entropy','Viewing time per video','Exposure topics']):
        g=d[d.metric.eq(metric)];a.plot(g.viewing_hour,g['mean'],'o-',ms=3,color=BLUE);a.set(xlabel='Viewing hour',ylabel=label)
    save(fig,'FigureS3')


def s4():
    fig,axs=canvas(1,2,w=4);d=read('empirical/boxplot_statistics.csv');box(axs[0],d,'watch_entropy_1st','First-hour entropy')
    d=read('empirical/distribution_cdfs.csv');cdf(axs[1],d[d.metric.eq('minimum_relative_entropy')],xlabel='Minimum relative entropy')
    axs[1].axvline(.7190319683690323,c='.5',ls='--');save(fig,'FigureS4')


def s5():
    d=stat('fig_s2_groups');fig,axs=canvas(3,3,w=3,h=2.5)
    order=['gender','mod_price_range','fre_city_level_eng',None,'fre_community_level_eng','entropy_quintile','ent_quintile','knowledge_quintile','news_quintile']
    labels=['Gender','Handset price','City tier','Relative disparity','Community type','Initial entropy','Entertainment share','Knowledge share','News share']
    for a,dim,label in zip(axs,order,labels):
        if dim:
            g=d[d.dimension.eq(dim)];bars(a,g,labels=g.group,scale=100);a.axhline(37.3507619138,c='.6',ls='--',lw=.7);a.set(ylabel='Deep-IC risk (%)',ylim=(0,70))
        else:
            for name,g in stat('fig_s2_contrasts').groupby('dimension',sort=False):
                a.plot(1-g.tail_probability,g.normalised_gap,marker='o',ms=2,label=name)
            a.set(xlabel='1 − α',ylabel='Disparity');a.legend(frameon=False,fontsize=6)
        a.set_title(label)
    save(fig,'FigureS5')


def s6():
    fig,axs=canvas(2,2,w=4);d=read('empirical/boxplot_statistics.csv')
    for a,metric,label in zip(axs,RAW,RAW_LABELS):box(a,d,metric,label)
    save(fig,'FigureS6')


def s7():
    fig,axs=canvas(1,3,w=3.4,h=3.1);d=read('empirical/indicator_rank_pairs.csv')
    for a,dim,ylabel,xlabel in zip(axs[:2],['Exploration','Exploitation'],['Search decile','Interested-viewing decile'],['Interaction decile','Top-decile viewing rank']):
        g=d[d.dimension.eq(dim)].rename(columns={'row_score':ylabel,'column_score':xlabel,'percent':'value'})
        heat(a,g,x=xlabel,y=ylabel,z='value',percent=False,title=dim,limits=(0,d.percent.max()),clabel='Users (%)')
    d=stat('behaviour_correlations');m=d.iloc[:,1:].to_numpy(float)
    im=axs[2].imshow(m,vmin=-1,vmax=1,cmap=MAP);axs[2].set_xticks(range(4),['Search','Interaction','Viewing','Top-decile'],rotation=40,ha='right');axs[2].set_yticks(range(4),['Search','Interaction','Viewing','Top-decile'])
    for i,j in np.ndindex(m.shape):axs[2].text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=axs[2],shrink=.7,label='Spearman correlation');save(fig,'FigureS7')


def s8():
    fig,axs=canvas(2,3,w=3,h=2.6);d=read('model/inputs/age_behaviour_distribution.csv')
    for a,age in zip(axs,AGES):heat(a,d[d.age.eq(age)],z='proportion',title=age,limits=(0,100*d.proportion.max()))
    save(fig,'FigureS8')


def s9():
    coef=stat('regression_paths').set_index(['model','term']).estimate
    fig,axs=canvas(w=8,h=2.1);a=axs[0];a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
    for x,y,label,c in [(.1,.5,'Age band\n(0–5)','.4'),(.5,.82,'Exploration\n(0–9)',BLUE),(.5,.18,'Exploitation\n(0–9)',ORANGE),(.9,.5,'Deep IC','.4')]:
        a.add_patch(FancyBboxPatch((x-.085,y-.1),.17,.2,boxstyle='round,pad=.01',ec=c,fc='white'))
        a.text(x,y,label,ha='center',va='center')
    arrows=[((.2,.57),(.4,.8),f'β={coef["explore_q","age_conti"]:.3f}',(.29,.81)),
            ((.2,.43),(.4,.2),f'β={coef["exploit_q","age_conti"]:.3f}',(.29,.17)),
            ((.6,.8),(.8,.57),f'OR={np.exp(coef["risk","explore_q"]):.3f}',(.72,.81)),
            ((.6,.2),(.8,.43),f'OR={np.exp(coef["risk","exploit_q"]):.3f}',(.72,.17)),
            ((.2,.5),(.8,.5),f'Adjusted age OR={np.exp(coef["risk","age_conti"]):.3f}',(.5,.57))]
    for start,end,label,xy in arrows:
        a.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'.35'});a.text(*xy,label,ha='center',fontsize=7)
    save(fig,'FigureS9')


def s10():
    # Preserve the author's vector layout rather than replacing a schematic.
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'reference/figures/Figure_S10.pdf',OUT/'FigureS10.pdf')


def s11():
    init=read('model/inputs/model_initialization.csv');topic=stat('note1_topic_profile')
    m=read('model/inputs/model_topic_correlations.csv').iloc[:,1:].to_numpy(float)
    order=[init.topic_id.tolist().index(v) for v in topic.topic_id]
    fig,axs=canvas(w=7,h=6);a=axs[0];im=a.imshow(m[np.ix_(order,order)],cmap=MAP,vmin=-1,vmax=1)
    a.set_xticks(range(35),topic.topic_name,rotation=90,fontsize=6);a.set_yticks(range(35),topic.topic_name,fontsize=6)
    fig.colorbar(im,ax=a,shrink=.65,label='Topic correlation');save(fig,'FigureS11')


def s12():
    d=stat('calibration_cell_support');fig,axs=canvas(1,3,w=3.4,h=3)
    heat(axs[0],d.rename(columns={'empirical_risk':'risk'}),title='Empirical risk')
    corners=d[d.exploration.isin([0,9])&d.exploitation.isin([0,9])]
    axs[1].plot(range(4),corners.empirical_risk,'o-',label='Empirical',color=BLUE);axs[1].plot(range(4),corners.simulated_risk,'s--',label='Simulated',color=ORANGE)
    axs[1].set_xticks(range(4),[f'({e},{x})' for e,x in zip(corners.exploration,corners.exploitation)],rotation=30);axs[1].set_ylabel('Deep-IC risk');axs[1].legend(frameon=False)
    error=d.pivot(index='exploration',columns='exploitation',values='error_pp').to_numpy()
    im=axs[2].imshow(error,origin='lower',cmap='PRGn',norm=TwoSlopeNorm(0,vmin=min(-1,error.min()),vmax=max(1,error.max())));fig.colorbar(im,ax=axs[2],label='Model − empirical (pp)',shrink=.7)
    axs[2].set(xlabel='Exploitation',ylabel='Exploration');save(fig,'FigureS12')


def s13():
    d=read('model/outputs/population_boxplots.csv');fig,axs=canvas(1,3,w=3.6,h=3)
    for a,metric,label in zip(axs,['entropy','timeInDEEP','narrowing'],['Stationary relative entropy','Fraction of time in ICs','Depth, 1 − mean relative entropy']):box(a,d,metric,label)
    save(fig,'FigureS13')


def s14():
    d=read('model/outputs/population_cdfs.csv');fig,axs=canvas(1,2,w=4)
    for a,metric,label in zip(axs,['timeInDEEP','narrowing'],['Fraction of time in ICs','Depth, 1 − mean relative entropy']):
        g=d[d.metric.eq(metric)].copy();g['group']=g['group'].map({0:'Other',1:'Deep IC'});cdf(a,g,xlabel=label)
    save(fig,'FigureS14')


def s15():
    d=stat('fig_s5');fig,axs=canvas(1,3,w=3.4)
    for a,panel,label in zip(axs,[1,2,3],['Weaker matching','Lower exploitation','Higher exploration']):
        for (s,x),g in d[d.panel.eq(panel)].groupby(['origin_s','origin_a']):a.plot(g.step,100*g.risk,'o-',ms=3,label=f'({s},{x})')
        a.set(xlabel='Grid adjustment steps',ylabel='Deep-IC risk (%)',title=label)
    axs[0].legend(frameon=False,fontsize=6);save(fig,'FigureS15')


def s16():
    d=read('model/outputs/phase_grid.csv');fig,axs=canvas(3,3,w=3,h=2.7)
    included=d[d.beta.between(1,9)]
    for a,b in zip(axs,range(1,10)):heat(a,d[d.beta.eq(b)],title=f'β = {b}',limits=(100*included.risk.min(),100*included.risk.max()))
    save(fig,'FigureS16')


def s17():
    d=read('model/outputs/intervention_cdfs.csv');fig,axs=canvas(1,3,w=3.5)
    for a,p in zip(axs,['beta','alpha','sigma']):
        cdf(a,d[d.parameter.eq(p)],group='delta',xlabel='Stationary relative entropy');a.axvline(.7190319683690323,c='.5',ls='--');a.set_title(p)
    save(fig,'FigureS17')


def tables():
    out=ROOT/'outputs/tables';out.mkdir(parents=True,exist_ok=True)
    for path in sorted((ROOT/'data/tables').glob('Table_S*.csv')):
        d=pd.read_csv(path,keep_default_na=False).astype(str)
        lines=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
        for row in d.itertuples(index=False,name=None):lines.append('| '+' | '.join(v.replace('|','\\|').replace('\n',' ') for v in row)+' |')
        (out/(path.stem+'.md')).write_text('\n'.join(lines)+'\n')
        shutil.copy2(path,out/path.name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main',choices=['all','1','2','3','4'])
    p.add_argument('--supplementary',default=None,help='all or a number from 1 to 17')
    p.add_argument('--tables',action='store_true')
    args=p.parse_args()
    if not any([args.main,args.supplementary,args.tables]):args.main='all';args.supplementary='all';args.tables=True
    if args.main:
        for i in (range(1,5) if args.main=='all' else [int(args.main)]):
            globals()[f'main_{i}']();print(f'Reproduced main figure {i} numerical panels.',flush=True)
    if args.supplementary:
        for i in (range(1,18) if args.supplementary=='all' else [int(args.supplementary)]):
            if not 1<=i<=17:p.error('Supplementary figure number must be 1–17.')
            globals()[f's{i}']();print(f'Prepared supplementary figure S{i}.',flush=True)
    if args.tables:tables();print('Prepared supplementary tables.',flush=True)


if __name__=='__main__':main()
