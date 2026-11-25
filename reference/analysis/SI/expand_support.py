"""Aggregate existing cohort and simulation data for the expanded supplement."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import norm
import statsmodels.api as sm

from build_figures import read_cohort, BLUE, ORANGE, AGE_KEYS, AGES, OUT, RESULTS, HERE

RAW = ['watched_search1', 'activeness', 'attention_span1', 'top10avg']
RANK = ['search_per_hour_q', 'activeness_q', 'attention_span_q', 'top10avg_q']
LABEL = ['Search frequency', 'Interaction frequency', 'Interested-video duration', 'Top-decile viewing duration']
SHORT = ['Search', 'Interaction', 'Interested\nviewing', 'Top-decile\nviewing']
AGE_COLORS = ['#355E95', '#6384B3', '#A9BCD7', '#F5C6A3', '#F08743', '#B65323']
MM = 1/25.4
MACROS = []


def macro(name, rows):
    MACROS.append('\\newcommand{\\'+name+'}{%\n'+'\n'.join(rows)+'\n}\n')


def row(values):
    return ' & '.join(map(str, values))+r' \\'


def style(ax, letter=None):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=.65)
    if letter:
        ax.text(-.16, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='bottom')


def save(fig, name):
    fig.savefig(OUT/f'ic-{name}.pdf')
    plt.close(fig)


def boxes(ax, arrays, colors, labels):
    b = ax.boxplot(arrays, positions=np.arange(len(arrays)), widths=.55,
                   whis=(5,95), showfliers=False, patch_artist=True,
                   medianprops={'color':'#27364A','linewidth':1.1},
                   whiskerprops={'color':'#666666','linewidth':.65},
                   capprops={'color':'#666666','linewidth':.65},
                   boxprops={'linewidth':.6})
    for patch,color in zip(b['boxes'],colors):
        patch.set(facecolor=color, edgecolor=color, alpha=.75)
    ax.set_xticks(np.arange(len(arrays)), labels)
    ax.tick_params(axis='x',labelsize=8)
    return b


def behaviours(df):
    cutrows, descriptives, values = [], [], []
    for key,q in zip(RAW,RANK):
        ranks,bins = pd.qcut(df[key],10,labels=False,retbins=True)
        np.testing.assert_array_equal(ranks,df[q])
        values.append(bins)
        descriptives.append(dict(indicator=key,mean=df[key].mean(),sd=df[key].std(),
                                 median=df[key].median(),q25=df[key].quantile(.25),
                                 q75=df[key].quantile(.75),zero_n=int(df[key].eq(0).sum())))
        for j in range(10):cutrows.append((key,j,bins[j],bins[j+1],int((ranks==j).sum())))
    np.testing.assert_array_equal((df[RANK[0]]+df[RANK[1]])//2,df.explore_q)
    np.testing.assert_array_equal((df[RANK[2]]+df[RANK[3]])//2,df.exploit_q)
    pd.DataFrame(cutrows,columns=['indicator','rank','lower','upper','n']).to_csv(RESULTS/'behaviour_deciles.csv',index=False)
    pd.DataFrame(descriptives).to_csv(RESULTS/'behaviour_descriptive.csv',index=False)
    macro('BehaviourCutRows',[row([f'{10*j}th' if j else 'Minimum']+[f'{v[j]:.4f}' for v in values]) for j in range(10)]+[row(['Maximum']+[f'{v[-1]:.4f}' for v in values])])
    macro('BehaviourSummaryRows',[row([label,f'{d["mean"]:.2f}',f'{d["sd"]:.2f}',
                  f'{d["median"]:.2f} [{d["q25"]:.2f}, {d["q75"]:.2f}]',f'{d["zero_n"]:,}'])
                  for label,d in zip(LABEL,descriptives)])
    fig,axes=plt.subplots(2,2,figsize=(180*MM,112*MM))
    fig.subplots_adjust(left=.10,right=.985,bottom=.11,top=.94,wspace=.27,hspace=.48)
    for i,(ax,key) in enumerate(zip(axes.flat,RAW)):
        arrays=[df.loc[df.age_range==age,key].to_numpy() for age in AGE_KEYS]
        boxes(ax,arrays,[BLUE if i<2 else ORANGE]*6,AGES)
        ax.set_title(LABEL[i],fontsize=9,pad=8)
        ax.set_ylabel('Per hour' if i<2 else 'Viewing duration (s)')
        ax.set_ylim(bottom=0);style(ax,'abcd'[i])
        ax.set_xlabel('Age (years)')
    save(fig,'behaviour-distributions')

    corr=df[RAW].corr(method='spearman')
    corr.to_csv(RESULTS/'behaviour_correlations.csv')
    fig=plt.figure(figsize=(180*MM,87*MM))
    axes=[fig.add_axes([x,.36,.235,.49]) for x in [.075,.40,.725]]
    matrices=[]
    for a,b in [(RANK[0],RANK[1]),(RANK[2],RANK[3])]:
        m=pd.crosstab(df[a],df[b]).reindex(index=range(10),columns=range(10),fill_value=0).to_numpy()/len(df)*100
        matrices.append(m)
    vmax=np.ceil(max(m.max() for m in matrices))
    for i,(ax,m) in enumerate(zip(axes,matrices)):
        im=ax.imshow(m,origin='lower',vmin=0,vmax=vmax,cmap='Blues',interpolation='nearest')
        ax.set(xticks=[0,3,6,9],yticks=[0,3,6,9])
        ax.set_title(['Exploration indicators','Exploitation indicators'][i],fontsize=9,pad=9)
        ax.set_xlabel(['Interaction decile','Top-decile viewing rank'][i],fontsize=8)
        ax.set_ylabel(['Search decile','Interested-viewing rank'][i],fontsize=8)
        ax.text(-.22,1.06,'ab'[i],transform=ax.transAxes,fontsize=11,fontweight='bold')
    cb=fig.colorbar(im,cax=fig.add_axes([.145,.19,.40,.022]),orientation='horizontal')
    cb.set_label('Users (%)',fontsize=8);cb.ax.tick_params(labelsize=7)
    palette=LinearSegmentedColormap.from_list('correlation',['#6384B3','#FFFFFF','#F08743'])
    im=axes[2].imshow(corr,cmap=palette,vmin=-1,vmax=1)
    axes[2].set_xticks(range(4),SHORT,rotation=45,ha='right',fontsize=7)
    axes[2].set_yticks(range(4),SHORT,fontsize=7)
    axes[2].set_title('Indicator correlations',fontsize=9,pad=9)
    axes[2].text(-.22,1.06,'c',transform=axes[2].transAxes,fontsize=11,fontweight='bold')
    for a,b in np.ndindex((4,4)):
        axes[2].text(b,a,f'{corr.iloc[a,b]:.2f}',ha='center',va='center',fontsize=7,
                     color='white' if abs(corr.iloc[a,b])>.75 else '#202020')
    cb=fig.colorbar(im,cax=fig.add_axes([.76,.10,.18,.022]),orientation='horizontal',ticks=[-1,0,1])
    cb.set_label('Spearman correlation',fontsize=7,labelpad=1);cb.ax.tick_params(labelsize=7)
    save(fig,'behaviour-relations')
    rows=[]
    for age,label in zip(AGE_KEYS,AGES):
        g=df[df.age_range==age]
        d={'age':label,'n':len(g)}
        for key in ['explore_q','exploit_q']:
            d[key+'_mean']=g[key].mean();d[key+'_sd']=g[key].std()
        d['score_correlation']=g[['explore_q','exploit_q']].corr(method='spearman').iloc[0,1]
        rows.append(d)
    pd.DataFrame(rows).to_csv(RESULTS/'behaviour_age_scores.csv',index=False)
    macro('BehaviourAgeRows',[row([d['age'].replace('–','--'),f'{d["n"]:,}',
              f'{d["explore_q_mean"]:.2f} ({d["explore_q_sd"]:.2f})',
              f'{d["exploit_q_mean"]:.2f} ({d["exploit_q_sd"]:.2f})',f'{d["score_correlation"]:.3f}']) for d in rows])


def mediation():
    d=pd.read_csv(RESULTS/'joint_mediation_risks.csv')
    macro('StandardizedRiskRows',[row([f'$R_{{{int(r.outcome_age)}{int(r.exploration_age)}{int(r.exploitation_age)}}}$',
              int(r.outcome_age),int(r.exploration_age),int(r.exploitation_age),f'{100*r.standardised_risk:.2f}']) for r in d.itertuples()])
    r=d.standardised_risk.to_numpy()*100
    steps=[('Exploration first',r[2]-r[1],r[4]-r[2]),('Exploitation first',r[4]-r[3],r[3]-r[1])]
    macro('MediationOrderRows',[row([n,f'{e:.2f}',f'{x:.2f}',f'{e+x:.2f}']) for n,e,x in steps])
    j=pd.read_csv(RESULTS/'joint_mediation.csv').set_index('component')
    assert np.isclose(np.mean([v[1] for v in steps]),100*j.loc['Exploration','risk_difference'])
    assert np.isclose(np.mean([v[2] for v in steps]),100*j.loc['Exploitation','risk_difference'])


def initialization():
    init=pd.read_csv(HERE/'source_data/model_initialization.csv')
    corr=pd.read_csv(HERE/'source_data/model_topic_correlations.csv',index_col=0)
    np.testing.assert_allclose(init[['mean_first_hour_share','candidate_topic_probability']].sum(),1)
    np.testing.assert_allclose(np.diag(corr),1)
    fig=plt.figure(figsize=(180*MM,95*MM))
    a=fig.add_axes([.10,.60,.46,.30]);b=fig.add_axes([.10,.15,.46,.30]);c=fig.add_axes([.67,.25,.29,.55])
    for ax,key,label,color,l in [(a,'candidate_topic_probability','Candidate items (%)',BLUE,'a'),
                                    (b,'mean_first_hour_share','First-hour viewing (%)',ORANGE,'b')]:
        ax.bar(np.arange(35),init[key]*100,width=.75,color=color)
        ax.set(xlim=(-.7,34.7),ylabel=label,xticks=[0,9,19,29,34],
               xticklabels=init.topic_id.iloc[[0,9,19,29,34]].tolist())
        ax.set_xlabel('Topic ID',labelpad=2);style(ax)
        ax.text(-.19,1.05,l,transform=ax.transAxes,fontsize=11,fontweight='bold')
    im=c.imshow(corr,cmap=LinearSegmentedColormap.from_list('topiccorr',['#6384B3','white','#F08743']),
                norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),interpolation='nearest')
    ticks=[0,9,19,29,34];c.set(xticks=ticks,yticks=ticks,xticklabels=init.topic_id.iloc[ticks],
                  yticklabels=init.topic_id.iloc[ticks],xlabel='Topic ID',ylabel='Topic ID')
    c.text(-.19,1.04,'c',transform=c.transAxes,fontsize=11,fontweight='bold')
    cb=fig.colorbar(im,cax=fig.add_axes([.70,.13,.23,.025]),orientation='horizontal',ticks=[-1,0,1]);cb.set_label('Topic correlation',fontsize=8,labelpad=1)
    save(fig,'model-inputs')
    macro('ModelInputRows',[row([int(r.topic_id),f'{100*r.candidate_topic_probability:.3f}',
                    f'{100*r.mean_first_hour_share:.3f}',f'{r.dirichlet_concentration:.4f}']) for r in init.itertuples()])


def calibration_support(df):
    g=df.groupby(['explore_q','exploit_q']).deep.agg(['size','sum','mean'])
    recomputed=g['mean'].unstack().reindex(index=range(10),columns=range(10)).to_numpy()
    em=pd.read_pickle(HERE.parent/'processed_data/heatmap_em_new.pkl')
    sim=pd.read_pickle(HERE.parent/'processed_data/heatmap_model.pkl')['mat']
    dist=pd.read_pickle(HERE.parent/'processed_data/age_range_all_dist.pkl')
    n=np.sum([dist[key]*dist['age_dist'][i] for i,key in enumerate(AGE_KEYS)],axis=0)
    np.testing.assert_allclose(n,np.round(n),rtol=0,atol=1e-9)
    np.testing.assert_allclose(em*n,np.round(em*n),rtol=0,atol=1e-9)
    assert n.sum()==len(df) and int(np.round((em*n).sum()))==int(df.deep.sum())
    n=n.astype(int);err=(sim-em)*100
    # Keep the matching cell counts and outcomes used in the published model grid.
    # CSV round-trip ranks move three users at tied decile boundaries, while totals
    # and all displayed score means are unchanged. Preserve the audit locally.
    (RESULTS/'calibration_rank_audit.json').write_text(json.dumps({
        'cohort_n':len(df),'deep_n':int(df.deep.sum()),
        'cells_with_count_difference_from_csv_scores':int(np.sum(n!=g['size'].unstack().to_numpy())),
        'users_redistributed':int(np.abs(n-g['size'].unstack().to_numpy()).sum()/2),
        'max_risk_difference_pp':float(np.max(np.abs(em-recomputed))*100),
        'calibration_counts_source':'age_range_all_dist.pkl; exact integer counts and cases verified'
    },indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(180*MM,78*MM),gridspec_kw={'width_ratios':[1,1]})
    fig.subplots_adjust(left=.09,right=.96,bottom=.23,top=.90,wspace=.48)
    im=axes[0].imshow(n/1000,origin='lower',cmap='Blues',interpolation='nearest')
    axes[0].set(xticks=[0,3,6,9],yticks=[0,3,6,9],xlabel='Exploitation score',ylabel='Exploration score')
    cb=fig.colorbar(im,ax=axes[0],fraction=.045,pad=.035);cb.set_label('Users (thousands)',fontsize=8)
    axes[0].text(-.22,1.04,'a',transform=axes[0].transAxes,fontsize=11,fontweight='bold')
    axes[1].scatter(em.ravel()*100,err.ravel(),s=8+n.ravel()/1000*3,c=BLUE,alpha=.7,edgecolors='none')
    for e,x in [(0,0),(0,9),(9,0),(9,9)]:axes[1].plot(em[e,x]*100,err[e,x],'s',mfc='none',mec=ORANGE,ms=6,mew=1)
    axes[1].axhline(0,color='#555555',lw=.8,ls=(0,(3,3)))
    axes[1].set(xlabel='Empirical deep-IC risk (%)',ylabel='Model − empirical (pp)',ylim=(-8,8))
    style(axes[1],'b')
    save(fig,'calibration-support')
    rows=[]
    for e,x in np.ndindex(em.shape):rows.append([e,x,int(n[e,x]),em[e,x],sim[e,x],err[e,x]])
    pd.DataFrame(rows,columns=['exploration','exploitation','n','empirical_risk','simulated_risk','error_pp']).to_csv(RESULTS/'calibration_cell_support.csv',index=False)
    macro('MappingRows',[row([i,f'{.018+.0003458*i:.7f}',f'{5.6+.1228*i:.4f}']) for i in range(10)])
    sums=dict(min_cell_n=int(n.min()),max_cell_n=int(n.max()),median_cell_n=float(np.median(n)),
              mean_error_pp=float(err.mean()),mae_all_pp=float(np.abs(err).mean()),
              rmse_all_pp=float(np.sqrt((err**2).mean())),weighted_mae_pp=float(np.average(np.abs(err),weights=n)),
              r2=float(1-((sim-em)**2).sum()/((em-em.mean())**2).sum()))
    (RESULTS/'calibration_expanded_summary.json').write_text(json.dumps(sums,indent=2)+'\n')
    return sums


def population():
    d=pd.read_pickle(HERE.parent/'processed_data/samples_10000_model.pkl').copy()
    assert len(d)==10000 and d[['entropy','timeInDEEP','depth','age_range']].notna().all().all()
    d['narrowing']=1-d.depth
    fig,axes=plt.subplots(1,3,figsize=(180*MM,75*MM))
    fig.subplots_adjust(left=.08,right=.985,bottom=.23,top=.90,wspace=.44)
    for i,(ax,key,label) in enumerate(zip(axes,['entropy','timeInDEEP','narrowing'],
                     ['Stationary relative entropy','Time in ICs (%)','Depth, 1 − mean relative entropy'])):
        ar=[d.loc[d.age_range==age,key].to_numpy()*(100 if key=='timeInDEEP' else 1) for age in AGE_KEYS]
        boxes(ax,ar,AGE_COLORS,AGES)
        ax.tick_params(axis='x',rotation=45,labelsize=7)
        ax.set_ylabel(label,fontsize=8);ax.set_xlabel('Age (years)');style(ax,'abc'[i])
        if key=='entropy':ax.axhline(.719032,c='.4',lw=.7,ls=(0,(3,3)))
        elif key=='timeInDEEP':ax.set_ylim(-3,103)
        else:ax.axhline(0,c='.6',lw=.7,ls=(0,(3,3)))
    save(fig,'population-outcomes')
    rows=[]
    for age,label in zip(AGE_KEYS,AGES):
        g=d[d.age_range==age]
        rows.append([label,len(g),g.indicator.mean(),g.entropy.mean(),g.timeInDEEP.mean(),g.narrowing.mean()])
    pd.DataFrame(rows,columns=['age','n','risk','final_entropy','persistence','depth']).to_csv(RESULTS/'population_outcomes.csv',index=False)
    macro('PopulationOutcomeRows',[row([r[0].replace('–','--'),f'{r[1]:,}',f'{r[2]*100:.2f}',f'{r[3]:.3f}',f'{r[4]*100:.2f}',f'{r[5]:.3f}']) for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(180*MM,70*MM))
    fig.subplots_adjust(left=.10,right=.97,bottom=.19,top=.87,wspace=.35)
    for i,(ax,key,label) in enumerate(zip(axes,['timeInDEEP','narrowing'],['Time in ICs (%)','Depth, 1 − mean relative entropy'])):
        for state,color,name in [(1,BLUE,'Deep IC'),(0,ORANGE,'Other users')]:
            v=np.sort(d.loc[d.indicator==state,key].to_numpy())*(100 if key=='timeInDEEP' else 1)
            # Exact empirical CDF, collapsed only where x values are tied.
            unique,last=np.unique(v,return_counts=True)
            ax.step(unique,np.cumsum(last)/len(v)*100,where='post',color=color,lw=1.5,label=name)
        ax.set(xlabel=label,ylabel='Cumulative users (%)',ylim=(0,100));style(ax,'ab'[i])
    axes[0].legend(frameon=False,loc='upper left',fontsize=8)
    save(fig,'population-distributions')
    groups=d.groupby('indicator').agg(n=('entropy','size'),risk_entropy=('entropy','mean'),persistence=('timeInDEEP','mean'),depth=('narrowing','mean'))
    macro('PopulationGroupRows',[row(['Deep IC' if k else 'Other users',f'{int(r.n):,}',f'{r.risk_entropy:.3f}',f'{100*r.persistence:.2f}',f'{r.depth:.3f}']) for k,r in groups.loc[[1,0]].iterrows()])


def baseline(df):
    fig,axes=plt.subplots(1,2,figsize=(180*MM,84*MM))
    fig.subplots_adjust(left=.09,right=.985,bottom=.19,top=.91,wspace=.32)
    boxes(axes[0],[df.loc[df.age_range==a,'watch_entropy_1st'].to_numpy() for a in AGE_KEYS],AGE_COLORS,AGES)
    axes[0].set(xlabel='Age (years)',ylabel='First-hour entropy');style(axes[0],'a')
    for a,label,color in zip(AGE_KEYS,AGES,AGE_COLORS):
        v=np.sort(df.loc[df.age_range==a,'m_ratio'].to_numpy())
        unique,cnt=np.unique(v,return_counts=True)
        ix=np.unique(np.r_[np.linspace(0,len(unique)-1,1600).astype(int),np.searchsorted(unique,.719032)])
        axes[1].plot(unique[ix],np.cumsum(cnt)[ix]/len(v)*100,c=color,lw=1.2,label=label)
    axes[1].axvline(.719032,c='.4',lw=.7,ls=(0,(3,3)))
    axes[1].set(xlabel='Minimum relative entropy',ylabel='Cumulative users (%)',xlim=(0,1.5),ylim=(0,100))
    axes[1].legend(frameon=False,ncol=2,fontsize=7,loc='lower right');style(axes[1],'b')
    save(fig,'baseline-distributions')
    rows=[]
    for a,label in zip(AGE_KEYS,AGES):
        g=df[df.age_range==a]
        rows.append(row([label.replace('–','--'),f'{len(g):,}',f'{g.watch_entropy_1st.mean():.3f}',f'{g.watch_entropy_1st.std():.3f}',f'{g.m_ratio.mean():.3f}']))
    macro('BaselineAgeRows',rows)


def thresholds(df):
    rows=[];detail=[]
    for a in [.5,.35,.25,.2,.15,.1,.05,.025,.005]:
        c=1+norm.ppf(a)*df.m_ratio.std(ddof=0);y=df.m_ratio<c
        risks=[y[df.age_range==k].mean() for k in AGE_KEYS]
        rows.append(row([f'{a:g}',f'{c:.6f}',f'{int(y.sum()):,}',f'{100*y.mean():.2f}',f'{100*(risks[-1]-risks[0]):.2f}',f'{risks[-1]/risks[0]:.3f}']))
        detail.append(row([f'{a:g}']+[f'{100*v:.2f}' for v in risks]))
    macro('ThresholdRows',rows);macro('ThresholdAgeRows',detail)


def full_regressions(df):
    x=pd.get_dummies(df,columns=['gender','fre_city_level_eng','fre_community_level_eng'],dtype=float)
    social=['price_conti','fre_city_level_eng_first','fre_city_level_eng_second','fre_city_level_eng_third','fre_city_level_eng_fourth','fre_community_level_eng_County','fre_community_level_eng_Countryside']
    interests=['entertainment_ratio','news_ratio','watch_entropy_1st']
    specs=[[],['gender_F'],social,interests,['gender_F']+social+interests]
    terms=['age_conti','gender_F']+social+interests+['const']
    labels=['Age band','Female','Handset price band','City tier 1','City tier 2','City tier 3','City tier 4','County','Countryside','Entertainment share','News share','First-hour entropy','Intercept']
    rows=[];report={}
    for outcome in ['risk','depth']:
        fits=[]
        for j,extra in enumerate(specs):
            design=sm.add_constant(x[['age_conti']+extra])
            fit=(sm.GLM(x.deep,design,family=sm.families.Binomial()).fit(cov_type='HC0') if outcome=='risk'
                 else sm.OLS(x.m_ratio,design).fit(cov_type='HC0'))
            fits.append(fit)
            for term,v in fit.params.items():rows.append([outcome,j+1,term,v,*fit.conf_int().loc[term],int(fit.nobs)])
        table=[]
        for term,label in zip(terms,labels):
            cells=[]
            for fit in fits:
                if term not in fit.params:cells.append(r'\textemdash');continue
                vals=np.r_[fit.params[term],fit.conf_int().loc[term]]
                if term in ['entertainment_ratio','news_ratio']:vals=vals*.1
                if outcome=='risk' and term!='const':vals=np.exp(vals)
                precision=3 if outcome=='risk' else 4
                cells.append(r'\shortstack{'+f'{vals[0]:.{precision}f}'+r'\\{'+f'[{vals[1]:.{precision}f}, {vals[2]:.{precision}f}]'+'}}')
            table.append(row([label]+cells))
        macro('FullRiskSpecs' if outcome=='risk' else 'FullDepthSpecs',table)
    pd.DataFrame(rows,columns=['outcome','specification','term','estimate','lo','hi','n']).to_csv(RESULTS/'regression_all_specifications.csv',index=False)


def interventions():
    summary=[];frames=[]
    for p,key,scale in [('beta','reduced_beta',1),('alpha','reduced_alpha',10),('sigma','increased_sigma',10)]:
        d=pd.read_pickle(HERE.parent/'processed_data'/f'varying_{p}.pkl').copy()
        d['delta']=np.round(d[key]/scale,1);d['parameter']=p;frames.append(d)
        g=d.groupby('delta').indicator.agg(['size','mean']);assert g['size'].eq(1596).all()
        for delta,r in g.iterrows():summary.append([p,delta,int(r['size']),r['mean']])
    s=pd.DataFrame(summary,columns=['parameter','delta','n','risk']);s.to_csv(RESULTS/'intervention_full_grid.csv',index=False)
    wide=s.pivot(index='delta',columns='parameter',values='risk');young=661/1889;base=664/1596
    np.testing.assert_allclose(wide.loc[0],base)
    macro('FullDoseRows',[row([f'{delta:.1f}']+[f'{100*r[p]:.2f}' for p in ['beta','alpha','sigma']]+[f'{100*(base-r[p])/(base-young):.1f}' for p in ['beta','alpha','sigma']]) for delta,r in wide.iterrows()])
    fig,axes=plt.subplots(1,3,figsize=(180*MM,77*MM))
    fig.subplots_adjust(left=.09,right=.985,bottom=.21,top=.81,wspace=.34)
    for i,(ax,d,title) in enumerate(zip(axes,frames,['Weaker matching','Lower exploitation','Higher exploration'])):
        for delta,color,ls in [(0,BLUE,'-'),(.3,ORANGE,'-'),(1,'#62666D','--')]:
            v=np.sort(d.loc[d.delta==delta,'entropy'].to_numpy());unique,cnt=np.unique(v,return_counts=True)
            ax.step(unique,np.cumsum(cnt)/len(v)*100,where='post',color=color,ls=ls,lw=1.2,label=f'δ = {delta:g}')
        ax.axvline(.719032,c='.45',ls=(0,(2,3)),lw=.7)
        ax.set(xlim=(0,1.6),ylim=(0,100),xlabel='Stationary relative entropy',ylabel='Cumulative users (%)')
        ax.set_title(title,fontsize=9,pad=9);style(ax,'abc'[i])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.53,1.0))
    save(fig,'intervention-distributions')


if __name__=='__main__':
    df=read_cohort()
    behaviours(df);mediation();initialization();cal=calibration_support(df);population()
    baseline(df);thresholds(df);full_regressions(df);interventions()
    included=[m for m in MACROS if not any(key in m.splitlines()[0] for key in ['ModelInputRows','MappingRows'])]
    (OUT/'ic-expanded-tables.tex').write_text('% Generated aggregate results; no individual records.\n'+''.join(included))
    print('Saved expanded figures and tables. Calibration:',cal)
