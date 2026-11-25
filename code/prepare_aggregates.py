"""Author-side export of aggregate data. No empirical person-level rows are written.

This optional script needs the private research directory. Public reproduction uses
the already exported CSV files and does not call this script.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind_from_stats

ROOT = Path(__file__).resolve().parents[1]
AGES = ['18-23', '24-30', '31-40', '41-50', '51-60', '60+']
LABELS = ['19–23', '24–30', '31–40', '41–50', '51–60', '61–79']
RAW = ['watched_search1', 'activeness', 'attention_span1', 'top10avg']
RANKS = ['search_per_hour_q', 'activeness_q', 'attention_span_q', 'top10avg_q']
TAILS = np.array([.005, .025, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5])
PROVENANCE = []


def write(frame, name, sources, description):
    path = ROOT / 'data' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    PROVENANCE.append(dict(file='data/'+name, sources=sources,
                           description=description, rows=len(frame)))


def describe(v):
    v = np.asarray(v, float)
    assert len(v) and np.isfinite(v).all()
    n = len(v)
    sd = np.std(v, ddof=1) if n > 1 else 0.
    se = sd / np.sqrt(n)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return dict(n=n, mean=v.mean(), sd=sd, se=se,
                lo=v.mean()-norm.ppf(.975)*se, hi=v.mean()+norm.ppf(.975)*se,
                q05=q[0], q25=q[1], median=q[2], q75=q[3], q95=q[4])


def cdf_rows(v, grid, **group):
    v = np.sort(np.asarray(v, float))
    count = np.searchsorted(v, grid, side='right')
    return [dict(**group, x=x, count_le=int(c), n=len(v), cdf=c/len(v))
            for x, c in zip(grid, count)]


def public_empirical(source, df):
    src = ['cleaned_data_v2.csv', 'cleaned_data_1216.csv']
    sigma = df.m_ratio.std(ddof=0)
    thresholds, age_risks = [], []
    for alpha in TAILS:
        criterion = 1 + norm.ppf(alpha)*sigma
        y = df.m_ratio < criterion
        p = y.mean()
        risks = []
        for key, label in zip(AGES, LABELS):
            mask = df.age_range.eq(key)
            n, cases = int(mask.sum()), int(y[mask].sum())
            risk = cases/n
            se = np.sqrt(risk*(1-risk)/n)
            age_risks.append(dict(alpha=alpha, criterion=criterion, age=label,
                                 n=n, cases=cases, risk=risk,
                                 lo=max(0., risk-1.95996398454*se),
                                 hi=min(1., risk+1.95996398454*se)))
            risks.append(risk)
        thresholds.append(dict(alpha=alpha, one_minus_alpha=1-alpha,
                               criterion=criterion, n=len(df), cases=int(y.sum()), risk=p,
                               disparity=(risks[-1]-risks[0])/p,
                               risk_ratio=risks[-1]/risks[0],
                               oldest_minus_youngest_pp=100*(risks[-1]-risks[0])))
    write(pd.DataFrame(thresholds), 'empirical/thresholds.csv', src,
          'One-sided tail probabilities, criteria, prevalence and age contrasts.')
    write(pd.DataFrame(age_risks), 'empirical/age_risks.csv', src,
          'Six age groups at each threshold; normal binomial confidence intervals.')
    count, edges = np.histogram(df.m_ratio, bins=100)
    write(pd.DataFrame(dict(left=edges[:-1], right=edges[1:],
                           centre=(edges[:-1]+edges[1:])/2, count=count,
                           density=count/(len(df)*np.diff(edges)))),
          'empirical/min_entropy_histogram.csv', src, '100-bin minimum relative entropy distribution.')
    behaviour, boxes, cdfs, baselines = [], [], [], []
    for key, label in zip(AGES, LABELS):
        g = df[df.age_range.eq(key)]
        for metric in RAW+['watch_entropy_1st']:
            boxes.append(dict(age=label, metric=metric, **describe(g[metric])))
        baselines.append(dict(age=label, n=len(g), first_hour_mean=g.watch_entropy_1st.mean(),
                              first_hour_sd=g.watch_entropy_1st.std(), minimum_mean=g.m_ratio.mean()))
        cdfs.extend(cdf_rows(g.m_ratio, np.linspace(0, 2, 401), metric='minimum_relative_entropy', group=label))
        for state in [0, 1]:
            for metric in RAW:
                behaviour.append(dict(age=label, deep=state, metric=metric,
                                      **describe(g.loc[g.deep.eq(state), metric])))
    for metric in ['entertainment_ratio', 'knowledge_ratio', 'news_ratio']:
        cdfs.extend(cdf_rows(df[metric], np.linspace(0, 1, 201), metric=metric, group='All'))
    write(pd.DataFrame(behaviour), 'empirical/behaviour_by_age_status.csv', src,
          'Age-by-IC-status means; public mean intervals use the normal approximation, not the legacy unseeded bootstrap.')
    write(pd.DataFrame(boxes), 'empirical/boxplot_statistics.csv', src,
          'Group-level 5th, 25th, 50th, 75th and 95th percentiles; no fliers or individual points.')
    write(pd.DataFrame(cdfs), 'empirical/distribution_cdfs.csv', src,
          'CDF counts on fixed public grids, rather than individual sorted observations.')
    write(pd.DataFrame(baselines), 'empirical/baseline_by_age.csv', src, 'Age-group summaries for Table S4.')
    pairs = []
    for name, a, b in [('Exploration', RANKS[0], RANKS[1]), ('Exploitation', RANKS[2], RANKS[3])]:
        counts = pd.crosstab(df[a], df[b]).reindex(index=range(10), columns=range(10), fill_value=0)
        for i,j in np.ndindex((10,10)):
            pairs.append(dict(dimension=name, row_score=i, column_score=j,
                              count=int(counts.iloc[i,j]), percent=100*counts.iloc[i,j]/len(df)))
    write(pd.DataFrame(pairs), 'empirical/indicator_rank_pairs.csv', src, 'Joint rank counts for Supplementary Figure S7.')

    # Reproduce the existing Fig. 1h,i curves using saved coefficients and cohort
    # means. The boundary construction is retained rather than silently refitted.
    coefs = pd.read_csv(source/'SI/results/regression_all_specifications.csv')
    x = pd.get_dummies(df, columns=['gender','fre_city_level_eng','fre_community_level_eng'], dtype=float)
    means = x.select_dtypes(include='number').mean()
    axis_age = np.arange(18, 62)
    age_code = np.r_[np.arange(6)/6, 1+np.arange(7)/7,
                     2+np.arange(10)/10, 3+np.arange(10)/10,
                     4+np.arange(10)/10, 5.]
    curves = []
    for (outcome,spec), part in coefs.groupby(['outcome','specification']):
        part=part.set_index('term')
        others=sum(r.estimate*means[term] for term,r in part.iterrows() if term not in ['const','age_conti'])
        b0,b1=part.loc['const'],part.loc['age_conti']
        centre=b0.estimate+b1.estimate*age_code+others
        bounds=np.vstack([a+b*age_code+others for a in [b0.lo,b0.hi] for b in [b1.lo,b1.hi]])
        if outcome=='risk':
            centre=1/(1+np.exp(-centre));bounds=1/(1+np.exp(-bounds))
        for age,code,mu,lo,hi in zip(axis_age,age_code,centre,bounds.min(0),bounds.max(0)):
            curves.append(dict(outcome=outcome,specification=int(spec),age_axis=age,
                               age_code=code,estimate=mu,lo=lo,hi=hi))
    write(pd.DataFrame(curves), 'statistical_results/regression_curves.csv', src+['SI/results/regression_all_specifications.csv'],
          'Legacy age-axis mapping and coefficient-bound envelope from Fig. 1 notebook; no model is refitted.')
    return sigma


def longitudinal(source, df, sigma):
    pieces=[]
    ids=set(df.user_id)
    for name in ['ratio_for_cleaned_p1.pkl','ratio_for_cleaned_p2.pkl']:
        print('Aggregating longitudinal partition:', name, flush=True)
        d=pd.read_pickle(source/name)
        d['user_id']=d.user_id.astype(str)
        d=d.loc[d.user_id.isin(ids), ['user_id','ratio']]
        assert d.ratio.notna().all()
        d['narrow']=d.ratio.lt(1).astype('int8')
        pieces.append(d.groupby('user_id').agg(windows=('ratio','size'), total=('ratio','sum'), narrow=('narrow','sum')))
        del d;gc.collect()
    totals=pd.concat(pieces).groupby(level=0).sum()
    z=df[['user_id','m_ratio','deep']].merge(totals,left_on='user_id',right_index=True,validate='one_to_one')
    assert len(z)==653486
    z['average_entropy']=z.total/z.windows
    z['persistence']=z.narrow/z.windows
    src=['ratio_for_cleaned_p1.pkl','ratio_for_cleaned_p2.pkl','cleaned_data_v2.csv']
    rows=[]
    for alpha in TAILS:
        criterion=1+norm.ppf(alpha)*sigma
        g=z[z.m_ratio<criterion]
        rows.append(dict(alpha=alpha,one_minus_alpha=1-alpha,criterion=criterion,
                         n=len(g),conditional_mean_minimum=g.m_ratio.mean(),
                         conditional_mean_average=g.average_entropy.mean()))
    write(pd.DataFrame(rows), 'empirical/threshold_depth.csv', src,
          'Figure 1b: criterion curve and mean follow-up entropy among classified users. Conditional mean minimum is provided separately.')
    rows=[dict(deep=state,**describe(z.loc[z.deep.eq(state),'persistence'])) for state in [1,0]]
    a,b=rows
    result=ttest_ind_from_stats(a['mean'],a['sd'],a['n'],b['mean'],b['sd'],b['n'],equal_var=True)
    for row in rows:row.update(t_statistic=result.statistic,p_value=result.pvalue)
    write(pd.DataFrame(rows), 'empirical/persistence_by_status.csv', src,
          'Mean proportion of follow-up windows in ICs; two-sided equal-variance t test; normal mean intervals.')


def model_outputs(source):
    sim=source/'processed_data'
    dist=pd.read_pickle(sim/'age_range_all_dist.pkl')
    age_grid=[]
    for i,(key,label) in enumerate(zip(AGES,LABELS)):
        for e,x in np.ndindex((10,10)):
            age_grid.append(dict(age=label,exploration=e,exploitation=x,
                                 count=int(round(dist[key][e,x]*dist['age_dist'][i])),
                                 proportion=dist[key][e,x],age_n=int(dist['age_dist'][i])))
    write(pd.DataFrame(age_grid), 'model/inputs/age_behaviour_distribution.csv',
          ['processed_data/age_range_all_dist.pkl'], 'Stored calibration-aligned age-group joint frequencies.')
    phase=pd.read_pickle(sim/'phase_diagram.pkl')
    rows=[dict(beta=b,exploitation=x,exploration=e,alpha=5.6+.1228*x,
               sigma=.018+.0003458*e,risk=phase[b,x,e])
          for b,x,e in np.ndindex(phase.shape)]
    write(pd.DataFrame(rows), 'model/outputs/phase_grid.csv', ['processed_data/phase_diagram.pkl'],
          'Saved 10×10×10 risk grid; beta=0 retained for legacy Fig. 4 slices, but excluded from the documented nine-level rerun.')
    d=pd.read_pickle(sim/'samples_10000_model.pkl').copy()
    d['narrowing']=1-d.depth
    boxes=[];summaries=[];cdfs=[]
    for key,label in zip(AGES,LABELS):
        g=d[d.age_range.eq(key)]
        for metric in ['entropy','timeInDEEP','narrowing']:
            boxes.append(dict(age=label,metric=metric,**describe(g[metric])))
        risk=g.indicator.mean();se=np.sqrt(risk*(1-risk)/len(g))
        summaries.append(dict(age=label,n=len(g),cases=int(g.indicator.sum()),risk=risk,
                              lo=risk-1.95996398454*se,hi=risk+1.95996398454*se))
    groups=[]
    for state in [1,0]:
        for metric,grid in [('timeInDEEP',np.linspace(0,1,101)),('narrowing',np.linspace(-2,1,601))]:
            v=d.loc[d.indicator.eq(state),metric]
            groups.append(dict(deep=state,metric=metric,**describe(v)))
            cdfs.extend(cdf_rows(v,grid,metric=metric,group=state))
    write(pd.DataFrame(boxes),'model/outputs/population_boxplots.csv',['processed_data/samples_10000_model.pkl'],'Grouped simulated outcome quantiles; no simulated-agent records.')
    write(pd.DataFrame(summaries),'model/outputs/population_risks.csv',['processed_data/samples_10000_model.pkl'],'Age-group risks and normal binomial intervals.')
    write(pd.DataFrame(groups),'model/outputs/population_status.csv',['processed_data/samples_10000_model.pkl'],'Persistence and depth summaries by deep-IC status.')
    write(pd.DataFrame(cdfs),'model/outputs/population_cdfs.csv',['processed_data/samples_10000_model.pkl'],'Simulated distribution CDFs on fixed grids.')
    grid=pd.read_pickle(sim/'intervention_beta.pkl').groupby(['sigma','alpha','beta']).indicator.agg(['size','sum','mean']).reset_index()
    grid.columns=['sigma','alpha','beta','n','cases','risk']
    write(grid,'model/outputs/intervention_cells.csv',['processed_data/intervention_beta.pkl'],'Group-level intervention grid results.')
    common=sorted(pd.read_pickle(sim/'count_explore_exploit.pkl'),key=lambda r:-r[2])[:5]
    write(pd.DataFrame(common,columns=['exploration','exploitation','count']), 'model/inputs/common_profiles.csv',
          ['processed_data/count_explore_exploit.pkl'],'Five most common empirical behavioural profiles.')
    rows=[];cdfs=[]
    for parameter,column in [('beta','reduced_beta'),('alpha','reduced_alpha'),('sigma','increased_sigma')]:
        g=pd.read_pickle(sim/f'varying_{parameter}.pkl')
        for dose,part in g.groupby(column):
            delta=float(dose if parameter=='beta' else dose/10)
            delta=round(delta,10)
            p=part.indicator.mean();se=np.sqrt(p*(1-p)/len(part))
            rows.append(dict(parameter=parameter,delta=delta,n=len(part),cases=int(part.indicator.sum()),
                             risk=p,lo=p-1.95996398454*se,hi=p+1.95996398454*se))
            if any(np.isclose(delta,v) for v in [0,.3,1]):
                cdfs.extend(cdf_rows(part.entropy,np.linspace(0,2,401),parameter=parameter,delta=delta))
    write(pd.DataFrame(rows),'model/outputs/intervention_curves.csv',['processed_data/varying_alpha.pkl','processed_data/varying_beta.pkl','processed_data/varying_sigma.pkl'],'Normalized doses; beta=7-delta, alpha=alpha0-1.228delta, sigma=sigma0+0.003458delta.')
    write(pd.DataFrame(cdfs),'model/outputs/intervention_cdfs.csv',['processed_data/varying_alpha.pkl','processed_data/varying_beta.pkl','processed_data/varying_sigma.pkl'],'Stationary entropy CDFs at three doses; no per-agent outputs.')


def copy_statistics(source):
    selected=['age_in_years','behaviour_age_scores','behaviour_correlations','behaviour_deciles',
              'behaviour_descriptive','calibration_cell_support','fig_s1_hourly_digitized',
              'fig_s2_contrasts','fig_s2_groups','fig_s2_quintiles','fig_s3','fig_s5',
              'fig_s7_parameter_scan','intervention_dose_definition','intervention_full_grid',
              'joint_mediation','joint_mediation_risks','note1_cohort_profile',
              'note1_content_age_means','note1_topic_profile','population_outcomes',
              'regression_all_specifications','regression_full','regression_paths',
              'simulation_population_summary','table_s1','table_s2']
    for name in selected:
        p=source/'SI/results'/f'{name}.csv'
        frame=pd.read_csv(p)
        assert not any('user_id' in str(c).lower() or str(c).lower()=='userid' for c in frame.columns)
        write(frame,f'statistical_results/{name}.csv',[f'SI/results/{name}.csv'], 'Saved aggregate result from the final supplement.')
    for name in ['model_initialization.csv','model_topic_correlations.csv','topic_labels.csv']:
        frame=pd.read_csv(source/'SI/source_data'/name)
        description='Aggregated empirical input to the model.'
        if name=='topic_labels.csv':
            frame=frame.drop(columns=['topic_name_zh'],errors='ignore')
            description='Topic-level metadata with English labels only; the original-language label column is excluded.'
        write(frame,'model/inputs/'+name,['SI/source_data/'+name],description)
    p=source/'server_backup/population_simulation/Data'
    out=ROOT/'data/model/inputs'
    for name in ['phi_video_v2.npy','item_popularity_video_v2.npy','user_alpha_mean_video_v3.npy']:
        a=np.load(p/name,allow_pickle=False)
        assert a.size<=35*35
        np.save(out/name,a,allow_pickle=False)
        PROVENANCE.append(dict(file='data/model/inputs/'+name,sources=['server_backup/population_simulation/Data/'+name],description='Numeric aggregate model initialization; no identifiers.',shape=list(a.shape)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root',type=Path,required=True)
    args=parser.parse_args();source=args.private_root.resolve()
    df=pd.read_csv(source/'cleaned_data_v2.csv',dtype={'user_id':str})
    final=pd.read_csv(source/'cleaned_data_1216.csv',usecols=['user_id','m_ratio'],dtype={'user_id':str})
    assert len(df)==653486 and df.user_id.is_unique
    check=df[['user_id','m_ratio']].merge(final,on='user_id',validate='one_to_one')
    assert len(check)==len(df) and np.allclose(check.m_ratio_x,check.m_ratio_y)
    df['deep']=(df.m_ratio<1+norm.ppf(.1)*df.m_ratio.std(ddof=0)).astype(int)
    assert int(df.deep.sum())==244082
    copy_statistics(source)
    sigma=public_empirical(source,df)
    model_outputs(source)
    longitudinal(source,df,sigma)
    (ROOT/'docs').mkdir(exist_ok=True)
    (ROOT/'docs/data_provenance.json').write_text(json.dumps(PROVENANCE,indent=2,ensure_ascii=False)+'\n')
    print('Exported aggregate files:',len(PROVENANCE))


if __name__=='__main__':main()
