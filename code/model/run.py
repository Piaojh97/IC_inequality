"""Run the manuscript dynamics from aggregate inputs; write aggregate outputs only.

The numerical interaction implementation is retained in legacy_simulator.py.
The adapter replaces disk-based per-agent tracing with window aggregation.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import entropy
import legacy_simulator as legacy

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'data/model/inputs'
AGES=['19–23','24–30','31–40','41–50','51–60','61–79']
CRITERION=0.7190319683690323


class AggregateSimulator(legacy.simulator):
    def init_users(self):
        intrinsic=[];observed=[]
        for _ in range(self.num_users):
            intrinsic.append(np.random.dirichlet(self.mu_users))
            observed.append(np.random.dirichlet(self.mu_users))
        self.user_groups=legacy.users_groups(self.alpha,np.asarray(intrinsic),np.asarray(observed),self.sim_type)
        self.baseline=entropy(self.user_groups.interest_int,axis=1)
        self.positive=np.zeros_like(self.user_groups.interest_int)
        self.noise=np.zeros_like(self.positive)
        self.window_entropy=[]

    def tracer(self,user_group_tracer):
        # Match draw_pic.py: round each exploration increment to five decimals,
        # sum over a window, clip the exploration and total consumption vectors.
        for i,row in enumerate(user_group_tracer):
            accepted=row['feedback'].reshape(-1).astype(bool)
            self.positive[i]+=row['item2rec'][accepted].sum(axis=0)
            self.noise[i]+=np.around(row['random_term'],5)
        if (self.ts+1)%self.other_para['window_steps']==0:
            consumption=np.clip(self.positive*self.dt+np.clip(self.noise,0,1),0,1)
            ratio=entropy(consumption,axis=1)/self.baseline
            if not np.isfinite(ratio).all():
                raise ValueError('A consumption window has undefined entropy; use the documented window length.')
            self.window_entropy.append(ratio)
            self.positive.fill(0);self.noise.fill(0)


def run_group(alpha,sigma,beta,args):
    np.random.seed(args.seed)
    model=AggregateSimulator(num_users=len(alpha),num_items=args.items,num_topics=35,
        num_iter=args.steps,num_rec=10,phi_path=str(INPUT/'phi_video_v2.npy'),
        item_path=str(INPUT/'item_popularity_video_v2.npy'),dt=.01,sim_type='dot',
        alpha=np.asarray(alpha,float),sigma=np.asarray(sigma,float),beta=float(beta),
        gamma_plus=1.,gamma_minus=-.1,user_method='real',
        mu_user_path=str(INPUT/'user_alpha_mean_video_v3.npy'),optimized=True,
        window_steps=100)
    for _ in model:pass
    windows=np.asarray(model.window_entropy)
    final=windows[-1];deep=final<CRITERION
    return dict(n=len(alpha),cases=int(deep.sum()),risk=float(deep.mean()),
                mean_stationary_relative_entropy=float(final.mean()),
                mean_persistence=float((windows<1).mean()),
                mean_depth=float((1-windows.mean(axis=0)).mean()))


def sample_profiles(age,n,seed):
    dist=pd.read_csv(INPUT/'age_behaviour_distribution.csv')
    part=dist[dist.age.eq(age)]
    rng=np.random.RandomState(seed)
    sampled=rng.choice(len(part),size=n,p=part.proportion.to_numpy()/part.proportion.sum())
    e=part.exploration.to_numpy()[sampled]
    x=part.exploitation.to_numpy()[sampled]
    return 5.6+.1228*x,.018+.0003458*e


def numbers(s,kind=float):return [kind(x) for x in s.split(',')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['demo','grid','population','interventions'],default='demo')
    p.add_argument('--users',type=int,default=None,help='Per grid cell, total population, or oldest-group size.')
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--items',type=int,default=10000)
    p.add_argument('--seed',type=int,default=4)
    p.add_argument('--betas',default='7')
    p.add_argument('--exploration-scores',default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--exploitation-scores',default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--doses',default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/simulation')
    args=p.parse_args()
    if args.steps<100 or args.steps%100 or args.items<10:
        p.error('Steps must be a positive multiple of 100 and the item pool must contain at least 10 items.')
    n=args.users or {'demo':20,'grid':1000,'population':10000,'interventions':1596}[args.mode]
    if n<2:p.error('Use at least two simulated users.')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.time()
    if args.mode=='demo':
        alpha,sigma=sample_profiles(AGES[-1],n,args.seed)
        rows.append(dict(age=AGES[-1],beta=7.,**run_group(alpha,sigma,7.,args)))
    elif args.mode=='grid':
        for beta in numbers(args.betas):
            if beta<1 or beta>9:p.error('The documented parameter scan covers beta=1,...,9.')
            for e in numbers(args.exploration_scores,int):
                for x in numbers(args.exploitation_scores,int):
                    if not (0<=e<=9 and 0<=x<=9):p.error('Scores must be 0,...,9.')
                    row=dict(beta=beta,exploration=e,exploitation=x,
                             **run_group(np.full(n,5.6+.1228*x),np.full(n,.018+.0003458*e),beta,args))
                    rows.append(row);print(json.dumps(row),flush=True)
    elif args.mode=='population':
        distribution=pd.read_csv(INPUT/'age_behaviour_distribution.csv').groupby('age',sort=False).age_n.first().reindex(AGES)
        counts=np.random.RandomState(args.seed).multinomial(n,distribution/distribution.sum())
        for age,size in zip(AGES,counts):
            if not size:continue
            alpha,sigma=sample_profiles(age,size,args.seed)
            row=dict(age=age,beta=7.,**run_group(alpha,sigma,7.,args))
            rows.append(row);print(json.dumps(row),flush=True)
    else:
        alpha,sigma=sample_profiles(AGES[-1],n,args.seed)
        for parameter in ['beta','alpha','sigma']:
            for dose in numbers(args.doses):
                if not 0<=dose<=1:p.error('Normalized doses must lie between 0 and 1.')
                a=alpha-1.228*dose if parameter=='alpha' else alpha
                s=sigma+.003458*dose if parameter=='sigma' else sigma
                b=7-dose if parameter=='beta' else 7
                row=dict(parameter=parameter,delta=dose,**run_group(a,s,b,args))
                rows.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(args.output/(args.mode+'.csv'),index=False)
    metadata={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.output.resolve().is_relative_to(ROOT):
        metadata['output']=str(args.output.resolve().relative_to(ROOT))
    metadata.update(elapsed_seconds=time.time()-start,criterion=CRITERION,window_steps=100,
                    realized_users=n,output_granularity='aggregate only',
                    run_type='new seeded realization; not a replay of historical agent assignments')
    (args.output/(args.mode+'_settings.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Wrote {len(rows)} aggregate rows in {time.time()-start:.1f} seconds.')


if __name__=='__main__':main()
