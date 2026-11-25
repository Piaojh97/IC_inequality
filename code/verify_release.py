"""Check the aggregate-only release and its numerical cross-references."""
from pathlib import Path
import ast
import hashlib
import io
import json
import re
import tokenize
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    checks=[]
    def check(name,condition):
        if not bool(condition):raise AssertionError(name)
        checks.append(name)
    read=lambda p:pd.read_csv(ROOT/'data'/p)
    forbidden={'user_id','userid','uid','device_id','video_id','item_id','account_id','session_id'}
    all_files=[p for p in ROOT.rglob('*') if p.is_file()
               and not any(x in p.relative_to(ROOT).parts for x in ['.venv','__pycache__','.git'])]
    files=[p for p in all_files if p.relative_to(ROOT).parts[0]!='outputs']
    check('No links to external data',not any(p.is_symlink() for p in ROOT.rglob('*')))
    check('No notebook checkpoints or backup directories',not any('.ipynb_checkpoints' in p.parts for p in all_files))
    check('No pickle, R workspace or raw archive files',not any(p.suffix.lower() in ['.pkl','.pickle','.rdata','.rhistory','.zip','.tar','.gz','.npz','.parquet','.feather','.sqlite','.db'] for p in all_files))
    text_extensions={'.py','.ipynb','.md','.txt','.csv','.json','.tex','.r','.yml','.yaml','.toml','.ini','.cfg','.sh'}
    for p in all_files:
        if p.suffix.lower() not in text_extensions and p.name!='.gitignore':continue
        text=p.read_text(encoding='utf-8')
        if p.suffix in {'.json','.ipynb'}:text=json.dumps(json.loads(text),ensure_ascii=False)
        check('No Chinese text: '+str(p.relative_to(ROOT)),not re.search(r'[\u3400-\u9fff]',text))
    schema=json.loads((ROOT/'docs/data_schema.json').read_text())
    check('All data files have a reviewed schema',
          {r['file'] for r in schema}=={str(p.relative_to(ROOT)) for p in (ROOT/'data').rglob('*') if p.is_file()})
    schema={r['file']:r for r in schema}
    for p in (ROOT/'data').rglob('*.csv'):
        d=pd.read_csv(p)
        name=str(p.relative_to(ROOT));review=schema[name]
        check('No individual identifier columns: '+name,not forbidden.intersection(str(c).lower() for c in d.columns))
        check('Reviewed columns and row count: '+name,list(d.columns)==review['columns'] and len(d)==review['rows'])
        check('Unique aggregate grouping keys: '+name,not d.duplicated(review['grouping_keys']).any())
    for p in (ROOT/'data').rglob('*.npy'):
        a=np.load(p,allow_pickle=False)
        check('Small numeric model input: '+p.name,a.dtype.kind in 'fiu' and a.size<=35*35 and np.isfinite(a).all())
    for p in ROOT.rglob('*.ipynb'):
        d=json.loads(p.read_text())
        check('Notebook has only standard top-level fields: '+p.name,set(d)=={'cells','metadata','nbformat','nbformat_minor'})
        check('Notebook metadata contains no saved state: '+p.name,d['metadata']=={'language_info':{'name':'python'}})
        check('Notebook outputs removed: '+p.name,all(not c.get('outputs') and not c.get('attachments') for c in d['cells']))
        check('Notebook execution state removed: '+p.name,all(c.get('execution_count') is None for c in d['cells']))
        check('Cell metadata and additional payloads removed: '+p.name,
              all(not c.get('metadata') and set(c).issubset({'cell_type','metadata','source','outputs','execution_count','id'}) for c in d['cells']))
        check('Notebook comments and explanatory text are English: '+p.name,
              all(not re.search(r'[\u3400-\u9fff]', ''.join(c.get('source',[]))) for c in d['cells']))
        for c in d['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),str(p),'exec')
    for p in ROOT.rglob('*.py'):
        text=p.read_text();tree=ast.parse(text)
        comments=[t.string for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type==tokenize.COMMENT]
        docstrings=[ast.get_docstring(n) or '' for n in ast.walk(tree) if isinstance(n,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
        check('Python comments and docstrings are English: '+str(p.relative_to(ROOT)),
              not any(re.search(r'[\u3400-\u9fff]',s) for s in comments+docstrings))
    for p in (ROOT/'reference/table_sources').glob('*.tex'):
        comments=[line.split('%',1)[1] for line in p.read_text().splitlines() if re.search(r'(?<!\\)%',line)]
        check('LaTeX comments are English: '+p.name,not any(re.search(r'[\u3400-\u9fff]',s) for s in comments))
    for p in (ROOT/'outputs').rglob('*.csv'):
        d=pd.read_csv(p);name=str(p.relative_to(ROOT))
        check('Generated output has no identifier columns: '+name,not forbidden.intersection(str(c).lower() for c in d.columns))
        if p.parent.name=='tables':
            check('Generated table is a copy of public aggregate values: '+name,p.read_bytes()==(ROOT/'data/tables'/p.name).read_bytes())
        elif p.parent.name=='simulation':
            check('Generated simulation rows summarize groups: '+name,{'n','cases','risk'}.issubset(d.columns) and d.n.ge(1).all() and np.allclose(d.risk,d.cases/d.n))
        else:check('Generated CSV has a documented role: '+name,False)
    t=read('empirical/thresholds.csv');main=t[np.isclose(t.alpha,.1)].iloc[0]
    check('Final cohort N=653486',main.n==653486)
    check('Deep IC cases=244082',main.cases==244082)
    risk=read('empirical/age_risks.csv');risk=risk[np.isclose(risk.alpha,.1)]
    check('Age-group counts sum to cohort',risk.n.sum()==653486 and risk.cases.sum()==244082)
    check('Youngest and oldest risks match manuscript',np.allclose(risk.risk.iloc[[0,-1]],[.3334,.4081],atol=.0001))
    p=read('empirical/persistence_by_status.csv').set_index('deep')
    check('Persistence comparison matches 76% versus 28%',np.allclose(p.loc[[1,0],'mean'],[.760165,.277069],atol=1e-6))
    check('Two-sided t statistic matches manuscript',np.isclose(p.t_statistic.iloc[0],609.854543,atol=1e-5))
    grid=read('statistical_results/calibration_cell_support.csv')
    r2=1-((grid.empirical_risk-grid.simulated_risk)**2).sum()/((grid.empirical_risk-grid.empirical_risk.mean())**2).sum()
    check('Calibration R2 matches stored model',np.isclose(r2,.8980142679576322))
    check('Calibration group counts sum to cohort',grid.n.sum()==653486)
    dist=read('model/inputs/age_behaviour_distribution.csv')
    check('Age-profile grid contains 600 aggregate cells',len(dist)==600 and dist['count'].sum()==653486)
    check('Each age-profile distribution is normalized',np.allclose(dist.groupby('age').proportion.sum(),1))
    phase=read('model/outputs/phase_grid.csv')
    check('Parameter grid contains 1000 saved cells and 900 documented scan cells',len(phase)==1000 and len(phase[phase.beta.between(1,9)])==900)
    joined=phase[phase.beta.eq(7)].merge(grid,on=['exploration','exploitation'],validate='one_to_one')
    check('Baseline phase slice matches calibration',np.allclose(joined.risk,joined.simulated_risk))
    med=read('statistical_results/joint_mediation.csv').set_index('component')
    check('Joint decomposition reproduces 69.3%',np.isclose(med.loc['Both behaviours','share'],.6930965964724006))
    check('Behavioural contributions add to the joint contribution',np.isclose(med.loc[['Exploration','Exploitation'],'risk_difference'].sum(),med.loc['Both behaviours','risk_difference']))
    curves=read('model/outputs/intervention_curves.csv')
    check('Three intervention curves contain eleven doses each',len(curves)==33 and (curves.groupby('parameter').size()==11).all())
    at30=curves[np.isclose(curves.delta,.3)].set_index('parameter').risk
    check('Saved 30% intervention risks preserved',np.allclose(at30.loc[['beta','alpha','sigma']],[.36466165413533835,.34273182957393483,.3634085213032581]))
    check('Intervention estimates are aggregate groups of 1596',curves.n.eq(1596).all())
    init=read('model/inputs/model_initialization.csv')
    check('35 topics and normalized empirical input distributions',len(init)==35 and np.allclose(init[['mean_first_hour_share','candidate_topic_probability']].sum(),1))
    table_manifest=json.loads((ROOT/'docs/table_manifest.json').read_text())
    check('All 24 supplementary table numbers mapped',len({r['number'] for r in table_manifest})==24)
    check('All table files exist',all((ROOT/r['file']).exists() for r in table_manifest))
    mapping=pd.read_csv(ROOT/'docs/reproduction_map.csv').fillna('')
    missing=[]
    for row in mapping.itertuples():
        for path in (row.data+';'+row.supporting_data).split(';'):
            if path and not (ROOT/path).exists():missing.append(path)
        for target in row.code.split(';'):
            filename,_,function=target.partition(':')
            check('Referenced code exists: '+target,(ROOT/filename).is_file())
            if function:
                tree=ast.parse((ROOT/filename).read_text())
                check('Referenced function exists: '+target,function in {n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))})
    check('All figure/table data references resolve',not missing)
    audit=json.loads((ROOT/'docs/release_audit.json').read_text())
    for function,inputs in audit['plotting_dependencies'].items():
        declared=set()
        for value in mapping.loc[mapping.code.eq('code/reproduce.py:'+function),'data']:
            declared.update(value.split(';'))
        check('Plotting dependencies match the audited inputs: '+function,declared==set(inputs))
    if (ROOT/'SHA256SUMS.txt').exists():
        listed=set()
        for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
            digest,name=line.split('  ',1)
            listed.add(name)
            check('File checksum: '+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest)
        check('Checksum manifest covers every release file',listed=={str(p.relative_to(ROOT)) for p in files if p.name!='SHA256SUMS.txt'})
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    report=dict(status='passed',checks_passed=len(checks),empirical_individual_rows_included=False,
                simulated_individual_rows_included=False,calibration_r_squared=float(r2),checks=checks)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Passed {len(checks)} checks. No person-level data files are included.')


if __name__=='__main__':main()
