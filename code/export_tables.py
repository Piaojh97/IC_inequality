"""Export manuscript table values from the included LaTeX table sources.

No model fitting or individual-level data are involved. The table labels are
frozen in data/table_labels.json; table text retains LaTeX mathematical notation.
"""
from pathlib import Path
import re
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/'reference/table_sources'


def escaped(s,i):
    n=0;i-=1
    while i>=0 and s[i]=='\\':n+=1;i-=1
    return n%2==1


def balanced(s,start):
    assert s[start]=='{'
    depth=1;i=start+1
    while depth:
        if s[i]=='{' and not escaped(s,i):depth+=1
        if s[i]=='}' and not escaped(s,i):depth-=1
        i+=1
    return s[start+1:i-1],i


def macros(s):
    out={}
    for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}\s*\{',s):
        out[m.group(1)]=balanced(s,m.end()-1)[0]
    return out


def rows(s):
    s=re.sub(r'(?<!\\)%[^\n]*','',s)
    s=re.sub(r'\\(?:toprule|midrule|bottomrule)\b','',s).strip()
    data=[];row=[];cell=[];depth=0;i=0
    while i<len(s):
        c=s[i]
        if c=='{' and not escaped(s,i):depth+=1
        if c=='}' and not escaped(s,i):depth-=1
        if depth==0 and s[i:i+2]=='\\\\':
            row.append(''.join(cell).strip());cell=[]
            if any(row):data.append(row)
            row=[];i+=2;continue
        if depth==0 and c=='&' and not escaped(s,i):
            row.append(''.join(cell).strip());cell=[]
        else:cell.append(c)
        i+=1
    if ''.join(cell).strip():row.append(''.join(cell).strip())
    if row:data.append(row)
    return data


def clean(cell):
    cell=cell.replace('\\%','%').replace('\\&','&').replace('\\textemdash','—')
    cell=re.sub(r'\\shortstack\{(.*)\}',r'\1',cell,flags=re.S)
    cell=cell.replace('\\\\',' ').replace('\n',' ')
    return re.sub(r'\s+',' ',cell).strip()


def main():
    definitions={}
    for p in SOURCES.glob('*.tex'):definitions.update(macros(p.read_text()))
    labels=json.loads((ROOT/'data/table_labels.json').read_text())
    out=ROOT/'data/tables';out.mkdir(parents=True,exist_ok=True)
    mapping={
      'tab:age':('AgeRows',['Age','Users','Share (%)']),
      'tab:baseline-age':('BaselineAgeRows',['Age','Users','First-hour entropy mean','First-hour entropy SD','Minimum relative entropy mean']),
      'tab:risk-specs':('FullRiskSpecs',['Predictor','Age only','Gender','Economic status','Interests','All covariates']),
      'tab:thresholds':('ThresholdRows',['Alpha','Entropy criterion','Deep-IC cases','Risk (%)','Age gap (pp)','Risk ratio']),
      'tab:depth-specs':('FullDepthSpecs',['Predictor','Age only','Gender','Economic status','Interests','All covariates']),
      'tab:threshold-age':('ThresholdAgeRows',['Alpha','19–23','24–30','31–40','41–50','51–60','61–79']),
      'tab:behaviour-summary':('BehaviourSummaryRows',['Indicator','Mean','SD','Median [Q25,Q75]','Zero count']),
      'tab:deciles':('BehaviourCutRows',['Percentile','Search','Interaction','Interested viewing','Top-decile viewing']),
      'tab:age-scores':('BehaviourAgeRows',['Age','Users','Exploration mean (SD)','Exploitation mean (SD)','Correlation']),
      'tab:paths':('PathRows',['Predictor','Exploration coefficient [95% CI]','Exploitation coefficient [95% CI]','Risk OR [95% CI]']),
      'tab:standardized-risks':('StandardizedRiskRows',['Risk setting','Outcome age code','Exploration age code','Exploitation age code','Risk (%)']),
      'tab:replacement-orders':('MediationOrderRows',['Order','Exploration (pp)','Exploitation (pp)','Joint (pp)']),
      'tab:med':('MediationRows',['Component','Risk difference (pp) [95% CI]','Share (%) [95% CI]']),
      'tab:simulation-age':('SimulationAgeRows',['Age','Simulated users','Empirical risk (%)','Simulated risk (%)']),
      'tab:population-groups':('PopulationGroupRows',['Group','Simulated users','Stationary relative entropy','Time in ICs (%)','Depth']),
      'tab:population-outcomes':('PopulationOutcomeRows',['Age','Simulated users','Risk (%)','Stationary relative entropy','Time in ICs (%)','Depth']),
      'tab:intervention':('DoseRows',['Intervention','Parameter rule','Risk at 30% (%)','Risk at 100% (%)']),
      'tab:full-interventions':('FullDoseRows',['Normalized adjustment','Weaker matching risk (%)','Lower exploitation risk (%)','Higher exploration risk (%)','Matching gap removed (%)','Exploitation gap removed (%)','Exploration gap removed (%)'])}
    manifest=[]
    def emit(key,body,headers,suffix='',source=''):
        data=[[clean(c) for c in row] for row in rows(body)]
        assert all(len(row)==len(headers) for row in data),(key,headers,data[:2])
        number=labels[key]['number'] if isinstance(labels[key],dict) else labels[key]
        filename=f'Table_{number}{suffix}.csv'
        pd.DataFrame(data,columns=headers).to_csv(out/filename,index=False)
        manifest.append(dict(label=key,number=number,file='data/tables/'+filename,source=source,rows=len(data)))
    for key,(macro,headers) in mapping.items():emit(key,definitions[macro],headers,source=macro)
    for suffix,macro,headers in [('a','RegressionRows',['Adjustment','Age OR [95% CI]','Age entropy coefficient [95% CI]']),
       ('b','FullAgeRows',['Predictor','Risk OR [95% CI]','Entropy coefficient [95% CI]']),
       ('c','DecadeRows',['Outcome','10-year contrast [95% CI]','Users in contrast'])]:
        emit('tab:reg',definitions[macro],headers,suffix,macro)
    text=(SOURCES/'static_tables.tex').read_text()
    for match in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',text,re.S):
        block=match.group();key=re.search(r'\\label\{([^}]+)\}',block).group(1)
        section=block.split('\\toprule',1)[1].split('\\bottomrule',1)[0]
        header,body=section.split('\\midrule',1)
        headers=[clean(c) for c in rows(header)[0]]
        emit(key,body,headers,source='static_tables.tex')
    assert len({r['number'] for r in manifest})==24
    (ROOT/'docs/table_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('Exported 24 supplementary tables in',len(manifest),'CSV files.')


if __name__=='__main__':main()
