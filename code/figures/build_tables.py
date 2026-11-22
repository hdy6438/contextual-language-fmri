"""Build manuscript tables from the same final-specification result tables."""
import numpy as np
import pandas as pd
from figure_common import SD, OUTPUT

TABLES=OUTPUT/'tables'

def tex(s): return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
def ptext(p):
    if p>=.001:return f'{p:.4f}'
    a,b=f'{p:.2e}'.split('e');return '$'+a+r'\times10^{'+str(int(b))+'}$'
def table(name,caption,columns,rows):
    spec='l'+'r'*(len(columns)-1)
    lines=[r'\begin{table}[htbp]\centering\small',r'\caption{'+caption+'}',
           r'\begin{tabular}{'+spec+r'}\toprule',' & '.join(columns)+r' \\\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (TABLES/(name+'.tex')).write_text('\n'.join(lines)+'\n')

def main():
    TABLES.mkdir(parents=True,exist_ok=True)
    g=pd.read_csv(SD/'text_glm_all_tests.tsv',sep='\t')
    e=pd.read_csv(SD/'encoding_all_assessment_tests.tsv',sep='\t')
    counts=[]
    for feature in ['surprisal','entropy','semantic_update']:
        for group,label,nfamilies in [(True,'Group response',1),(False,'Assessment association',11)]:
            a=g[(g.text_feature==feature)&((g.contrast=='group_mean')==group)]
            counts.append([tex(feature.replace('_',' ').title()+': '+label),nfamilies,214,len(a),int((a.q_bh<=.05).sum())])
    for metric,label in [('incremental_r2',r'Encoding: partial $R^2$'),('pearson_r',r'Encoding: Pearson $r$')]:
        a=e[e.metric==metric];counts.append([label,11,232,len(a),int((a.q_bh<=.05).sum())])
    assert sum(r[3] for r in counts)==12808
    assert sum(r[4] for r in counts)==35
    table('test_counts','Regional test families and BH-FDR results.',
          ['Analysis','Families','ROIs','Tests',r'$q\leq0.05$'],counts+[['Total',58,'',12808,35]])
    sig=g[(g.contrast=='group_mean')&(g.q_bh<=.05)]
    rows=[]
    for r in sig.sort_values(['text_feature','roi_id']).itertuples():
        rows.append([tex(r.text_feature+': '+r.roi),f'{r.estimate:.4f}',f'[{r.ci_95_low:.4f}, {r.ci_95_high:.4f}]',f'{r.t:.2f}',ptext(r.p_permutation),f'{r.q_bh:.4f}'])
    table('group_responses','All 30 significant regional text-feature group responses.',
          ['Feature and ROI',r'Mean $\beta$',r'95\% CI','$t$','$p$','$q$'],rows)
    rows=[]
    for r in e[e.q_bh<=.05].sort_values(['metric','roi_id']).itertuples():
        rows.append([tex(r.roi),tex(r.metric),tex(r.assessment),f'{r.effect:.4f}',ptext(r.p_raw),f'{r.q_bh:.4f}'])
    for r in g[(g.contrast!='group_mean')&(g.q_bh<=.05)].sort_values('roi_id').itertuples():
        rows.append([tex(r.roi),r'$\beta_{\mathrm{SU}}$',tex(r.contrast),f'{r.estimate:.4f}',ptext(r.p_permutation),f'{r.q_bh:.4f}'])
    table('associations','All five significant assessment associations.',
          ['ROI','Brain measure','Assessment','Effect','$p$','$q$'],rows)
    demo=pd.read_csv(SD/'cohort_demographics.tsv',sep='\t')
    a=demo[demo.cohort=='Assessment']
    score=pd.read_csv(SD/'cohort_assessment_scores.tsv',sep='\t')
    rows=[]
    for key,label in [('age','Age (years)'),('school_grade','School grade')]:
        x=a[key].to_numpy(float);rows.append([label,f'{x.mean():.2f} ({x.std(ddof=1):.2f})',f'{x.min():g}--{x.max():g}'])
    rows.insert(1,['Sex (female/male)',f'{(a.sex=="F").sum()}/{(a.sex=="M").sum()}','---'])
    for name,grp in score.groupby('assessment',sort=False):
        x=grp.score.to_numpy(float);rows.append([tex(name),f'{x.mean():.2f} ({x.std(ddof=1):.2f})',f'{x.min():g}--{x.max():g}'])
    table('cohort','Characteristics of the 16-child assessment cohort.',
          ['Characteristic','Mean (SD) or count','Range'],rows)
    print(f'Tables: {TABLES}')

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
