"""Export the final paper's figure/table inputs from freshly computed results.

This stage reads private participant/assessment tables locally. Its outputs may
contain individual data and belong in the ignored outputs/ directory.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from figure_common import INPUTS, OUTPUT, SD

ASSESSMENTS = [
    'fsiq','literacy','chinese_reading','chinese_dictation','cantonese_grammar_ss',
    'discourse_comprehension_ss','word_definition_ss','semantic_relations_ss',
    'story_retelling_ss','noun_expression_ss','hkgcnt_ss']
CFG = dict(model='gemma3_1b',variant='pc128',atlas='schaefer200_tian_s2',top_pct=25)

def read(path): return pd.read_csv(path,sep='\t',dtype={'subject':str})
def save(frame,name): frame.to_csv(SD/name,sep='\t',index=False)

def main():
    demo=read(INPUTS/'participants.tsv').set_index('subject')
    scores=read(INPUTS/'assessments.tsv').set_index('subject')
    assert demo.index.is_unique and scores.index.is_unique
    enc=OUTPUT/'encoding'
    glm=OUTPUT/'story_text_glm'
    tests=read(enc/'encoding_associations.tsv')
    tests=tests.rename(columns={'spearman_rho':'effect','p_exact':'p_raw','q_bh_232_rois':'q_bh'})
    assert len(tests)==2*11*232
    save(tests,'encoding_all_assessment_tests.tsv')
    sig=tests[tests.q_bh<=.05].copy().sort_values(['metric','roi_id'])
    assert len(sig)==2, 'The final paper plots two encoding associations; inspect changed results.'
    save(sig,'encoding_associations_significant.tsv')
    with np.load(enc/'roi_metrics.npz') as z:
        subjects=z['subjects'].tolist()
        assert len(subjects)==16
        score=scores.loc[subjects,ASSESSMENTS]
        regions=[]
        for ri,(rid,name) in enumerate(zip(z['roi_ids'],z['roi_names'])):
            r=z['pearson_r'][:,ri];r2=z['incremental_r2'][:,ri]
            regions.append(dict(roi_id=rid,roi=name,r_mean=r.mean(),r_sd=r.std(ddof=1),
                partial_r2_mean=r2.mean(),partial_r2_sd=r2.std(ddof=1)))
        save(pd.DataFrame(regions),'encoding_roi_performance_all.tsv')
        individual=pd.DataFrame(dict(subject=subjects,r=z['pearson_r'].mean(axis=1),partial_r2=z['incremental_r2'].mean(axis=1)))
        save(individual,'encoding_atlas_performance_subjects.tsv')
        points=[]
        for i,row in enumerate(sig.itertuples()):
            ri=z['roi_ids'].tolist().index(row.roi_id)
            values=z[row.metric][:,ri]
            x=score[row.assessment].to_numpy(float)
            np.testing.assert_allclose(spearmanr(x,values).statistic,row.effect,rtol=1e-10)
            points.extend(dict(subject=s,metric=row.metric,assessment=row.assessment,roi_id=row.roi_id,
                roi=row.roi,score=a,brain_metric=b) for s,a,b in zip(subjects,x,values))
        save(pd.DataFrame(points),'figure_encoding_behaviour_participants.tsv')
    g=read(glm/'second_level/roi_glm_gemma3_1b.tsv')
    assert len(g)==3*12*214
    save(g,'text_glm_all_tests.tsv')
    gs=g[g.q_bh<=.05].copy();assert len(gs)==33
    save(gs,'text_glm_significant.tsv')
    with np.load(glm/'roi/roi_betas_gemma3_1b_schaefer200_tian_s2.npz') as z:
        order=[z['subjects'].tolist().index(s) for s in subjects]
        fi=z['features'].tolist().index('semantic_update')
        pi=z['top_pct'].tolist().index(25)
        points=[]
        for row in gs[gs.contrast=='cantonese_grammar_ss'].sort_values('roi_id').itertuples():
            ri=z['roi_ids'].tolist().index(row.roi_id)
            b=z['beta'][:,fi,pi,ri][order]
            x=score['cantonese_grammar_ss'].to_numpy(float)
            np.testing.assert_allclose(linregress((x-x.mean())/x.std(ddof=1),b).slope,row.estimate,rtol=1e-10)
            points.extend(dict(subject=s,roi_id=row.roi_id,roi=row.roi,score=a,beta=v,
                assessment='cantonese_grammar_ss') for s,a,v in zip(subjects,x,b))
        save(pd.DataFrame(points),'semantic_behaviour_participants.tsv')
    assert len(demo)==23
    demo=demo.copy();demo['cohort']=['Assessment' if s in subjects else 'Training' for s in demo.index]
    assert demo[['age','sex','school_grade']].notna().all().all()
    save(demo.reset_index(),'cohort_demographics.tsv')
    save(score.rename_axis('subject').reset_index().melt(id_vars='subject',var_name='assessment',value_name='score'),'cohort_assessment_scores.tsv')
    (SD/'configuration.json').write_text(json.dumps(CFG,indent=2)+'\n')
    print(f'Figure source tables: {SD}')

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
