"""Observed cohort demographics and native-scale assessment distributions.

Each dot is a child. Boxes show medians and interquartile ranges; whiskers
span the observed range. Off-axis displacement only separates tied points.
All 23 story children contribute to demographic panels. All 16 assessment-
complete children contribute all 11 scores. No distributional inference.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment

from figure_common import ROOT, SD, OUT, QA

def export(fig,name,**alignment):
    destination=OUT/'figS1'
    destination.parent.mkdir(parents=True,exist_ok=True)
    audit=QA/name.replace('/','_')
    audit.parent.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(fig,json_out=str(audit)+'.alignment.json',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,**alignment)
    fig.savefig(str(destination)+'.pdf',dpi=450)
    fig.savefig(str(destination)+'.svg',dpi=450)
    fig.savefig(str(destination)+'.png',dpi=600)
    plt.close(fig)

BLUE,GREY='#0072B2','#92989E'
LABELS={'fsiq':'Full-scale IQ','literacy':'Literacy','chinese_reading':'Character reading',
        'chinese_dictation':'Character dictation','cantonese_grammar_ss':'Cantonese grammar',
        'discourse_comprehension_ss':'Discourse comprehension','word_definition_ss':'Word definition',
        'semantic_relations_ss':'Semantic relations','story_retelling_ss':'Story retelling',
        'noun_expression_ss':'Noun expression','hkgcnt_ss':'HKGCNT'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6.5,
    'ytick.labelsize':6.5,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none',
    'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.5,'ytick.major.size':2.5})


def dots(values):
    """Deterministic vertical offsets preserve the x-coordinate of every score."""
    values=np.asarray(values)
    result=np.zeros(len(values))
    for value in np.unique(values):
        at=np.flatnonzero(values==value)
        result[at]=np.linspace(-.22,.22,len(at)) if len(at)>1 else 0
    return result


def box_points(ax,values,y,color=BLUE):
    values=np.asarray(values,float)
    ax.boxplot(values,positions=[y],orientation='horizontal',widths=.40,whis=(0,100),
        showfliers=False,manage_ticks=False,patch_artist=True,
        boxprops=dict(facecolor='white',edgecolor='#505860',linewidth=.65),
        medianprops=dict(color='#202020',linewidth=1.1),
        whiskerprops=dict(color='#777777',linewidth=.65),capprops=dict(color='#777777',linewidth=.65))
    ax.scatter(values,y+dots(values),s=13,c=color,alpha=.8,edgecolors='white',linewidths=.25,zorder=3)


def style(ax,label,title):
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(title,loc='left',pad=10)
    ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-18,11),
                textcoords='offset points',fontweight='bold',fontsize=8,annotation_clip=False)


def main():
    demo=pd.read_csv(SD/'cohort_demographics.tsv',sep='\t').set_index('subject')
    long=pd.read_csv(SD/'cohort_assessment_scores.tsv',sep='\t')
    assert len(demo)==23 and (demo.cohort=='Assessment').sum()==16
    scores={key:group.score.to_numpy(float) for key,group in long.groupby('assessment',sort=False)}
    assert len(scores)==11 and all(len(v)==16 for v in scores.values())

    # The upper row describes the imaging sample; the large lower panel shows
    # all nine scale-compatible scores, with IQ and HKGCNT on their native axes.
    fig=plt.figure(figsize=(183/25.4,120/25.4))
    a=fig.add_axes([.10,.73,.24,.19]);b=fig.add_axes([.415,.73,.24,.19]);c=fig.add_axes([.73,.73,.24,.19])
    d=fig.add_axes([.24,.12,.40,.43]);e=fig.add_axes([.77,.40,.20,.15]);f=fig.add_axes([.77,.12,.20,.15])
    for yi,(group,color) in enumerate([('Assessment',BLUE),('Training',GREY)]):
        box_points(a,demo.loc[demo.cohort==group,'age'],1-yi,color)
    a.set(yticks=[0,1],yticklabels=['Training','Assessment'],ylim=(-.6,1.6),xlim=(7.5,13.5),
          xticks=[8,10,12],xlabel='Age (years)');style(a,'a','Age at scanning')
    for ax,field,levels,title,letter in [(b,'school_grade',list(range(2,7)),'School grade','b'),
                                          (c,'sex',['F','M'],'Sex','c')]:
        offset=np.arange(len(levels));bottom=np.zeros(len(levels))
        for group,color in [('Assessment',BLUE),('Training',GREY)]:
            counts=demo[demo.cohort==group][field].value_counts().reindex(levels,fill_value=0).to_numpy()
            ax.bar(offset,counts,bottom=bottom,width=.65,color=color,edgecolor='white',linewidth=.4)
            bottom+=counts
        ax.set_xticks(offset,[str(x) for x in levels] if field=='school_grade' else ['Female','Male'])
        ax.set_ylabel('Children (n)');ax.set_ylim(0,14 if field=='sex' else 10)
        ax.set_yticks([0,4,8,12] if field=='sex' else [0,2,4,6,8,10]);style(ax,letter,title)
    keys=[k for k in LABELS if k not in ('fsiq','hkgcnt_ss')]
    for i,key in enumerate(keys):box_points(d,scores[key],len(keys)-i-1)
    d.set(yticks=np.arange(len(keys)),yticklabels=[LABELS[k] for k in keys[::-1]],
          ylim=(-.7,len(keys)-.3),xlim=(0,20),xticks=[0,5,10,15,20],xlabel='Recorded standardized score')
    d.tick_params(axis='y',length=0,pad=5);style(d,'d','Literacy and language scores (n = 16)')
    for ax,key,lim,ticks,letter in [(e,'fsiq',(80,135),[80,100,120],'e'),
                                   (f,'hkgcnt_ss',(-6,3),[-6,-3,0,3],'f')]:
        box_points(ax,scores[key],0)
        ax.set(xlim=lim,xticks=ticks,ylim=(-.6,.6),yticks=[],xlabel='Recorded standardized score')
        ax.spines['left'].set_visible(False);style(ax,letter,LABELS[key]+' (n = 16)')
    handles=[Line2D([0],[0],marker='o',linestyle='',markerfacecolor=BLUE,markeredgecolor='none',
                    label='Assessment cohort (n = 16)'),
             Line2D([0],[0],marker='o',linestyle='',markerfacecolor=GREY,markeredgecolor='none',
                    label='Additional training children (n = 7)')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.53,.635),ncol=2,frameon=False,
               fontsize=7,handletextpad=.4,columnspacing=2)
    export(fig,'figure0_cohort',axes=[a,b,c,d,e,f],panel_ids=list('abcdef'),
           row_groups=[['a','b','c']],column_groups=[['e','f']],
           exemptions=[{'panels':['d'],'checks':['panel-width'],
                        'reason':'Nine language measures occupy the main lower panel; IQ and HKGCNT use separate native scales.'}])

    rows=[]
    for group in ('All','Assessment','Training'):
        data=demo if group=='All' else demo[demo.cohort==group]
        rows.append(dict(cohort=group,n=len(data),female=int((data.sex=='F').sum()),male=int((data.sex=='M').sum()),
                         age_mean=data.age.mean(),age_sd=data.age.std(),age_min=data.age.min(),age_max=data.age.max(),
                         grade_min=int(data.school_grade.min()),grade_max=int(data.school_grade.max())))
    pd.DataFrame(rows).to_csv(SD/'cohort_descriptives.tsv',sep='\t',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
