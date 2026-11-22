"""Single-configuration encoding figures from saved results.

Gemma 3 1B / Schaefer 200 + Tian scale 2 / top25 / PC128.
No model comparison, inference or voxel selection is rerun.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr, linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize
from figure_common import ROOT, SD, OUT, QA, ATLAS, CMAPS, NORMS, labels_for_surface
from surface_anchors import anchors_for_atlas
from surface_maps import SurfaceRenderer, cortical_ids
from audit_panel_alignment import require_matplotlib_panel_alignment
from association_style import joint_association

BLUE='#0072B2'
W=183.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.linewidth':.5,'xtick.labelsize':6,
    'ytick.labelsize':6,'pdf.fonttype':42,'svg.fonttype':'none'})


def read(path):
    return pd.read_csv(path,sep='\t')


def panel_tools(fig,h):
    def axes(x,y,w,height):return fig.add_axes([x/W,1-(y+height)/h,w/W,height/h])
    def text(x,y,s,**kw):return fig.text(x/W,1-y/h,s,va='center',**kw)
    return axes,text


def export(fig,name,plot_axes,ids,rows,columns):
    require_matplotlib_panel_alignment(fig,axes=plot_axes,panel_ids=ids,
        row_groups=rows,column_groups=columns,tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        strict=True,json_out=QA/f'selected_{name}.alignment.json')
    name={'encoding':'fig2','encoding_behaviour':'fig4'}[name]
    fig.savefig(OUT/f'{name}.pdf',dpi=450)
    fig.savefig(OUT/f'{name}.svg',dpi=450)
    fig.savefig(OUT/f'{name}.png',dpi=450)
    plt.close(fig)


def performance(renderer,cfg):
    regional=read(SD/'encoding_roi_performance_all.tsv')
    indiv=read(SD/'encoding_atlas_performance_subjects.tsv')
    assert len(regional)==232 and len(indiv)==16
    path=ATLAS
    labels=renderer.project_labels(path,cortical_ids(cfg['atlas'],regional.roi_id.unique()))
    nlabels=int(nib.load(path).get_fdata().max())+1
    h=132.
    fig=plt.figure(figsize=(W/25.4,h/25.4));ax,txt=panel_tools(fig,h)
    txt(W/2,3,'Gemma 3 1B | PC128 | Schaefer–Tian | Top 25%',ha='center',fontsize=7.5,fontweight='bold')
    plots=[]
    for k,(x,letter,title,col,individual) in enumerate([
        (7,'a','Prediction correspondence','r_mean','r'),
        (100,'b','Incremental prediction','partial_r2_mean','partial_r2')]):
        txt(x-5,12,letter,fontsize=8,fontweight='bold')
        txt(x+37,12,title,ha='center',fontweight='bold')
        values=np.full(nlabels,np.nan);values[regional.roi_id.to_numpy(int)]=regional[col]
        im=renderer.render(labels,values,CMAPS[k],NORMS[k],layout='grid')
        width=76;hh=width*im.shape[0]/im.shape[1]
        a=ax(x,17,width,hh);a.imshow(im,interpolation='antialiased');a.axis('off');plots.append(a)
        cax=ax(x+7,72,62,2)
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORMS[k],cmap=CMAPS[k]),cax=cax,orientation='horizontal')
        cb.set_ticks([-.05,0,.1,.2,.3,.4] if k==0 else [-.01,0,.025,.05,.075,.1])
        cb.ax.tick_params(labelsize=6,length=2,pad=2)
        txt(x-5,86,'cd'[k],fontsize=8,fontweight='bold')
        vals=indiv[individual].to_numpy()
        txt(x+37,86,f'Mean = {vals.mean():.4f}; SD = {vals.std(ddof=1):.4f}',ha='center',fontsize=6.5)
        a=ax(x+7,92,62,22)
        # Deterministic offsets separate the 16 children without adding a
        # sampling dimension or changing the measured x-coordinate.
        a.scatter(vals,np.linspace(-.16,.16,16),s=20,color=BLUE,edgecolors='white',linewidth=.4,zorder=3)
        a.errorbar(vals.mean(),.48,xerr=vals.std(ddof=1),fmt='o',color='#222222',ms=3.5,capsize=3,lw=.8)
        a.set_ylim(-.35,.75);a.set_yticks([]);a.set_xlabel('Pearson r' if k==0 else 'Partial R²',labelpad=3)
        a.spines[['top','right','left']].set_visible(False)
        a.tick_params(length=2,pad=2)
        plots.append(a)
    txt(W/2,128,'Blue dots: children (n = 16). Black point and bar: mean ± SD. Descriptive performance.',ha='center',fontsize=6.4)
    # Maps and child-summary plots form two separately comparable rows.
    export(fig,'encoding',plots,['a','c','b','d'],[['a','b'],['c','d']],[])
    regional.to_csv(SD/'figure_encoding_surface_values.tsv',sep='\t',index=False)
    indiv.to_csv(SD/'figure_encoding_subject_values.tsv',sep='\t',index=False)


def associations(renderer,cfg):
    sig=read(SD/'encoding_associations_significant.tsv')
    assert len(sig)==2 and (sig.q_bh<=.05).all()
    points=read(SD/'figure_encoding_behaviour_participants.tsv')
    names={'chinese_dictation':'Left frontal operculum / insula',
           'literacy':'Left visual-network parcel\n(Schaefer L Vis 13)'}
    score_names={'chinese_dictation':'Character dictation','literacy':'Literacy'}
    face,nlabels=labels_for_surface(renderer,cfg['atlas'])
    anchors=anchors_for_atlas(renderer,face,sig.roi_id.tolist())
    h=145.
    fig=plt.figure(figsize=(W/25.4,h/25.4));ax,txt=panel_tools(fig,h)
    plot_axes=[];plot_ids=[];records=[];annotations=[];plot_provenance=[]
    for i,r in enumerate(sig.itertuples()):
        y=3+70*i
        txt(1,y+2,'ac'[i],fontweight='bold',fontsize=8)
        txt(44,y+6,names[r.assessment],ha='center',fontsize=7,linespacing=1.1)
        vals=np.full(nlabels,np.nan);vals[int(r.roi_id)]=1
        im=renderer.render(face,vals,ListedColormap([BLUE]),Normalize(0,1),layout='grid')
        width=69.;hh=width*im.shape[0]/im.shape[1]
        a=ax(9.5,y+13,width,hh);a.imshow(im);a.axis('off');plot_axes.append(a);plot_ids.append('ac'[i])
        anchor=anchors[int(r.roi_id)];assert anchor['pixel_roi']==r.roi_id
        target=((9.5+anchor['x']*width)/W,1-(y+13+anchor['y']*hh)/h)
        fig.add_artist(Line2D([44/W,target[0]],[1-(y+10.5)/h,target[1]],
                              transform=fig.transFigure,lw=.45,color='#606060'))
        selected=points[(points.metric==r.metric)&(points.assessment==r.assessment)&(points.roi_id==r.roi_id)]
        subjects=selected.subject.tolist()
        values=selected.brain_metric.to_numpy(float)
        scores=selected.score.to_numpy(float)
        assert np.isclose(spearmanr(scores,values).statistic,r.effect,rtol=1e-10)
        txt(94,y+2,'bd'[i],fontweight='bold',fontsize=8)
        txt(137.5,y+2,f'Spearman ρ = {r.effect:.3f}  |  n = 16',ha='center',fontsize=6.5)
        txt(137.5,y+7,f'p = {r.p_raw:.3g}  |  BH q = {r.q_bh:.4f}',ha='center',fontsize=6.3)
        joint=joint_association(fig,(105,y+20,65,39),scores,values,
            score_names[r.assessment]+' score',
            'Prediction r' if r.metric=='pearson_r' else 'Partial R²',figure_height_mm=h)
        plot_axes.extend(joint['axes'])
        plot_ids.extend(['bd'[i], 'bd'[i]+'_top', 'bd'[i]+'_right'])
        plot_provenance.append(dict(panel='bd'[i],roi_id=int(r.roi_id),
            effect=float(r.effect),p_raw=float(r.p_raw),q_bh=float(r.q_bh),**joint['provenance']))
        for subject,x,v in zip(subjects,scores,values):records.append(dict(panel='bd'[i],subject=subject,score=x,brain_metric=v,metric=r.metric,assessment=r.assessment,roi_id=r.roi_id,roi=r.roi))
        annotations.append(dict(panel='ac'[i],roi_id=r.roi_id,roi=r.roi,label=names[r.assessment],anchor_pixel_roi=anchor['pixel_roi']))
    export(fig,'encoding_behaviour',plot_axes,plot_ids,[],
        [['a','c'],['b','d'],['b_top','d_top'],['b_right','d_right']])
    pd.DataFrame(records).to_csv(SD/'figure_encoding_behaviour_participants.tsv',sep='\t',index=False)
    pd.DataFrame(annotations).to_csv(SD/'figure_encoding_behaviour_annotations.tsv',sep='\t',index=False)
    (QA/'selected_encoding_behaviour.contract.json').write_text(json.dumps(dict(
        template_adaptation='Style-only inheritance: scatter, descriptive fitted line with mean CI, marginal densities',
        scientific_claim='Regional encoding measures are associated with dictation and literacy scores',
        all_original_statistics_unchanged=True,n=16,panels=plot_provenance,
        inference='Original exact Spearman permutation p and within-metric/assessment ROI-wise BH q',
        cortical_maps='Original standard inflated surfaces with directly named and verified ROI leaders'),indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SD/'configuration.json').read_text())
    assert cfg['model']=='gemma3_1b' and cfg['atlas']=='schaefer200_tian_s2' and cfg['top_pct']==25 and cfg['variant']=='pc128'
    renderer=SurfaceRenderer(width=420)
    performance(renderer,cfg)
    associations(renderer,cfg)
    (QA/'selected_encoding.contract.json').write_text(json.dumps(dict(
        configuration=cfg,n=16,performance='Descriptive means and SD across the same 16 held-out children; regional top25 values selected by held-out r.',
        association_inference='Original exact two-sided permutation p and within-configuration ROI BH q, unchanged.',
        child_scatter_lines='Descriptive OLS fits and pointwise 95% Student-t mean intervals; inference remains the original Spearman permutation test. Scott Gaussian KDE marginals and rugs show observed distributions.',
        cortical_maps='Standard inflated fsaverage; direct ROI-name labels and verified leader target pixels for associations.',
        group_panels={'encoding':'Two metric maps plus child-level atlas summaries','encoding_behaviour':'Both significant encoding-score associations in the selected configuration'},
        original_statistics_unchanged=True),indent=2)+'\n')


if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
