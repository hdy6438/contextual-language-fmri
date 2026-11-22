"""Single-configuration GLM figures from unchanged stored estimates.

Gemma 3 1B / Schaefer 200 + Tian scale 2 / within-ROI top 25%.
Only existing within-configuration BH discoveries are shown. The configuration
was selected after comparison of prior results; this script adds no inference.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize

from figure_common import ROOT, SD, OUT, QA, setup, labels_for_surface, CMAP, NORM, star
from surface_anchors import anchors_for_atlas
from surface_maps import SurfaceRenderer, cortical_ids
from audit_panel_alignment import require_matplotlib_panel_alignment
from association_style import joint_association

ATLAS = 'schaefer200_tian_s2'
W = 183
BLUE = '#0072B2'

NAMES = {
  115:'Left default-mode network\nPFC 1',
  180:'Right salience/ventral-attention network\nTempOccPar 1',
  192:'Right limbic network\nOFC 2',
  201:'Right frontoparietal control network\nPFCv 1',
  206:'Right frontoparietal control network\nPFCl 5',
  212:'Right frontoparietal control network\nPFCmp 1',
  222:'Right default-mode network\nPFCv 1',
  105:'Left frontoparietal control network\nCing 2',
  108:'Left default-mode network\nTemp 3',
  179:'Right dorsal-attention network\nPrCv 1',
  184:'Right salience/ventral-attention network\nFrOperIns 1',
  204:'Right frontoparietal control network\nPFCl 3',
  211:'Right frontoparietal control network\nCing 2',
  224:'Right default-mode network\nPFCdPFCm 2',
  95:'Left frontoparietal control network\nPar 3',
  156:'Right somatomotor network\nParcel 9',
  171:'Right dorsal-attention network\nPost 5',
}
DEEP = {2:'Right posterior hippocampus',4:'Right medial amygdala',5:'Right dorsal-posterior thalamus',
        18:'Left posterior hippocampus',19:'Left lateral amygdala',20:'Left medial amygdala',
        26:'Left nucleus accumbens core',31:'Left anterior caudate'}


def read(path):
    return pd.read_csv(path,sep='\t')


def ax_mm(fig,x,y,w,h,height):
    return fig.add_axes([x/W,1-(y+h)/height,w/W,h/height])


def label(fig,x,y,s,height,**kwargs):
    return fig.text(x/W,1-y/height,s,va='center',**kwargs)


def save(fig,stem,axes,ids,**kwargs):
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,
        json_out=str(QA/f'selected_{stem}.alignment.json'),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,**kwargs)
    stem={'text_features':'fig3','semantic_behaviour':'fig5'}[stem]
    fig.savefig(OUT/f'{stem}.pdf',dpi=450)
    fig.savefig(OUT/f'{stem}.svg',dpi=450)
    fig.savefig(OUT/f'{stem}.png',dpi=450)
    plt.close(fig)


def group_figure(renderer,face,nlabels,rows):
    height=178
    fig=plt.figure(figsize=(183/25.4,height/25.4))
    label(fig,2,3,'Joint text-feature responses during story listening',height,fontsize=8,fontweight='bold')
    label(fig,2,8,'Gemma 3 1B | Schaefer–Tian | top 25% within each ROI | n = 16',height,fontsize=7)
    cortical=rows[rows.roi_id.isin(cortical_ids(ATLAS,rows.roi_id))]
    anchors=anchors_for_atlas(renderer,face,cortical.roi_id.unique())
    annotations=[];maps=[]
    for i,feature in enumerate(['surprisal','entropy']):
        top=16+i*77
        a=rows[rows.text_feature==feature]
        cortex=a[a.roi_id>32];deep=a[a.roi_id<=32].sort_values('roi_id')
        label(fig,2,top,'ab'[i],height,fontsize=8,fontweight='bold')
        label(fig,8,top,feature.capitalize()+f' | {len(a)} regional responses',height,fontsize=7.5,fontweight='bold')
        lut=np.full(nlabels,np.nan);lut[cortex.roi_id.to_numpy(int)]=cortex.t.to_numpy(float)
        image=renderer.render(face,lut,CMAP,NORM,layout='grid')
        x,y,w=52,top+7,77;h=w*image.shape[0]/image.shape[1]
        ax=ax_mm(fig,x,y,w,h,height);ax.imshow(image);ax.axis('off');maps.append(ax)
        for side in ['Left','Right']:
            items=[r for r in cortex.itertuples() if NAMES[r.roi_id].startswith(side)]
            items.sort(key=lambda r:anchors[r.roi_id]['y'])
            ys=(np.linspace(top+12,top+57,len(items)) if side=='Right'
                else np.linspace(top+13,top+25,len(items)))
            for row,text_y in zip(items,ys):
                rid=int(row.roi_id);anchor=anchors[rid];left=side=='Left'
                tx=48 if left else 132;line_x=tx+.7 if left else tx-.7
                target_x=x+anchor['x']*w;target_y=y+anchor['y']*h
                fig.add_artist(Line2D([line_x/W,target_x/W],[1-text_y/height,1-target_y/height],
                               transform=fig.transFigure,color='#606060',lw=.35))
                display=NAMES[rid]+' '+star(row.q_bh)
                label(fig,tx,text_y,display,height,fontsize=5.9,linespacing=1.12,ha='right' if left else 'left')
                assert anchor['pixel_roi']==rid
                annotations.append(dict(text_feature=feature,roi_id=rid,roi=row.roi,display_name=NAMES[rid],q_bh=row.q_bh,anchor_pixel_roi=anchor['pixel_roi']))
        label(fig,2,top+34,'Subcortical regions',height,fontsize=6,fontweight='bold')
        axm=ax_mm(fig,44,top+38,7,3.2*len(deep),height)
        axm.imshow(deep.t.to_numpy()[:,None],cmap=CMAP,norm=NORM,aspect='auto')
        axm.set_xticks([]);axm.set_yticks(range(len(deep)),[DEEP[rid] for rid in deep.roi_id],fontsize=5.7)
        axm.tick_params(length=0,pad=2)
        for s in axm.spines.values():s.set_visible(False)
        for j,row in enumerate(deep.itertuples()):axm.text(0,j,star(row.q_bh),ha='center',va='center',color='white',fontsize=6.5)
    cax=ax_mm(fig,52,165,77,2,height)
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=cax,orientation='horizontal')
    bar.set_ticks([-9,-6,-3,0,3,6,9]);bar.ax.tick_params(labelsize=6,length=2,pad=1)
    label(fig,131,166,'Group t statistic',height,fontsize=6.5)
    label(fig,2,175,'Maps: left / right hemispheres; lateral / medial views.  * q ≤ 0.05   ** q ≤ 0.01   *** q ≤ 0.001',height,fontsize=6)
    save(fig,'text_features',maps,['a','b'],column_groups=[['a','b']],row_groups=[])
    pd.DataFrame(annotations).to_csv(SD/'text_feature_surface_annotations.tsv',sep='\t',index=False)
    rows.to_csv(SD/'text_feature_group_display.tsv',sep='\t',index=False)
    (QA/'selected_text_features.contract.json').write_text(json.dumps(dict(
        model='gemma3_1b',atlas=ATLAS,top_pct=25,children=16,displayed_tests=30,
        cortical_tests=18,subcortical_tests=12,signed_profile='14 negative surprisal; 16 positive entropy',
        group_inference='Existing two-sided exact sign flips and within-feature ROI BH q',
        selection='Outcome-informed specification selected after prior comparisons; no cross-configuration selection adjustment',
        cortical_annotation='Full functional-network names with canonical Schaefer parcel suffixes; leaders verified inside source atlas parcel',
        no_new_inference=True),indent=2)+'\n')


def semantic_figure(renderer,face,nlabels,rows):
    height=218
    fig=plt.figure(figsize=(183/25.4,height/25.4));axes=[];ids=[];points=[];styles=[]
    participant_values=read(SD/'semantic_behaviour_participants.tsv')
    rows=rows.sort_values('roi_id')
    anchors=anchors_for_atlas(renderer,face,rows.roi_id.unique())
    for i,row in enumerate(rows.itertuples()):
        top=3+i*72;rid=int(row.roi_id)
        name=NAMES[rid]
        label(fig,2,top,'ace'[i],height,fontsize=8,fontweight='bold')
        label(fig,37,top+4,name,height,fontsize=6.5,ha='center',linespacing=1.15)
        lut=np.full(nlabels,np.nan);lut[rid]=1
        image=renderer.render(face,lut,ListedColormap([BLUE]),Normalize(0,1),layout='grid')
        x,y,w=7,top+17,65;h=w*image.shape[0]/image.shape[1]
        ax=ax_mm(fig,x,y,w,h,height);ax.imshow(image);ax.axis('off');axes.append(ax);ids.append('ace'[i])
        anchor=anchors[rid]
        fig.add_artist(Line2D([37/W,(x+w*anchor['x'])/W],[1-(top+9)/height,1-(y+h*anchor['y'])/height],
                             transform=fig.transFigure,color='#606060',lw=.4))
        assert anchor['pixel_roi']==rid
        selected=participant_values[participant_values.roi_id==rid]
        subjects=selected.subject.tolist()
        score=selected.score.to_numpy(float)
        beta=selected.beta.to_numpy(float)
        estimate=linregress((score-score.mean())/score.std(ddof=1),beta).slope
        assert np.isclose(estimate,row.estimate,rtol=1e-10)
        joint=joint_association(fig,rect_mm=(104,top+22,66,39),xdata=score,ydata=beta,
            xlabel='Cantonese grammar score',ylabel='Semantic-update β (PSC units)',
            figure_height_mm=height,figure_width_mm=W)
        for suffix,axis in zip(['','_top','_right'],joint['axes']):
            axes.append(axis);ids.append('bdf'[i]+suffix)
        styles.append(dict(panel='bdf'[i],roi_id=rid,**joint['provenance']))
        label(fig,85,top,'bdf'[i],height,fontsize=8,fontweight='bold')
        label(fig,137,top+1,
              f'b = {row.estimate:.4f} (95% CI {row.ci_95_low:.4f}, {row.ci_95_high:.4f})',
              height,fontsize=6.5,ha='center')
        label(fig,137,top+6,
              f'p = {row.p_permutation:.4f}  |  BH q = {row.q_bh:.4f}  |  n = 16',
              height,fontsize=6.5,ha='center')
        for subject,s,b in zip(subjects,score,beta):
            points.append(dict(subject=subject,roi_id=rid,roi=row.roi,score=s,beta=b,assessment='cantonese_grammar_ss',model='gemma3_1b',top_pct=25))
    # Brain and scatter panels are distinct evidence types; compare their
    # own columns, not their intentionally different aspect ratios.
    save(fig,'semantic_behaviour',axes,ids,
         column_groups=[['a','c','e'],['b','d','f'],['b_top','d_top','f_top'],
                        ['b_right','d_right','f_right']],row_groups=[])
    pd.DataFrame(points).to_csv(SD/'semantic_behaviour_participants.tsv',sep='\t',index=False)
    rows.to_csv(SD/'semantic_behaviour_display.tsv',sep='\t',index=False)
    (QA/'selected_semantic_behaviour.contract.json').write_text(json.dumps(dict(
        model='gemma3_1b',atlas=ATLAS,top_pct=25,children=16,displayed_tests=3,
        contrast='Cantonese grammar vs signed semantic-update regional coefficients',
        slope_units='PSC coefficient per sample SD of the grammar score',
        confidence_intervals='Original analytic two-sided 95% Student t intervals, df=14',
        probabilities='Existing two-sided score permutation p and within-contrast ROI BH q',
        selection='Outcome-informed specification selected after prior comparisons; no cross-configuration selection adjustment',
        scatter_lines='Descriptive OLS fit, one unjittered point per child',
        shaded_bands='Pointwise 95% Student t confidence bands for the fitted OLS conditional mean; descriptive, distinct from original permutation p and BH q',
        marginal_distributions='Scott-bandwidth Gaussian KDE, displayed only within observed min/max, with one rug mark per child',
        panel_styles=styles,
        map_color='Categorical negative association direction; not effect magnitude',
        no_new_inference=True),indent=2)+'\n')


def main():
    setup();OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'pdf.fonttype':42,'svg.fonttype':'none','font.size':6.5,'axes.linewidth':.5})
    data=read(SD/'text_glm_significant.tsv')
    assert len(data)==33 and set(data.model)=={'gemma3_1b'} and set(data.atlas)=={ATLAS} and set(data.top_pct)=={25}
    renderer=SurfaceRenderer(width=500);face,nlabels=labels_for_surface(renderer,ATLAS)
    group_figure(renderer,face,nlabels,data[data.contrast=='group_mean'])
    semantic_figure(renderer,face,nlabels,data[data.contrast=='cantonese_grammar_ss'])


if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
