"""Portable paths, fixed manuscript styles, and surface labels."""
from pathlib import Path
import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.environ.get('PAPER_OUTPUT', REPO/'outputs')).expanduser().resolve()
INPUTS = Path(os.environ.get('PAPER_INPUTS', REPO/'data/inputs')).expanduser().resolve()
ROOT = OUTPUT
SD = OUTPUT/'figure_data'
OUT = OUTPUT/'figures'
QA = OUTPUT/'figure_qa'
ASSETS = REPO/'assets/surface'
ATLAS = OUTPUT/'atlas/labels.nii.gz'
if not ATLAS.exists(): ATLAS = REPO/'assets/atlas/labels.nii.gz'
for folder in (SD, OUT, QA): folder.mkdir(parents=True, exist_ok=True)
CMAPS = [plt.get_cmap('viridis'), LinearSegmentedColormap.from_list('gain', ['#0072B2','#F4F4F4','#D55E00'])]
NORMS = [Normalize(-.05,.4), TwoSlopeNorm(vmin=-.01,vcenter=0,vmax=.10)]
CMAP = LinearSegmentedColormap.from_list('signed_group_t',['#0072B2','#FAFAFA','#D55E00'])
NORM = Normalize(-9,9)

def setup():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.linewidth':.5,'xtick.labelsize':6,'ytick.labelsize':6,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def labels_for_surface(renderer, atlas):
    from surface_maps import cortical_ids
    if atlas != 'schaefer200_tian_s2': raise ValueError(atlas)
    mask = nib.load(OUTPUT/'story_text_glm/common_mask.nii.gz')
    atlas_img = nib.load(ATLAS)
    if atlas_img.shape != mask.shape or not np.allclose(atlas_img.affine,mask.affine):
        atlas_img = resample_from_to(atlas_img,mask,order=0)
    labels = np.asanyarray(atlas_img.dataobj).astype(np.int16)
    covered = np.where(np.asanyarray(mask.dataobj)>0,labels,0).astype(np.int16)
    face = renderer.project_labels(nib.Nifti1Image(covered,mask.affine),cortical_ids(atlas,np.unique(labels)))
    return face,int(labels.max())+1

def star(q):
    return '***' if q<=.001 else '**' if q<=.01 else '*'
