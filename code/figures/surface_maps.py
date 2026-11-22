"""Fixed-view fsaverage surface rendering with neuromaps registration fusion.

The saved registration-fusion coordinates map MNI152 parcel labels onto
fsaverage (10,242 vertices/hemisphere). All statistical calculations remain
in the original volume; this module only changes their display.
"""
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from PIL import Image, ImageDraw
import matplotlib as mpl

from figure_common import ASSETS


class SurfaceRenderer:
    def __init__(self, width=420):
        self.geometry = {}
        self.views = {}
        for hemi in ('L', 'R'):
            stem = ASSETS / f'tpl-fsaverage_den-10k_hemi-{hemi}'
            coords, faces = nib.load(str(stem) + '_inflated.surf.gii').agg_data()
            sulc = nib.load(str(stem) + '_desc-sulc_midthickness.shape.gii').agg_data()
            cortex = nib.load(str(stem) + '_desc-nomedialwall_dparc.label.gii').agg_data() > 0
            ras = np.loadtxt(ASSETS / f'tpl-MNI152_space-fsaverage_den-10k_hemi-{hemi}_regfusion.txt')
            self.geometry[hemi] = dict(coords=coords, faces=faces, sulc=sulc,
                                       cortex=cortex, ras=ras)
            for view in ('lateral', 'medial'):
                self.views[hemi, view] = self._view(hemi, view, width)

    def _view(self, hemi, view, width):
        g = self.geometry[hemi]
        xyz, faces = g['coords'], g['faces']
        sign = -1 if (hemi, view) in (('L', 'lateral'), ('R', 'medial')) else 1
        xy = np.column_stack([sign * xyz[:, 1], -xyz[:, 2]])
        xy -= xy.min(axis=0)
        scale = (width - 12) / xy[:, 0].max()
        xy = xy * scale + 6
        height = int(np.ceil(xy[:, 1].max())) + 6
        depth = sign * xyz[:, 0]
        image = Image.new('I', (width, height), 0)
        draw = ImageDraw.Draw(image)
        for i in np.argsort(depth[faces].mean(axis=1)):
            draw.polygon([tuple(p) for p in xy[faces[i]]], fill=int(i + 1))
        pix = np.asarray(image).copy()
        # Smooth illumination from an upper, camera-facing light; sulci retain
        # their anatomical texture on the inflated standard surface.
        tri = xyz[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        light = np.array([sign, 0.0, 0.55]); light /= np.linalg.norm(light)
        shade = .78 + .22 * np.maximum(normal @ light, 0)
        depthshade = np.clip(g['sulc'][faces].mean(axis=1), -2, 2)
        grey = (.83 - .075 * depthshade) * shade
        background = np.column_stack([grey, grey, grey])
        return dict(pixel_faces=pix, background=background, shade=shade)

    def project_labels(self, atlas, allowed_ids=None):
        atlas = nib.load(str(atlas)) if isinstance(atlas, (str, Path)) else atlas
        volume = np.asanyarray(atlas.dataobj)
        out = {}
        for hemi, g in self.geometry.items():
            vox = nib.affines.apply_affine(np.linalg.inv(atlas.affine), g['ras'])
            labels = map_coordinates(volume, vox.T, order=0, mode='constant', cval=0).astype(int)
            labels[~g['cortex']] = 0
            if allowed_ids is not None:
                labels[~np.isin(labels, allowed_ids)] = 0
            # Use the majority label at each face (rather than averaging parcel
            # numbers); ties use the first vertex. Labels stay discrete.
            a = labels[g['faces']]
            face = np.where(a[:, 1] == a[:, 2], a[:, 1], a[:, 0])
            out[hemi] = face
        return out

    def render(self, face_labels, values, cmap, norm, layout='row'):
        cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        colors = cmap(norm(values))[:, :3]
        active = np.isfinite(values)
        active[0] = False
        images = {}
        for (hemi, view), info in self.views.items():
            labels = face_labels[hemi]
            rgb = info['background'].copy()
            on = active[labels]
            rgb[on] = colors[labels[on]] * (.88 + .12 * info['shade'][on, None])
            lut = np.vstack([np.ones((1, 3)), rgb])
            images[hemi, view] = (np.clip(lut[info['pixel_faces']], 0, 1) * 255).astype(np.uint8)
        if layout == 'row':
            order = [('L', 'lateral'), ('L', 'medial'), ('R', 'medial'), ('R', 'lateral')]
            rows = [order]
        else:
            rows = [[('L', 'lateral'), ('R', 'lateral')], [('L', 'medial'), ('R', 'medial')]]
        height = max(a.shape[0] for a in images.values())
        width = max(a.shape[1] for a in images.values())
        canvas = np.full((height * len(rows), width * len(rows[0]), 3), 255, np.uint8)
        for ri, row in enumerate(rows):
            for ci, key in enumerate(row):
                im = images[key]; off = (height - im.shape[0]) // 2
                canvas[ri * height + off:ri * height + off + im.shape[0], ci * width:ci * width + im.shape[1]] = im
        return canvas


def cortical_ids(atlas, ids):
    """Combined labels: Tian 1--32; Schaefer cortex 33--232."""
    if atlas != 'schaefer200_tian_s2':
        raise ValueError('This release uses Schaefer 200 + Tian S2 only.')
    return np.asarray(ids)[np.asarray(ids) > 32]
