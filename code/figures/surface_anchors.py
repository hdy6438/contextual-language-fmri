import numpy as np
from scipy.ndimage import distance_transform_edt

def anchors_for_atlas(renderer, face, ids):
    """An interior pixel of each ROI, in the clearest of the four views."""
    views = [('L', 'lateral'), ('R', 'lateral'), ('L', 'medial'), ('R', 'medial')]
    height = max(info['pixel_faces'].shape[0] for info in renderer.views.values())
    width = next(iter(renderer.views.values()))['pixel_faces'].shape[1]
    pixels = {view: np.r_[0, face[view[0]]][renderer.views[view]['pixel_faces']] for view in views}
    anchors = {}
    for rid in ids:
        view = max(views, key=lambda v: (pixels[v] == rid).sum())
        mask = pixels[view] == rid
        assert mask.any(), rid
        y, x = np.unravel_index(np.argmax(distance_transform_edt(mask)), mask.shape)
        row, column = divmod(views.index(view), 2)
        anchors[rid] = dict(x=(x + column * width) / (2 * width),
                            y=(y + row * height + (height - mask.shape[0]) // 2) / (2 * height),
                            pixel_roi=int(pixels[view][y, x]))
    return anchors
