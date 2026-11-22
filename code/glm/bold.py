"""4-mm normalized smoothing and percent signal change, preserving all GLM frames."""
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
TR = 0.75
FIRST_FRAME = 20
FULL_FRAMES = 336
FWHM = 4.0

def image_on_grid(values, reference, dtype):
    image = nib.Nifti1Image(values, reference.affine, reference.header.copy())
    image.set_data_dtype(dtype)
    image.header.set_slope_inter(1, 0)
    return image

def prepare_bold(record, work, good):
    source = nib.load(record["bold"])
    source_mask = nib.load(record["mask"])
    if (
        source.shape != (*source_mask.shape, FULL_FRAMES)
        or not np.allclose(source.affine, source_mask.affine, atol=1e-5)
        or not np.allclose(source.header.get_zooms(), [2, 2, 2, TR], atol=1e-5)
    ):
        raise ValueError("Unexpected BOLD or mask grid.")

    values = np.array(source.dataobj[..., FIRST_FRAME:], dtype=np.float32, copy=True, order="F")
    mask = np.asarray(source_mask.dataobj) > 0
    # Derive an effective mask before normalized spatial smoothing.
    mean_signal = values.mean(axis=3, dtype=np.float64, where=good)
    mask &= np.isfinite(values).all(axis=3) & (mean_signal > 0)
    sigma = FWHM / (np.sqrt(8 * np.log(2)) * np.asarray(source.header.get_zooms()[:3]))
    weights = gaussian_filter(mask.astype(np.float32), sigma, mode="constant")[mask]
    for frame in range(values.shape[3]):
        volume = np.where(mask, values[..., frame], 0)
        volume = gaussian_filter(volume, sigma, mode="constant")
        values[..., frame] = 0
        values[..., frame][mask] = volume[mask] / weights

    in_mask = values[mask]
    baseline = in_mask[:, good].mean(axis=1, dtype=np.float64).astype(np.float32)
    if not np.isfinite(baseline).all() or (baseline <= 0).any():
        raise ValueError("Invalid percent signal change baseline.")
    in_mask /= baseline[:, None]
    in_mask -= 1
    in_mask *= 100
    values[mask] = in_mask
    nib.save(image_on_grid(values, source, np.float32), work / "bold.nii")
    nib.save(image_on_grid(mask.astype(np.uint8), source_mask, np.uint8), work / "mask.nii")
    return source, mask

def check_matrix(path, names, matrix):
    label_line = next(line for line in path.read_text().splitlines() if "ColumnLabels" in line)
    labels = [label.strip().removesuffix("#0") for label in label_line.split('"')[1].split(";")]
    if labels != names:
        raise ValueError(f"Unexpected AFNI matrix column order: {labels}")
    generated = np.loadtxt(path)
    np.testing.assert_allclose(generated, matrix, atol=1e-7, rtol=1e-5)
    if np.linalg.matrix_rank(generated) != matrix.shape[1]:
        raise ValueError("AFNI matrix is rank deficient.")
