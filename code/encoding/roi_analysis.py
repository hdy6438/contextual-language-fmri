"""Top-25% Schaefer--Tian encoding summaries and exact ROI-wise score associations."""
import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "glm"))
from paths import ASSESSMENTS, INPUTS, OUTPUT, REPOSITORY, cohort, read_tsv, write_tsv
from exact_spearman import permutation_pvalues

METRICS = ("pearson_r", "incremental_r2")


def subject_top_metrics(values, groups, counts):
    """Rank by held-out raw r, then average both signed metrics in identical voxels."""
    scores = np.empty((len(groups), values.shape[1]))
    for i, voxels in enumerate(groups):
        ordered = voxels[np.argsort(-values[voxels, 0], kind="stable")]
        accumulated = np.cumsum(values[ordered[:counts[i]]], axis=0, dtype=np.float64)
        scores[i] = accumulated[counts[i] - 1] / counts[i]
    return scores


def image_values(path, reference):
    image = nib.load(path)
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
        raise ValueError(f"Grid mismatch: {path}")
    return np.asarray(image.dataobj)


def aggregate(root, atlas, lut, output, subjects):
    reference = nib.load(root / "common_mask.nii.gz")
    mask = np.asarray(reference.dataobj) > 0
    grid = image_values(atlas, reference)
    if not np.isfinite(grid).all() or not np.array_equal(grid, np.rint(grid)):
        raise ValueError("Atlas labels must be finite integers.")
    lookup = {int(r["roi_id"]): r["roi"] for r in read_tsv(lut)}
    labels = grid[mask].astype(int)
    ids = np.unique(labels[labels > 0])
    if len(lookup) != 232 or not np.array_equal(ids, np.arange(1, 233)) or set(lookup) != set(ids):
        raise ValueError("Expected all 232 represented Schaefer--Tian ROIs.")
    groups = [np.flatnonzero(labels == i) for i in ids]
    n_voxels = np.array([len(g) for g in groups])
    n_selected = (25 * n_voxels + 99) // 100
    scores = []
    for subject in subjects:
        voxel_values = np.column_stack([image_values(root / subject / f"{name}.nii.gz", reference)[mask] for name in ("r", "r2")])
        if not np.isfinite(voxel_values[labels > 0]).all():
            raise ValueError(f"Nonfinite atlas voxel metrics: {subject}")
        scores.append(subject_top_metrics(voxel_values, groups, n_selected))
    scores = np.array(scores)
    output.mkdir(parents=True, exist_ok=True)
    names = [lookup[i] for i in ids]
    np.savez_compressed(output / "roi_metrics.npz", subjects=subjects, roi_ids=ids, roi_names=names,
                        n_roi_voxels=n_voxels, n_selected=n_selected,
                        pearson_r=scores[:, :, 0], incremental_r2=scores[:, :, 1])
    rows = []
    for s, subject in enumerate(subjects):
        for r, roi_id in enumerate(ids):
            rows.append(dict(subject=subject, roi_id=roi_id, roi=names[r], n_roi_voxels=n_voxels[r],
                             n_selected=n_selected[r], pearson_r=scores[s, r, 0], incremental_r2=scores[s, r, 1]))
    write_tsv(output / "roi_subject_metrics.tsv", rows)
    rows = []
    for m, metric in enumerate(METRICS):
        for r, roi_id in enumerate(ids):
            rows.append(dict(metric=metric, roi_id=roi_id, roi=names[r], mean=scores[:, r, m].mean(), sd=scores[:, r, m].std(ddof=1)))
    write_tsv(output / "roi_performance.tsv", rows)
    means = scores.mean(axis=1)
    write_tsv(output / "atlas_subject_metrics.tsv", [dict(subject=s, pearson_r=means[i, 0], incremental_r2=means[i, 1]) for i, s in enumerate(subjects)])
    return scores, ids, names, n_voxels, n_selected


def associations(scores, subjects, ids, names, n_voxels, n_selected, assessments_path, output):
    records = read_tsv(assessments_path)
    by_subject = {r["subject"]: r for r in records}
    if len(records) != len(by_subject) or not set(subjects) <= set(by_subject):
        raise ValueError("Missing or duplicated assessment participants.")
    assessments, display_names = zip(*ASSESSMENTS)
    behavioral = np.array([[float(by_subject[s][name]) for name in assessments] for s in subjects])
    if not np.isfinite(behavioral).all() or (behavioral.std(axis=0) == 0).any():
        raise ValueError("All 16 children require finite nonconstant scores on all 11 assessments.")
    rows = []
    for m, metric in enumerate(METRICS):
        rho, p = permutation_pvalues(scores[:, :, m], behavioral)
        # axis=0 comprises the 232 ROIs; each metric--assessment pair is one family.
        q = false_discovery_control(p, axis=0, method="bh")
        for a, assessment in enumerate(assessments):
            for r, roi_id in enumerate(ids):
                rows.append(dict(assessment=assessment, assessment_label=display_names[a], model="gemma3_1b",
                                 variant="pc128", top_pct=25, n_subjects=16, roi_id=roi_id, roi=names[r],
                                 n_roi_voxels=n_voxels[r], n_selected_voxels=n_selected[r],
                                 spearman_rho=rho[r, a], p_exact=p[r, a], q_bh_232_rois=q[r, a], metric=metric))
    write_tsv(output / "encoding_associations.tsv", rows)
    print(f"{len(rows)} exact tests; {sum(r['q_bh_232_rois'] <= .05 for r in rows)} BH-FDR discoveries.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", type=Path, default=OUTPUT / "encoding/pc128")
    parser.add_argument("--atlas", type=Path, default=REPOSITORY / "assets/atlas/labels.nii.gz")
    parser.add_argument("--lut", type=Path, default=REPOSITORY / "assets/atlas/lut.tsv")
    parser.add_argument("--assessments", type=Path, default=INPUTS / "assessments.tsv")
    parser.add_argument("--output", type=Path, default=OUTPUT / "encoding")
    args = parser.parse_args()
    _, subjects = cohort()
    scores, ids, names, n_voxels, n_selected = aggregate(args.encoding, args.atlas, args.lut, args.output, subjects)
    associations(scores, subjects, ids, names, n_voxels, n_selected, args.assessments, args.output)


if __name__ == "__main__":
    main()
