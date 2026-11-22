"""Pool signed GLM betas in each child's top 25% absolute-t voxels per ROI."""
import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from paths import ROOT, MODEL, ATLAS, INPUTS, OUTPUT, REPOSITORY, cohort, read_tsv, write_tsv
FEATURES = ("surprisal", "entropy", "semantic_update")
STAT_COLUMNS = tuple(f"t_{feature}" for feature in FEATURES)
TOP_PCT = np.array([25])
OUT = ROOT / "roi"
COMMON_MASK = ROOT / "common_mask.nii.gz"


@dataclass(frozen=True)
class Atlas:
    name: str
    labels: np.ndarray
    ids: np.ndarray
    names: np.ndarray
    nominal_count: np.ndarray
    digest: str


def load_grid(path, reference):
    image = nib.load(path)
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5):
        raise ValueError(f"Grid mismatch: {path}")
    return image


def read_mask(path, reference):
    values = np.asarray(load_grid(path, reference).dataobj)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"Expected binary mask: {path}")
    return values > 0


def load_atlases(reference):
    directory = INPUTS / "atlas"
    if not (directory / "labels.nii.gz").is_file():
        directory = OUTPUT / "atlas"
    if not (directory / "labels.nii.gz").is_file():
        directory = REPOSITORY / "assets/atlas"
    path = directory / "labels.nii.gz"
    image = nib.load(path)
    values = np.asarray(image.dataobj)
    canonical = nib.as_closest_canonical(image)
    exact = canonical.shape == reference.shape and np.allclose(canonical.affine, reference.affine, rtol=0, atol=1e-5)
    target = canonical if exact else resample_from_to(image, reference, order=0)
    rows = sorted(read_tsv(directory / "lut.tsv"), key=lambda r: int(r["roi_id"]))
    ids = np.array([int(r["roi_id"]) for r in rows], dtype=np.int32)
    labels = np.asarray(target.dataobj)
    if (len(ids) != 232 or len(set(ids)) != 232 or not np.array_equal(np.unique(values), np.r_[0, ids])
            or not np.array_equal(np.unique(labels), np.r_[0, ids])):
        raise ValueError("Atlas must contain the 200 Schaefer and 32 Tian labels.")
    labels = labels.astype(np.int32).ravel(order="F")
    return [Atlas(ATLAS, labels, ids, np.array([r["roi"] for r in rows]),
                  np.bincount(labels)[ids], sha256(path.read_bytes()).hexdigest())]

def prepare_atlases(reference, mask):
    layouts, coverage = [], []
    for atlas in load_atlases(reference):
        covered = atlas.labels[mask]
        n_common = np.bincount(covered, minlength=int(atlas.ids.max()) + 1)[atlas.ids]
        eligible = (5 * n_common >= 4 * atlas.nominal_count) & (n_common > 0)
        if not eligible.any():
            raise ValueError(f"No ROI meets common coverage in {atlas.name}.")
        n_selected = (TOP_PCT[:, None] * n_common[None, :] + 99) // 100
        fractions = n_common / atlas.nominal_count
        for j, roi_id in enumerate(atlas.ids):
            coverage.append({
                "atlas": atlas.name, "roi_id": roi_id, "roi": atlas.names[j],
                "n_atlas_voxels": atlas.nominal_count[j], "n_roi_voxels": n_common[j],
                "coverage_fraction": fractions[j], "eligible": int(eligible[j]),
                **{f"n_top{pct}": count if eligible[j] else ""
                   for pct, count in zip(TOP_PCT, n_selected[:, j], strict=True)},
            })
        indexes = [np.flatnonzero(covered == roi_id) for roi_id in atlas.ids[eligible]]
        layouts.append((atlas.name, indexes, {
            "roi_ids": atlas.ids[eligible], "roi_names": atlas.names[eligible],
            "n_atlas_voxels": atlas.nominal_count[eligible], "n_roi_voxels": n_common[eligible],
            "coverage_fraction": fractions[eligible], "n_selected": n_selected[:, eligible],
            "atlas_sha256": atlas.digest,
        }))
        print(f"{atlas.name}: {eligible.sum()} / {len(eligible)} ROIs have at least 80% coverage.", flush=True)
    return layouts, coverage

def read_paths(manifest, subjects, original):
    rows = read_tsv(manifest)
    lookup = {row["subject"]: row for row in rows}
    if len(rows) != len(lookup) or set(lookup) not in (set(subjects), set(original)):
        raise ValueError(f"Cohort mismatch in {manifest}.")
    return {s: {name: manifest.parent / lookup[s][name] for name in ("mask", *FEATURES, *STAT_COLUMNS)}
            for s in lookup}

def model_betas(manifest, reference, common, subjects, original):
    paths = read_paths(manifest, subjects, original)
    if any(not read_mask(paths[s]["mask"], reference).ravel(order="F")[common].all() for s in subjects):
        raise ValueError(f"A subject mask omits original common voxels in {manifest}.")

    def feature_values(columns):
        return np.stack([
            [np.asarray(load_grid(paths[s][column], reference).dataobj).ravel(order="F")[common]
             for column in columns]
            for s in subjects
        ]).astype(np.float64, copy=False)

    tested, statistic = feature_values(FEATURES), feature_values(STAT_COLUMNS)
    if not np.isfinite(tested).all() or not np.isfinite(statistic).all():
        raise ValueError(f"Nonfinite coefficients or t values in the common mask in {manifest}.")
    numeric = (np.abs(tested) > 1e-9) & (np.abs(statistic) > 1e-9)
    if np.any(np.sign(tested[numeric]) != np.sign(statistic[numeric])):
        raise ValueError(f"GLM beta and t signs disagree in {manifest}.")
    return tested, statistic

def aggregate(tested, statistic, indexes, counts):
    n_subjects, n_features, _ = tested.shape
    result = np.empty((n_subjects, n_features, len(TOP_PCT), len(indexes)))
    offsets = np.r_[0, np.cumsum(counts[-1])]
    selected = np.empty((n_subjects, n_features, offsets[-1]), dtype=np.int32)
    for j, voxels in enumerate(indexes):
        # The mask index breaks exact ties deterministically.
        rank = np.argsort(-np.abs(statistic[:, :, voxels]), axis=2, kind="stable")[:, :, :counts[-1, j]]
        selected[:, :, offsets[j]:offsets[j + 1]] = voxels[rank]
        top = np.take_along_axis(tested[:, :, voxels], rank, axis=2)
        for i, count in enumerate(counts[:, j]):
            result[:, :, i, j] = top[:, :, :count].mean(axis=2)
    return result, offsets, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    original, subjects = cohort()
    reference = nib.load(COMMON_MASK)
    common = read_mask(COMMON_MASK, reference).ravel(order="F")
    if common.sum() != 224756:
        raise ValueError("Unexpected published GLM common-mask size.")
    layouts, coverage = prepare_atlases(reference, common)
    if len(layouts[0][1]) != 214:
        raise ValueError("Expected 214 ROIs satisfying at least 80% coverage.")
    tested, statistic = model_betas(ROOT / MODEL / "first_level.tsv", reference, common, subjects, original)
    atlas, indexes, metadata = layouts[0]
    beta, offsets, selected = aggregate(tested, statistic, indexes, metadata["n_selected"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"roi_betas_{MODEL}_{ATLAS}.npz", beta=beta, model=MODEL, atlas=atlas,
                        subjects=subjects, features=FEATURES, top_pct=TOP_PCT,
                        selection_offsets=offsets, selected_voxel_indices=selected, **metadata)
    write_tsv(OUT / "roi_coverage.tsv", coverage)
    (OUT / "aggregation.json").write_text(json.dumps({
        "model": MODEL, "atlas": ATLAS, "top_pct": [25], "features": list(FEATURES),
        "subjects": subjects, "common_mask_subjects": original, "n_common_voxels": int(common.sum()),
        "criterion": "At least 80% of atlas label voxels in the intersection of 23 effective masks.",
        "ranking": "Each child's own absolute first-level t; stable order breaks exact ties.",
        "pooling": "Arithmetic mean of signed coefficients in selected voxels.",
        "selected_index": "Zero-based voxel index within the Fortran-order flattened common mask; selection_offsets identifies ROI segments.",
    }, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
