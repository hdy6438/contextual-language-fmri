"""Prepare 4-mm smoothed nuisance-adjusted BOLD on the preserved story timeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "glm"))
from design import speech_design, MOTION, TISSUE, SPEECH, read_tsv, values

FULL_FRAMES, FIRST_FRAME, N_FRAMES = 336, 12, 324
TR, FWHM = .75, 4.
NUISANCE = (*SPEECH, *MOTION, *TISSUE, "dct_01", "dct_02", "dct_03", "intercept")


def nuisance_basis(speech, confounds):
    rows = read_tsv(confounds)
    if len(rows) != FULL_FRAMES:
        raise ValueError("Expected 336 fMRIPrep confound rows.")
    rows = rows[FIRST_FRAME:]
    fd, dvars = values(rows, "framewise_displacement"), values(rows, "std_dvars")
    if not np.isfinite(fd).all() or not np.isfinite(dvars).all():
        raise ValueError("Retained frames require finite FD and standardized DVARS.")
    flags = (fd > .5) | (dvars > 1.5)
    for name in rows[0]:
        if name.startswith("non_steady_state_outlier"):
            flags |= values(rows, name) == 1
    drifts = np.cos(np.pi * (np.arange(N_FRAMES)[:, None] + .5) * np.arange(1, 4) / N_FRAMES)
    design = np.column_stack([speech[FIRST_FRAME:], *[values(rows, name) for name in (*MOTION, *TISSUE)], drifts, np.ones(N_FRAMES)])
    if design.shape != (N_FRAMES, len(NUISANCE)) or not np.isfinite(design).all():
        raise ValueError("Invalid nuisance design.")
    keep = ~flags
    x = design[keep].copy()
    norms = np.linalg.norm(x, axis=0)
    if np.any(norms <= 0):
        raise ValueError("Empty nuisance term.")
    x /= norms
    if np.linalg.matrix_rank(x) != len(NUISANCE):
        raise ValueError("Rank-deficient nuisance design.")
    basis = np.zeros_like(design)
    basis[keep] = np.linalg.qr(x, mode="reduced")[0]
    np.testing.assert_allclose(basis.T @ basis, np.eye(len(NUISANCE)), atol=1e-12)
    return keep, basis


def resolve_records(path, dataset):
    records = read_tsv(path)
    if len(records) != 23 or len({r["subject"] for r in records}) != 23:
        raise ValueError("Expected the 23 unique selected story runs.")
    for record in records:
        for key in ("bold", "mask", "confounds"):
            p = Path(record[key])
            record[key] = str(p if p.is_absolute() else dataset / p)
    return records


def effective_mask(record, keep, output):
    image = nib.load(record["bold"])
    source_mask = nib.load(record["mask"])
    check_grid(image, source_mask)
    if image.shape[-1] != FULL_FRAMES or not np.allclose(image.header.get_zooms(), [2, 2, 2, TR], atol=1e-5):
        raise ValueError("Expected 336 BOLD frames on a 2-mm grid with TR=.75 s.")
    raw = np.asarray(image.dataobj[..., FIRST_FRAME:], dtype=np.float32)
    mask = np.asarray(source_mask.dataobj) > 0
    mask &= np.isfinite(raw).all(axis=3) & (raw.mean(axis=3, dtype=np.float64, where=keep) > 0)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), source_mask.affine, source_mask.header), output)
    record["effective_mask"] = str(output)
    return mask, source_mask


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def check_grid(image, reference):
    if image.shape[:3] != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
        raise ValueError(f"Grid mismatch: {image.get_filename()}")

def psc_in_common(record, reference, common, keep):
    image = nib.load(record["bold"])
    source_mask = nib.load(record["mask"])
    effective = nib.load(record["effective_mask"])
    for dataset in (image, source_mask, effective):
        check_grid(dataset, reference)
    if image.shape[-1] != FULL_FRAMES or not np.allclose(image.header.get_zooms(), [2, 2, 2, TR], atol=1e-5):
        raise ValueError(f"Unexpected BOLD dimensions or TR for {record['subject']}.")
    raw = np.asarray(image.dataobj[..., FIRST_FRAME:], dtype=np.float32)
    mask = np.asarray(source_mask.dataobj) > 0
    mask &= np.isfinite(raw).all(axis=3) & (raw.mean(axis=3, dtype=np.float64, where=keep) > 0)
    if not np.array_equal(mask, np.asarray(effective.dataobj) > 0):
        raise ValueError(f"Effective mask changed for {record['subject']}.")
    if np.any(common & ~mask):
        raise ValueError(f"Common mask extends beyond {record['subject']}.")
    sigma = FWHM / (np.sqrt(8 * np.log(2)) * np.asarray(image.header.get_zooms()[:3]))
    weights = gaussian_filter(mask.astype(np.float32), sigma, mode="constant")[common]
    values = np.empty((N_FRAMES, int(common.sum())), dtype=np.float32, order="C")
    for i in range(N_FRAMES):
        if keep[i]:
            volume = gaussian_filter(np.where(mask, raw[..., i], 0), sigma, mode="constant")
            values[i] = volume[common] / weights
        else:
            values[i] = 0
    baseline = values.mean(axis=0, dtype=np.float64, where=keep[:, None]).astype(np.float32)
    if not np.isfinite(baseline).all() or np.any(baseline <= 0):
        raise ValueError(f"Invalid percent signal change baseline for {record['subject']}.")
    values /= baseline
    values -= 1
    values *= 100
    values[~keep] = 0
    return values

def residualize(values, keep, basis, path):
    saved = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=values.shape)
    q = basis[keep]
    max_error = 0.0
    max_relative_error = 0.0
    for start in range(0, values.shape[1], 8192):
        columns = slice(start, start + 8192)
        y = values[keep, columns].astype(np.float64)
        residual = (y - q @ (q.T @ y)).astype(np.float32)
        if not np.isfinite(residual).all():
            raise ValueError("Nonfinite nuisance-adjusted BOLD.")
        error = float(np.max(np.abs(q.T @ residual)))
        relative_error = error / max(1.0, float(np.max(np.abs(residual))))
        if relative_error > 2e-7:
            raise ValueError(f"Residual failed orthogonality check: {relative_error}.")
        max_error = max(max_error, error)
        max_relative_error = max(max_relative_error, relative_error)
        saved[:, columns] = 0
        saved[keep, columns] = residual
    saved.flush()
    return max_error, max_relative_error

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = Path(os.getenv("PAPER_INPUTS", "data/inputs"))
    parser.add_argument("--dataset", type=Path, default=Path(os.getenv("PAPER_DATASET", "data/dataset")))
    parser.add_argument("--runs", type=Path, default=inputs / "runs.tsv")
    parser.add_argument("--spans", type=Path, default=inputs / "spans.tsv")
    parser.add_argument("--audio", type=Path, default=Path(os.getenv("PAPER_AUDIO", str(inputs / "story.wav"))))
    parser.add_argument("--cache", type=Path, default=Path(os.getenv("PAPER_CACHE", str(Path.home() / ".cache/narrative-paper"))) / "encoding")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cache = args.cache.resolve()
    if (cache / "prepared.json").exists():
        raise FileExistsError("A prepared cache already exists; use a new cache for a fresh preparation.")
    cache.mkdir(parents=True, exist_ok=True)
    masks = cache / "effective_masks"
    masks.mkdir(exist_ok=True)
    records = resolve_records(args.runs, args.dataset.resolve())
    speech = speech_design(args.spans, args.audio, args.ffmpeg)
    if speech.shape != (FULL_FRAMES, 4):
        raise ValueError("Expected four full-timeline speech controls.")
    keep_bases = []
    reference, common = None, None
    for record in records:
        keep, basis = nuisance_basis(speech, Path(record["confounds"]))
        keep_bases.append((keep, basis))
        mask, image = effective_mask(record, keep, masks / (record["subject"] + ".nii.gz"))
        if reference is None:
            reference, common = image, mask.copy()
        else:
            check_grid(image, reference)
            common &= mask
    if common.sum() != 224757:
        raise ValueError(f"Published encoding intersection has 224757 voxels; found {common.sum()}.")
    common_path = cache / "common_mask.nii.gz"
    nib.save(nib.Nifti1Image(common.astype(np.uint8), reference.affine, reference.header), common_path)
    target = cache / "subjects"
    target.mkdir(exist_ok=True)
    def prepare(item):
        record, (keep, basis) = item
        subject_dir = target / record["subject"]
        if subject_dir.exists():
            raise FileExistsError(subject_dir)
        with TemporaryDirectory(dir=target, prefix=".prepare-") as temp:
            stage = Path(temp) / "ready"
            stage.mkdir()
            bold = psc_in_common(record, reference, common, keep)
            error, relative = residualize(bold, keep, basis, stage / "residual.npy")
            np.save(stage / "keep.npy", keep, allow_pickle=False)
            np.save(stage / "nuisance_basis.npy", basis, allow_pickle=False)
            save_json(stage / "metadata.json", {"subject": record["subject"], "n_keep": int(keep.sum()), "n_flagged": int((~keep).sum()), "max_projection_error": error, "max_relative_projection_error": relative})
            stage.rename(subject_dir)
        print(f"Prepared {record['subject']}: {keep.sum()} unflagged frames.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(prepare, zip(records, keep_bases, strict=True)))
    save_json(cache / "prepared.json", {
        "subjects": [r["subject"] for r in records], "n_frames": N_FRAMES,
        "n_voxels": int(common.sum()), "first_time_s": FIRST_FRAME * TR, "tr_s": TR,
        "frame_times_s": (np.arange(FIRST_FRAME, FULL_FRAMES) * TR).tolist(),
        "common_mask": common_path.name, "common_mask_sha256": digest(common_path),
        "nuisance_columns": list(NUISANCE),
        "scaling": "Percent signal change relative to each voxel's mean across unflagged retained frames.",
        "smoothing": "4-mm Gaussian, normalized within each child's effective mask.",
        "runs_sha256": digest(args.runs), "spans_sha256": digest(args.spans), "audio_sha256": digest(args.audio),
    })


if __name__ == "__main__":
    main()
