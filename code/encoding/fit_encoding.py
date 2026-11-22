import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Semaphore
from time import monotonic

import nibabel as nib
import numpy as np

from publish_results import publish_directory
from accelerated_solver import fit_nested_loso


PREDICTION_WRITES = Semaphore(2)
REFERENCE_COLUMNS = 1536
REFERENCE_ALPHAS = np.array([.001, .01, .1, 1., 10., 100., 1000., 10000.])


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subject_data(cache, subject, stimulus, n_voxels):
    folder = cache / "subjects" / subject
    response = np.load(folder / "residual.npy", mmap_mode="r", allow_pickle=False)
    basis = np.load(folder / "nuisance_basis.npy", allow_pickle=False)
    keep = np.load(folder / "keep.npy", allow_pickle=False)
    if (response.shape != (len(stimulus), n_voxels) or response.dtype != np.float32
            or keep.shape != (len(stimulus),) or keep.dtype != bool or basis.shape[0] != len(stimulus)):
        raise ValueError(f"Invalid cache layout for {subject}.")
    if np.any(basis[~keep]):
        raise ValueError(f"Flagged nuisance rows must be zero for {subject}.")
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
    # Partial out unpenalized nuisance from predictors in the same way as BOLD.
    projected = stimulus - basis @ (basis.T @ stimulus)
    projected[~keep] = 0
    return response, projected, keep


def cross_products(cache, subjects, stimulus, n_voxels, provenance):
    target = cache / f"ridge_inputs_{provenance['features_sha256'][:16]}"
    paths = ("cross.npy", "gram.npy", "n_good.npy")
    if target.exists():
        if json.loads((target / "provenance.json").read_text()) != provenance:
            raise ValueError("Cached ridge inputs use a different source or feature design.")
        return tuple(np.load(target / name, mmap_mode="r", allow_pickle=False) for name in paths)
    with TemporaryDirectory(prefix=".ridge-inputs-", dir=cache) as work:
        stage = Path(work)
        shape = (len(subjects), stimulus.shape[1], n_voxels)
        cross = np.lib.format.open_memmap(stage / paths[0], mode="w+", dtype=np.float32, shape=shape)
        gram = np.empty((len(subjects), stimulus.shape[1], stimulus.shape[1]), dtype=np.float64)
        counts = np.empty(len(subjects), dtype=int)
        for i, subject in enumerate(subjects):
            response, projected, keep = subject_data(cache, subject, stimulus, n_voxels)
            gram[i] = projected.T @ projected
            cross[i] = projected.astype(np.float32).T @ response
            counts[i] = keep.sum()
            if not np.isfinite(cross[i]).all():
                raise ValueError(f"Nonfinite ridge cross-products for {subject}.")
            print(f"{subject}: ridge inputs ready ({counts[i]} scored frames).", flush=True)
        cross.flush()
        del cross
        np.save(stage / paths[1], gram, allow_pickle=False)
        np.save(stage / paths[2], counts, allow_pickle=False)
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        publish_directory(stage, target)
    return tuple(np.load(target / name, mmap_mode="r", allow_pickle=False) for name in paths)


def score(response, projected, beta, cross, gram, n_good):
    n_voxels = response.shape[1]
    r, r2 = np.full(n_voxels, np.nan), np.full(n_voxels, np.nan)
    design_sum = projected.sum(axis=0)
    for start in range(0, n_voxels, 4096):
        block = slice(start, start + 4096)
        measured = response[:, block]
        b = beta[:, block].astype(np.float64)
        yy = np.einsum("tv,tv->v", measured, measured, dtype=np.float64)
        pp = np.einsum("pv,pv->v", b, gram @ b)
        yp = np.einsum("pv,pv->v", b, cross[:, block], dtype=np.float64)
        y_sum = measured.sum(axis=0, dtype=np.float64)
        p_sum = design_sum @ b
        denom_y = np.maximum(0, yy - y_sum**2 / n_good)
        denom_p = np.maximum(0, pp - p_sum**2 / n_good)
        valid = (denom_y > 0) & (denom_p > 0)
        np.divide(yp - y_sum * p_sum / n_good, np.sqrt(denom_y * denom_p),
                  out=r[block], where=valid)
        np.divide(2 * yp - pp, yy, out=r2[block], where=yy > 0)
    return np.clip(r, -1, 1), r2


def save_image(path, values, mask, reference, tr=None, start_time=None):
    shape = mask.shape + (() if values.ndim == 1 else (values.shape[0],))
    volume = np.zeros(shape, dtype=np.float32, order="F")
    volume[mask] = values if values.ndim == 1 else values.T
    image = nib.Nifti1Image(volume, reference.affine)
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.header.set_xyzt_units("mm", "sec")
    if tr is not None:
        image.header.set_zooms((*reference.header.get_zooms()[:3], tr))
        image.header["toffset"] = start_time
    nib.save(image, path)


def predict_fold(outer, context):
    (cache, out, subjects, complete, stimulus, cross, gram, counts,
     mask, reference, metadata, provenance, alphas) = context
    subject = subjects[outer]
    target = out / subject
    filenames = ["prediction.nii.gz", "r.nii.gz", "r2.nii.gz", "alpha.nii.gz", "scored_frames.tsv"]
    if target.exists():
        row = json.loads((target / "metrics.json").read_text())
        if row["provenance"] != provenance or not all((target / name).is_file() for name in filenames):
            raise ValueError(f"Stale or incomplete fold in {target}.")
        print(f"{subject}: verified existing fold.", flush=True)
        return row["summary"]
    start = monotonic()
    diagnostics = {}
    beta, alpha_index = fit_nested_loso(gram, cross, counts, outer, complete, alphas, diagnostics=diagnostics)
    response, projected, keep = subject_data(cache, subject, stimulus, mask.sum())
    r, r2 = score(response, projected, beta, cross[outer], gram[outer], int(counts[outer]))
    if not np.any(np.isfinite(r)) or not np.any(np.isfinite(r2)):
        raise ValueError(f"No defined test scores for {subject}.")
    summary = {
        "subject": subject, "n_train": len(subjects) - 1, "n_scored_frames": int(keep.sum()),
        "n_valid_r_voxels": int(np.isfinite(r).sum()), "n_valid_r2_voxels": int(np.isfinite(r2).sum()),
        "median_r": float(np.nanmedian(r)), "p95_r": float(np.nanquantile(r, .95)),
        "median_r2": float(np.nanmedian(r2)), "fraction_r2_positive": float((r2 > 0).sum() / np.isfinite(r2).sum()),
        **{f"n_voxels_alpha_{a:g}": int((alpha_index == i).sum()) for i, a in enumerate(alphas)},
    }
    with TemporaryDirectory(prefix=f".{subject}-", dir=out) as work:
        stage = Path(work)
        with PREDICTION_WRITES:
            prediction = stimulus.astype(np.float32) @ beta
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite predictions for {subject}.")
            save_image(stage / filenames[0], prediction, mask, reference,
                       metadata["tr_seconds"], metadata["first_time_seconds"])
            del prediction
        save_image(stage / filenames[1], r, mask, reference)
        save_image(stage / filenames[2], r2, mask, reference)
        save_image(stage / filenames[3], alphas[alpha_index], mask, reference)
        write_tsv(stage / filenames[4], [
            {"frame_one_based": 13 + i, "story_time_s": metadata["first_time_seconds"] + i * metadata["tr_seconds"],
             "scored": int(valid)} for i, valid in enumerate(keep)
        ])
        (stage / "metrics.json").write_text(json.dumps({"summary": summary, "provenance": provenance, "solver_diagnostics": diagnostics},
                                                       indent=2) + "\n")
        publish_directory(stage, target)
    print(f"{subject}: fold finished in {monotonic() - start:.1f}s, median r={summary['median_r']:.5f}.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(os.getenv("PAPER_OUTPUT", "outputs")) / "encoding/pc128")
    parser.add_argument("--cache", type=Path, default=Path(os.getenv("PAPER_CACHE", str(Path.home() / ".cache/narrative-paper"))) / "encoding")
    parser.add_argument("--participants", type=Path, default=Path(os.getenv("PAPER_INPUTS", "data/inputs")) / "participants.tsv")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    out, cache = args.output, args.cache
    prep_file = cache / "prepared.json"
    prep = json.loads(prep_file.read_text())
    metadata = json.loads((out / "features.json").read_text())
    subjects = prep["subjects"]
    complete_subjects = {row["subject"] for row in read_tsv(args.participants) if row["complete_assessments"] == "1"}
    if len(subjects) != len(set(subjects)) or len(subjects) != 23 or len(complete_subjects) != 16 or not complete_subjects <= set(subjects):
        raise ValueError("Expected exactly sixteen complete cases and seven training-only listeners.")
    complete = np.array([i for i, subject in enumerate(subjects) if subject in complete_subjects])
    stimulus = np.load(out / "features.npy", allow_pickle=False)
    if (stimulus.shape != (prep["n_frames"], metadata["n_features"]) or not np.isfinite(stimulus).all()
            or prep["first_time_s"] != metadata["first_time_seconds"] or prep["tr_s"] != metadata["tr_seconds"]):
        raise ValueError("The stimulus timeline and prepared BOLD disagree.")
    np.testing.assert_allclose(prep["frame_times_s"], metadata["first_time_seconds"] + np.arange(len(stimulus)) * prep["tr_s"])
    mask_path = cache / prep["common_mask"]
    if digest(mask_path) != prep["common_mask_sha256"]:
        raise ValueError("Common-mask hash mismatch.")
    reference = nib.load(mask_path)
    mask = np.asarray(reference.dataobj) > 0
    if int(mask.sum()) != prep["n_voxels"]:
        raise ValueError("Common-mask size mismatch.")
    if not (out / "common_mask.nii.gz").exists():
        (out / "common_mask.nii.gz").write_bytes(mask_path.read_bytes())
    folds = []
    for i in complete:
        train = [subject for j, subject in enumerate(subjects) if i != j]
        inner = [subject for subject in train if subject in complete_subjects]
        if len(train) != 22 or len(inner) != 15 or not set(subjects) - complete_subjects <= set(train):
            raise ValueError("Invalid LOSO partition.")
        folds.append({"test_subject": subjects[i], "n_train": len(train), "n_inner_folds": len(inner),
                      "train_subjects": ",".join(train), "inner_validation_subjects": ",".join(inner)})
    write_tsv(out / "folds.tsv", folds)
    sources = {"features_sha256": digest(out / "features.npy"), "preparation_sha256": digest(prep_file),
               "subjects": subjects}
    cross, gram, counts = cross_products(cache, subjects, stimulus, int(mask.sum()), sources)
    if metadata["n_expanded_features"] != REFERENCE_COLUMNS or not metadata["embedding_pca"]:
        raise ValueError("This release uses only the published 128-PC, 1536-column design.")
    alphas = REFERENCE_ALPHAS.copy()
    provenance = {**sources, "test_subjects": [subjects[i] for i in complete], "alphas": alphas.tolist()}
    (out / "encoding.json").write_text(json.dumps({
        **provenance,
        "output_prediction": "Embedding-only contribution to PSC over all 324 frames, with no measured held-out BOLD used to fit the ridge coefficients",
        "performance": "Test both target and prediction in each held-out subject's nuisance-adjusted PSC space. r is Pearson correlation; R2 uses zero residual prediction as baseline, retains negative values. Scoring uses frames without spike flags.",
        "nuisance": prep["nuisance_columns"], "scaling": prep["scaling"], "smoothing": prep["smoothing"],
        "spatial_mask": "Fixed geometric intersection of the 23 QC-passing runs",
        "ridge": "Minimize summed squared error divided by training frame count plus alpha times squared coefficients. Each voxel selects alpha by summed inner LOSO SSE, excluding the outer subject.",
        "embedding_pca": metadata["embedding_pca"], "embedding_dimensions": metadata["embedding_dimensions"],
        "expanded_features": metadata["n_expanded_features"], "exact_ridge_coordinates": metadata["n_features"],
        "scope": "Across-subject generalization on the same story, with seven incomplete-assessment participants in every training fold.",
    }, indent=2) + "\n")
    context = (cache, out, subjects, complete, stimulus, cross, gram, counts, mask, reference, metadata, provenance, alphas)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        summaries = list(pool.map(lambda index: predict_fold(index, context), complete))
    write_tsv(out / "heldout_prediction_summary.tsv", summaries)
    print(f"Completed all {len(summaries)} LOSO folds.", flush=True)


if __name__ == "__main__":
    main()
