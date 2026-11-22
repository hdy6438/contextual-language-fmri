"""Summarize selected story runs for manual QC without changing data or inclusion.

This publication adapter reports the quantitative checks described in the paper;
it is not an archived copy of the original interactive QC procedure. Temporal
SNR is mean / sample SD (ddof=1) across every acquired volume, calculated within
the fMRIPrep brain mask for finite voxels with nonzero temporal SD. It is computed
before smoothing, nuisance regression or frame exclusion. Original volume order
is preserved. No inclusion threshold is applied to any summary.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def read_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def number(value):
    return np.nan if value in (None, "", "n/a", "NA") else float(value)


def summarize_frames(rows):
    fd = np.array([number(r.get("framewise_displacement")) for r in rows])
    dvars = np.array([number(r.get("std_dvars")) for r in rows])
    nonsteady = np.zeros(len(rows), dtype=bool)
    for name in rows[0]:
        if name.startswith("non_steady_state_outlier"):
            nonsteady |= np.array([number(r[name]) for r in rows]) == 1
    flags = (fd > .5) | (dvars > 1.5) | nonsteady
    def finite_summary(x, reducer):
        take = x[np.isfinite(x)]
        return float(reducer(take)) if len(take) else "n/a"
    result = {
        "fd_mean_mm": finite_summary(fd, np.mean), "fd_max_mm": finite_summary(fd, np.max),
        "std_dvars_mean": finite_summary(dvars, np.mean), "std_dvars_max": finite_summary(dvars, np.max),
        "n_fd_missing": int((~np.isfinite(fd)).sum()),
        "n_std_dvars_missing": int((~np.isfinite(dvars)).sum()),
        "n_fd_gt_0_5": int((fd > .5).sum()), "n_std_dvars_gt_1_5": int((dvars > 1.5).sum()),
        "n_nonsteady": int(nonsteady.sum()), "n_flagged_union": int(flags.sum()),
        "fraction_flagged_union": float(flags.mean()),
    }
    # These crops preserve the original acquisition timeline in both analyses.
    for first, label in ((12, "encoding"), (20, "text_glm")):
        retained = flags[first:]
        result[f"{label}_retained_frames"] = len(retained)
        result[f"{label}_flagged_frames"] = int(retained.sum())
        result[f"{label}_unflagged_frames"] = int((~retained).sum())
    return result


def temporal_snr(volumes):
    """Welford mean/SD from an iterator of equally sized masked 1-D volumes.

    A voxel is retained only if every acquired observation is finite and its
    sample SD is nonzero. The iterator keeps memory proportional to one volume.
    """
    mean = second = valid = None
    count = 0
    for volume in volumes:
        x = np.asarray(volume, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Expected one vector of masked voxels per volume.")
        if mean is None:
            mean = np.zeros_like(x)
            second = np.zeros_like(x)
            valid = np.ones(x.shape, dtype=bool)
        if x.shape != mean.shape:
            raise ValueError("Volume dimensions changed.")
        valid &= np.isfinite(x)
        safe = np.where(np.isfinite(x), x, 0.)
        count += 1
        delta = safe - mean
        mean += delta / count
        second += delta * (safe - mean)
    if count < 2:
        raise ValueError("At least two acquired volumes are required for tSNR.")
    sd = np.sqrt(np.maximum(second / (count - 1), 0))
    take = valid & (sd > 0)
    return mean[take] / sd[take], int((~valid).sum()), int((valid & (sd == 0)).sum())


def report_path(output, dataset, subject):
    for root in (output / "fmriprep", dataset / "derivatives/fmriprep"):
        path = root / f"{subject}.html"
        if path.is_file():
            return str(path.resolve())
        matches = sorted(root.glob(f"{subject}*.html"))
        if len(matches) == 1:
            return str(matches[0].resolve())
    return "n/a"


def main():
    repo = Path(__file__).resolve().parents[2]
    inputs = Path(os.getenv("PAPER_INPUTS", repo / "data/inputs"))
    output = Path(os.getenv("PAPER_OUTPUT", repo / "outputs"))
    dataset = Path(os.getenv("PAPER_DATASET", repo / "data/dataset"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=inputs / "runs.tsv")
    parser.add_argument("--output", type=Path, default=output / "qc/story_summary.tsv")
    parser.add_argument("--skip-tsnr", action="store_true", help="Report motion/length only, without reading BOLD voxels.")
    args = parser.parse_args()
    records = read_tsv(args.runs)
    if not records or len({r["subject"] for r in records}) != len(records):
        raise ValueError("Expected unique subjects in a nonempty selected-run manifest.")
    results = []
    for record in records:
        paths = {}
        for key in ("bold", "mask", "confounds"):
            path = Path(record[key]).expanduser()
            paths[key] = path if path.is_absolute() else dataset / path
        # Keep the file handle open, so compressed sequential volumes are read
        # forward from the same stream instead of reopening the gzip file.
        image = nib.load(paths["bold"], keep_file_open=True)
        mask_image = nib.load(paths["mask"])
        if len(image.shape) != 4 or image.shape[:3] != mask_image.shape or not np.allclose(image.affine, mask_image.affine, atol=1e-5, rtol=0):
            raise ValueError(f"BOLD/mask grid mismatch for {record['subject']}.")
        mask = np.asarray(mask_image.dataobj) > 0
        confounds = read_tsv(paths["confounds"])
        if len(confounds) != image.shape[3] or not mask.any():
            raise ValueError("Confound length mismatch or empty brain mask.")
        tr = float(image.header.get_zooms()[3])
        row = dict(subject=record["subject"], session=record.get("session", ""),
                   n_volumes=image.shape[3], tr_s=tr, acquisition_duration_s=image.shape[3] * tr,
                   last_volume_onset_s=(image.shape[3] - 1) * tr,
                   matches_paper_336_volumes_0_75s=int(image.shape[3] == 336 and np.isclose(tr, .75)),
                   n_mask_voxels=int(mask.sum()), **summarize_frames(confounds))
        row.update(tsnr_n_voxels="n/a", tsnr_mean="n/a", tsnr_median="n/a", tsnr_p05="n/a", tsnr_p95="n/a",
                   n_nonfinite_voxels="n/a", n_constant_voxels="n/a")
        if not args.skip_tsnr:
            values, nonfinite, constant = temporal_snr(np.asarray(image.dataobj[..., t], dtype=np.float64)[mask] for t in range(image.shape[3]))
            if not len(values):
                raise ValueError(f"No finite nonconstant tSNR voxel for {record['subject']}.")
            row.update(tsnr_n_voxels=len(values), tsnr_mean=float(values.mean()), tsnr_median=float(np.median(values)),
                       tsnr_p05=float(np.quantile(values, .05)), tsnr_p95=float(np.quantile(values, .95)),
                       n_nonfinite_voxels=nonfinite, n_constant_voxels=constant)
        row["fmriprep_html_report"] = report_path(output, dataset, record["subject"])
        results.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(results)
    args.output.with_suffix(".json").write_text(json.dumps({
        "role": "Publication QC summary adapter; manual inclusion decisions are supplied in the run manifest.",
        "tsnr": "Per-voxel temporal mean / sample SD (ddof=1), all acquired frames, fMRIPrep mask; finite observations and nonzero SD required.",
        "frame_flag": "FD > 0.5 mm or standardized DVARS > 1.5 or nonsteady-state indicator. Missing values reported separately.",
        "tsnr_computed": not args.skip_tsnr, "automatic_exclusion": False,
        "source_volume_timeline_modified": False,
    }, indent=2) + "\n")
    print(f"Wrote {len(results)} run summaries to {args.output}")


if __name__ == "__main__":
    main()
