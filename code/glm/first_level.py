"""Fit the three-feature Gemma GLM with AFNI ARMA(1,1) for the 16 assessed children.

Construct the common effective mask from all 23 story runs. No volumes are
removed: flagged retained volumes are represented by spike regressors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import nibabel as nib
import numpy as np
from paths import ROOT, MODEL, INPUTS, CACHE, EVENTS, cohort, runs, read_tsv, write_tsv
from design import FEATURES, individual_design, speech_design, text_design
from bold import TR, FIRST_FRAME, FULL_FRAMES, check_matrix, image_on_grid, prepare_bold

def digest(path):
    with path.open("rb") as stream:
        result = sha256()
        while chunk := stream.read(4 * 2**20):
            result.update(chunk)
    return result.hexdigest()

def json_save(path, data):
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    stage.replace(path)

def publish_fit(stage, result):
    # A Windows scan of newly written NIfTIs can temporarily lock the directory.
    for delay in (0.25, 0.5, 1, 2, 4):
        try:
            stage.rename(result)
            return
        except PermissionError:
            if result.exists():
                raise
            time.sleep(delay)
    stage.rename(result)

def read_matrix(path):
    with path.open(encoding="utf-8") as stream:
        return stream.readline().strip().split("\t"), np.loadtxt(stream, delimiter="\t", ndmin=2)

def report(subject, message):
    print(f"{time.strftime('%H:%M:%S')} {subject} {message}", flush=True)

def map_name(subject, feature, statistic):
    return f"{subject}_task-listening_desc-{feature}_{statistic}.nii.gz"

def fit(record, output, scratch, afni, threads):
    subject = record["subject"]
    subject_dir = output / subject
    names, matrix = read_matrix(output / record["design"])
    design_hash = digest(output / record["design"])
    result = subject_dir / "fit"
    output_names = [map_name(subject, "glm", "mask")] + [
        map_name(subject, feature, statistic) for feature in FEATURES for statistic in ("beta", "t")
    ]
    if result.exists():
        metrics = json.loads((result / "fit_metrics.json").read_text())
        if metrics["design_sha256"] != design_hash or not all((result / n).is_file() for n in output_names):
            raise ValueError(f"Incomplete or stale fit for {output.name}/{subject}")
        return metrics

    bold_dir = prepare(record, scratch, output)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), AFNI_NIFTI_TYPE_WARN="NO")
    start = time.monotonic()
    with TemporaryDirectory(dir=scratch, prefix=f"{subject}-") as folder, (subject_dir / "fit.log").open("w") as log:
        work = Path(folder)
        np.savetxt(work / "columns.1D", matrix, fmt="%.12g")
        command = [str(afni / "3dDeconvolve"), "-nodata", str(len(matrix)), str(TR),
                   "-polort", "-1", "-num_stimts", str(len(names)), "-x1D", "design.xmat.1D", "-x1D_stop"]
        for i, name in enumerate(names, start=1):
            command.extend(("-stim_file", str(i), f"columns.1D[{i - 1}]", "-stim_label", str(i), name))
            if i > len(FEATURES):
                command.extend(("-stim_base", str(i)))
        subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        check_matrix(work / "design.xmat.1D", names, matrix)
        def run_afni(*parts):
            return subprocess.run(parts, cwd=work, env=env, check=True, text=True, capture_output=True).stdout
        report(f"{output.name}/{subject}", f"fitting {len(matrix)} original frames")
        command = [
            str(afni / "3dREMLfit"), "-matrix", "design.xmat.1D", "-input", str(bold_dir / "bold.nii"),
            "-mask", str(bold_dir / "mask.nii"), "-nobout", "-Rbeta", "beta.nii", "-Rbuck", "stats.nii", "-tout",
        ]
        subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        beta_labels = run_afni(str(afni / "3dinfo"), "-label", "beta.nii").strip().split("|")
        stat_labels = run_afni(str(afni / "3dinfo"), "-label", "stats.nii").strip().split("|")
        if beta_labels != [f"{name}#0" for name in FEATURES]:
            raise ValueError(f"Unexpected AFNI beta labels {beta_labels}")
        reference = nib.load(bold_dir / "bold.nii")
        mask_image = nib.load(bold_dir / "mask.nii")
        mask = np.asarray(mask_image.dataobj) > 0
        beta_image, stat_image = (nib.load(work / f"{name}.nii") for name in ("beta", "stats"))
        for image in (beta_image, stat_image):
            if image.shape[:3] != mask.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
                raise ValueError(f"Unexpected AFNI grid for {image.get_filename()}")
        beta = np.asarray(beta_image.dataobj, dtype=np.float32).reshape((*mask.shape, len(FEATURES)))
        stat = np.asarray(stat_image.dataobj, dtype=np.float32).reshape((*mask.shape, len(stat_labels)))
        with TemporaryDirectory(dir=subject_dir, prefix=".fit-") as stage_dir:
            stage = Path(stage_dir) / "fit"
            stage.mkdir()
            nib.save(image_on_grid(mask.astype(np.uint8), mask_image, np.uint8), stage / output_names[0])
            for i, name in enumerate(FEATURES):
                beta_map = beta[..., i]
                t_map = stat[..., stat_labels.index(f"{name}#0_Tstat")]
                coefficient = stat[..., stat_labels.index(f"{name}#0_Coef")]
                np.testing.assert_array_equal(beta_map, coefficient)
                for statistic, volume in (("beta", beta_map), ("t", t_map)):
                    if not np.isfinite(volume[mask]).all() or volume[mask].std() <= 0 or np.any(volume[~mask] != 0):
                        raise ValueError(f"Invalid {statistic} map for {output.name}/{subject}/{name}")
                    nib.save(image_on_grid(volume, reference, np.float32), stage / map_name(subject, name, statistic))
            metrics = {
                "subject": subject, "voxels": int(mask.sum()), "design_columns": len(names),
                "spike_columns": int(record["n_spikes"]), "nominal_residual_df": int(record["residual_df"]),
                "design_sha256": design_hash, "elapsed_seconds": round(time.monotonic() - start, 2),
            }
            json_save(stage / "fit_metrics.json", metrics)
            publish_fit(stage, result)
    report(f"{output.name}/{subject}", f"complete in {metrics['elapsed_seconds']:.1f}s")
    return metrics


def prepare(record, scratch, output):
    """Prepare the original normalized-smoothed PSC volume using GLM good frames."""
    dest = scratch / "prepared" / record["subject"]
    names, design = read_matrix(output / record["design"])
    spikes = np.array([c.startswith("spike") for c in names])
    good = ~np.any(design[:, spikes] != 0, axis=1)
    signature = {"design": digest(output / record["design"])}
    for key in ("bold", "mask"):
        path = Path(record[key]); st = path.stat()
        signature[key] = [str(path), st.st_size, st.st_mtime_ns]
    if dest.is_dir():
        if json.loads((dest / "source.json").read_text()) != signature:
            raise ValueError(f"Cached source changed for {record['subject']}; use a fresh PAPER_CACHE.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".prepare-") as tmp:
        stage = Path(tmp) / "images"; stage.mkdir()
        prepare_bold(record, stage, good)
        np.testing.assert_array_equal(np.asarray(nib.load(stage / "mask.nii").dataobj),
                                      np.asarray(nib.load(ROOT / "effective_masks" / f"{record['subject']}.nii.gz").dataobj))
        json_save(stage / "source.json", signature)
        stage.rename(dest)
    return dest


def build_designs(records, output, ffmpeg, audio):
    text = text_design(ROOT / "features" / MODEL / EVENTS, INPUTS / "spans.tsv")
    speech = speech_design(INPUTS / "spans.tsv", audio, ffmpeg)[FIRST_FRAME:]
    common = np.column_stack((text, speech))
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    intersection = None
    reference = None
    masks = ROOT / "effective_masks"; masks.mkdir(parents=True, exist_ok=True)
    for original in records:
        record = dict(original)
        subject = record["subject"]
        names, matrix, n_spikes, rank, residual_df, vif = individual_design(common, Path(record["confounds"]), FIRST_FRAME)
        path = Path(subject) / f"{subject}_design.tsv"
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output / path, matrix, delimiter="\t", header="\t".join(names), comments="", fmt="%.16g")
        record.update(design=str(path), n_spikes=n_spikes, rank=rank, residual_df=residual_df,
                      n_volumes=len(matrix), n_columns=len(names))
        record.update({f"vif_{f}": f"{v:.6f}" for f, v in zip(FEATURES, vif)})
        prepared.append(record)
        good = ~matrix[:, [j for j, name in enumerate(names) if name.startswith("spike")]].any(axis=1)
        image, mask_image = nib.load(record["bold"]), nib.load(record["mask"])
        if image.shape != (*mask_image.shape, FULL_FRAMES) or not np.allclose(image.affine, mask_image.affine, atol=1e-5) or not np.allclose(image.header.get_zooms(), [2,2,2,TR], atol=1e-5):
            raise ValueError(f"Unexpected fMRIPrep geometry: {subject}")
        data = np.asarray(image.dataobj[..., FIRST_FRAME:], dtype=np.float32)
        effective = (np.asarray(mask_image.dataobj) > 0) & np.isfinite(data).all(axis=3) & (data.mean(axis=3, dtype=np.float64, where=good) > 0)
        if intersection is None:
            reference = mask_image
            intersection = effective.copy()
        else:
            if mask_image.shape != reference.shape or not np.allclose(mask_image.affine, reference.affine, rtol=0, atol=1e-5):
                raise ValueError("Story runs do not share the MNI grid.")
            intersection &= effective
        nib.save(image_on_grid(effective.astype(np.uint8), mask_image, np.uint8), masks / f"{subject}.nii.gz")
    if int(intersection.sum()) != 224756:
        raise ValueError(f"Published GLM common mask has 224756 voxels; found {intersection.sum()}.")
    nib.save(image_on_grid(intersection.astype(np.uint8), reference, np.uint8), ROOT / "common_mask.nii.gz")
    write_tsv(output / "design_manifest.tsv", prepared)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="Subjects processed concurrently.")
    parser.add_argument("--threads", type=int, default=4, help="AFNI OpenMP threads per subject.")
    parser.add_argument("--afni-bin", type=Path, default=Path(shutil.which("3dREMLfit") or "/usr/local/bin/3dREMLfit").parent)
    parser.add_argument("--scratch", type=Path, default=CACHE / "glm")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--audio", type=Path, default=Path(os.environ.get("PAPER_AUDIO", INPUTS / "story.wav")))
    parser.add_argument("--design-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.threads < 1:
        parser.error("--jobs and --threads must be positive.")
    _, complete = cohort()
    output = ROOT / MODEL
    records = build_designs(runs(), output, args.ffmpeg, args.audio)
    if args.design_only:
        return
    for binary in ("3dDeconvolve", "3dREMLfit", "3dinfo"):
        if not (args.afni_bin / binary).is_file():
            parser.error(f"Missing AFNI binary: {args.afni_bin / binary}")
    records = [r for r in records if r["subject"] in complete]
    args.scratch.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [pool.submit(fit, r, output, args.scratch, args.afni_bin, args.threads) for r in records]
        metrics = [j.result() for j in jobs]
    write_tsv(output / "fit_summary.tsv", metrics)
    manifest = []
    for r in records:
        subject = r["subject"]; folder = Path(subject) / "fit"
        manifest.append({"subject": subject, "mask": str(folder / map_name(subject, "glm", "mask")),
                         **{f: str(folder / map_name(subject, f, "beta")) for f in FEATURES},
                         **{f"t_{f}": str(folder / map_name(subject, f, "t")) for f in FEATURES}})
    write_tsv(output / "first_level.tsv", manifest)
    json_save(output / "first_level_provenance.json", {
        "model": MODEL, "subjects": complete, "mask_subjects": cohort()[0],
        "scalar_features_sha256": digest(ROOT / "features" / MODEL / EVENTS),
        "afni_3dREMLfit_sha256": digest(args.afni_bin / "3dREMLfit"),
        "frames": 316, "first_frame_zero_based": 20, "TR_s": TR,
        "fit": "AFNI 3dREMLfit ARMA(1,1); all acquired retained frames and motion spike regressors.",
    })


if __name__ == "__main__":
    main()
