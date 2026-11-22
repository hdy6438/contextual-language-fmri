"""Locate one QC-selected story run per participant in fMRIPrep outputs.

This release adapter writes paths consumed by the original analysis algorithms.
When multiple story runs exist, supply a session in participants.tsv; selection
is never made from the observed association statistics.
"""
import argparse
import csv
import os
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    inputs = Path(os.getenv("PAPER_INPUTS", repo / "data/inputs"))
    output = Path(os.getenv("PAPER_OUTPUT", repo / "outputs"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fmriprep", type=Path, default=output / "fmriprep")
    p.add_argument("--participants", type=Path, default=inputs / "participants.tsv")
    p.add_argument("--output", type=Path, default=inputs / "runs.tsv")
    args = p.parse_args()
    with args.participants.open() as f:
        participants = list(csv.DictReader(f, delimiter="\t"))
    if len(participants) != 23 or len({r['subject'] for r in participants}) != 23:
        raise ValueError("Expected the 23 unique paper participants.")
    if sum(int(r['complete_assessments']) for r in participants) != 16:
        raise ValueError("Expected 16 assessment-complete participants.")
    rows = []
    for r in participants:
        subject = r["subject"]
        if not subject.startswith('sub-') or not subject[4:].isalnum():
            raise ValueError(f"Invalid BIDS subject: {subject}")
        candidates = list((args.fmriprep / subject).rglob(
            "*_task-listening*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"))
        session = r.get("session", "").strip()
        if session:
            session = session if session.startswith('ses-') else 'ses-' + session
            candidates = [f for f in candidates if session in f.name.split('_')]
        if len(candidates) != 1:
            raise ValueError(f"{subject}: found {len(candidates)} runs; specify a QC-selected session in participants.tsv or supply runs.tsv explicitly.")
        bold = candidates[0]
        stem = bold.name.split('_space-')[0]
        mask = bold.with_name(bold.name.replace('desc-preproc_bold', 'desc-brain_mask'))
        confounds = bold.with_name(stem + '_desc-confounds_timeseries.tsv')
        if not mask.is_file() or not confounds.is_file():
            raise FileNotFoundError(f"Missing mask or confounds for {bold}")
        session = next((s for s in stem.split('_') if s.startswith('ses-')), '')
        rows.append(dict(subject=subject, session=session, bold=str(bold.resolve()),
                         mask=str(mask.resolve()), confounds=str(confounds.resolve())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} selected runs to {args.output}")


if __name__ == "__main__":
    main()
