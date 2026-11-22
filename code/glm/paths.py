"""Filesystem settings and input readers for the published analysis."""
import csv
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DATASET = Path(os.environ.get("PAPER_DATASET", REPOSITORY / "data/dataset")).expanduser().resolve()
INPUTS = Path(os.environ.get("PAPER_INPUTS", REPOSITORY / "data/inputs")).expanduser().resolve()
OUTPUT = Path(os.environ.get("PAPER_OUTPUT", REPOSITORY / "outputs")).expanduser().resolve()
CACHE = Path(os.environ.get("PAPER_CACHE", Path.home() / ".cache/narrative-paper")).expanduser().resolve()
ROOT = OUTPUT / "story_text_glm"
MODEL = "gemma3_1b"
ATLAS = "schaefer200_tian_s2"
EVENTS = "task-listening_desc-textFeatures_events.tsv"
ASSESSMENTS = (
    ("fsiq", "FSIQ"), ("literacy", "Literacy"),
    ("chinese_reading", "Word reading"), ("chinese_dictation", "Word dictation"),
    ("cantonese_grammar_ss", "Cantonese grammar"),
    ("discourse_comprehension_ss", "Discourse comprehension"),
    ("word_definition_ss", "Word definition"), ("semantic_relations_ss", "Semantic relations"),
    ("story_retelling_ss", "Story retelling"), ("noun_expression_ss", "Noun expression"),
    ("hkgcnt_ss", "HKGCNT"),
)


def read_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cohort():
    rows = read_tsv(INPUTS / "participants.tsv")
    subjects = [r["subject"] for r in rows]
    complete = [r["subject"] for r in rows if str(r["complete_assessments"]) == "1"]
    if len(subjects) != 23 or len(set(subjects)) != 23 or len(complete) != 16:
        raise ValueError("participants.tsv must contain 23 unique subjects, including 16 complete_assessments=1.")
    return subjects, complete


def runs():
    subjects, _ = cohort()
    rows = read_tsv(INPUTS / "runs.tsv")
    lookup = {r["subject"]: r for r in rows}
    if len(rows) != 23 or set(lookup) != set(subjects):
        raise ValueError("runs.tsv must contain exactly one selected story run for every participant.")
    for row in rows:
        for field in ("bold", "mask", "confounds"):
            p = Path(row[field]).expanduser()
            row[field] = str(p if p.is_absolute() else DATASET / p)
    return [lookup[s] for s in subjects]
