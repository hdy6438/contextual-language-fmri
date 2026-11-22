# Input formats

Run the scripts from the repository root. Private inputs belong in `PAPER_INPUTS`
(default `data/inputs/`); that directory is ignored by Git. Header-only templates
are provided in `data/templates/`.

## Participants and imaging

`participants.tsv` contains the 23 included children, ordered by ascending BIDS
subject label. Exactly 16 rows have `complete_assessments=1`; the other seven
remain in every encoding training fold. Keep subject labels as strings.

| Column | Meaning |
| --- | --- |
| `subject` | BIDS label, including `sub-` and leading zeros |
| `complete_assessments` | `1` for an outer-fold evaluation/brain–behavior participant, otherwise `0` |
| `session` | Optional selected functional session, e.g. `ses-02`; required by the manifest builder when several listening runs exist |
| `age` | Age in years, derived from the study's rounded mean scan-month age divided by 12; used only for the cohort figure/table |
| `sex` | `F` or `M`; used only for the cohort figure/table |
| `school_grade` | Recorded school grade; used only for the cohort figure/table |

`runs.tsv` contains one previously QC-selected listening run per child, with
columns `subject`, `session`, `bold`, `mask`, and `confounds`. The last three
columns are file paths, either absolute or relative to `PAPER_DATASET`.
The inputs are the fMRIPrep MNI152NLin6Asym 2-mm preprocessed BOLD (336 volumes,
TR 0.75 s), its brain mask, and its confounds TSV. The manifest builder resolves
these from `PAPER_OUTPUT/fmriprep`; `--fmriprep` can point to existing derivatives.
It stops if run selection is ambiguous. `summarize_qc.py` writes length, TR, motion, frame-flag and tSNR summaries to `PAPER_OUTPUT/qc/`; `--skip-tsnr` performs the lightweight checks only. These summaries do not change cohort inclusion. Visual inspection of fMRIPrep HTML
reports establishes normalization, coregistration, coverage and artefact quality.

## Story timing and audio

`character_events.tsv` has 869 rows and columns `trial_type`, `text`,
`character_index`, `onset`, and `duration`. `trial_type` is `character`;
`character_index` is one-based. Times are in seconds from the first acquired
BOLD volume (audio onset delay 0 s).

`spans.tsv` is the original shared 732-span segmentation. It contains:

| Column | Meaning |
| --- | --- |
| `embedding_row` | Zero-based row in the embedding array |
| `character_start`, `character_end` | One-based, inclusive character range |
| `text` | Exact characters in that span |
| `onset`, `offset`, `duration` | Original character-derived timing in seconds |
| `token_start`, `token_end` | Original token span bounds; the speech-control token count is `token_end - token_start + 1` |

Preserve the exact original files, decimal timing and segmentation. The original
span token counts are used for speech controls; Gemma native tokenization is
used separately for causal model features. Creating new spans with Gemma would
change the study's design.

Use the original story audio recording. Set `PAPER_AUDIO` to its path, or supply
`--audio` to both `code/encoding/prepare_bold.py` and
`code/glm/first_level.py`. The default is `PAPER_INPUTS/story.wav`. FFmpeg accepts
the original format (including MP3) and decodes directly to mono 16-kHz float32
PCM before the 20-ms RMS calculation; no intermediate re-encoding is needed.

## Assessments

`assessments.tsv` has one row per complete-assessment child and columns `subject`
plus these 11 scores, in the study's supplied standardized-score units:

| Column | Paper label |
| --- | --- |
| `fsiq` | FSIQ |
| `literacy` | Literacy |
| `chinese_reading` | Word reading |
| `chinese_dictation` | Word dictation |
| `cantonese_grammar_ss` | Cantonese grammar |
| `discourse_comprehension_ss` | Discourse comprehension |
| `word_definition_ss` | Word definition |
| `semantic_relations_ss` | Semantic relations |
| `story_retelling_ss` | Story retelling |
| `noun_expression_ss` | Noun expression |
| `hkgcnt_ss` | HKGCNT |

All 16 rows must have finite values in all 11 score columns. Scripts join by
subject identifier. Score standardization for GLM slopes is performed internally
using the sample SD; encoding associations use average tied ranks.

## Model and atlas

Provide the complete locally downloaded `google/gemma-3-1b-pt` checkpoint at
revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, including its tokenizer.
Both embedding and scalar extraction must use that same directory.

The ready-to-use public atlas and label table are in `assets/atlas/`. IDs 1–32
are Tian S2 subcortex; IDs 33–232 are Schaefer 200 cortex. The atlas preparation
script verifies upstream file hashes and reproduces the canonical RAS orientation.

## Generated outputs

| Directory | Contents |
| --- | --- |
| `PAPER_OUTPUT/features/` | 732 × 1,152 contextual embeddings and source metadata |
| `PAPER_OUTPUT/encoding/pc128/` | PCA/FIR design, 16 LOSO predictions and voxelwise scores |
| `PAPER_OUTPUT/encoding/` | 16 × 232 regional metrics and 5,104 assessment tests |
| `PAPER_OUTPUT/story_text_glm/` | Common mask, scalar features, 16 first-level fits, regional coefficients and 7,704 tests |
| `PAPER_OUTPUT/figure_data/` | Generated figure inputs, including individual observations |
| `PAPER_OUTPUT/figures/`, `tables/` | Regenerated figures and LaTeX tables |
| `PAPER_CACHE/` | Intermediate BOLD arrays and numerical caches |

The original data are available from the corresponding author upon reasonable
request, as stated in the manuscript. Individual imaging, assessment and plot-input
files are supplied separately; the repository distributes code, final figures,
public atlas assets and regional summary statistics.
