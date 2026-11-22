# Code provenance and verification

This release was assembled against the final author-supplied manuscript. The
scientific functions were extracted from the scripts that generated its results.
Local paths and inter-script dependencies were adapted to the `PAPER_*` settings;
the released execution paths implement the paper's Gemma 3 1B/PC128,
Schaefer–Tian and top-25% specification. Source-file hashes are recorded in
`config/code_provenance.json`.

## Equivalence checks performed during packaging

| Component | Check against the original saved analysis |
| --- | --- |
| Encoding nuisance projection | All 23 participants; maximum absolute basis difference 1.27 × 10⁻¹³ |
| PC128 multiscale/Lanczos/FIR design | All 324 × 1,536 expanded values; maximum absolute difference 3.83 × 10⁻¹² |
| Encoding regional aggregation | Both 16 × 232 metric arrays reproduced exactly from original held-out maps |
| Text GLM design | All 16 first-level designs; maximum absolute difference 5.69 × 10⁻¹⁴ |
| GLM regional inference | All 7,704 permutation p values and BH q values reproduced exactly from original regional coefficients |
| Scientific figures | Export and rendering of Figures 2–5 and S1 completed using original private inputs; 35 BH-positive results retained |

Synthetic numerical checks cover tied-rank exact permutation probabilities,
nested ridge versus direct dense solutions, outer-fold exclusion, prediction
scores, within-child voxel selection, sign-flip probabilities and score-permutation
inference. These checks can be run without participant data:

```bash
python code/encoding/test_numerics.py
python code/glm/test_numerics.py
python -m unittest discover -s tests -v
```

Packaging validation used existing study outputs and small numerical examples.
It did not repeat fMRIPrep, full language-model extraction or AFNI first-level
fitting. The full pipeline requires the original inputs described in
[inputs.md](inputs.md). Floating-point differences may arise across BLAS, CPU,
PyTorch and rendering environments; source hashes, checkpoint revision and
analysis parameters are retained to support comparison.

The release also provides a quantitative QC adapter for length, TR, FD, standardized DVARS and tSNR. It summarizes the selected runs without changing inclusion decisions. Its tSNR is the temporal mean divided by the sample temporal SD across all acquired volumes, summarized over finite, nonconstant in-mask voxels. This adapter is documented separately from the historical analysis functions.

## Figure provenance

The PDFs under `docs/figures/` are byte-for-byte copies of the final manuscript
figures. The PNGs are rendered from those PDFs. Data-driven plotting routines
for Figures 2–5 and S1 retain the original surface projection, ROI labels,
scatterplot fitting and density estimation. Portable font fallback uses
DejaVu Sans when Arial is unavailable; regenerated typography can therefore
differ from the supplied final figures.

Figure 1 is available as `docs/figures/fig1.pdf`, with a PNG preview displayed in the README.
