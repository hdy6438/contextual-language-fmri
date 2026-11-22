"""Group and assessment inference for the paper's 214 eligible GLM ROIs."""
import argparse
import json
import numpy as np
from scipy.stats import false_discovery_control, t
from threadpoolctl import threadpool_limits
from paths import ROOT, MODEL, ATLAS, INPUTS, ASSESSMENTS, cohort, read_tsv, write_tsv
ROI = ROOT / "roi"
OUT = ROOT / "second_level"
ATLASES = (ATLAS,)
FEATURES = ("surprisal", "entropy", "semantic_update")
TIERS = (25,)
DRAWS = 19999
SEED = 1162026
BATCH = 512
ALPHA = .05


def load_cohort():
    _, subjects = cohort()
    rows = read_tsv(INPUTS / "assessments.tsv")
    lookup = {r["subject"]: r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != set(subjects):
        raise ValueError("assessments.tsv must contain exactly the 16 complete participants.")
    scores = {}
    for code, label in ASSESSMENTS:
        x = np.array([float(lookup[s][code]) for s in subjects])
        sd = x.std(ddof=1)
        if not np.isfinite(x).all() or not sd > 0:
            raise ValueError(f"Invalid score: {code}")
        scores[code] = label, (x - x.mean()) / sd
    return subjects, scores

def null_designs(scores, n):
    # Opposite sign patterns have equal two-sided statistics.
    masks = np.arange(1 << (n - 1), dtype=np.uint32)[:, None]
    bits = (masks >> np.arange(n - 1, dtype=np.uint32)) & 1
    signs = np.column_stack((np.ones(len(bits)), 1.0 - 2.0 * bits))
    yield "group_mean", "Group story response", np.ones(n), signs, True
    rng = np.random.default_rng(SEED)
    orders = np.stack([rng.permutation(n) for _ in range(DRAWS)])
    for code, (label, x) in scores.items():
        yield code, label, x, x[orders], False

def load_roi(model, atlas, subjects):
    path = ROI / f"roi_betas_{model}_{atlas}.npz"
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["roi_ids"]
        names = archive["roi_names"]
        counts = archive["n_roi_voxels"]
        selected = archive["n_selected"]
        beta = archive["beta"].astype(float)
        stored = archive["subjects"].tolist()
        if (archive["features"].tolist() != list(FEATURES)
                or archive["top_pct"].tolist() != list(TIERS)
                or len(stored) != len(subjects) or set(stored) != set(subjects)
                or beta.shape != (len(subjects), len(FEATURES), len(TIERS), len(ids))
                or not np.isfinite(beta).all()
                or ids.ndim != 1 or len(set(ids)) != len(ids) or names.shape != ids.shape
                or counts.shape != ids.shape or selected.shape != (len(TIERS), len(ids))
                or np.any(selected < 1) or np.any(selected > counts)):
            raise ValueError(f"Unexpected cohort, ROI labels or effect layout in {path}.")
    return beta[[stored.index(s) for s in subjects]], ids, names, counts, selected

def fit(beta, x, permutations, exact):
    n, n_features, n_tiers, n_rois = beta.shape
    matrix = beta.reshape(n, -1)
    design_norm = np.linalg.norm(x)
    if exact:
        # Zero is the null mean. The uncentered norm is unchanged by sign flipping.
        y = matrix
        df = n - 1
    else:
        # The intercept is fitted by centering the dependent variable.
        y = matrix - matrix.mean(axis=0)
        df = n - 2
    norm = np.linalg.norm(y, axis=0)
    standardized = np.divide(y, norm, out=np.zeros_like(y), where=norm > 0)
    observed = x @ standardized
    abs_stat = np.maximum(0, np.abs(observed) - 32 * np.finfo(float).eps * n)

    effect = (x @ y) / design_norm**2
    residual = y - x[:, None] * effect
    se = np.linalg.norm(residual, axis=0) / (np.sqrt(df) * design_norm)
    observed_t = np.divide(effect, se, out=np.zeros_like(effect), where=se > 0)
    observed_t[(se == 0) & (effect != 0)] = np.sign(effect[(se == 0) & (effect != 0)]) * np.inf
    cutoff = t.ppf(.975, df)
    analytic_p = 2 * t.sf(np.abs(observed_t), df)

    counts = np.zeros(abs_stat.size, dtype=np.int64)
    for start in range(0, len(permutations), BATCH):
        null = permutations[start:start + BATCH] @ standardized
        np.abs(null, out=null)
        counts += (null >= abs_stat).sum(axis=0)
    add = int(not exact)
    p = (counts + add) / (len(permutations) + add)
    q = false_discovery_control(p.reshape(-1, n_rois), axis=1, method="bh")
    if not np.all((p > 0) & (p <= 1)):
        raise ValueError("Invalid permutation probabilities.")
    if np.any(q.ravel() < p - 1e-14):
        raise ValueError("Invalid BH probabilities.")
    fields = (effect, se, effect - cutoff * se, effect + cutoff * se, observed_t,
              analytic_p, p, q)
    return df, tuple(field.reshape(n_features, n_tiers, n_rois) for field in fields)

def run_model(model, subjects, designs):
    result_path = OUT / f"roi_glm_{model}.tsv"
    summary_path = OUT / f"roi_glm_summary_{model}.tsv"
    rows, summary = [], []
    for atlas in ATLASES:
        beta, ids, names, counts, selected = load_roi(model, atlas, subjects)
        for code, label, x, permutations, exact in designs:
            df, fields = fit(beta, x, permutations, exact)
            for f, feature in enumerate(FEATURES):
                for k, tier in enumerate(TIERS):
                    response = [a[f, k] for a in fields]
                    p, q = response[-2:]
                    common = {
                        "model": model, "atlas": atlas, "top_pct": tier,
                        "text_feature": feature, "contrast": code, "assessment_label": label,
                        "n_subjects": len(subjects), "df": df,
                    }
                    summary.append({
                        **common, "n_rois": len(ids), "min_p_permutation": min(p),
                        "min_q_bh": min(q),
                        "n_q_bh_le_0_05": int(np.count_nonzero(q <= ALPHA)),
                    })
                    stat_names = (
                        "estimate", "standard_error", "ci_95_low", "ci_95_high", "t",
                        "p_analytic", "p_permutation", "q_bh",
                    )
                    for j, roi_id in enumerate(ids):
                        stars = "***" if q[j] <= .001 else "**" if q[j] <= .01 else "*" if q[j] <= ALPHA else ""
                        rows.append({
                            **common, "roi_id": roi_id, "roi": names[j], "n_roi_voxels": counts[j],
                            "n_selected_voxels_per_subject": selected[k, j],
                            **dict(zip(stat_names, (x[j] for x in response), strict=True)),
                            "fdr_significance": stars,
                        })
        print(f"{model} {atlas}: {len(ids)} ROIs tested.", flush=True)
    write_tsv(result_path, rows)
    write_tsv(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    subjects, scores = load_cohort()
    designs = list(null_designs(scores, len(subjects)))
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2, user_api="blas"):
        run_model(MODEL, subjects, designs)
    (OUT / "analysis_parameters.json").write_text(json.dumps({
        "model": MODEL, "atlas": ATLAS, "top_pct": 25,
        "subjects": subjects, "assessments": list(scores), "features": list(FEATURES),
        "group_effect": "Intercept-only OLS; exact two-sided subject sign flips with 32768 opposite-sign representatives.",
        "assessment_effect": "Intercept plus score z-standardized by the complete cohort sample SD.",
        "assessment_permutations": DRAWS, "add_observed_assignment": True, "seed": SEED,
        "correction": "BH-FDR across 214 ROIs separately for each feature and group/assessment contrast.",
        "confidence_intervals": "Unadjusted analytic 95% Student t intervals.",
    }, indent=2) + "\n")
    print(OUT / f"roi_glm_{MODEL}.tsv")


if __name__ == "__main__":
    main()
