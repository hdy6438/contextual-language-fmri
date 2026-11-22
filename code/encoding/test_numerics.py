"""Fast numerical checks using synthetic data only; run directly with Python."""
import itertools
import numpy as np
from scipy.stats import rankdata
from accelerated_solver import fit_nested_loso
from exact_spearman import permutation_pvalues
from roi_analysis import subject_top_metrics
from fit_encoding import score


def test_exact_spearman():
    x = np.array([1, 1, 2, 3, 4, 5.])
    y = np.array([2, 3, 1, 4, 4, 6.])
    rho, p = permutation_pvalues(x[:, None], y[:, None])
    rx, ry = rankdata(x), rankdata(y)
    observed = np.corrcoef(rx, ry)[0, 1]
    null = [np.corrcoef(rx, np.array(perm))[0, 1] for perm in itertools.permutations(ry)]
    expected = np.mean(np.abs(null) >= abs(observed) - 1e-12)
    np.testing.assert_allclose(rho[0, 0], observed, atol=1e-14)
    np.testing.assert_allclose(p[0, 0], expected, atol=1e-14)


def test_nested_ridge():
    rng = np.random.default_rng(2026)
    x = [rng.normal(size=(15 + i, 4)) for i in range(5)]
    y = [rng.normal(size=(len(a), 7)) for a in x]
    grams = np.array([a.T @ a for a in x])
    cross = np.array([a.T @ b for a, b in zip(x, y)], dtype=np.float32)
    counts = np.array([len(a) for a in x])
    alphas = np.array([.001, .1, 1., 100.])
    complete, outer = np.array([0, 1, 2, 3]), 1
    beta, best = fit_nested_loso(grams, cross, counts, outer, complete, alphas)
    train = [i for i in range(5) if i != outer]
    loss = np.zeros((len(alphas), 7))
    for heldout in complete[complete != outer]:
        inner = [i for i in train if i != heldout]
        for ai, alpha in enumerate(alphas):
            fitted = np.linalg.solve(grams[inner].sum(axis=0) + alpha * counts[inner].sum() * np.eye(4),
                                     cross[inner].astype(float).sum(axis=0))
            loss[ai] += np.einsum('pv,pv->v', fitted, grams[heldout] @ fitted) - 2 * np.einsum('pv,pv->v', fitted, cross[heldout])
    expected_best = loss.argmin(axis=0)
    np.testing.assert_array_equal(best, expected_best)
    expected = np.column_stack([np.linalg.solve(grams[train].sum(axis=0) + alphas[best[v]] * counts[train].sum() * np.eye(4),
                                cross[train, :, v].astype(float).sum(axis=0)) for v in range(7)])
    np.testing.assert_allclose(beta, expected, atol=1e-7, rtol=1e-6)
    changed = cross.copy()
    changed[outer] *= 10000
    beta2, best2 = fit_nested_loso(grams, changed, counts, outer, complete, alphas)
    np.testing.assert_array_equal(beta, beta2)
    np.testing.assert_array_equal(best, best2)


def test_scoring_and_selection():
    rng = np.random.default_rng(8)
    x, y, beta = rng.normal(size=(31, 5)), rng.normal(size=(31, 7)), rng.normal(size=(5, 7))
    pred = x @ beta
    r, r2 = score(y, x, beta, x.T @ y, x.T @ x, len(x))
    direct_r = [np.corrcoef(y[:, v], pred[:, v])[0, 1] for v in range(y.shape[1])]
    direct_r2 = 1 - ((y - pred) ** 2).sum(axis=0) / (y ** 2).sum(axis=0)
    np.testing.assert_allclose(r, direct_r, atol=1e-12)
    np.testing.assert_allclose(r2, direct_r2, atol=1e-12)
    values = np.array([[.1, -2.], [.8, -.2], [.7, -.3], [.2, 9.], [.9, -.4]])
    selected = subject_top_metrics(values, [np.arange(4), np.array([4])], np.array([1, 1]))
    np.testing.assert_array_equal(selected, [[.8, -.2], [.9, -.4]])


if __name__ == '__main__':
    test_exact_spearman()
    test_nested_ridge()
    test_scoring_and_selection()
    print('PASS: exact tied-rank permutations, nested ridge, held-out exclusion, scoring and shared-voxel selection.')
