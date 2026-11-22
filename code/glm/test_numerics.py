"""Small synthetic checks of the released GLM statistics; no research data needed."""
from itertools import permutations, product
import unittest
import numpy as np
from scipy.stats import false_discovery_control
from aggregate import aggregate
from design import hrf_at
from second_level import fit


class Numerics(unittest.TestCase):
    def test_own_t_selection_preserves_signed_beta(self):
        rng = np.random.default_rng(12)
        betas = rng.normal(size=(3, 3, 20))
        statistic = rng.normal(size=betas.shape)
        regions = [np.arange(10), np.arange(10, 20)]
        counts = np.array([[3, 3]])  # ceil(25% of 10)
        response, offsets, selected = aggregate(betas, statistic, regions, counts)
        np.testing.assert_array_equal(offsets, [0, 3, 6])
        for s in range(3):
            for f in range(3):
                for r, voxels in enumerate(regions):
                    top = voxels[np.argsort(-abs(statistic[s, f, voxels]), kind="stable")[:3]]
                    np.testing.assert_equal(response[s, f, 0, r], betas[s, f, top].mean())
                    np.testing.assert_array_equal(selected[s, f, offsets[r]:offsets[r+1]], top)

    def test_exact_signflip_and_roi_fdr(self):
        rng = np.random.default_rng(59)
        beta = rng.normal(size=(6, 3, 1, 4))
        representatives = np.array([(1, *s) for s in product((-1, 1), repeat=5)])
        df, result = fit(beta, np.ones(6), representatives, True)
        self.assertEqual(df, 5)
        y = beta.reshape(6, -1)
        unit = y / np.linalg.norm(y, axis=0)
        full = np.array(list(product((-1, 1), repeat=6))) @ unit
        threshold = np.maximum(0, abs(unit.sum(axis=0)) - 32*np.finfo(float).eps*6)
        expected_p = np.mean(abs(full) >= threshold, axis=0).reshape(3, 1, 4)
        np.testing.assert_array_equal(result[-2], expected_p)
        np.testing.assert_array_equal(result[-1], false_discovery_control(expected_p, axis=2))
        np.testing.assert_allclose(result[0], beta.mean(axis=0), atol=1e-15)

    def test_score_permutation_plus_observed_and_slope(self):
        rng = np.random.default_rng(6)
        beta = rng.normal(size=(6, 3, 1, 4))
        raw = np.array([1., 3., 2., 4., 7., 5.])
        x = (raw - raw.mean()) / raw.std(ddof=1)
        # Enumerate all permutations for this small synthetic example, reserving
        # identity as the observed assignment handled by the +1 correction.
        orders = np.array(list(permutations(range(6))))
        df, result = fit(beta, x, x[orders[1:]], False)
        self.assertEqual(df, 4)
        y = beta.reshape(6, -1); centered = y - y.mean(axis=0)
        unit = centered / np.linalg.norm(centered, axis=0)
        threshold = np.maximum(0, abs(x @ unit) - 32*np.finfo(float).eps*6)
        expected_p = np.mean(abs(x[orders] @ unit) >= threshold, axis=0).reshape(3, 1, 4)
        np.testing.assert_array_equal(result[-2], expected_p)
        ols = np.linalg.lstsq(np.column_stack((np.ones(6), x)), y, rcond=None)[0][1]
        np.testing.assert_allclose(result[0].ravel(), ols, atol=1e-15)

    def test_hrf_has_no_future_response(self):
        response = hrf_at(np.array([0., 1., 5., 40.]), np.array([2.]))[:, 0]
        np.testing.assert_array_equal(response[:2], 0)
        self.assertGreater(response[2], 0)
        self.assertEqual(response[3], 0)


if __name__ == "__main__":
    unittest.main()
