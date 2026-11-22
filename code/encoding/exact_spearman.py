"""Exact two-sided Spearman tests for complete samples of at most 16."""

from functools import lru_cache
from math import factorial, gcd

import numpy as np
from scipy.stats import rankdata


@lru_cache(maxsize=16)
def _mask_layers(n):
    layers = [[] for _ in range(n + 1)]
    for mask in range(1, 1 << n):
        layers[mask.bit_count()].append(mask)
    return layers[1:]


def _reduced_ranks(pattern):
    """Translate and rescale ranks to shrink the integer dot-product support."""
    base = pattern[0]
    scale = gcd(*(value - base for value in pattern))
    return base, scale, [(value - base) // scale for value in pattern]


@lru_cache(maxsize=512)
def _two_sided_tail(pattern_x, pattern_y):
    """Return P(|permuted rank dot product| >= d) for each integer d."""
    n = len(pattern_x)
    x_base, x_scale, x = _reduced_ranks(pattern_x)
    y_base, y_scale, y = _reduced_ranks(pattern_y)

    # A mask tracks the identities of assigned y ranks. Counting identities
    # gives the correct multiplicity even when several ranks have equal values.
    layer = {0: (0, np.ones(1, dtype=np.uint64))}
    for rank, masks in zip(x, _mask_layers(n)):
        shifts = [rank * value for value in y]
        next_layer = {}
        for mask in masks:
            low, high = None, 0
            terms = []
            remaining = mask
            while remaining:
                bit = remaining & -remaining
                previous_low, previous = layer[mask ^ bit]
                start = previous_low + shifts[bit.bit_length() - 1]
                terms.append((start, previous))
                low = start if low is None else min(low, start)
                high = max(high, start + previous.size)
                remaining ^= bit
            counts = np.zeros(high - low, dtype=np.uint64)
            for start, previous in terms:
                index = start - low
                counts[index:index + previous.size] += previous
            next_layer[mask] = (low, counts)
        layer = next_layer

    low, counts = layer[(1 << n) - 1]
    total = factorial(n)
    if int(counts.sum()) != total:
        raise RuntimeError("Incorrect total count for the exact permutation distribution.")
    offset = n * x_base * y_base + x_scale * y_base * sum(x) + x_base * y_scale * sum(y)
    numerators = np.abs(offset + x_scale * y_scale * (low + np.arange(counts.size)))
    absolute = np.zeros(int(numerators.max()) + 1, dtype=np.uint64)
    np.add.at(absolute, numerators, counts)
    return np.cumsum(absolute[::-1], dtype=np.uint64)[::-1] / total


def _canonical(pattern):
    # Reflection and swapping the two samples preserve the absolute null.
    return min(pattern, tuple(-value for value in reversed(pattern)))


def _rank_patterns(values):
    ranks = (2 * rankdata(values, axis=0) - values.shape[0] - 1).astype(np.int32)
    patterns, groups = np.unique(np.sort(ranks, axis=0).T, axis=0, return_inverse=True)
    return ranks, patterns, groups


def permutation_pvalues(x, y):
    """Return (rho, p) matrices, with rows for x columns and columns for y columns.

    Subjects are rows of x and y. Each p-value includes all permutations whose
    absolute Spearman correlation equals or exceeds the observed magnitude.
    Constant columns have undefined correlations and p-values.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or not 2 <= x.shape[0] <= 16:
        raise ValueError("x and y must be 2-D matrices with the same 2 to 16 subject rows.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Exact permutation tests require finite observations.")

    rx, x_patterns, x_groups = _rank_patterns(x)
    ry, y_patterns, y_groups = _rank_patterns(y)
    dot = rx.T @ ry
    denominator = np.sqrt(np.sum(rx ** 2, axis=0)[:, None] * np.sum(ry ** 2, axis=0))
    rho = np.full(dot.shape, np.nan)
    np.divide(dot, denominator, out=rho, where=denominator > 0)
    p = np.full(dot.shape, np.nan)
    for i, pattern_x in enumerate(x_patterns):
        if pattern_x[0] == pattern_x[-1]:
            continue
        x_columns = np.flatnonzero(x_groups == i)
        for j, pattern_y in enumerate(y_patterns):
            if pattern_y[0] == pattern_y[-1]:
                continue
            y_columns = np.flatnonzero(y_groups == j)
            indexes = np.ix_(x_columns, y_columns)
            key = sorted((_canonical(tuple(map(int, pattern_x))), _canonical(tuple(map(int, pattern_y)))))
            p[indexes] = _two_sided_tail(*key)[np.abs(dot[indexes])]
    return rho, p
