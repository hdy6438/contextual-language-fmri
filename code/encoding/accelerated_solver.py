"""CPU nested LOSO ridge with bounded single-precision validation losses."""

import operator

import numpy as np


def _eigenbasis(gram, n_train, alphas):
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    denominator = eigenvalues[:, None] + n_train * alphas
    if not np.all(denominator > 0):
        raise ValueError("A penalized training Gram matrix is not positive definite.")
    return np.ascontiguousarray(eigenvectors.T), np.reciprocal(denominator)


def _validation_factor(gram):
    """Factor the numerical range and measure the discarded residual."""
    eigvals, eigvecs = np.linalg.eigh(gram)
    keep = eigvals > eigvals[-1] * max(gram.shape) * np.finfo(np.float64).eps
    factor = np.sqrt(eigvals[keep, None]) * eigvecs[:, keep].T
    factor_norm = np.linalg.norm(factor)
    residual = np.linalg.norm(gram - factor.T @ factor)
    return np.ascontiguousarray(factor, dtype=np.float32), factor_norm, residual


def _select_block(folds, outer_cross, cross, block, n_alphas):
    """Use float64 if rounded validation losses could change the winner."""
    n_features, n_voxels = outer_cross.shape
    loss = np.zeros((n_alphas, n_voxels))
    error = np.zeros_like(loss)
    single = np.finfo(np.float32)
    # The matmul bound includes casts of both operands and one multiply.
    # The extra factor of two also covers the float64 reference and the
    # float64 accumulation of the fold losses.
    operations = n_features + 2 * len(folds) + 12
    nu = operations * (single.eps / 2)
    gamma = 2 * nu / (1 - nu)
    absolute_error = 4 * single.tiny * (operations**2 + 1)
    fit = np.empty_like(outer_cross, dtype=np.float32)
    bins = np.arange(0, n_features, max(1, n_features // 32))
    for validation, basis, shrink, gram, factor, factor_norm, residual in folds:
        heldout = np.ascontiguousarray(cross[validation, :, block], dtype=np.float64)
        train_projection = basis @ (outer_cross - heldout)
        heldout_projection = basis @ heldout
        linear = shrink.T @ (train_projection * heldout_projection)
        train_single = train_projection.astype(np.float32)
        shrink_single = shrink.astype(np.float32)
        # Within each bin the first coefficient is the largest because
        # the eigenvalues are increasing. Bound the fitted norm for all
        # alphas with one pass through the double-precision projection.
        bin_energy = np.add.reduceat(train_projection**2, bins, axis=0)
        norm_squared = (shrink[bins]**2).T @ bin_energy
        norm = np.sqrt(norm_squared)
        projected_error = gamma * factor_norm * norm + absolute_error * (
            1 + factor_norm
        ) * (1 + norm)
        product = np.empty((len(factor), n_voxels), dtype=np.float32)
        for candidate, penalty in enumerate(shrink_single.T):
            np.multiply(train_single, penalty[:, None], out=fit)
            np.matmul(factor, fit, out=product)
            quadratic = np.einsum("pv,pv->v", product, product).astype(np.float64)
            loss[candidate] += quadratic - 2.0 * linear[candidate]
            difference = projected_error[candidate]
            # ||Hf||^2 differs from the computed norm by at most
            # delta(2 ||fl(Hf)|| + delta), before the final dot roundoff.
            error[candidate] += (
                difference * (2.0 * np.sqrt(quadratic) + difference)
                + gamma * quadratic + residual * norm_squared[candidate]
            )
        # The exact reference evaluates the linear term in another order.
        # Include its double-precision rounding bound even under cancellation.
        double_eps = np.finfo(np.float64).eps
        error += 8 * operations * double_eps * (
            norm * np.linalg.norm(heldout_projection, axis=0)
            + (factor_norm**2 + residual) * norm_squared
        )

    best = np.argmin(loss, axis=0)
    columns = np.arange(n_voxels)
    lower = loss - error
    lower[best, columns] = np.inf
    certified = (
        np.isfinite(loss).all(axis=0) & np.isfinite(error).all(axis=0)
        & (loss[best, columns] + error[best, columns] < lower.min(axis=0))
    )
    uncertain = np.flatnonzero(~certified)
    if uncertain.size:
        exact = np.zeros((n_alphas, uncertain.size))
        selected = block.start + uncertain
        train_outer = np.ascontiguousarray(outer_cross[:, uncertain])
        for validation, basis, shrink, gram, _, _, _ in folds:
            heldout = np.ascontiguousarray(cross[validation][:, selected], dtype=np.float64)
            train = basis @ (train_outer - heldout)
            heldout = basis @ heldout
            for candidate, penalty in enumerate(shrink.T):
                fitted = train * penalty[:, None]
                quadratic = np.einsum("pv,pv->v", fitted, gram @ fitted)
                linear = np.einsum("pv,pv->v", fitted, heldout)
                exact[candidate] += quadratic - 2.0 * linear
        best[uncertain] = np.argmin(exact, axis=0)
    return best, uncertain.size

def fit_nested_loso(
    gram, cross, n_good, outer_index, complete_indices, alphas, *, block_size=4096, diagnostics=None
):
    """Fit the same ridge objective as the float64 solver with guarded alpha selection.

    Matrices, eigenbases, BOLD cross-products and the final coefficients use
    the original precision. The validation Gram products run in float32,
    and overlapping loss intervals trigger float64 validation for that voxel.
    Diagnostics, when given a dict, records the number of such voxels.
    """
    gram = np.asarray(gram, dtype=np.float64)
    cross = np.asarray(cross)
    n_good = np.asarray(n_good, dtype=np.float64)
    complete = np.asarray(complete_indices)
    alphas = np.asarray(alphas, dtype=np.float64)
    outer_index = operator.index(outer_index)
    block_size = operator.index(block_size)
    if gram.ndim != 3 or gram.shape[1] != gram.shape[2]:
        raise ValueError("gram must have shape (subjects, features, features).")
    n_subjects, n_features, _ = gram.shape
    if cross.ndim != 3 or cross.shape[:2] != (n_subjects, n_features):
        raise ValueError("cross must have shape (subjects, features, voxels).")
    if not n_features or not cross.shape[2] or block_size < 1:
        raise ValueError("At least one feature, voxel, and voxel per block are required.")
    if n_good.shape != (n_subjects,) or not np.all(np.isfinite(n_good) & (n_good > 0)):
        raise ValueError("n_good must contain one positive finite count per subject.")
    if not np.isfinite(gram).all() or not np.allclose(gram, gram.swapaxes(1, 2)):
        raise ValueError("gram must contain finite symmetric matrices.")
    if alphas.ndim != 1 or not alphas.size or not np.all(np.isfinite(alphas) & (alphas > 0)):
        raise ValueError("alphas must be a nonempty vector of positive finite values.")
    if (
        complete.ndim != 1
        or complete.dtype.kind not in "iu"
        or complete.size < 2
        or np.unique(complete).size != complete.size
        or np.any((complete < 0) | (complete >= n_subjects))
        or outer_index not in complete
        or n_subjects < 3
    ):
        raise ValueError("Complete subjects must be unique valid indices and include the outer test subject.")
    alpha_order = np.argsort(-alphas, kind="stable")
    ordered_alphas = alphas[alpha_order]
    outer_train = np.flatnonzero(np.arange(n_subjects) != outer_index)
    outer_gram = gram[outer_train].sum(axis=0)
    outer_n = n_good[outer_train].sum()
    outer_basis, outer_shrink = _eigenbasis(outer_gram, outer_n, ordered_alphas)
    inner_folds = []
    for validation in complete[complete != outer_index]:
        basis, shrink = _eigenbasis(
            outer_gram - gram[validation], outer_n - n_good[validation], ordered_alphas
        )
        validation_gram = basis @ gram[validation] @ basis.T
        inner_folds.append((validation, basis, shrink, validation_gram,
                            *_validation_factor(validation_gram)))

    n_voxels = cross.shape[2]
    beta = np.empty((n_features, n_voxels), dtype=np.float32)
    alpha_index = np.empty(n_voxels, dtype=np.intp)
    n_uncertain = 0
    for start in range(0, n_voxels, block_size):
        block = slice(start, min(start + block_size, n_voxels))
        outer_cross = np.zeros((n_features, block.stop - block.start))
        for subject in outer_train:
            outer_cross += cross[subject, :, block]
        if not np.isfinite(outer_cross).all():
            raise ValueError("cross must be finite for every training subject.")
        best, uncertain = _select_block(inner_folds, outer_cross, cross, block, len(alphas))
        n_uncertain += uncertain
        beta[:, block] = outer_basis.T @ ((outer_basis @ outer_cross) * outer_shrink[:, best])
        alpha_index[block] = alpha_order[best]
    if diagnostics is not None:
        diagnostics["float64_fallback_voxels"] = n_uncertain
    return beta, alpha_index
