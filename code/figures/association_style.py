"""Shared joint-plot style for measured child-level brain--behaviour data.

Style-only adaptation of the user's reference image. Original inference remains
in the supplied result tables. Bands describe the OLS conditional mean; they are
not confidence intervals for Spearman rho or multiplicity-adjusted intervals.
"""
import numpy as np
from scipy.stats import gaussian_kde, linregress, t

BLUE = '#0072B2'
ORANGE = '#E69F00'
INK = '#30343B'


def joint_association(fig, rect_mm, xdata, ydata, xlabel, ylabel,
                      figure_height_mm, figure_width_mm=183):
    """Draw all observations, OLS mean and pointwise 95% CI, and marginal KDEs.

    rect_mm gives x, top, width, height of the scatter plot in mm. Marginals
    extend 8.5 mm above and to its right (7 mm panel plus 1.5 mm gap).
    """
    x = np.asarray(xdata, dtype=float)
    y = np.asarray(ydata, dtype=float)
    assert x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) > 2
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert len(np.unique(x)) > 1 and len(np.unique(y)) > 1
    left, top, width, height = rect_mm

    def axes(a, b, c, d):
        return fig.add_axes([a / figure_width_mm,
                             1 - (b + d) / figure_height_mm,
                             c / figure_width_mm, d / figure_height_mm])

    main = axes(left, top, width, height)
    marginal_x = axes(left, top - 8.5, width, 7)
    marginal_y = axes(left + width + 1.5, top, 7, height)
    fit = linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 240)
    fitted = fit.intercept + fit.slope * xx
    residual = y - fit.intercept - fit.slope * x
    residual_variance = np.dot(residual, residual) / (len(x) - 2)
    sxx = np.sum((x - x.mean()) ** 2)
    half_width = t.ppf(.975, len(x) - 2) * np.sqrt(
        residual_variance * (1 / len(x) + (xx - x.mean()) ** 2 / sxx))
    low, high = fitted - half_width, fitted + half_width
    main.fill_between(xx, low, high, color=BLUE, alpha=.14, linewidth=0, zorder=1)
    main.plot(xx, fitted, color=BLUE, lw=1.15, zorder=2)
    main.scatter(x, y, s=19, color=INK, edgecolors='white', linewidth=.45,
                 zorder=3, clip_on=True)
    # Limits cover every observation and the entire displayed mean interval.
    xpad = np.ptp(x) * .09
    ymin, ymax = min(y.min(), low.min()), max(y.max(), high.max())
    ypad = (ymax - ymin) * .08
    main.set_xlim(x.min() - xpad, x.max() + xpad)
    main.set_ylim(ymin - ypad, ymax + ypad)
    main.set_xlabel(xlabel, labelpad=3, fontsize=6.7)
    main.set_ylabel(ylabel, labelpad=4, fontsize=6.7)
    main.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        main.spines[side].set_color('#81858B')
        main.spines[side].set_linewidth(.55)
    main.tick_params(length=2.2, pad=2, labelsize=6.1, color='#81858B')
    main.set_axisbelow(True)
    main.grid(True, color='#E9EBEE', linewidth=.4)

    kde_info = {}
    for name, data, axis, color in [('x', x, marginal_x, ORANGE),
                                    ('y', y, marginal_y, BLUE)]:
        kde = gaussian_kde(data, bw_method='scott')
        grid = np.linspace(data.min(), data.max(), 240)
        density = kde(grid)
        assert np.isfinite(density).all() and (density >= 0).all()
        peak = density.max()
        if name == 'x':
            axis.fill_between(grid, 0, density, color=color, alpha=.8, linewidth=0)
            axis.plot(grid, density, color=color, lw=.8)
            axis.vlines(data, -peak * .16, -peak * .045, color=INK, lw=.55)
            axis.set_xlim(main.get_xlim())
            axis.set_ylim(-peak * .19, peak * 1.08)
        else:
            axis.fill_betweenx(grid, 0, density, color=color, alpha=.75, linewidth=0)
            axis.plot(density, grid, color=color, lw=.8)
            axis.hlines(data, -peak * .16, -peak * .045, color=INK, lw=.55)
            axis.set_ylim(main.get_ylim())
            axis.set_xlim(-peak * .19, peak * 1.08)
        axis.axis('off')
        kde_info[name] = dict(method='Gaussian KDE', bandwidth='Scott',
                              bandwidth_factor=float(kde.factor),
                              display_domain='Observed minimum to maximum',
                              n_observations=len(data), n_unique=len(np.unique(data)))

    provenance = dict(n=len(x), observations='All measured children; no jitter or exclusions',
                      line='OLS fit on displayed raw coordinates',
                      band='Pointwise 95% Student-t interval for the OLS mean response',
                      degrees_of_freedom=len(x)-2,
                      band_role='Descriptive; original permutation p and ROI-wise BH q unchanged',
                      kde=kde_info, x_limits=list(main.get_xlim()), y_limits=list(main.get_ylim()),
                      slope_raw=float(fit.slope), intercept=float(fit.intercept))
    return dict(main=main, axes=[main, marginal_x, marginal_y], provenance=provenance)
