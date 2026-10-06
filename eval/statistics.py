from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon


def holm_correction(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("invalid p-values")
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.
    for i, index in enumerate(order):
        running = max(running, (len(order)-i)*pvalues[index])
        adjusted[index] = min(1., running)
    return adjusted


def paired_comparison(baseline, candidate):
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or len(baseline) < 2:
        raise ValueError("paired vectors must have identical shape and length >=2")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("non-finite costs")
    delta = candidate-baseline
    # Round numerical noise before ranking to avoid unequal ranks from subtraction.
    delta = np.round(delta, 8)
    nonzero = delta[delta != 0]
    if len(nonzero):
        test = wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto")
        ranks = rankdata(np.abs(nonzero))
        effect = float((ranks[nonzero < 0].sum()-ranks[nonzero > 0].sum())/ranks.sum())
        p = float(test.pvalue)
    else:
        p, effect = 1., 0.
    return dict(n_instances=len(delta), pvalue=p, rank_biserial_improvement=effect,
                mean_cost_delta=float(delta.mean()), median_cost_delta=float(np.median(delta)),
                mean_relative_improvement_percent=float(np.mean(100*(baseline-candidate)/baseline)),
                wins=int((delta < 0).sum()), ties=int((delta == 0).sum()), losses=int((delta > 0).sum()))
