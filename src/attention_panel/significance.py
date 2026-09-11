"""Multiple-testing control: FDR, block bootstrap, and Hansen's SPA.

THE PROBLEM, RESTATED WITH NUMBERS (DESIGN.md section 6)

Screening 600 stocks against 30 features is 18,000 tests. At a 5% threshold
roughly 900 of them look significant *by chance alone* -- in both directions,
every one of them with a plausible story attached. This is the fastest available
route to a false result, and it does not announce itself.

The panel is the first line of defence: pooling turns N per-stock tests into one
coefficient estimated on N x T observations, which is simultaneously more
powerful and more honest. What remains after pooling is a handful of
pre-registered hypotheses, and this module is what keeps *those* honest.

THREE TOOLS, EACH FOR A DIFFERENT QUESTION

  benjamini_hochberg   "of the hypotheses I rejected, what share are false?"
  stationary_bootstrap "what does the null look like for a SERIALLY CORRELATED
                        series, where the textbook standard error lies?"
  hansen_spa           "is the BEST of N specifications better than chance,
                        given that I looked at all N?"

The middle one underpins the other two here. Financial series are strongly
autocorrelated, and a classical t-test treats every day as independent evidence.
On volatility data that overstates the effective sample size by a large factor,
so a p-value of 0.01 from a textbook test can correspond to a true probability
well above 0.05. Resampling in BLOCKS preserves the dependence instead of
destroying it.

numpy only, for the same reason as `validation`: this must run on a Python where
scipy has not shipped wheels yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# False discovery rate
# ---------------------------------------------------------------------------


@dataclass
class FDRResult:
    """Benjamini-Hochberg output, aligned with the input order."""

    q_values: np.ndarray
    rejected: np.ndarray
    threshold: float
    n_rejected: int

    def summary(self) -> str:
        return (
            f"{self.n_rejected}/{len(self.q_values)} hypotheses rejected at "
            f"q<={self.threshold:g} (Benjamini-Hochberg)"
        )


def benjamini_hochberg(p_values, q: float = 0.05) -> FDRResult:
    """Control the false discovery rate across a family of tests.

    WHY FDR AND NOT BONFERRONI. Bonferroni controls the probability of *any*
    false positive, which across 18,000 tests is so strict that nothing
    survives, including real effects. FDR controls the expected *share* of
    rejections that are false, which is the quantity a reader of an exploratory
    screen actually cares about: "of the twelve things you found, roughly how
    many are noise?"

    **q-values are what gets reported, never raw p-values.** A raw p-value from
    a screen is not interpretable, because it does not know how many other
    tests were run alongside it.
    """
    p = np.asarray(p_values, dtype="float64")
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return FDRResult(np.full(p.shape, np.nan), np.zeros(p.shape, bool), q, 0)

    order = np.argsort(np.where(finite, p, np.inf))
    ranks = np.arange(1, m + 1)
    sorted_p = p[order][:m]

    # Step-up: q_(i) = min over j>=i of (m/j) * p_(j), which enforces the
    # monotonicity a raw (m/i)*p_(i) does not have.
    raw = sorted_p * m / ranks
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    q_sorted = np.clip(monotone, 0.0, 1.0)

    q_values = np.full(p.shape, np.nan)
    q_values[order[:m]] = q_sorted

    rejected = np.zeros(p.shape, bool)
    rejected[order[:m]] = q_sorted <= q
    return FDRResult(q_values, rejected, q, int(rejected.sum()))


# ---------------------------------------------------------------------------
# Stationary block bootstrap (Politis & Romano, 1994)
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int, expected_block: float = 20.0, n_boot: int = 1000, seed: int = 0
) -> np.ndarray:
    """Resample indices in blocks of random length, wrapping at the end.

    WHY BLOCKS. Resampling observations one at a time destroys serial
    dependence, which is precisely what makes financial data hard: a naive
    bootstrap would produce a null distribution far narrower than the truth and
    turn ordinary noise into significance. Blocks of contiguous observations
    keep the local dependence intact.

    WHY *STATIONARY* BLOCKS. With fixed-length blocks the resampled series is
    not stationary -- observations near a block boundary behave differently from
    those in the middle. Politis and Romano draw each block length from a
    geometric distribution with mean `expected_block`, which restores
    stationarity and removes the sensitivity to a single arbitrary block length.

    `expected_block` should be on the order of the dependence length in the
    data. For daily volatility a few weeks is the usual choice; 20 trading days
    is the default here, matching the HAR monthly component.
    """
    if n <= 0:
        return np.zeros((n_boot, 0), dtype="int64")

    rng = np.random.default_rng(seed)
    p = 1.0 / max(expected_block, 1.0)

    starts = rng.integers(0, n, size=(n_boot, n))
    # A new block begins wherever the geometric coin comes up; otherwise the
    # previous index is advanced by one, wrapping at the end of the sample.
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True

    indices = np.empty((n_boot, n), dtype="int64")
    indices[:, 0] = starts[:, 0]
    for t in range(1, n):
        continued = (indices[:, t - 1] + 1) % n
        indices[:, t] = np.where(new_block[:, t], starts[:, t], continued)
    return indices


def block_bootstrap_mean_pvalue(
    x, expected_block: float = 20.0, n_boot: int = 1000, seed: int = 0, alternative: str = "greater"
) -> float:
    """Probability of a mean this large under a null of zero mean.

    The sample is recentred before resampling, so the bootstrap distribution is
    generated under H0 rather than around the observed mean. Skipping that step
    is a common error that makes the test reject almost always.
    """
    values = np.asarray(x, dtype="float64")
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 30:
        return float("nan")

    observed = values.mean()
    centred = values - observed  # generate the null, not the alternative
    indices = stationary_bootstrap_indices(n, expected_block, n_boot, seed)
    means = centred[indices].mean(axis=1)

    if alternative == "greater":
        # The +1 in both places is Davison-Hinkley: a bootstrap p-value of
        # exactly zero claims more precision than n_boot resamples can support.
        return float((np.sum(means >= observed) + 1) / (n_boot + 1))
    if alternative == "less":
        return float((np.sum(means <= observed) + 1) / (n_boot + 1))
    return float((np.sum(np.abs(means) >= abs(observed)) + 1) / (n_boot + 1))


# ---------------------------------------------------------------------------
# Hansen's Superior Predictive Ability test
# ---------------------------------------------------------------------------


@dataclass
class SPAResult:
    statistic: float
    p_value: float
    n_models: int
    best_model: int
    note: str = ""

    def summary(self) -> str:
        return (
            f"SPA over {self.n_models} specifications: T={self.statistic:.3f}, "
            f"p={self.p_value:.4f}, best = #{self.best_model}. {self.note}"
        )


def hansen_spa(
    loss_differentials: np.ndarray,
    expected_block: float = 20.0,
    n_boot: int = 1000,
    seed: int = 0,
) -> SPAResult:
    """Is the BEST of N specifications better than the benchmark, given that all
    N were examined?

    THE QUESTION THIS ANSWERS, AND WHY NOTHING ELSE DOES

    Having tried twenty attention specifications and reported the best one, its
    individual p-value is meaningless -- the maximum of twenty draws from a null
    distribution is large by construction. Hansen's SPA tests the maximum
    directly: the null is that NO specification beats the benchmark, and the
    null distribution is the distribution of the maximum, which already accounts
    for the search.

    Args:
        loss_differentials: an `n x k` array where column `k` holds
            `loss(benchmark) - loss(model_k)` per observation. Positive means
            the model did better.

    Hansen's consistent variant is used: specifications performing far worse
    than the benchmark are recentred to exactly zero rather than kept at their
    observed mean. Keeping them would let a portfolio of obviously terrible
    models inflate the critical value and make the test too easy to pass -- the
    known weakness of White's earlier Reality Check.
    """
    d = np.atleast_2d(np.asarray(loss_differentials, dtype="float64"))
    if d.shape[0] < d.shape[1]:
        d = d.T
    n, k = d.shape
    if n < 30 or k < 1:
        return SPAResult(float("nan"), float("nan"), k, -1, "too few observations")

    means = d.mean(axis=0)
    indices = stationary_bootstrap_indices(n, expected_block, n_boot, seed)

    # The scale of each column, taken from the bootstrap itself so that serial
    # dependence is reflected rather than assumed away.
    boot_means = d[indices].mean(axis=1)  # (n_boot, k)
    omega = boot_means.std(axis=0, ddof=1)
    omega = np.where(omega > 1e-12, omega, np.nan)

    standardised = means / omega
    statistic = float(np.nanmax(np.append(standardised, 0.0)))
    best = int(np.nanargmax(standardised)) if np.isfinite(standardised).any() else -1

    # Hansen's consistent recentring threshold: keep a model's observed mean
    # only if it is not far below zero.
    threshold = -math.sqrt(2.0 * math.log(math.log(n))) if n > 15 else 0.0
    keep = standardised >= threshold
    centre = np.where(keep, means, 0.0)

    null_stats = np.nanmax(
        np.concatenate([(boot_means - centre) / omega, np.zeros((n_boot, 1))], axis=1), axis=1
    )
    p_value = float((np.sum(null_stats >= statistic) + 1) / (n_boot + 1))

    return SPAResult(
        statistic=statistic,
        p_value=p_value,
        n_models=k,
        best_model=best,
        note="H0: no specification beats the benchmark",
    )
