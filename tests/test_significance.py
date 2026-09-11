"""Tests for the multiple-testing machinery.

Every tool here is tested from both sides: it must detect a real effect, and it
must refuse a fabricated one. A correction that never rejects is not
conservative, it is broken, and a correction that always rejects is the problem
it was meant to solve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_panel.significance import (
    benjamini_hochberg,
    block_bootstrap_mean_pvalue,
    hansen_spa,
    stationary_bootstrap_indices,
)


# -- Benjamini-Hochberg -----------------------------------------------------


def test_bh_rejects_the_genuinely_small_p_values():
    p = np.array([0.001, 0.002, 0.2, 0.5, 0.9])
    result = benjamini_hochberg(p, q=0.05)
    assert result.rejected[:2].all()
    assert not result.rejected[2:].any()


def test_bh_rejects_almost_nothing_under_a_uniform_null():
    """With no real effects, p-values are uniform and roughly 5% would pass an
    uncorrected threshold. FDR must cut that to nearly none."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)

    uncorrected = (p < 0.05).sum()
    corrected = benjamini_hochberg(p, q=0.05).n_rejected

    assert uncorrected > 50          # the problem, in this sample
    assert corrected <= 2            # the correction


def test_bh_still_finds_real_effects_among_many_nulls():
    """A correction that never rejects is broken, not conservative."""
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(size=980), np.full(20, 1e-6)])
    assert benjamini_hochberg(p, q=0.05).n_rejected >= 20


def test_q_values_are_monotone_in_p():
    """A raw (m/i)*p is not monotone; the step-up procedure enforces it."""
    rng = np.random.default_rng(2)
    p = np.sort(rng.uniform(size=200))
    q = benjamini_hochberg(p).q_values
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all(q <= 1.0)


def test_q_values_keep_the_input_order():
    p = np.array([0.9, 0.001, 0.5])
    q = benjamini_hochberg(p).q_values
    assert q[1] < q[2] < q[0]


def test_nan_p_values_are_ignored_rather_than_counted():
    """Counting a failed test as a hypothesis would weaken the correction for
    every other test in the family."""
    result = benjamini_hochberg(np.array([0.001, np.nan, 0.9]))
    assert np.isnan(result.q_values[1])
    assert not result.rejected[1]


# -- stationary block bootstrap ---------------------------------------------

def test_resampled_indices_stay_in_range_and_have_the_right_shape():
    indices = stationary_bootstrap_indices(50, expected_block=10, n_boot=25, seed=0)
    assert indices.shape == (25, 50)
    assert indices.min() >= 0 and indices.max() < 50


def test_longer_blocks_preserve_more_serial_structure():
    """The point of blocks: a one-at-a-time bootstrap destroys the dependence
    that makes financial data hard."""
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(size=2000)).rolling(30).mean().dropna().to_numpy()

    def resampled_autocorrelation(block):
        indices = stationary_bootstrap_indices(len(series), block, n_boot=40, seed=1)
        samples = series[indices]
        return np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in samples])

    assert resampled_autocorrelation(50) > resampled_autocorrelation(1) + 0.2


def test_the_naive_bootstrap_understates_uncertainty_on_correlated_data():
    """Why blocks are not optional: with block length 1 the null distribution
    is far too narrow, which is how ordinary noise becomes significant."""
    rng = np.random.default_rng(4)
    series = pd.Series(rng.normal(size=1500)).rolling(30).mean().dropna().to_numpy()

    def spread(block):
        indices = stationary_bootstrap_indices(len(series), block, n_boot=300, seed=2)
        return series[indices].mean(axis=1).std()

    assert spread(40) > spread(1) * 1.5


# -- bootstrap p-value ------------------------------------------------------

def test_the_bootstrap_detects_a_real_positive_mean():
    rng = np.random.default_rng(5)
    assert block_bootstrap_mean_pvalue(rng.normal(loc=0.5, size=600), n_boot=400) < 0.05


def test_the_bootstrap_does_not_reject_a_zero_mean():
    rng = np.random.default_rng(6)
    assert block_bootstrap_mean_pvalue(rng.normal(loc=0.0, size=600), n_boot=400) > 0.05


def test_a_bootstrap_p_value_is_never_exactly_zero():
    """Zero would claim more precision than the number of resamples supports."""
    rng = np.random.default_rng(7)
    p = block_bootstrap_mean_pvalue(rng.normal(loc=50.0, size=600), n_boot=200)
    assert p > 0.0 and p == pytest.approx(1 / 201, rel=1e-9)


# -- Hansen's SPA -----------------------------------------------------------

def test_spa_does_not_reject_when_no_specification_is_genuinely_better():
    """The question it exists for: having tried twenty specifications, the best
    one's own p-value is meaningless, because the maximum of twenty null draws
    is large by construction."""
    rng = np.random.default_rng(8)
    differentials = rng.normal(loc=0.0, scale=1.0, size=(500, 20))
    assert hansen_spa(differentials, n_boot=300).p_value > 0.05


def test_spa_rejects_when_one_specification_really_does_beat_the_benchmark():
    rng = np.random.default_rng(9)
    differentials = rng.normal(loc=0.0, scale=1.0, size=(500, 20))
    differentials[:, 7] += 0.5                     # one genuinely better model

    result = hansen_spa(differentials, n_boot=300)
    assert result.p_value < 0.05
    assert result.best_model == 7


def test_hopeless_specifications_do_not_make_the_test_easier_to_pass():
    """Hansen's correction to White's Reality Check: a portfolio of obviously
    terrible models must not inflate the critical value."""
    rng = np.random.default_rng(10)
    base = rng.normal(loc=0.0, scale=1.0, size=(500, 5))
    base[:, 0] += 0.4                              # one modestly better model

    padded = np.concatenate([base, rng.normal(loc=-5.0, scale=1.0, size=(500, 40))], axis=1)

    tight = hansen_spa(base, n_boot=300).p_value
    padded_p = hansen_spa(padded, n_boot=300).p_value
    # Adding forty hopeless models must not rescue or wreck the verdict.
    assert abs(padded_p - tight) < 0.1


def test_too_few_observations_returns_nan_rather_than_a_verdict():
    assert np.isnan(hansen_spa(np.zeros((10, 3))).p_value)
