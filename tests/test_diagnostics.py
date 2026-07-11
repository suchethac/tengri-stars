"""Tests for the effective-sample-size estimator."""

import numpy as np

from tengri_stars import effective_sample_size


def test_ess_of_iid_samples_is_near_n():
    x = np.random.default_rng(0).normal(size=4000)

    ess = effective_sample_size(x)

    assert 0.7 * x.size < ess <= 1.05 * x.size


def test_ess_of_autocorrelated_chain_matches_theory():
    """AR(1) with lag-1 correlation rho has tau = (1+rho)/(1-rho), ESS = N/tau."""
    rho = 0.8
    rng = np.random.default_rng(1)
    n = 20_000
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(scale=np.sqrt(1 - rho**2))

    ess = effective_sample_size(x)

    expected = n * (1 - rho) / (1 + rho)  # N / tau, tau = 9 here
    assert 0.6 * expected < ess < 1.6 * expected


def test_ess_of_constant_chain_is_one():
    assert effective_sample_size(np.full(500, 3.0)) == 1.0
