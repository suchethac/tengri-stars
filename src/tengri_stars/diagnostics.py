"""Posterior diagnostics: effective sample size."""

from __future__ import annotations

import numpy as np


def effective_sample_size(chain) -> float:
    """Effective sample size of a correlated MCMC chain.

    Estimates :math:`N_{\\rm eff} = N / \\tau` where the integrated
    autocorrelation time :math:`\\tau = 1 + 2\\sum_t \\rho_t` is truncated by
    Geyer's initial-positive-sequence rule (sum consecutive autocorrelation
    pairs until the first negative pair) — the estimator Stan uses. The
    autocovariance is computed by FFT, so cost is O(N log N).

    A gradient sampler that returns 1000 draws with :math:`\\tau = 10` bought
    ~100 independent samples; a nested-sampling run's ESS instead comes from
    its importance weights (:attr:`NSSResult.ess`). Both are directly
    comparable as *ESS per second* — the sampler cost metric that survives
    changes in chain length.

    Parameters
    ----------
    chain : array_like, shape (n_samples,)
        Successive draws of one parameter from one chain.

    Returns
    -------
    float
        Effective sample size, in [1, n_samples].
    """
    x = np.asarray(chain, dtype=float).ravel()
    n = x.size
    if n < 2:
        return float(n)

    centered = x - x.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(centered, nfft)
    acov = np.fft.irfft(spec * np.conjugate(spec), nfft)[:n].real / n

    if acov[0] <= 0.0:  # constant chain: no information beyond one sample
        return 1.0

    rho = acov / acov[0]
    sum_pairs = 0.0
    t = 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0.0:
            break
        sum_pairs += pair
        t += 2

    tau = max(1.0 + 2.0 * sum_pairs, 1.0)
    return float(min(n / tau, n))


def ess_summary(samples: dict) -> dict[str, float]:
    """Per-parameter effective sample size for a posterior sample dict."""
    return {name: effective_sample_size(chain) for name, chain in samples.items()}
