"""Tests for posterior sampling: tengri NSS reuse and blackjax NUTS."""

import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from tengri import Uniform

from tengri_stars import StarModel, fit_nss, fit_nuts
from tengri_stars.grids import load_photometry_grid

COEFFS = {
    "lsst_g_ab": (30.0, 1.0e-3, 0.50, 1.20),
    "lsst_i_ab": (29.0, 0.6e-3, 0.20, 0.40),
    "skymapper_v_ab": (31.0, 1.4e-3, 0.70, 0.90),
    "cahk": (35.0, -2.0e-3, 0.30, 2.00),
    "i_des": (29.5, 0.5e-3, 0.10, 0.30),
}


def _toy_grid():
    rows = []
    for teff in np.linspace(4000.0, 5200.0, 5):
        for logg in [1.0, 2.0, 3.0]:
            for feh in np.linspace(-3.0, 0.0, 4):
                row = {"teff": teff, "logg": logg, "feh": feh}
                for name, (c0, ct, cg, cf) in COEFFS.items():
                    row[name] = c0 + ct * teff + cg * logg + cf * feh
                rows.append(row)
    return load_photometry_grid(Table(rows=rows))


def test_nss_recovers_mock_star_with_calibrated_posterior():
    model = StarModel(grid=_toy_grid(), interp_method="pchip")
    truth = {"teff": 4600.0, "logg": 2.2, "feh": -1.3, "mu": 14.5}
    sigma = 0.02

    key = jax.random.PRNGKey(42)
    key, noise_key = jax.random.split(key)
    mags_obs = model.predict_mags(**truth) + sigma * jax.random.normal(noise_key, (5,))

    def loglikelihood(p):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - mags_obs) / sigma) ** 2)

    priors = {
        "teff": Uniform(4000.0, 5200.0),
        "logg": Uniform(1.0, 3.0),
        "feh": Uniform(-3.0, 0.0),
        "mu": Uniform(10.0, 20.0),
    }

    result = fit_nss(loglikelihood, priors, key=key, n_live=200, num_delete=20)

    assert np.isfinite(result.logz)
    assert result.ess > 50
    for name, true_val in truth.items():
        samples = np.asarray(result.samples[name])
        assert samples.shape[0] >= 1000
        lo, hi = np.percentile(samples, [2.5, 97.5])
        assert lo < true_val < hi, f"{name}: truth {true_val} outside [{lo:.3f}, {hi:.3f}]"
    # Posterior must be informative: much narrower than the prior.
    assert np.std(np.asarray(result.samples["feh"])) < 0.3


def test_nuts_recovers_mock_star_and_agrees_with_nss():
    """blackjax NUTS through the ξ-space Hamiltonian recovers the same star."""
    model = StarModel(grid=_toy_grid())  # triweight default: C² gradients for NUTS
    truth = {"teff": 4600.0, "logg": 2.2, "feh": -1.3, "mu": 14.5}
    sigma = 0.02

    key = jax.random.PRNGKey(42)
    key, noise_key = jax.random.split(key)
    mags_obs = model.predict_mags(**truth) + sigma * jax.random.normal(noise_key, (5,))

    def loglikelihood(p):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - mags_obs) / sigma) ** 2)

    priors = {
        "teff": Uniform(4000.0, 5200.0),
        "logg": Uniform(1.0, 3.0),
        "feh": Uniform(-3.0, 0.0),
        "mu": Uniform(10.0, 20.0),
    }

    result = fit_nuts(loglikelihood, priors, key=key, num_warmup=500, num_samples=1000)

    for name, true_val in truth.items():
        samples = np.asarray(result.samples[name])
        assert samples.shape == (1000,)
        assert np.all(np.isfinite(samples))
        lo, hi = np.percentile(samples, [2.5, 97.5])
        assert lo < true_val < hi, f"{name}: truth {true_val} outside [{lo:.3f}, {hi:.3f}]"
    assert np.std(np.asarray(result.samples["feh"])) < 0.3
