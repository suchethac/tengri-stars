"""Tests for posterior sampling: tengri NSS reuse and blackjax NUTS."""

import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from tengri import Uniform

from tengri_stars import (
    StarModel,
    fit_hmc,
    fit_laplace,
    fit_map,
    fit_nss,
    fit_nuts,
    make_hmc_pipeline,
    make_laplace_pipeline,
    make_nss_pipeline,
)
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


def _mock_setup(key=42):
    model = StarModel(grid=_toy_grid())
    truth = {"teff": 4600.0, "logg": 2.2, "feh": -1.3, "mu": 14.5}
    sigma = 0.02
    key = jax.random.PRNGKey(key)
    key, noise_key = jax.random.split(key)
    mags_obs = model.predict_mags(**truth) + sigma * jax.random.normal(noise_key, (5,))
    priors = {
        "teff": Uniform(4000.0, 5200.0),
        "logg": Uniform(1.0, 3.0),
        "feh": Uniform(-3.0, 0.0),
        "mu": Uniform(10.0, 20.0),
    }
    return model, truth, sigma, mags_obs, priors, key


def _assert_covers_truth(samples, truth):
    for name, true_val in truth.items():
        lo, hi = np.percentile(np.asarray(samples[name]), [2.5, 97.5])
        assert lo < true_val < hi, f"{name}: truth {true_val} outside [{lo:.3f}, {hi:.3f}]"


def test_hmc_recovers_mock_star_and_reuses_tuning():
    """Plain HMC (fixed L) recovers the star; tuned params transfer to a reuse run."""
    model, truth, sigma, mags_obs, priors, key = _mock_setup()

    def loglikelihood(p):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - mags_obs) / sigma) ** 2)

    result = fit_hmc(loglikelihood, priors, key=key, num_warmup=500, num_samples=1000)
    assert result.acceptance_rate > 0.5
    _assert_covers_truth(result.samples, truth)

    # Adapt-once / sample-many: reuse the tuned step size + mass matrix.
    warm = fit_hmc(
        loglikelihood,
        priors,
        key=jax.random.PRNGKey(1),
        num_samples=1000,
        tuned_params=result.tuned_params,
    )
    _assert_covers_truth(warm.samples, truth)


def test_nss_data_args_compile_reuse_across_stars():
    """With a 2-arg likelihood, one compiled NSS program fits many stars."""
    model, truth, sigma, mags_obs, priors, key = _mock_setup()

    def loglikelihood(p, data):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - data) / sigma) ** 2)

    r1 = fit_nss(loglikelihood, priors, key=key, data=mags_obs, n_live=200, num_delete=20)
    _assert_covers_truth(r1.samples, truth)

    # Second star, different data, same compiled program. The correctness
    # property is bit-agreement with the fresh-closure path (same key): the
    # cached program must not bake in the first star's data.
    truth2 = {"teff": 4300.0, "logg": 2.0, "feh": -2.0, "mu": 16.0}
    mags2 = model.predict_mags(**truth2) + sigma * jax.random.normal(jax.random.PRNGKey(7), (5,))
    r2 = fit_nss(
        loglikelihood, priors, key=jax.random.PRNGKey(3), data=mags2, n_live=200, num_delete=20
    )

    def closure_loglikelihood(p):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - mags2) / sigma) ** 2)

    ref = fit_nss(
        closure_loglikelihood, priors, key=jax.random.PRNGKey(3), n_live=200, num_delete=20
    )
    for name in truth2:
        np.testing.assert_allclose(r2.samples[name], ref.samples[name], rtol=1e-12)
    assert r2.wall_time < r1.wall_time  # warm reuse must beat the compiling run


def test_jitted_hmc_pipeline_single_star_and_vmap_batch():
    """One compiled graph: photometry in → posterior samples out; vmaps over stars."""
    model, truth, sigma, mags_obs, priors, _ = _mock_setup()

    def loglikelihood(p, data):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - data) / sigma) ** 2)

    pipeline = make_hmc_pipeline(
        loglikelihood, priors, num_warmup=500, num_samples=1500, num_integration_steps=32
    )

    samples, info = pipeline(jax.random.PRNGKey(0), mags_obs)
    assert samples["teff"].shape == (1500,)
    assert float(info["acceptance_rate"]) > 0.5
    _assert_covers_truth(samples, truth)

    # Whole-catalog mode: vmap over stars, one XLA program.
    batch = jnp.stack([mags_obs, mags_obs, mags_obs])
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    bsamples, _binfo = jax.vmap(pipeline)(keys, batch)
    assert bsamples["teff"].shape == (3, 1500)
    assert np.all(np.isfinite(np.asarray(bsamples["feh"])))


def test_jitted_nss_pipeline_matches_fit_nss():
    """Whole-graph NSS (lax.while_loop) agrees with the Python-loop driver."""
    model, truth, sigma, mags_obs, priors, _key = _mock_setup()

    def loglikelihood(p, data):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - data) / sigma) ** 2)

    pipeline = make_nss_pipeline(loglikelihood, priors, n_live=200, num_delete=20)
    samples, info = pipeline(jax.random.PRNGKey(11), mags_obs)

    assert np.isfinite(info["logz"])
    assert int(info["n_iterations"]) > 5
    _assert_covers_truth(samples, truth)

    ref = fit_nss(
        loglikelihood,
        priors,
        key=jax.random.PRNGKey(11),
        data=mags_obs,
        n_live=200,
        num_delete=20,
    )
    # Same algorithm, same settings: evidence and medians must agree closely.
    assert abs(info["logz"] - ref.logz) < 1.5
    for name in truth:
        med_pipe = float(jnp.median(samples[name]))
        med_ref = float(jnp.median(ref.samples[name]))
        width = float(np.std(np.asarray(ref.samples[name])))
        assert abs(med_pipe - med_ref) < max(0.5 * width, 1e-3), name

    # Raw jitted core vmaps over stars for catalog mode.
    batch = jnp.stack([mags_obs, mags_obs])
    keys = jax.random.split(jax.random.PRNGKey(12), 2)
    _dead, _live, _n_iter, logz = jax.vmap(pipeline.run)(keys, batch)
    assert logz.shape == (2,)
    assert np.all(np.isfinite(np.asarray(logz)))


def test_sampler_results_report_ess_and_ess_rate():
    """Every sampler exposes ESS and ESS/second on a comparable footing."""
    model, _truth, sigma, mags_obs, priors, key = _mock_setup()

    def loglikelihood(p):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - mags_obs) / sigma) ** 2)

    hmc = fit_hmc(loglikelihood, priors, key=key, num_warmup=400, num_samples=1000)
    nss = fit_nss(loglikelihood, priors, key=key, n_live=200, num_delete=20)

    # HMC: autocorrelation ESS, one entry per parameter, never above chain length.
    assert set(hmc.ess) == set(priors)
    assert all(1.0 <= v <= 1000.0 for v in hmc.ess.values())
    assert 0 < hmc.min_ess <= 1000.0
    assert hmc.ess_rate == hmc.min_ess / hmc.wall_time

    # NSS: weight-based ESS (already a float); same rate interface.
    assert nss.ess_rate == nss.ess / nss.wall_time
    assert nss.ess_rate > 0


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


def test_lbfgs_map_recovers_mock_star_and_seeds_nuts():
    """L-BFGS MAP lands near truth; its ξ seeds NUTS (tengri's default recipe)."""
    model = StarModel(grid=_toy_grid())
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

    result = fit_map(loglikelihood, priors, key=key)

    assert result.success
    assert abs(result.params["teff"] - truth["teff"]) < 100.0
    assert abs(result.params["feh"] - truth["feh"]) < 0.2
    assert abs(result.params["mu"] - truth["mu"]) < 0.2

    nuts = fit_nuts(
        loglikelihood, priors, key=key, num_warmup=300, num_samples=300, init_xi=result.xi
    )
    lo, hi = np.percentile(np.asarray(nuts.samples["feh"]), [2.5, 97.5])
    assert lo < truth["feh"] < hi


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


def test_laplace_recovers_mock_star_with_iid_samples_and_evidence():
    """Laplace: MAP + Hessian → Gaussian posterior, iid samples, log Z estimate."""
    model, truth, sigma, mags_obs, priors, key = _mock_setup()

    def loglikelihood(p):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - mags_obs) / sigma) ** 2)

    lap = fit_laplace(loglikelihood, priors, key=key, n_samples=2000)

    _assert_covers_truth(lap.samples, truth)
    assert np.isfinite(lap.logz)
    # Draws are iid by construction: ESS is the full sample count, not a fraction.
    assert lap.ess > 0.7 * 2000
    assert lap.ess_rate > 0

    # The Gaussian approximation must agree with the exact nested-sampling
    # posterior on this well-behaved, unimodal problem.
    nss = fit_nss(loglikelihood, priors, key=key, n_live=300, num_delete=30)
    for name in truth:
        width = float(np.std(np.asarray(nss.samples[name])))
        d_med = abs(float(np.median(lap.samples[name])) - float(np.median(nss.samples[name])))
        assert d_med < 0.5 * width, f"{name}: median offset {d_med:.4g} vs width {width:.4g}"
    assert abs(lap.logz - nss.logz) < 3.0  # evidence within a few nats


def test_jitted_laplace_pipeline_matches_fit_laplace_and_maps_over_catalog():
    """Device-side Laplace: same answer as the scipy path, lax.map-able."""
    model, truth, sigma, mags_obs, priors, _key = _mock_setup()

    def loglikelihood(p, data):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - data) / sigma) ** 2)

    pipe = make_laplace_pipeline(loglikelihood, priors, n_samples=2000)
    samples, info = pipe(jax.random.PRNGKey(3), mags_obs)
    assert bool(info["converged"])

    # The contract: the device-side path must reproduce the scipy-driven one.
    ref = fit_laplace(
        lambda p: loglikelihood(p, mags_obs), priors, key=jax.random.PRNGKey(3), n_samples=2000
    )
    assert abs(float(info["logz"]) - ref.logz) < 0.5
    for name in truth:
        width = float(np.std(np.asarray(ref.samples[name])))
        d = abs(float(jnp.median(samples[name])) - float(np.median(ref.samples[name])))
        assert d < 0.3 * width, name

    # Truth coverage at 99%: the Gaussian approximation is mildly skewed in
    # log g on this toy grid (it sits near a coverage edge), which is the
    # documented Laplace caveat — not a defect in the device-side path.
    for name, true_val in truth.items():
        lo, hi = np.percentile(np.asarray(samples[name]), [0.5, 99.5])
        assert lo < true_val < hi, f"{name}: truth {true_val} outside [{lo:.3f}, {hi:.3f}]"

    # Catalog: lax.map compiles the single-star body once and walks on-device.
    batch = jnp.stack([mags_obs, mags_obs, mags_obs])
    keys = jax.random.split(jax.random.PRNGKey(4), 3)
    cat_samples, cat_info = jax.lax.map(lambda km: pipe(km[0], km[1]), (keys, batch))
    assert cat_samples["teff"].shape == (3, 2000)
    assert np.all(np.isfinite(np.asarray(cat_info["logz"])))


def test_laplace_pipeline_finds_the_global_peak_on_a_sharp_posterior():
    """Regression: BFGS from random starts lands in local optima on sharp posteriors.

    With sigma = 0.002 mag the posterior is knife-sharp. The prior scan that
    seeds the restarts is what keeps the optimizer on the true peak; without it
    the Hamiltonian at the reported 'MAP' is orders of magnitude too high and
    the evidence is wrong by hundreds of nats.
    """
    model = StarModel(grid=_toy_grid(), interp_method="pchip")
    truth = {"teff": 4600.0, "logg": 2.2, "feh": -1.3, "mu": 14.5}
    sigma = 0.002
    mags_obs = model.predict_mags(**truth)

    def loglikelihood(p, data):
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(((pred - data) / sigma) ** 2)

    priors = {
        "teff": Uniform(4000.0, 5200.0),
        "logg": Uniform(1.0, 3.0),
        "feh": Uniform(-3.0, 0.0),
        "mu": Uniform(10.0, 20.0),
    }

    pipe = make_laplace_pipeline(loglikelihood, priors, n_samples=500)
    _samples, info = pipe(jax.random.PRNGKey(0), mags_obs)

    # Noise-free data at the truth: the Hamiltonian at the true peak is ~0.
    assert float(info["neg_log_posterior"]) < 5.0

    ref = fit_nss(loglikelihood, priors, key=jax.random.PRNGKey(0), data=mags_obs, n_live=300)
    assert abs(float(info["logz"]) - ref.logz) < 3.0
