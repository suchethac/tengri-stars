"""Per-star sampler cost on the real TSLTE grid (4 parameters, 21 bands).

Reports wall time *and* effective sample size — a sampler is only as fast as
the independent samples it produces per second, so ESS/s is the metric that
survives changes in chain length.

Run from the repo root:
    PYTHONPATH=src JAX_PLATFORMS=cpu python bench/benchmark_samplers.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time

import jax
import jax.numpy as jnp
import numpy as np
from tengri import Uniform

from tengri_stars import (
    StarModel,
    effective_sample_size,
    fit_hmc,
    fit_map,
    fit_nss,
    fit_nuts,
    load_photometry_grid,
    make_hmc_pipeline,
    make_nss_pipeline,
)

jax.config.update("jax_enable_x64", True)

N_WARM = 3
SIG_MAG = 0.02

grid = load_photometry_grid("data/TSLTE_combined_photometry.fits", fill="nearest")
model = StarModel(grid=grid, interp_method="pchip")
priors = {
    "teff": Uniform(float(grid.axes[0][0]), float(grid.axes[0][-1])),
    "logg": Uniform(float(grid.axes[1][0]), float(grid.axes[1][-1])),
    "feh": Uniform(float(grid.axes[2][0]), float(grid.axes[2][-1])),
    "mu": Uniform(-28.0, -8.0),
}

rng = np.random.default_rng(5)
stars = []
for _ in range(N_WARM + 1):
    truth = {
        "teff": rng.uniform(4000.0, 6500.0),
        "logg": rng.uniform(0.5, 4.5),
        "feh": rng.uniform(-3.5, -0.5),
        "mu": -18.0,
    }
    clean = np.asarray(model.predict_mags(**truth))
    stars.append(jnp.asarray(clean + rng.normal(0.0, SIG_MAG, clean.size)))


def loglikelihood(p, data):
    pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
    return -0.5 * jnp.sum(((pred - data) / SIG_MAG) ** 2)


def closure(data):
    def loglik(p):
        return loglikelihood(p, data)

    return loglik


key = jax.random.PRNGKey(0)
rows = []


def bench(name, cold_fn, warm_fn, ess_fn):
    """Time a cold run, then N_WARM warm runs; report median time and ESS/s."""
    t0 = time.time()
    cold_out = cold_fn(stars[0])
    cold = time.time() - t0

    times, esss = [], []
    for mags in stars[1:]:
        t0 = time.time()
        out = warm_fn(mags, cold_out)
        times.append(time.time() - t0)
        esss.append(ess_fn(out))

    t_med = float(np.median(times))
    ess_med = float(np.median(esss)) if esss[0] is not None else float("nan")
    rate = ess_med / t_med if ess_med == ess_med else float("nan")
    rows.append((name, cold, t_med, ess_med, rate))
    print(
        f"{name:30s} cold {cold:6.1f} s   warm {t_med:6.2f} s   "
        f"ESS {ess_med:7.0f}   {rate:7.1f} ESS/s"
    )


bench(
    "MAP (L-BFGS, 4 restarts)",
    lambda m: fit_map(closure(m), priors, key=key),
    lambda m, _c: fit_map(closure(m), priors, key=key),
    lambda _out: None,  # point estimate: no posterior samples
)
bench(
    "NSS (data-args cache)",
    lambda m: fit_nss(loglikelihood, priors, key=key, data=m, n_live=400, num_delete=40),
    lambda m, _c: fit_nss(loglikelihood, priors, key=key, data=m, n_live=400, num_delete=40),
    lambda out: out.ess,
)
bench(
    "NUTS (adapt every star)",
    lambda m: fit_nuts(closure(m), priors, key=key, num_warmup=800, num_samples=1500),
    lambda m, _c: fit_nuts(closure(m), priors, key=key, num_warmup=800, num_samples=1500),
    lambda out: out.min_ess,
)
bench(
    "NUTS (adapt-once reuse)",
    lambda m: fit_nuts(closure(m), priors, key=key, num_warmup=800, num_samples=1500),
    lambda m, c: fit_nuts(
        closure(m), priors, key=key, num_samples=1500, tuned_params=c.tuned_params
    ),
    lambda out: out.min_ess,
)
bench(
    "HMC L=32 (adapt-once reuse)",
    lambda m: fit_hmc(closure(m), priors, key=key, num_warmup=500, num_samples=1500),
    lambda m, c: fit_hmc(
        closure(m), priors, key=key, num_samples=1500, tuned_params=c.tuned_params
    ),
    lambda out: out.min_ess,
)

hmc_pipe = make_hmc_pipeline(loglikelihood, priors, num_warmup=500, num_samples=1000)
bench(
    "jitted HMC pipeline",
    lambda m: jax.block_until_ready(hmc_pipe(key, m)),
    lambda m, _c: jax.block_until_ready(hmc_pipe(key, m)),
    lambda out: min(effective_sample_size(v) for v in out[0].values()),
)

nss_pipe = make_nss_pipeline(loglikelihood, priors, n_live=400, num_delete=40)
bench(
    "jitted NSS pipeline",
    lambda m: nss_pipe(key, m),
    lambda m, _c: nss_pipe(key, m),
    lambda out: out[1]["ess"],
)

print("\nESS/s is the cost metric: a chain of 1500 draws with an autocorrelation")
print("time of 15 bought ~100 independent samples, no matter how fast it ran.")
