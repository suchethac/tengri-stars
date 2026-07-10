"""Per-star sampler timings on the real TSLTE grid (4 parameters, 21 bands).

Run from the repo root:
    PYTHONPATH=src JAX_PLATFORMS=cpu python bench/benchmark_samplers.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from tengri import Uniform

from tengri_stars import StarModel, fit_hmc, fit_map, fit_nss, fit_nuts, load_photometry_grid

jax.config.update("jax_enable_x64", True)

N_WARM = 5
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


def bench(name, cold_fn, warm_fn):
    t0 = time.time()
    cold_out = cold_fn(stars[0])
    cold = time.time() - t0
    warms = []
    for mags in stars[1:]:
        t0 = time.time()
        warm_fn(mags, cold_out)
        warms.append(time.time() - t0)
    print(f"{name:28s} cold {cold:6.1f} s   warm median {np.median(warms):6.2f} s/star")


key = jax.random.PRNGKey(0)

bench(
    "MAP (L-BFGS, 4 restarts)",
    lambda m: fit_map(closure(m), priors, key=key),
    lambda m, _c: fit_map(closure(m), priors, key=key),
)
bench(
    "NSS (data-args cache)",
    lambda m: fit_nss(loglikelihood, priors, key=key, data=m, n_live=400, num_delete=40),
    lambda m, _c: fit_nss(loglikelihood, priors, key=key, data=m, n_live=400, num_delete=40),
)
bench(
    "NUTS (adapt every star)",
    lambda m: fit_nuts(closure(m), priors, key=key, num_warmup=800, num_samples=1500),
    lambda m, _c: fit_nuts(closure(m), priors, key=key, num_warmup=800, num_samples=1500),
)
bench(
    "NUTS (adapt-once reuse)",
    lambda m: fit_nuts(closure(m), priors, key=key, num_warmup=800, num_samples=1500),
    lambda m, c: fit_nuts(
        closure(m), priors, key=key, num_samples=1500, tuned_params=c.tuned_params
    ),
)
bench(
    "HMC L=32 (adapt-once reuse)",
    lambda m: fit_hmc(closure(m), priors, key=key, num_warmup=500, num_samples=1500),
    lambda m, c: fit_hmc(
        closure(m), priors, key=key, num_samples=1500, tuned_params=c.tuned_params
    ),
)
