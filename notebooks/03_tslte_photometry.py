# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stellar parameters from the TSLTE pre-integrated photometry grid
#
# The real thing: `TSLTE_combined_photometry.fits` (A. Chiti's Turbospectrum LTE
# synthetic photometry — the MAGIC grid) loaded into tengri-stars, a mock star
# drawn from it, and the full posterior workflow of notebook 01 — NSS and NUTS
# side by side — on pre-filter-integrated magnitudes. No wavelength integral
# anywhere: fit-time photometry is one differentiable grid lookup.
#
# Get the grid (from a Sherlock login):
# ```bash
# scp <sunetid>@login.sherlock.stanford.edu:\
#     /oak/stanford/orgs/kipac/users/achiti/grid/TSLTE_combined_photometry.fits \
#     ~/Projects/tengri-stars/data/
# ```

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence XLA/PJRT C++ chatter

import time
from pathlib import Path

# Notebook kernels launch in notebooks/; scripts run from the repo root.
DATA = (
    Path("data") if (Path("data") / "TSLTE_combined_photometry.fits").exists() else Path("../data")
)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tengri import Uniform

from tengri_stars import (
    StarModel,
    fit_nss,
    fit_nuts,
    load_photometry_grid,
    make_laplace_pipeline,
    overlay_corner,
)

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(17)

GRID_PATH = DATA / "TSLTE_combined_photometry.fits"

# %% [markdown]
# ## 1. Load the grid
#
# The loader prefers `averaged=True` rows on duplicate nodes and rejects the
# duplicated label columns (`Teff_1`, `[Fe/H]_1`, ...). Synthetic grids are
# rarely perfect boxes, so fall back to nearest-neighbor hole filling and keep
# the coverage mask.

# %%
try:
    grid = load_photometry_grid(GRID_PATH)
except ValueError as err:
    print(f"strict load: {err}\n→ retrying with fill='nearest'")
    grid = load_photometry_grid(GRID_PATH, fill="nearest")

n_teff, n_logg, n_feh = (a.shape[0] for a in grid.axes)
covered = float(grid.coverage.mean())
print(f"axes: {n_teff} Teff × {n_logg} logg × {n_feh} [Fe/H] nodes, {covered:.1%} covered")
print(f"Teff  [{float(grid.axes[0][0]):7.0f}, {float(grid.axes[0][-1]):7.0f}] K")
print(f"logg  [{float(grid.axes[1][0]):7.2f}, {float(grid.axes[1][-1]):7.2f}]")
print(f"[Fe/H][{float(grid.axes[2][0]):7.2f}, {float(grid.axes[2][-1]):7.2f}]")
print(f"{len(grid.filter_names)} filters: {grid.filter_names}")

# %% [markdown]
# ## 2. Coverage map
#
# Fraction of [Fe/H] nodes present per (Teff, log g) cell — the white regions are
# where `fill='nearest'` extrapolated and posteriors should not be trusted.

# %%
frac = np.asarray(grid.coverage).mean(axis=2)
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.pcolormesh(
    np.asarray(grid.axes[0]),
    np.asarray(grid.axes[1]),
    frac.T,
    cmap="viridis",
    vmin=0,
    vmax=1,
    shading="nearest",
)
fig.colorbar(im, label="fraction of [Fe/H] nodes covered")
ax.set(xlabel=r"$T_{\rm eff}$ [K]", ylabel=r"$\log g$")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Filters and a mock star
#
# Edit `FILTERS_USE` to the survey combination you care about (default: all
# columns in the file). The mock star sits at interior grid values; magnitudes
# get survey-like noise and the dilution μ shifts them to apparent scale.

# %%
FILTERS_USE = list(grid.filter_names)  # e.g. ["lsst_g_ab", "skymapper_v_ab", "cahk", ...]
fidx = jnp.asarray([grid.filter_names.index(f) for f in FILTERS_USE])

model = StarModel(grid=grid, interp_method="pchip")

TRUTH = {
    "teff": float(np.median(np.asarray(grid.axes[0]))),
    "logg": float(np.asarray(grid.axes[1])[n_logg // 3]),
    "feh": float(np.asarray(grid.axes[2])[n_feh // 3]),
    "mu": -18.0,  # grid zero-point → apparent-mag offset, absorbed by the fit
}
SIG_MAG = 0.02

mags_clean = model.predict_mags(**TRUTH)[fidx]
mags_obs = jnp.asarray(np.asarray(mags_clean) + rng.normal(0.0, SIG_MAG, len(FILTERS_USE)))
print("mock star:", {k: round(v, 2) for k, v in TRUTH.items()})
print("observed mags:", dict(zip(FILTERS_USE, np.round(np.asarray(mags_obs), 2))))

# %% [markdown]
# ## 4. Fit: NSS and NUTS on the same posterior

# %%
lo_hi = [(float(a[0]), float(a[-1])) for a in grid.axes]
priors = {
    "teff": Uniform(*lo_hi[0]),
    "logg": Uniform(*lo_hi[1]),
    "feh": Uniform(*lo_hi[2]),
    "mu": Uniform(TRUTH["mu"] - 10.0, TRUTH["mu"] + 10.0),
}


def loglikelihood(p):
    pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])[fidx]
    return -0.5 * jnp.sum(((pred - mags_obs) / SIG_MAG) ** 2)


t0 = time.time()
nss = fit_nss(loglikelihood, priors, key=jax.random.PRNGKey(2), n_live=400, num_delete=40)
print(f"NSS:  {time.time() - t0:5.1f} s  log Z = {nss.logz:.1f}  ESS = {nss.ess:.0f}")

t0 = time.time()
nuts = fit_nuts(loglikelihood, priors, key=jax.random.PRNGKey(3), num_warmup=800, num_samples=1500)
print(
    f"NUTS: {time.time() - t0:5.1f} s  acceptance {nuts.acceptance_rate:.2f}, "
    f"{nuts.num_divergent} divergences"
)

# %% [markdown]
# ## 5. Posteriors

# %%
names = list(priors)
labels_tex = [r"$T_{\rm eff}$ [K]", r"$\log g$", "[Fe/H]", r"$\mu$"]
truth_vec = [TRUTH[n] for n in names]

fig = overlay_corner(
    [nss.samples, nuts.samples],
    names=names,
    labels=labels_tex,
    colors=["C0", "C1"],
    legend_labels=[f"NSS ({nss.wall_time:.0f} s)", f"NUTS ({nuts.wall_time:.0f} s)"],
    truths=TRUTH,
)
plt.show()

med, ci = nss.median(), nss.interval(0.68)
print(f"{'':6s}{'truth':>9s}{'NSS median':>12s}{'68% interval':>22s}")
for n in names:
    print(f"{n:6s}{TRUTH[n]:9.2f}{med[n]:12.2f}      [{ci[n][0]:8.2f}, {ci[n][1]:8.2f}]")

# %% [markdown]
# ## 6. The fast path: Laplace on the real grid
#
# For a well-constrained star with 21 bands, the posterior is close to Gaussian
# — so we can skip sampling entirely: optimize to the MAP, take the exact
# Hessian, and read the covariance off its inverse. `make_laplace_pipeline`
# does the whole thing (multi-restart BFGS + Hessian + evidence + draws) in one
# jitted graph with no host round-trip, and `jax.lax.map` walks a catalog
# through it. See notebook 05 for the full cost analysis.


# %% [markdown]
# **The optimizer is the weak link — and it bit us.** `jax.scipy`'s BFGS,
# launched from random starts, converged to a *local* optimum on this 21-band
# posterior: it reported a Hamiltonian of 640 where the true peak is 12, and an
# evidence 600 nats off. Nested sampling, which explores, was immune.
#
# The fix exploits what a grid model is good at: the likelihood is ~10 µs, so
# `make_laplace_pipeline` **scans the prior box first** (512 points, a few
# milliseconds) and seeds the BFGS restarts from the best points found. With
# that, log Z agrees with NSS and the MAP matches scipy's L-BFGS exactly. The
# lesson generalizes: a Gaussian approximation is only as good as the peak it
# is expanded around, so always cross-check log Z against an exact sampler on a
# subset before trusting Laplace across a catalog.


# %%
# The pipeline takes the data as a traced argument (that is what lets one
# compiled graph serve every star), so give it the two-argument likelihood.
def loglikelihood_data(p, data):
    pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])[fidx]
    return -0.5 * jnp.sum(((pred - data) / SIG_MAG) ** 2)


laplace = make_laplace_pipeline(loglikelihood_data, priors, n_samples=2000)
_ = jax.block_until_ready(laplace(jax.random.PRNGKey(6), mags_obs))  # compile

t0 = time.time()
lap_samples, lap_info = jax.block_until_ready(laplace(jax.random.PRNGKey(7), mags_obs))
t_lap = time.time() - t0
print(f"Laplace: {t_lap * 1000:.0f} ms/star (2000 iid draws)")
print(f"  log Z = {float(lap_info['logz']):.1f}   (NSS gave {nss.logz:.1f})")
print(f"  → {2000 / t_lap:,.0f} effective samples per second")
print(
    f"NSS was {nss.wall_time:.1f} s at {nss.ess_rate:.0f} ESS/s "
    f"— Laplace is {(2000 / t_lap) / nss.ess_rate:.0f}x more efficient here."
)

fig = overlay_corner(
    [nss.samples, lap_samples],
    names=names,
    labels=labels_tex,
    colors=["C0", "C3"],
    legend_labels=[
        f"NSS — exact ({nss.wall_time:.1f} s)",
        f"Laplace — Gaussian ({t_lap * 1000:.0f} ms)",
    ],
    truths=TRUTH,
)
plt.show()

# %% [markdown]
# Where the contours agree, Laplace bought the same answer for a fraction of the
# compute — the catalog workhorse. Where they part (grid edges, or the
# dwarf/giant bimodality of notebook 04) only the exact sampler can be trusted:
# a Gaussian sees one mode and quotes a confident, wrong error bar.

# %% [markdown]
# ## Next steps on real data
#
# - **Real stars**: replace the mock block with catalog magnitudes (DELVE/Gaia
#   cross-match) and per-band errors; add SFD dereddening (0.65 × E(B−V)) via
#   per-filter reddening coefficients (`StarModel(extinction_coeffs=...)`).
# - **Convention audit**: recompute a few grid magnitudes from raw TSLTE spectra
#   through tengri's filter integration (photon-counting 1/λ default) and compare
#   against the FITS values before mixing this grid with other filter sets.
# - **`averaged` flag**: confirm semantics with Ani — the loader currently
#   prefers `averaged=True` rows when nodes are duplicated.
