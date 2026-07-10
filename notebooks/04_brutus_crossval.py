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
# # Cross-validation against brutus on the TSLTE grid
#
# [brutus](https://github.com/joshspeagle/brutus) (Speagle et al.) fits stellar
# photometry against MIST isochrone grids with a brute-force Bayesian scan — the
# closest published analog to tengri-stars' photometry channel. Here the mock star
# comes **from the TSLTE grid itself** (A. Chiti's Turbospectrum LTE synthetic
# photometry, notebook 03), in the 8 bands both systems share (SDSS ugri + DECam
# ugri), so the comparison measures two things at once:
#
# 1. **prior structure** — brutus confines models to the MIST manifold with IMF /
#    age / Galactic priors; tengri-stars fits a flat atmosphere box;
# 2. **grid-to-grid systematics** — brutus predicts photometry from its C3K/MIST
#    machinery while the data were generated from Turbospectrum LTE, so any
#    physics difference between the synthetic grids lands in brutus's residuals
#    (tengri-stars fits the generating grid and carries no mismatch by
#    construction).
#
# No parallax is given to brutus: the TSLTE magnitude zero-point is a
# convention (absorbed by tengri-stars' μ), so an implied distance would be
# meaningless. Only (Teff, log g, [Fe/H]) are compared.

# %%
import os

# JAX's XLA and numba's default OpenMP threading layer both load libomp —
# a duplicate-runtime segfault on macOS. numba's workqueue layer avoids libomp.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from pathlib import Path

# Notebook kernels launch in notebooks/; scripts run from the repo root.
DATA = (
    Path("data") if (Path("data") / "TSLTE_combined_photometry.fits").exists() else Path("../data")
)

import corner
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brutus import utils as brutus_utils
from brutus.analysis import BruteForce
from brutus.core.individual import StarGrid
from brutus.data.download import fetch_grids
from brutus.data.loader import load_models
from tengri import Uniform

from tengri_stars import StarModel, fit_nss, load_photometry_grid

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(9)

# %% [markdown]
# ## 1. Grids and the shared bands

# %%
BRUTUS_GRID = str(fetch_grids(grid="mist_v9", target_dir=str(DATA / "brutus")))
tslte = load_photometry_grid(DATA / "TSLTE_combined_photometry.fits", fill="nearest")

# TSLTE column → brutus filter name, for the bands both systems provide.
BAND_MAP = {
    "SLOAN-SDSS.u_ab": "SDSS_u",
    "SLOAN-SDSS.g_ab": "SDSS_g",
    "SLOAN-SDSS.r_ab": "SDSS_r",
    "SLOAN-SDSS.i_ab": "SDSS_i",
    "DECCAM_u_des_ab": "DECam_u",
    "DECCAM_g_des_ab": "DECam_g",
    "DECCAM_r_des_ab": "DECam_r",
    "DECCAM_i_des_ab": "DECam_i",
}
TSLTE_BANDS = list(BAND_MAP)
BRUTUS_BANDS = list(BAND_MAP.values())
fidx = jnp.asarray([tslte.filter_names.index(f) for f in TSLTE_BANDS])
print(f"{len(BAND_MAP)} shared bands:", BRUTUS_BANDS)

# %% [markdown]
# ## 2. Mock star from the TSLTE table
#
# A metal-poor giant on grid nodes (guaranteed inside coverage), zero
# extinction (the grid is a bare atmosphere), and a magnitude offset μ that
# stands in for distance/radius/zero-point.

# %%
TRUTH = {"teff": 4600.0, "logg": 2.0, "feh": -1.5, "mu": -14.0}
SIG_MAG = 0.02

model = StarModel(grid=tslte, interp_method="pchip")
mags_clean = model.predict_mags(**TRUTH)[fidx]
mags_obs = np.asarray(mags_clean) + rng.normal(0.0, SIG_MAG, len(TSLTE_BANDS))
print("observed mags:", dict(zip(BRUTUS_BANDS, np.round(mags_obs, 3))))

# %% [markdown]
# ## 3. brutus fit (BruteForce over MIST v9, no parallax)

# %%
models, labels, label_mask = load_models(BRUTUS_GRID, filters=BRUTUS_BANDS, verbose=False)
star_grid = StarGrid(models, labels, filters=BRUTUS_BANDS, verbose=False)
bf = BruteForce(star_grid, verbose=False)

flux_obs, flux_err = brutus_utils.inv_magnitude(mags_obs, np.full(len(BRUTUS_BANDS), SIG_MAG))

t0 = time.time()
bf.fit(
    flux_obs[None, :],
    flux_err[None, :],
    np.ones((1, len(BRUTUS_BANDS)), dtype=bool),
    np.array([0]),  # object ID (h5py rejects unicode labels)
    str(DATA / "brutus" / "tslte_mock_fit"),
    data_coords=np.array([[90.0, 60.0]]),  # halo sightline for the Galactic prior
    av_gauss=(0.0, 0.01),  # bare-atmosphere mock: extinction pinned at zero
    Ndraws=1000,
    rstate=np.random.RandomState(4),
    verbose=True,
)
print(f"brutus wall time: {time.time() - t0:.1f} s")

with h5py.File(DATA / "brutus" / "tslte_mock_fit.h5", "r") as f:
    model_idx = f["model_idx"][0]
brutus_samples = {
    "teff": 10.0 ** labels["logt"][model_idx],
    "logg": labels["logg"][model_idx],
    "feh": labels["feh"][model_idx],
}
print("brutus medians:", {k: round(float(np.median(v)), 3) for k, v in brutus_samples.items()})

# %% [markdown]
# ## 4. tengri-stars fit on the same TSLTE grid

# %%
lo_hi = [(float(a[0]), float(a[-1])) for a in tslte.axes]
priors = {
    "teff": Uniform(*lo_hi[0]),
    "logg": Uniform(*lo_hi[1]),
    "feh": Uniform(*lo_hi[2]),
    "mu": Uniform(-24.0, -4.0),
}
obs = jnp.asarray(mags_obs)


def loglikelihood(p):
    pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])[fidx]
    return -0.5 * jnp.sum(((pred - obs) / SIG_MAG) ** 2)


t0 = time.time()
ts = fit_nss(loglikelihood, priors, key=jax.random.PRNGKey(21), n_live=400, num_delete=40)
print(
    f"tengri-stars NSS wall time: {time.time() - t0:.1f} s,  "
    f"log Z = {ts.logz:.1f}, ESS = {ts.ess:.0f}"
)
print("tengri-stars medians:", {k: round(v, 3) for k, v in ts.median().items()})

# %% [markdown]
# ## 5. Posterior overlay

# %%
shared = ["teff", "logg", "feh"]
labels_tex = [r"$T_{\rm eff}$ [K]", r"$\log g$", "[Fe/H]"]
truth_vec = [TRUTH["teff"], TRUTH["logg"], TRUTH["feh"]]

b_stack = np.column_stack([np.asarray(brutus_samples[k], dtype=float) for k in shared])
t_stack = np.column_stack([np.asarray(ts.samples[k]) for k in shared])

fig = corner.corner(
    t_stack,
    labels=labels_tex,
    truths=truth_vec,
    color="C0",
    hist_kwargs={"density": True},
    plot_datapoints=False,
    smooth=1.0,
)
corner.corner(
    b_stack,
    fig=fig,
    color="C2",
    hist_kwargs={"density": True},
    plot_datapoints=False,
    smooth=1.0,
)
fig.legend(
    handles=[
        plt.Line2D([], [], color="C0", label="tengri-stars (NSS on the TSLTE grid)"),
        plt.Line2D([], [], color="C2", label="brutus (MIST v9 + priors, C3K photometry)"),
    ],
    loc="upper right",
    frameon=False,
)
plt.show()

# %%
print(f"{'':8s}{'truth':>9s}{'brutus':>22s}{'tengri-stars':>22s}")
for k, tv in zip(shared, truth_vec):
    b = np.percentile(np.asarray(brutus_samples[k], dtype=float), [16, 50, 84])
    t = np.percentile(np.asarray(ts.samples[k]), [16, 50, 84])
    print(
        f"{k:8s}{tv:9.2f}{b[1]:10.2f} [{b[0]:7.2f},{b[2]:7.2f}]"
        f"{t[1]:10.2f} [{t[0]:7.2f},{t[2]:7.2f}]"
    )

# %% [markdown]
# ## Reading the comparison (observed on this mock)
#
# - **The dwarf/giant degeneracy is the headline.** ugri colors alone barely
#   separate a metal-poor giant from a dwarf, and with no parallax the two codes
#   resolve the bimodality by prior: brutus's IMF + Galactic priors (dwarfs
#   heavily outnumber giants) collapse its posterior onto a *dwarf* solution —
#   log g = 4.72 ± 0.03, confidently excluding the true 2.0 — while tengri-stars'
#   flat prior settles on the giant side (0.85, 68% [−0.1, 1.8]) with honest
#   width. Both Teff medians ride the Teff–log g ridge ~180 K above truth. The
#   cures, in order of power: parallax (kills the dwarf solution outright), a
#   gravity-sensitive band, or — Bayesianly cleanest — running NSS once per
#   log g class and comparing evidences (Δlog Z), the model-comparison upgrade of
#   MAGIC's parallax z-score classification.
# - **[Fe/H] separates cleanly**: tengri-stars, fitting the generating grid,
#   recovers −1.54 (truth −1.5); brutus lands +0.3 dex metal-rich at −1.20
#   [−1.25, −0.95] — the C3K↔Turbospectrum u-band synthetic-photometry
#   systematic (compounded by its dwarf solution). Per-band zero-point offsets
#   are the standard remedy — brutus itself ships per-survey offset files — and
#   this is exactly the calibration MAGIC-scale surveys need.
# - tengri-stars carries no model mismatch by construction here; that's the
#   point of the design — brutus's residuals bundle its priors *and* the grid
#   physics differences, and this notebook measures the bundle.
# - The spectroscopic channel (notebook 01) is untouched by all of this: these
#   are photometric-grid systematics only. TSLTE synthetic *spectra* grids, when
#   available, slot into `SpectralGrid.from_arrays` for the joint fit.
