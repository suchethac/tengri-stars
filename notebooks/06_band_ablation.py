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
# # Which bands actually carry the information?
#
# A survey has to choose filters. This notebook answers the question
# empirically: **drop bands and watch the posterior widen.**
#
# Three views, all on the real TSLTE grid (notebook 03):
#
# 1. **Leave-one-out** — remove each band in turn from the full set and measure
#    how much each parameter's uncertainty inflates. A band that carries unique
#    information is expensive to lose; a redundant one is free.
# 2. **Sequential ablation** — greedily drop the least-informative band, again
#    and again, and watch the uncertainties climb as the filter set shrinks.
# 3. **Survey scenarios** — realistic filter sets (broadband only, +u, +CaHK)
#    side by side, with the posteriors overlaid.
#
# The whole sweep is ~60 fits. That is only affordable because of the Laplace
# pipeline (notebook 05): one jitted graph, ~100 ms per fit. But a Gaussian
# posterior cannot be trusted blindly — as bands are removed the posterior can
# go degenerate, and a Gaussian would report a confident, wrong answer. So every
# headline configuration is **cross-checked against nested sampling**, which
# explores instead of descending.

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tengri import Uniform

from tengri_stars import (
    StarModel,
    load_photometry_grid,
    make_laplace_pipeline,
    make_nss_pipeline,
    overlay_corner,
)

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(29)

DATA = (
    Path("data") if (Path("data") / "TSLTE_combined_photometry.fits").exists() else Path("../data")
)

# %% [markdown]
# ## 1. A metal-poor giant, observed in every band the grid provides

# %%
grid = load_photometry_grid(DATA / "TSLTE_combined_photometry.fits", fill="nearest")
model = StarModel(grid=grid, interp_method="pchip")
ALL_BANDS = list(grid.filter_names)
SIG_MAG = 0.02

TRUTH = {"teff": 4600.0, "logg": 1.5, "feh": -2.0, "mu": -18.0}
mags_all = np.asarray(model.predict_mags(**TRUTH)) + rng.normal(0.0, SIG_MAG, len(ALL_BANDS))
mags_all = jnp.asarray(mags_all)

lo_hi = [(float(a[0]), float(a[-1])) for a in grid.axes]
PARAMS = ["teff", "logg", "feh", "mu"]
priors = {
    "teff": Uniform(*lo_hi[0]),
    "logg": Uniform(*lo_hi[1]),
    "feh": Uniform(*lo_hi[2]),
    "mu": Uniform(TRUTH["mu"] - 10.0, TRUTH["mu"] + 10.0),
}
print(f"{len(ALL_BANDS)} bands, sigma = {SIG_MAG} mag")
print("truth:", {k: round(v, 2) for k, v in TRUTH.items()})


# %% [markdown]
# ## 2. One fit = one band subset
#
# The likelihood takes the *masked* magnitudes as its traced data argument, so
# a single compiled program serves every band subset: masked-out bands get a
# huge error bar (weight ~0) instead of changing the array shape, which would
# force a recompile per subset.


# %%
def make_fitter(pipeline_factory, **kwargs):
    """Build one jitted pipeline whose data argument is (mags, inverse-variance)."""

    def loglikelihood(p, data):
        mags, weights = data
        pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
        return -0.5 * jnp.sum(weights * (pred - mags) ** 2)

    return pipeline_factory(loglikelihood, priors, **kwargs)


laplace = make_fitter(make_laplace_pipeline, n_samples=2000)
nss = make_fitter(make_nss_pipeline, n_live=400, num_delete=40)


def weights_for(bands):
    """Inverse variance: 1/sigma^2 for bands in use, 0 for bands dropped."""
    keep = np.isin(ALL_BANDS, list(bands))
    return jnp.asarray(np.where(keep, 1.0 / SIG_MAG**2, 0.0))


def fit(bands, key=0, sampler=None):
    """Posterior for one band subset; returns (sigma per parameter, samples)."""
    sampler = sampler or laplace
    samples, _info = sampler(jax.random.PRNGKey(key), (mags_all, weights_for(bands)))
    sigma = {p: float(np.std(np.asarray(samples[p]))) for p in PARAMS}
    return sigma, samples


t0 = time.time()
sigma_full, samples_full = fit(ALL_BANDS)
print(f"full {len(ALL_BANDS)}-band fit: {time.time() - t0:.2f} s (incl. compile)")
t0 = time.time()
_ = fit(ALL_BANDS[:-1])
print(f"subsequent fits: {time.time() - t0 * 0 - t0:.2f} s each — one compiled graph, reused")
print("\nfull-set 1-sigma uncertainties:")
for p in PARAMS:
    print(f"  {p:5s} {sigma_full[p]:8.3f}")

# %% [markdown]
# ## 3. Leave-one-out: what does each band uniquely contribute?
#
# Drop one band at a time from the full set. The inflation of σ([Fe/H]) is that
# band's *unique* contribution — information no other filter supplies.

# %%
loo = {}
t0 = time.time()
for band in ALL_BANDS:
    subset = [b for b in ALL_BANDS if b != band]
    sig, _ = fit(subset)
    loo[band] = {p: sig[p] / sigma_full[p] for p in PARAMS}
print(f"{len(ALL_BANDS)} leave-one-out fits in {time.time() - t0:.1f} s")

order = sorted(ALL_BANDS, key=lambda b: -loo[b]["feh"])
fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(order))
ax.barh(y, [loo[b]["feh"] for b in order], color="C3", alpha=0.85, label="[Fe/H]")
ax.barh(y, [loo[b]["teff"] for b in order], height=0.45, color="C0", label=r"$T_{\rm eff}$")
ax.axvline(1.0, color="0.4", lw=1)
ax.set_yticks(y, order, fontsize=8)
ax.set_xlabel(r"$\sigma$ without the band $/$ $\sigma$ with it   (1 = redundant)")
ax.set_title("Leave-one-out: unique information per band")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

print("most costly bands to lose (by [Fe/H]):")
for b in order[:5]:
    print(f"  {b:28s} sigma([Fe/H]) x{loo[b]['feh']:.2f},  sigma(Teff) x{loo[b]['teff']:.2f}")

# %% [markdown]
# ## 4. Sequential ablation: greedily strip the filter set
#
# Repeatedly drop the band whose removal costs the least, and watch the
# uncertainties climb. This traces the *efficient frontier* of filter sets: at
# each size, the best subset that greedy search can find.

# %%
remaining = list(ALL_BANDS)
history = [(len(remaining), dict(sigma_full), None)]

t0 = time.time()
while len(remaining) > 2:
    candidates = []
    for band in remaining:
        sig, _ = fit([b for b in remaining if b != band])
        candidates.append((sig["feh"], band, sig))
    cost, drop, sig = min(candidates)  # cheapest band to lose
    remaining = [b for b in remaining if b != drop]
    history.append((len(remaining), sig, drop))
n_fits = sum(range(3, len(ALL_BANDS) + 1))
print(f"greedy ablation: {len(history)} steps, ~{n_fits} fits in {time.time() - t0:.0f} s")

sizes = [h[0] for h in history]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
for ax, p, c in zip(axes, ["teff", "logg", "feh"], ["C0", "C1", "C3"]):
    ax.plot(sizes, [h[1][p] for h in history], "o-", color=c, ms=4)
    ax.set(xlabel="bands retained", ylabel=rf"$\sigma$({p})", yscale="log")
    ax.invert_xaxis()
axes[0].set_title(r"$T_{\rm eff}$ [K]")
axes[1].set_title(r"$\log g$ [dex]")
axes[2].set_title("[Fe/H] [dex]")
plt.tight_layout()
plt.show()

print("greedy drop order (cheapest first):")
print("  " + " → ".join(h[2] for h in history[1:] if h[2]))

# %% [markdown]
# ## 5. Survey scenarios
#
# Realistic filter sets, from a broadband-only survey to one with a
# metallicity-sensitive narrow band. Every configuration is fit twice: with the
# fast Laplace pipeline **and** with nested sampling, so we can see whether the
# Gaussian approximation still holds as information is removed.


# %%
def pick(*fragments):
    return [b for b in ALL_BANDS if any(f in b for f in fragments)]


SCENARIOS = {
    "DECam gri": pick("DECCAM_g", "DECCAM_r", "DECCAM_i"),
    "DECam ugri": pick("DECCAM_u", "DECCAM_g", "DECCAM_r", "DECCAM_i"),
    "DECam gri + CaHK": pick("DECCAM_g", "DECCAM_r", "DECCAM_i", "CaHK"),
    "LSST ugri": pick("lsst_u", "lsst_g", "lsst_r", "lsst_i"),
    "all bands": ALL_BANDS,
}

rows = []
scen_samples = {}
for name, bands in SCENARIOS.items():
    sig_l, s_l = fit(bands, key=1, sampler=laplace)
    sig_n, s_n = fit(bands, key=1, sampler=nss)
    scen_samples[name] = (s_l, s_n)
    rows.append((name, len(bands), sig_l, sig_n))

print(
    f"{'scenario':<20s}{'n':>3s} | {'sigma[Fe/H] Laplace':>20s}{'NSS':>8s} | "
    f"{'sigma(logg) Laplace':>20s}{'NSS':>8s}"
)
for name, n, sl, sn in rows:
    print(
        f"{name:<20s}{n:3d} | {sl['feh']:20.3f}{sn['feh']:8.3f} | "
        f"{sl['logg']:20.3f}{sn['logg']:8.3f}"
    )

# %% [markdown]
# ## 6. The posteriors, side by side

# %%
show = ["DECam gri", "DECam gri + CaHK", "all bands"]
fig = overlay_corner(
    [scen_samples[s][1] for s in show],  # nested sampling: the trustworthy one
    names=PARAMS,
    labels=[r"$T_{\rm eff}$ [K]", r"$\log g$", "[Fe/H]", r"$\mu$"],
    colors=["C1", "C3", "C0"],
    legend_labels=[f"{s} ({len(SCENARIOS[s])} bands)" for s in show],
    truths=TRUTH,
)
plt.show()

# %% [markdown]
# ## 7. Does the Gaussian still hold when bands are removed?
#
# Compare Laplace against nested sampling for the sparsest configuration. If
# they part company, the posterior has gone non-Gaussian — and only the sampler
# that *explores* can be believed.

# %%
lap_gri, nss_gri = scen_samples["DECam gri"]
fig = overlay_corner(
    [nss_gri, lap_gri],
    names=PARAMS,
    labels=[r"$T_{\rm eff}$ [K]", r"$\log g$", "[Fe/H]", r"$\mu$"],
    colors=["C0", "C3"],
    legend_labels=["NSS — exact", "Laplace — Gaussian"],
    truths=TRUTH,
)
plt.show()

for p in PARAMS:
    ln = np.asarray(lap_gri[p])
    nn = np.asarray(nss_gri[p])
    print(
        f"{p:5s} Laplace {np.median(ln):9.2f} +- {np.std(ln):7.3f}   "
        f"NSS {np.median(nn):9.2f} +- {np.std(nn):7.3f}   truth {TRUTH[p]:9.2f}"
    )
