# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The stitched pipeline on real MAGIC data
#
# The two-hypothesis model average (`tengri_stars.branch`, validated on mocks
# in `bench/magic_stitch_campaign.py`) applied to the MAGIC paper's own
# validation samples — CaHK, DECam g, r, i only:
#
# - **g195 reference table** (1,453 stars in dwarf-galaxy fields):
#   literature `metallicity_ref`, reference distances `dmod_ref`, and RGB/MS
#   labels `star_class` — scores classification and the known-distance mode.
# - **APOGEE cross-match** (2,235 field stars): spectroscopic `M_H` and
#   `LOGG_SPEC` (branch ground truth), Gaia parallaxes, and MAGIC's own
#   per-branch estimates `fehs_rgb`/`fehs_ms` — the branch-resolved sanity
#   gate, and the *re-weighting* variant (our weights × MAGIC's per-branch
#   metallicities: deployable on existing catalogs without re-fitting).
#
# Mock-campaign prediction to test (`notebooks/stitch_results/summary.md`):
# the mixture matches or beats both a hard parallax classification and a hard
# Δχ² vote at every parallax quality. Real data adds what mocks exclude:
# grid–isochrone mismatch, age spread, dereddening errors, real parallax
# noise, and stars on neither 12 Gyr branch.
#
# Data live outside the repo (paper directory); paths are absolute.

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence XLA/PJRT C++ chatter

from pathlib import Path

# Notebook kernels launch in notebooks/; scripts run from the repo root.
DATA = (
    Path("data") if (Path("data") / "TSLTE_combined_photometry.fits").exists() else Path("../data")
)
PAPER = Path("/Users/Ani/Dropbox (MIT)/my_papers/magic_overview")
ISO_DIR = Path.home() / "Documents/MIT_Work/Research/magic_scratch/isochrones"

import itertools

import jax
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from tengri_stars import StarModel, load_photometry_grid
from tengri_stars.branch import binary_scan, combine_mixture, load_dartmouth_branches

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 1. Model, branch tables, and the two samples (f9 cuts verbatim)

# %%
BANDS = ("CaHK_filter_ab", "DECCAM_g_des_ab", "DECCAM_r_des_ab", "DECCAM_i_des_ab")
IK, IG, IR, II = range(4)
CHI2_BANDS = (0, 1, 2, 3)

grid = load_photometry_grid(
    DATA / "TSLTE_combined_photometry.fits", filter_columns=BANDS, fill="nearest"
)
model = StarModel(grid=grid, interp_method="pchip")
ISO_FILES = {-2.5: "fehm25_12Gyr.txt", -2.0: "fehm20_12Gyr.txt",
             -1.5: "fehm15_12Gyr.txt", -1.0: "fehm10_12Gyr.txt"}
tables = load_dartmouth_branches(
    {feh: str(ISO_DIR / fn) for feh, fn in ISO_FILES.items()}, model, ig=IG, ii=II
)


def f9_cuts(t):
    gi = np.asarray(t["g_dered"] - t["i_dered"], float)
    return (
        np.asarray(t["broadband_valid"], bool)
        & ~np.asarray(t["gaia_var_flag"], bool)
        & (gi > 0.2) & (gi < 1.5)
        & (np.asarray(t["feh_extrapolation_flag"], float) == 0)
        & (np.asarray(t["e_feh"], float) < 0.5)
        & (np.asarray(t["feh"], float) > -4.0)
        & (np.asarray(t["ebv_sfd98"], float) < 0.2)
    )


def phot_arrays(t, cahk_err_col):
    obs = np.column_stack([np.asarray(t[c], float) for c in
                           ("CaHK_dered", "g_dered", "r_dered", "i_dered")])
    sig = np.column_stack([np.asarray(t[c], float) for c in
                           (cahk_err_col, "magerr_psf_g", "magerr_psf_r", "magerr_psf_i")])
    return obs, np.clip(sig, 0.005, None)


g195 = Table.read(PAPER / "figs/f9/g195_all_CMDcut_reference_distances.fits")
g195 = g195[f9_cuts(g195) & np.isfinite(np.asarray(g195["metallicity_ref"], float))]
obs_g, sig_g = phot_arrays(g195, "MAGERR_PSF")

apo = Table.read(
    PAPER / "figs/f9/data/processed/"
    "2023B_APOGEE_xmatch_no_interp_raw_catalog_to_usable_260524.fits"
)
apo = apo[f9_cuts(apo) & np.isfinite(np.asarray(apo["M_H"], float))
          & np.isfinite(np.asarray(apo["LOGG_SPEC"], float))]
obs_a, sig_a = phot_arrays(apo, "magerr_psf_cahk")
print(f"after cuts: g195 {len(g195)} stars, APOGEE {len(apo)} stars")

# %% [markdown]
# ## 2. Calibration gate: grid vs data color offsets
#
# The catalog photometry was calibrated against the same TSLTE grid the
# forward model uses (the paper's +0.12 mag CaHK adjustment is upstream, in
# `CaHK_dered`), so residual offsets should be small — but they are measured,
# not assumed. For each g195 star, the model SED at its *published* branch
# (`star_class`) and [Fe/H] (clipped to the isochrone range), at the observed
# color; residuals after removing the profiled normalization. The median per
# band is then subtracted from the observations for everything downstream.

# %%
star_class = np.char.decode(np.asarray(g195["star_class"], "S8")).astype(str)
feh_pub = np.clip(np.asarray(g195["feh"], float), tables.feh_nodes[0], tables.feh_nodes[-1])
gi_obs_g = obs_g[:, IG] - obs_g[:, II]

resid = np.full_like(obs_g, np.nan)
mags_at = jax.jit(jax.vmap(lambda t, g, f: model.predict_mags(teff=t, logg=g, feh=f)))
import jax.numpy as jnp

for b in ("RGB", "MS"):
    sel = star_class == b
    tab = tables.scan[b]
    k = np.clip(np.searchsorted(tables.feh_scan, feh_pub[sel]) - 1, 0, tables.feh_scan.size - 1)
    j = np.clip(np.searchsorted(tables.gi_grid, gi_obs_g[sel]) - 1, 0, tables.gi_grid.size - 2)
    teff, logg = tab["teff"][k, j], tab["logg"][k, j]
    pred = np.asarray(mags_at(jnp.asarray(teff), jnp.asarray(logg), jnp.asarray(feh_pub[sel])))
    d = obs_g[sel] - pred
    w = 1.0 / sig_g[sel] ** 2
    mu = (d * w).sum(axis=1) / w.sum(axis=1)
    resid[sel] = d - mu[:, None]

OFFSETS = np.nanmedian(resid, axis=0)
print("median (data − model) color offsets [mag], CaHK/g/r/i:",
      np.round(OFFSETS, 3))
obs_g_cal = obs_g - OFFSETS
obs_a_cal = obs_a - OFFSETS

# %% [markdown]
# ## 3. Branch-resolved sanity gate (APOGEE)
#
# Our scan's per-branch [Fe/H] against MAGIC's own `fehs_rgb` / `fehs_ms` —
# the two estimators share the grid but differ in log g source (our
# isochrone-at-observed-color vs MAGIC's isochrone+distance iteration), so
# the demand is *tracking*, not identity.

# %%
gi_sigma_a = np.hypot(sig_a[:, IG], sig_a[:, II])
scans_a = binary_scan(obs_a_cal, sig_a, tables, model, ig=IG, ii=II,
                      chi2_bands=CHI2_BANDS, gi_sigma=gi_sigma_a)

in_range = {
    b: (np.asarray(apo[f"fehs_{b.lower()}"], float) > tables.feh_nodes[0] + 0.05)
    & (np.asarray(apo[f"fehs_{b.lower()}"], float) < tables.feh_nodes[-1] - 0.05)
    for b in ("RGB", "MS")
}
gate_ok = True
for b in ("RGB", "MS"):
    ours = scans_a[b]["feh"]
    theirs = np.asarray(apo[f"fehs_{b.lower()}"], float)
    valid = in_range[b] & np.isfinite(scans_a[b]["chi2"])
    dd = (ours - theirs)[valid]
    med, sc = np.median(dd), 0.5 * np.subtract(*np.percentile(dd, [84, 16]))
    print(f"gate {b}: ours − MAGIC fehs_{b.lower()}: median {med:+.3f}, "
          f"68% half-width {sc:.3f} dex  (N={valid.sum()})")
    gate_ok &= abs(med) < 0.15
print("GATE:", "PASS" if gate_ok else "FAIL — do not interpret the policies below")

# %% [markdown]
# ## 4. Policies on the real samples
#
# For each star: hard Δχ² vote (no distance), mixture with the Gaia parallax,
# and — g195 only — the known-distance mode with `dmod_ref`. Baseline: the
# published MAGIC `feh` on the same stars. APOGEE additionally gets the
# *re-weighting* variant: P(RGB) from our scan × MAGIC's own per-branch
# metallicities.

# %%
def plx_arrays(t):
    plx = np.asarray(t["parallax"], float)
    err = np.asarray(t["parallax_error"], float)
    bad = ~np.isfinite(plx) | ~np.isfinite(err) | (err <= 0)
    return np.where(bad, 0.0, plx), np.where(bad, 1e6, err)


def stats(est, ref, sel=None):
    d = est - ref
    if sel is not None:
        d = d[sel]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return "—"
    return (f"{np.median(d):+.3f} / {0.5 * np.subtract(*np.percentile(d, [84, 16])):.3f}"
            f" (N={d.size})")


# ---- g195 (dwarf fields; reference metallicities + known distances)
ref_g = np.asarray(g195["metallicity_ref"], float)
gi_sigma_g = np.hypot(sig_g[:, IG], sig_g[:, II])
dmod_ref = np.asarray(g195["dmod_ref"], float)
scans_g = binary_scan(obs_g_cal, sig_g, tables, model, ig=IG, ii=II,
                      chi2_bands=CHI2_BANDS, gi_sigma=gi_sigma_g,
                      dm_known=dmod_ref, n_quad=25)
plx_g, plxe_g = plx_arrays(g195)

vote_rgb = scans_g["RGB"]["chi2"] < scans_g["MS"]["chi2"]
feh_vote = np.where(vote_rgb, scans_g["RGB"]["feh"], scans_g["MS"]["feh"])
mix = combine_mixture(scans_g, obs_g_cal[:, IG], tables.feh_scan,
                      parallax=plx_g, parallax_error=plxe_g)
mix_dm = combine_mixture(scans_g, obs_g_cal[:, IG], tables.feh_scan, use_dm=True)

is_rgb_true = star_class == "RGB"
in_ref = (ref_g > tables.feh_nodes[0] + 0.05) & (ref_g < tables.feh_nodes[-1] - 0.05)
print("g195 — Δ[Fe/H] vs metallicity_ref: median / 68% half-width "
      f"(reference within scan range only; {in_ref.sum()} of {len(g195)}):")
print(f"  published MAGIC feh   : {stats(np.asarray(g195['feh'], float), ref_g, in_ref)}")
print(f"  hard Δχ² vote         : {stats(feh_vote, ref_g, in_ref)}")
print(f"  mixture + parallax    : {stats(mix['feh'], ref_g, in_ref)}")
print(f"  known-DM mixture      : {stats(mix_dm['feh'], ref_g, in_ref)}")
print("g195 — branch classification vs star_class:")
print(f"  hard Δχ² vote      : {np.mean(vote_rgb == is_rgb_true):.2f}")
print(f"  mixture + parallax : {np.mean((mix['p_rgb'] >= 0.5) == is_rgb_true):.2f}")
print(f"  known-DM mixture   : {np.mean((mix_dm['p_rgb'] >= 0.5) == is_rgb_true):.2f}")
cov = np.mean((mix["feh_lo"] <= ref_g) & (ref_g <= mix["feh_hi"]))
print(f"  mixture 68% interval covers metallicity_ref: {cov:.2f} "
      "(inflated by reference-value errors)")

# ---- APOGEE (field stars; spectroscopic truth)
ref_a = np.asarray(apo["M_H"], float)
logg_spec = np.asarray(apo["LOGG_SPEC"], float)
rgb_true_a = logg_spec <= 3.6
plx_a, plxe_a = plx_arrays(apo)
mix_a = combine_mixture(scans_a, obs_a_cal[:, IG], tables.feh_scan,
                        parallax=plx_a, parallax_error=plxe_a)
vote_rgb_a = scans_a["RGB"]["chi2"] < scans_a["MS"]["chi2"]
feh_vote_a = np.where(vote_rgb_a, scans_a["RGB"]["feh"], scans_a["MS"]["feh"])
# re-weighting: our P(RGB) × MAGIC's per-branch metallicities
feh_rw = (mix_a["p_rgb"] * np.asarray(apo["fehs_rgb"], float)
          + (1 - mix_a["p_rgb"]) * np.asarray(apo["fehs_ms"], float))

in_ref_a = (ref_a > tables.feh_nodes[0] + 0.05) & (ref_a < tables.feh_nodes[-1] - 0.05)
print(f"\nAPOGEE — Δ[Fe/H] vs M_H (reference within scan range; {in_ref_a.sum()} stars):")
print(f"  published MAGIC feh    : {stats(np.asarray(apo['feh'], float), ref_a, in_ref_a)}")
print(f"  hard Δχ² vote          : {stats(feh_vote_a, ref_a, in_ref_a)}")
print(f"  mixture + parallax     : {stats(mix_a['feh'], ref_a, in_ref_a)}")
print(f"  re-weighting (MAGIC per-branch × our weights): {stats(feh_rw, ref_a, in_ref_a)}")
print("APOGEE — branch classification vs LOGG_SPEC (RGB = log g ≤ 3.6):")
print(f"  MAGIC is_rgb        : {np.mean(np.asarray(apo['is_rgb'], bool) == rgb_true_a):.2f}")
print(f"  hard Δχ² vote       : {np.mean(vote_rgb_a == rgb_true_a):.2f}")
print(f"  mixture + parallax  : {np.mean((mix_a['p_rgb'] >= 0.5) == rgb_true_a):.2f}")

# %% [markdown]
# ### The comparison, visually

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
for ax, est, label in (
    (axes[0], np.asarray(apo["feh"], float), "published MAGIC feh"),
    (axes[1], mix_a["feh"], "mixture + parallax (full stitch)"),
    (axes[2], feh_rw, "re-weighting (MAGIC per-branch)"),
):
    ax.plot([-2.6, -0.9], [-2.6, -0.9], color="0.7", lw=1)
    ax.scatter(ref_a[in_ref_a], est[in_ref_a], s=8, alpha=0.4,
               c=np.where(rgb_true_a[in_ref_a], "C3", "C0"))
    d = (est - ref_a)[in_ref_a]
    d = d[np.isfinite(d)]
    ax.set_title(f"{label}\nΔ = {np.median(d):+.2f}, σ68 = "
                 f"{0.5 * np.subtract(*np.percentile(d, [84, 16])):.2f}", fontsize=9)
    ax.set_xlabel("APOGEE [M/H]")
axes[0].set_ylabel("photometric [Fe/H]")
fig.suptitle("APOGEE field stars (red = spectroscopic giants, blue = dwarfs)")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(logg_spec, mix_a["p_rgb"], s=8, alpha=0.4,
           c=np.where(rgb_true_a, "C3", "C0"))
ax.axvline(3.6, color="0.7", lw=1)
ax.axhline(0.5, color="0.7", lw=1)
ax.set_xlabel("APOGEE spectroscopic log g")
ax.set_ylabel("P(RGB) — mixture + parallax")
ax.set_title("branch posterior vs spectroscopic gravity")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Diagnostics
#
# Five views of *how* the stitch behaves, not just its headline numbers:
# residual trends (metallicity- and color-dependent systematics), the
# reliability of the branch posterior (is P(RGB)=0.7 right 70% of the time?),
# where classification fails, interval calibration (the systematics floor),
# and the per-field breakdown on g195.

# %%
FIGDIR = DATA.parent / "notebooks" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# --- (a) residual trends vs reference [Fe/H] and vs color (APOGEE)
gi_a = obs_a[:, IG] - obs_a[:, II]
d_mix = mix_a["feh"] - ref_a
d_pub = np.asarray(apo["feh"], float) - ref_a


def running_median(x, y, bins):
    med, lo, hi, centers = [], [], [], []
    for b0, b1 in itertools.pairwise(bins):
        s = (x >= b0) & (x < b1) & np.isfinite(y)
        if s.sum() < 8:
            continue
        q = np.percentile(y[s], [16, 50, 84])
        centers.append(0.5 * (b0 + b1))
        lo.append(q[0]), med.append(q[1]), hi.append(q[2])
    return np.array(centers), np.array(med), np.array(lo), np.array(hi)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, x, xbins, xlabel in (
    (axes[0], ref_a, np.arange(-2.5, -0.95, 0.25), "APOGEE [M/H]"),
    (axes[1], gi_a, np.arange(0.4, 1.55, 0.15), "g − i (DECam)"),
):
    sel = in_ref_a
    ax.scatter(x[sel], d_mix[sel], s=6, alpha=0.25, color="C0")
    for d, color, label in ((d_mix, "C0", "mixture"), (d_pub, "C1", "published MAGIC")):
        c, m, lo, hi = running_median(x[sel], d[sel], xbins)
        ax.plot(c, m, "-o", ms=4, color=color, label=label)
        ax.fill_between(c, lo, hi, color=color, alpha=0.15)
    ax.axhline(0, color="0.7", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-1.2, 1.2)
axes[0].set_ylabel("Δ[Fe/H] (photometric − APOGEE)")
axes[0].legend(fontsize=8)
fig.suptitle("residual trends: running median ± 68% (APOGEE, reference within scan range)")
fig.tight_layout()
fig.savefig(FIGDIR / "stitch_residual_trends.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- (b) reliability of the branch posterior + (c) where classification fails
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

bins = np.linspace(0.0, 1.0, 11)
centers, frac, counts = [], [], []
for b0, b1 in itertools.pairwise(bins):
    s = (mix_a["p_rgb"] >= b0) & (mix_a["p_rgb"] < b1 if b1 < 1 else mix_a["p_rgb"] <= 1)
    if s.sum() < 5:
        continue
    centers.append(mix_a["p_rgb"][s].mean())
    frac.append(rgb_true_a[s].mean())
    counts.append(s.sum())
axes[0].plot([0, 1], [0, 1], color="0.7", lw=1)
axes[0].scatter(centers, frac, s=np.clip(np.array(counts) / 3, 10, 200), color="C0")
for c, f, n in zip(centers, frac, counts):
    axes[0].annotate(str(n), (c, f), fontsize=6.5, xytext=(3, 3),
                     textcoords="offset points")
axes[0].set_xlabel("P(RGB) — mixture + parallax")
axes[0].set_ylabel("empirical RGB fraction (LOGG_SPEC ≤ 3.6)")
axes[0].set_title("branch-posterior reliability (APOGEE)", fontsize=10)

mis_a = (mix_a["p_rgb"] >= 0.5) != rgb_true_a
plx_snr = np.abs(plx_a) / plxe_a
axes[1].scatter(gi_a[~mis_a], plx_snr[~mis_a], s=6, alpha=0.2, color="0.6",
                label="correct")
axes[1].scatter(gi_a[mis_a], plx_snr[mis_a], s=14, alpha=0.8, color="C3",
                label=f"misclassified ({mis_a.sum()})")
axes[1].set_yscale("log")
axes[1].axhline(5, color="0.8", lw=0.8, ls=":")
axes[1].set_xlabel("g − i (DECam)")
axes[1].set_ylabel("|parallax| / σ_ϖ")
axes[1].set_title("where the mixture fails (APOGEE)", fontsize=10)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "stitch_reliability_failures.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **The one real defect these two panels expose:** P(RGB) is nearly binary
# (weights saturate), and the confident failures are not random — of the
# misclassifications at parallax S/N > 5, ~94% are spectroscopic
# dwarfs/subgiants called RGB with p ≈ 1, with LOGG_SPEC median ≈ 4.1 and 60%
# in 3.0–4.2. That is exactly the **gap between the two hypothesis manifolds**
# (RGB branch capped at log g 3.6, MS starting ≈ 4.2): a subgiant belongs to
# neither, and a two-hypothesis χ² must confidently pick the closer impostor.
# The probability is uncalibrated only where the model family is incomplete —
# the fix is a third (SGB) hypothesis or a continuous log g family along the
# isochrone, plus tempered χ² weights for model mismatch, not a better
# parallax term.

# %%
# --- (d) interval calibration: the systematics floor made visible
half = 0.5 * (mix_a["feh_hi"] - mix_a["feh_lo"])
ok = in_ref_a & np.isfinite(d_mix) & (half > 0)
fig, ax = plt.subplots(figsize=(6.8, 4))
for floor, color in ((0.0, "C0"), (0.2, "C2"), (0.3, "C1")):
    z = d_mix[ok] / np.hypot(half[ok], floor)
    z = z[np.abs(z) < 8]
    label = "photometric interval only" if floor == 0 else f"+ {floor} dex floor"
    ax.hist(z, bins=np.linspace(-6, 6, 61), density=True, histtype="step",
            lw=1.5, color=color, label=f"{label} (σ_z = {np.std(z):.2f})")
xx = np.linspace(-6, 6, 200)
ax.plot(xx, np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi), "k--", lw=1, label="N(0,1)")
ax.set_xlabel("Δ[Fe/H] / total σ")
ax.set_ylabel("density")
ax.set_title("interval calibration vs APOGEE — the systematics floor", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "stitch_interval_calibration.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- (e) per-field breakdown on g195
fields = np.char.decode(np.asarray(g195["field_name"], "S32")).astype(str)
d_mix_g = mix["feh"] - ref_g
d_pub_g = np.asarray(g195["feh"], float) - ref_g
names = [f for f in np.unique(fields) if ((fields == f) & in_ref).sum() >= 10]
fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.5))
for yy, f in enumerate(names):
    s = (fields == f) & in_ref
    for d, color, dy in ((d_pub_g, "C1", -0.12), (d_mix_g, "C0", 0.12)):
        v = d[s][np.isfinite(d[s])]
        q16, q50, q84 = np.percentile(v, [16, 50, 84])
        ax.errorbar(q50, yy + dy, xerr=[[q50 - q16], [q84 - q50]], fmt="o", ms=4,
                    color=color, capsize=2)
    ax.annotate(f"N={s.sum()}", (1.02, yy), xycoords=("axes fraction", "data"),
                fontsize=7, va="center")
ax.axvline(0, color="0.7", lw=1)
ax.set_yticks(range(len(names)), names, fontsize=8)
ax.set_xlabel("Δ[Fe/H] (photometric − reference), median ± 68%")
ax.set_title("g195 per-field: published MAGIC (orange) vs mixture (blue)", fontsize=10)
fig.tight_layout()
fig.savefig(FIGDIR / "stitch_per_field.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Verdict
#
# - **Calibration is a non-issue** (median grid−data offsets ≤ 2 mmag — the
#   catalog was calibrated against this same grid), and the branch-resolved
#   gate passes: our per-branch [Fe/H] tracks MAGIC's `fehs_ms` to
#   −0.00 ± 0.06 dex and `fehs_rgb` to +0.08 ± 0.12 dex (the small RGB bias
#   is the log g-source difference: isochrone-at-observed-color vs MAGIC's
#   distance iteration).
# - **The mock ranking survives contact with real data.** On APOGEE field
#   stars with spectroscopic gravities, the shape-only Δχ² vote gets 0.67 —
#   almost exactly the mock prediction (0.69) — and the **mixture with Gaia
#   parallax reaches 0.94, beating the current pipeline's `is_rgb` (0.85)**.
#   On g195 the mixture agrees with `star_class` at 0.98 (with the caveat
#   that `star_class` itself is partly parallax-derived, so this is
#   consistency more than truth).
# - **Metallicity: bias removed, scatter matched or improved.** APOGEE:
#   mixture −0.01 ± 0.21 vs published −0.09 ± 0.21 (same stars). g195 vs
#   literature references: mixture 0.37, known-DM mode 0.34, vs published
#   0.38 dex. The re-weighting variant (our weights × MAGIC's per-branch
#   values) reproduces the published numbers almost exactly — the accuracy
#   gain comes from the estimator (isochrone-tied log g + profiled
#   normalization + [Fe/H] interpolation), while the classification gain
#   comes from the weights; deploying only the weights on existing catalogs
#   fixes classification but not the bias.
# - **Per-star intervals are photometric-only** and cover the mock truth at
#   the nominal rate, but cover real reference values at only 0.15 — the
#   real-data error budget is dominated by systematics (grid, isochrone age,
#   reference-value errors) that no per-star photometric interval contains.
#   A catalog release should add the systematic floor in quadrature
#   (≈ 0.2–0.3 dex, from these comparisons).
# - Scope caveats: the [Fe/H] scan is bounded by the isochrone set at
#   [−2.5, −1.0] (comparisons restricted accordingly); ages fixed at 12 Gyr;
#   ~8% of g195 stars fall outside both branch windows after noise and are
#   flagged invalid rather than forced.
#
# **Bottom line:** stitching MAGIC's own isochrone hypotheses into a
# two-component model average — weights from the five-band χ² and the Gaia
# parallax — upgrades branch classification from 0.85 to 0.94 on
# spectroscopic ground truth and removes the metallicity bias at equal
# scatter, using only CaHK, g, r, i and data products MAGIC already computes.
