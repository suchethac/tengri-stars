"""MAGIC stitch campaign: branch-policy × parallax-quality mock matrix (CaHK, g, r, i).

Twelve configs — policies {magic-hard, hard-vote, mixture} × parallax quality
{none, p50, p20, knownDM} — each scoring the same self-consistent isochrone mocks
(notebook 09 pattern) with `tengri_stars.branch`. One CSV per config; a config is
"done" iff its CSV exists (resumable; delete to redo). Real-data validation lives in
notebook 10, not here.

Run from the repo root in the ARM tengri env:
    JAX_PLATFORMS=cpu python bench/magic_stitch_campaign.py --next
    JAX_PLATFORMS=cpu python bench/magic_stitch_campaign.py --status
    JAX_PLATFORMS=cpu python bench/magic_stitch_campaign.py --summarize
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence XLA/PJRT C++ chatter

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from tengri_stars import StarModel, load_photometry_grid
from tengri_stars.branch import binary_scan, combine_mixture, load_dartmouth_branches

ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "data" / "TSLTE_combined_photometry.fits"
ISO_DIR = Path.home() / "Documents/MIT_Work/Research/magic_scratch/isochrones"
RESULTS = ROOT / "notebooks" / "stitch_results"

BANDS = ("CaHK_filter_ab", "DECCAM_g_des_ab", "DECCAM_r_des_ab", "DECCAM_i_des_ab")
IG, II = 1, 3
CHI2_BANDS = (0, 1, 2, 3)  # all bands; shape normalization profiled analytically
SIGMA = np.array([0.03, 0.02, 0.02, 0.02])
ISO_FILES = {-2.5: "fehm25_12Gyr.txt", -2.0: "fehm20_12Gyr.txt",
             -1.5: "fehm15_12Gyr.txt", -1.0: "fehm10_12Gyr.txt"}

POLICIES = ("magic-hard", "hard-vote", "mixture")
PLX_MODES = ("none", "p50", "p20", "knownDM")  # fractional parallax error / known DM
CONFIGS = [f"{pol}_{plx}" for pol in POLICIES for plx in PLX_MODES]

DM_TRUE = 16.5
N_POS, N_MC = 12, 60
LOGG_CUT = 3.3
MOCK_SEED = 47

FIELDS = ("config", "policy", "plx", "branch", "feh_true", "gi", "seed",
          "p_rgb", "feh_est", "feh_lo", "feh_hi", "correct")


def _setup():
    grid = load_photometry_grid(GRID_PATH, filter_columns=BANDS, fill="nearest")
    model = StarModel(grid=grid, interp_method="pchip")
    tables = load_dartmouth_branches(
        {feh: str(ISO_DIR / fn) for feh, fn in ISO_FILES.items()}, model, ig=IG, ii=II
    )
    return grid, model, tables


def _mocks(model, tables):
    """Isochrone-sampled truth stars + noisy observations (fixed seed everywhere)."""
    rng = np.random.default_rng(MOCK_SEED)
    truths = []
    for feh in tables.feh_nodes:
        krow = int(np.argmin(np.abs(tables.feh_scan - feh)))
        lo, hi = tables.overlap(krow)
        for gi0 in np.linspace(lo + 0.02, hi - 0.02, N_POS):
            j = np.searchsorted(tables.gi_grid, gi0)
            for b in ("RGB", "MS"):
                tab = tables.scan[b]
                teff0, logg0, mg0 = (float(tab[q][krow, j]) for q in ("teff", "logg", "mg"))
                phot = np.asarray(model.predict_mags(teff=teff0, logg=logg0, feh=float(feh)))
                truths.append({"branch": b, "feh": float(feh), "gi": float(gi0),
                               "mags": phot + (mg0 - phot[IG]) + DM_TRUE})
    obs = np.stack([t["mags"] for t in truths])[:, None, :] + \
        rng.normal(0.0, 1.0, (len(truths), N_MC, len(BANDS))) * SIGMA
    return truths, obs.reshape(-1, len(BANDS)), np.repeat(np.arange(len(truths)), N_MC)


def _parallax_data(plx, n_obs, rng):
    """(parallax, parallax_error) arrays for the mode, or (None, None)."""
    if plx in ("none", "knownDM"):
        return None, None
    frac = {"p50": 0.5, "p20": 0.2}[plx]
    plx_true = 10.0 ** (2.0 - DM_TRUE / 5.0)  # mas
    err = frac * plx_true
    return plx_true + rng.normal(0.0, err, n_obs), np.full(n_obs, err)


def run_config(name):
    policy, plx = name.rsplit("_", 1)
    _, model, tables = _setup()
    truths, obs, tidx = _mocks(model, tables)
    n_obs = obs.shape[0]
    rng = np.random.default_rng(MOCK_SEED + 1)

    t0 = time.time()
    dm_known = DM_TRUE if plx == "knownDM" else None
    gi_sigma = float(np.hypot(SIGMA[IG], SIGMA[II]))  # color-noise marginalization
    scans = binary_scan(obs, SIGMA, tables, model, ig=IG, ii=II,
                        chi2_bands=CHI2_BANDS, dm_known=dm_known, gi_sigma=gi_sigma,
                        n_quad=25 if dm_known is not None else 9)
    parallax, plx_err = _parallax_data(plx, n_obs, rng)

    chi2_key, feh_key = ("chi2_dm", "feh_dm") if plx == "knownDM" else ("chi2", "feh")
    if policy == "mixture":
        out = combine_mixture(scans, obs[:, IG], tables.feh_scan,
                              parallax=parallax, parallax_error=plx_err,
                              use_dm=(plx == "knownDM"))
        p_rgb, feh_est = out["p_rgb"], out["feh"]
        feh_lo, feh_hi = out["feh_lo"], out["feh_hi"]
    else:
        if policy == "hard-vote":
            pick_rgb = scans["RGB"][chi2_key] < scans["MS"][chi2_key]
        elif plx == "knownDM":  # magic-hard with a known distance = CMD position
            pick_rgb = scans["RGB"]["chi2_dm"] < scans["MS"]["chi2_dm"]
        elif parallax is not None:  # magic-hard: parallax-only classification
            dm_b = {b: obs[:, IG] - scans[b]["mg"][
                np.argmin(scans[b]["chi2_curve"], axis=0), np.arange(n_obs)]
                for b in ("RGB", "MS")}
            like = {b: -((parallax - 10.0 ** (2.0 - dm_b[b] / 5.0)) / plx_err) ** 2
                    for b in ("RGB", "MS")}
            pick_rgb = like["RGB"] > like["MS"]
        else:  # magic-hard with nothing: coin flip
            pick_rgb = rng.random(n_obs) < 0.5
        p_rgb = pick_rgb.astype(float)
        feh_est = np.where(pick_rgb, scans["RGB"][feh_key], scans["MS"][feh_key])
        feh_lo = feh_hi = np.full(n_obs, np.nan)

    rows = []
    for i in range(n_obs):
        t = truths[tidx[i]]
        correct = (p_rgb[i] >= 0.5) == (t["branch"] == "RGB")
        rows.append({
            "config": name, "policy": policy, "plx": plx, "branch": t["branch"],
            "feh_true": t["feh"], "gi": round(t["gi"], 3), "seed": i % N_MC,
            "p_rgb": round(float(p_rgb[i]), 4), "feh_est": round(float(feh_est[i]), 3),
            "feh_lo": round(float(feh_lo[i]), 3), "feh_hi": round(float(feh_hi[i]), 3),
            "correct": int(correct),
        })

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{name}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{name}] {n_obs} stars in {time.time() - t0:.0f}s -> {out_path}")


def _read(name):
    with open(RESULTS / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def summarize():
    done = [n for n in CONFIGS if (RESULTS / f"{n}.csv").exists()]
    if not done:
        print("no results yet")
        return
    lines = [
        "# MAGIC stitch campaign — policy × parallax-quality mock matrix",
        "",
        f"Bands CaHK,g,r,i; σ = {tuple(SIGMA)}; isochrone mocks at "
        f"[Fe/H] {{-2.5,-2,-1.5,-1}}, DM = {DM_TRUE}; {N_POS} positions × 2 branches × "
        f"{N_MC} noise draws. 'coverage' = fraction of truth inside the 68% interval "
        "(mixture only; the hard policies have no interval).",
        "",
        "| config | P(correct) | feh bias | feh 68% half-width | 68% coverage |",
        "|---|---|---|---|---|",
    ]
    for name in CONFIGS:
        if name not in done:
            lines.append(f"| {name} | pending | | | |")
            continue
        rows = _read(name)
        err = np.array([float(r["feh_est"]) - float(r["feh_true"]) for r in rows])
        ok = np.array([int(r["correct"]) for r in rows]).mean()
        lo, med, hi = np.percentile(err, [16, 50, 84])
        cov = ""
        if rows[0]["policy"] == "mixture":
            inside = np.array([float(r["feh_lo"]) <= float(r["feh_true"]) <= float(r["feh_hi"])
                               for r in rows])
            # interior [Fe/H] nodes only — at the scan edges (−2.5, −1.0) the
            # posterior is mechanically truncated and coverage is meaningless
            interior = np.array([float(r["feh_true"]) in (-2.0, -1.5) for r in rows])
            cov = f"{inside.mean():.2f} ({inside[interior].mean():.2f} interior)"
        lines.append(f"| {name} | {ok:.2f} | {med:+.3f} | {(hi - lo) / 2:.3f} | {cov} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--next", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    pending = [n for n in CONFIGS if not (RESULTS / f"{n}.csv").exists()]
    if args.status:
        print(f"done {len(CONFIGS) - len(pending)}/{len(CONFIGS)}; pending: "
              f"{', '.join(pending) or '—'}")
    elif args.summarize:
        summarize()
    else:
        if not pending:
            print("QUEUE EMPTY — all mock configs done")
            sys.exit(0)
        run_config(pending[0])
        print(f"remaining: {', '.join(pending[1:]) or 'NONE — queue empty after this'}")


if __name__ == "__main__":
    main()
