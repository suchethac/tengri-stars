"""Band-combination forecast campaign: which of (u, CaHK, g, r, i) carry the information?

Each configuration fits the notebook-06 mock sweep (4 Teff × 3 [Fe/H] × {RGB, MS} ×
3 noise seeds = 72 NSS fits, ~2 s each after compile) with one band subset, and writes
one CSV to notebooks/forecast_results/. A config is "done" iff its CSV exists, so the
campaign is resumable and safe to re-run; delete a CSV to redo it.

Run from the repo root in the ARM tengri env:
    JAX_PLATFORMS=cpu python bench/forecast_campaign.py --next       # run next pending
    JAX_PLATFORMS=cpu python bench/forecast_campaign.py --status     # queue state
    JAX_PLATFORMS=cpu python bench/forecast_campaign.py --summarize  # summary.md + png
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence XLA/PJRT C++ chatter

import argparse
import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tengri import Uniform

from tengri_stars import StarModel, load_photometry_grid, make_nss_pipeline

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "data" / "TSLTE_combined_photometry.fits"
RESULTS = ROOT / "notebooks" / "forecast_results"

# All five bands, with the assumed 1σ photometric errors [mag] (notebook 06).
ALL_BANDS = ("lsst_u_ab", "CaHK_filter_ab", "lsst_g_ab", "lsst_r_ab", "lsst_i_ab")
ALL_SIGMA = {"lsst_u_ab": 0.05, "CaHK_filter_ab": 0.03, "lsst_g_ab": 0.02,
             "lsst_r_ab": 0.02, "lsst_i_ab": 0.02}
SHORT = {"lsst_u_ab": "u", "CaHK_filter_ab": "CaHK", "lsst_g_ab": "g",
         "lsst_r_ab": "r", "lsst_i_ab": "i"}

# The queue: name -> band subset (indices into ALL_BANDS). Order = run order.
CONFIGS = {
    "all5": (0, 1, 2, 3, 4),      # ceiling
    "no-u": (1, 2, 3, 4),          # marginal value of u
    "no-cahk": (0, 2, 3, 4),       # marginal value of CaHK
    "gri": (2, 3, 4),              # broadband-only floor
    "magic3": (1, 2, 4),           # the actual MAGIC band set (CaHK, g, i)
    "u-magic3": (0, 1, 2, 4),      # MAGIC + u, drop r
    "ugi": (0, 2, 4),              # LSST-only 3-band, no CaHK
}

# Mock sweep, identical to notebook 06 §4.
TEFF_SWEEP = (4000.0, 4500.0, 5000.0, 5500.0)
FEH_SWEEP = (-3.0, -2.0, -1.0)
CLASSES = (("RGB", 2.0), ("MS", 4.6))
N_SEEDS = 3
LOGG_CUT = 3.3
MU_TRUE = -18.0
NOISE_SEED = 29  # one generator for all configs: same noise draws per star everywhere

FIELDS = ("config", "bands", "teff", "feh", "class", "seed",
          "p_dwarf", "sig_logg", "sig_feh", "feh_err", "logz")


def _load():
    grid = load_photometry_grid(GRID_PATH, filter_columns=ALL_BANDS, fill="nearest")
    return grid, StarModel(grid=grid, interp_method="pchip")


def _noise_draws(rng):
    """Per-star 5-band noise, drawn once in a fixed order shared by every config."""
    draws = {}
    for teff in TEFF_SWEEP:
        for feh in FEH_SWEEP:
            for cls, _logg in CLASSES:
                for seed in range(N_SEEDS):
                    draws[(teff, feh, cls, seed)] = rng.normal(0.0, 1.0, len(ALL_BANDS))
    return draws


def run_config(name: str) -> Path:
    idx = CONFIGS[name]
    bands = "+".join(SHORT[ALL_BANDS[i]] for i in idx)
    grid, model = _load()
    sub = jnp.asarray(idx)
    sigma_full = np.array([ALL_SIGMA[b] for b in ALL_BANDS])
    sigma = jnp.asarray(sigma_full[list(idx)])

    priors = {
        "teff": Uniform(float(grid.axes[0][0]), float(grid.axes[0][-1])),
        "logg": Uniform(float(grid.axes[1][0]), float(grid.axes[1][-1])),
        "feh": Uniform(float(grid.axes[2][0]), float(grid.axes[2][-1])),
        "mu": Uniform(-28.0, -8.0),
    }

    def loglik(params, data):
        pred = model.predict_mags(
            teff=params["teff"], logg=params["logg"], feh=params["feh"], mu=params["mu"]
        )[sub]
        return jnp.sum(-0.5 * ((data - pred) / sigma) ** 2)

    pipe = make_nss_pipeline(loglik, priors)
    draws = _noise_draws(np.random.default_rng(NOISE_SEED))
    key = jax.random.PRNGKey(7)

    rows = []
    t0 = time.time()
    for teff in TEFF_SWEEP:
        for feh in FEH_SWEEP:
            for cls, logg in CLASSES:
                mags_true = np.asarray(
                    model.predict_mags(teff=teff, logg=logg, feh=feh, mu=MU_TRUE)
                )
                for seed in range(N_SEEDS):
                    obs = (mags_true + draws[(teff, feh, cls, seed)] * sigma_full)[list(idx)]
                    key, k = jax.random.split(key)
                    samples, info = pipe(k, obs)
                    s_logg = np.asarray(samples["logg"])
                    s_feh = np.asarray(samples["feh"])
                    rows.append({
                        "config": name, "bands": bands, "teff": teff, "feh": feh,
                        "class": cls, "seed": seed,
                        "p_dwarf": round(float(np.mean(s_logg > LOGG_CUT)), 4),
                        "sig_logg": round(float(np.std(s_logg)), 4),
                        "sig_feh": round(float(np.std(s_feh)), 4),
                        "feh_err": round(float(np.median(s_feh) - feh), 4),
                        "logz": round(info["logz"], 2),
                    })
    elapsed = time.time() - t0

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{name}] ({bands}): {len(rows)} fits in {elapsed:.0f}s -> {out}")
    return out


def _read_all():
    data = {}
    for name in CONFIGS:
        path = RESULTS / f"{name}.csv"
        if path.exists():
            with open(path) as f:
                data[name] = list(csv.DictReader(f))
    return data


def _headline(rows):
    """Median σ([Fe/H]), P(correct RGB), P(correct MS) over the sweep."""
    sig_feh = np.median([float(r["sig_feh"]) for r in rows])
    p_rgb = np.mean([1.0 - float(r["p_dwarf"]) for r in rows if r["class"] == "RGB"])
    p_ms = np.mean([float(r["p_dwarf"]) for r in rows if r["class"] == "MS"])
    return sig_feh, p_rgb, p_ms


def summarize():
    data = _read_all()
    if not data:
        print("no results yet")
        return
    RESULTS.mkdir(exist_ok=True)

    lines = [
        "# Five-band forecast campaign — band-combination sweep",
        "",
        f"Mock sweep: Teff {[int(t) for t in TEFF_SWEEP]} × [Fe/H] {list(FEH_SWEEP)} × "
        f"{{RGB log g 2.0, MS log g 4.6}} × {N_SEEDS} seeds; μ free (no distance); "
        f"σ = (u 0.05, CaHK 0.03, gri 0.02); P(correct) = posterior mass on the "
        f"true side of log g = {LOGG_CUT}. Same noise realizations for every config.",
        "",
        "| config | bands | median σ([Fe/H]) [dex] | P(correct RGB) | P(correct MS) |",
        "|---|---|---|---|---|",
    ]
    table = {}
    for name in CONFIGS:
        if name not in data:
            lines.append(f"| {name} | — | pending | pending | pending |")
            continue
        sig_feh, p_rgb, p_ms = _headline(data[name])
        table[name] = (sig_feh, p_rgb, p_ms)
        bands = data[name][0]["bands"]
        lines.append(f"| {name} | {bands} | {sig_feh:.3f} | {p_rgb:.2f} | {p_ms:.2f} |")

    if "all5" in table:
        lines += ["", "Δ vs all5 ceiling:", ""]
        s0, r0, m0 = table["all5"]
        for name, (s, r, m) in table.items():
            if name == "all5":
                continue
            lines.append(
                f"- **{name}**: σ([Fe/H]) {s - s0:+.3f} dex, "
                f"P(RGB) {r - r0:+.2f}, P(MS) {m - m0:+.2f}"
            )

    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Comparison figure: one panel per headline metric, configs on the x axis.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in CONFIGS if n in table]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, vals) in zip(
        axes,
        (
            ("median σ([Fe/H]) [dex]", [table[n][0] for n in names]),
            ("P(correct | true RGB)", [table[n][1] for n in names]),
            ("P(correct | true MS)", [table[n][2] for n in names]),
        ),
    ):
        ax.bar(range(len(names)), vals, color=["C0" if n == "all5" else "C1" for n in names])
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_title(title)
        if "P(" in title:
            ax.set_ylim(0.4, 1.0)
            ax.axhline(0.5, color="k", lw=0.8, ls=":")
    fig.suptitle("band-combination forecast (mock sweep medians/means)")
    fig.tight_layout()
    fig.savefig(RESULTS / "summary.png", dpi=150)
    print(f"\nwrote {RESULTS / 'summary.md'} and summary.png")


def status():
    done = [n for n in CONFIGS if (RESULTS / f"{n}.csv").exists()]
    pending = [n for n in CONFIGS if n not in done]
    print(f"done ({len(done)}/{len(CONFIGS)}): {', '.join(done) or '—'}")
    print(f"pending: {', '.join(pending) or '—'}")
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--next", action="store_true", help="run the next pending config")
    group.add_argument("--status", action="store_true", help="list done/pending configs")
    group.add_argument("--summarize", action="store_true", help="write summary.md + summary.png")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.summarize:
        summarize()
    else:  # --next
        pending = [n for n in CONFIGS if not (RESULTS / f"{n}.csv").exists()]
        if not pending:
            print("QUEUE EMPTY — all configs done")
            sys.exit(0)
        run_config(pending[0])
        remaining = [n for n in pending[1:]]
        print(f"remaining: {', '.join(remaining) or 'NONE — queue empty after this'}")


if __name__ == "__main__":
    main()
