# CLAUDE.md — tengri-stars

## Purpose

Differentiable Bayesian inference of single-star parameters (Teff, log g, [Fe/H], distance
modulus μ; later extinction, RV) from photometry and spectra, built as a library client of
[tengri](https://github.com/suchethac/tengri) (SED-fitting stack, pinned git dep). Input: a
pre-computed synthetic grid — primarily `TSLTE_combined_photometry.fits` (A. Chiti's
Turbospectrum LTE grid, axes teff×logg×feh → per-filter AB mags). Output: full posteriors
(NSS / NUTS / HMC / MAP) + log-evidence per star. It is the Bayesian generalization of the
deterministic MAGIC pipeline (Chiti et al. 2026, arXiv:2605.26581). Joint project of
Suchetha Cooray (tengri) and Ani Chiti (stars/MAGIC); lives on suchethac's GitHub for now.

## Key files

- `src/tengri_stars/inference.py` — the bulk (772 loc): `fit_map/nss/nuts/hmc`, `make_nss_pipeline`, `make_hmc_pipeline` (one XLA graph, vmap-able over a catalog), result dataclasses.
- `src/tengri_stars/model.py` — `StarModel` (grid + priors → log-prob; `interp_method="pchip"`).
- `src/tengri_stars/grids/photometry_grid.py` — `load_photometry_grid` FITS→LUT; prefers `averaged=True` rows, rejects duplicated label columns (`Teff_1`, …), `fill="nearest"` for coverage holes.
- `src/tengri_stars/grids/spectral_grid.py` — `SpectralGrid.from_arrays` (spectroscopy channel).
- `src/tengri_stars/plotting.py` — `overlay_corner` (full-range, wider corner plots).
- `src/tengri_stars/branch.py` — the MAGIC stitch: two-hypothesis (RGB/MS) isochrone-pinned [Fe/H] scan + Bayesian model average (`load_dartmouth_branches` in grid-color coords, `binary_scan` with profiled-μ shape χ² + color marginalization + per-star errors, `combine_mixture` with parallax term). Tested in `tests/test_branch.py`.
- `bench/benchmark_samplers.py` — per-star sampler timings on the real grid.
- `docs/design-2026-07-09.md` — design decisions, milestones M0–M5, open items. Read before architectural changes.

## Notebooks (jupytext-paired `py:percent` ↔ `.ipynb`, outputs committed)

1. `01_spectrum_to_stellar_parameters` — spectroscopy channel on a physically-motivated toy grid (Hβ→Teff, Mg b→logg, Fe I comb→[Fe/H]); NSS posteriors + evidence.
2. `02_magic_reproduction` — clones `ac8119/magic-processing-scratch@refactor-rebuild` at runtime and runs the *actual* `getFeHs_v2` estimator vs the NSS posterior, same 3 bands, real TSLTE grid.
3. `03_tslte_photometry` — real TSLTE grid, mock star, NSS + NUTS side by side.
4. `04_brutus_crossval` — brutus (MIST/C3K) vs tengri-stars on TSLTE mock, 8 shared bands (SDSS+DECam ugri); measures prior structure + grid-to-grid systematics.
5. `05_fast_pipeline` — `make_hmc_pipeline`: adaptation+sampling in one compiled graph, vmap catalog engine.
6. `06_band_ablation` — (Suchetha) band-ablation study across survey groups and filters, Laplace/ESS machinery.
7. `07_five_band_forecast` — the science question: [Fe/H] + log g information content of (LSST u, CaHK, g, r, i); Fisher/CRLB maps (marginal+conditional, prior-capped) + NSS mock sweep + RGB/MS classification + head-to-head vs the exact MAGIC `getFeHs_v2` (clones `magic-processing-scratch`, needs `dustmaps`). Companion campaign: `bench/forecast_campaign.py` (band-subset sweep → `notebooks/forecast_results/`).
8. `08_rgb_ms_separability` — can photometry alone determine log g? min-χ² dwarf/giant separability maps D(Teff, [Fe/H]) over the impostor manifold (μ analytic), per-band signal, u-depth scaling, NSS validation; §7 Balmer-jump/degeneracy addendum. Answer: giants ID-able only at [Fe/H] ≳ −1; dwarfs never provable; metal-poor regime fundamentally confused.
9. `09_branch_discrimination` — the MAGIC binary test: two isochrone-pinned (log g, μ) hypotheses at the observed (g, g−i), 1-D χ² scan in [Fe/H] per branch (no sampler). Needs Dartmouth isochrones at `~/Documents/MIT_Work/Research/magic_scratch/isochrones/`. Answer: constrained test works (P≈0.9–1.0 cool, ≥0.7 overall); u contributes; known distance decisive; Δχ²-selected [Fe/H] nearly matches the perfect-knowledge ceiling.
10. `10_magic_stitch_validation` — the stitch (`branch.py` mixture) on REAL MAGIC paper data (paths under `/Users/Ani/Dropbox (MIT)/my_papers/magic_overview`, f9 cuts verbatim). Result: classification 0.94 vs MAGIC is_rgb 0.85 (APOGEE LOGG_SPEC truth); bias removed at equal scatter. Mock policy matrix: `bench/magic_stitch_campaign.py` → `notebooks/stitch_results/`.

Edit the `.py` side; keep the pair in sync with jupytext.

## How to run

```bash
# Use the ARM conda env (Python 3.12): ~/miniforge3-arm64/envs/tengri
# NEVER the default ~/anaconda3 — it is x86/Rosetta Python 3.7; jax's AVX wheels crash on it.
conda activate tengri             # via ~/miniforge3-arm64/bin/conda
pip install -e ".[dev]"          # pulls astro-tengri from git at a pinned SHA
pytest                            # unit tests, 300 s timeout, no grid needed
ruff check src tests
# real-grid work needs the FITS (never committed; data/ is gitignored):
scp <sunetid>@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/achiti/grid/TSLTE_combined_photometry.fits data/
PYTHONPATH=src JAX_PLATFORMS=cpu python bench/benchmark_samplers.py
```

Notebooks resolve `data/` vs `../data/` themselves (kernel launches in `notebooks/`).

## Conventions

- Figure style: `~/.claude/skills/paper-fig`. Validation checks: `validation/checks.py` (see `validate` skill). Don't modify paths/labels not named in a request.
- tengri pin bumps are deliberate PRs; anything survey-agnostic (filters, likelihoods, inference) goes upstream to tengri — only star-specific code lives here.
- x64 JAX everywhere (`jax.config.update("jax_enable_x64", True)` before other imports — hence the E402 ruff ignore).
- Wavelengths vacuum Å; distances in parsecs; American English; ruff config mirrors tengri.

## Gotchas

- `astro-tengri` is NOT on PyPI (name squatted); git-URL dep only until first release.
- Notebook 04: JAX/XLA and numba both load libomp → segfault on macOS. Guards required: `NUMBA_THREADING_LAYER=workqueue`, `KMP_DUPLICATE_LIB_OK=TRUE` (set at top of nb04).
- `TF_CPP_MIN_LOG_LEVEL=2` set in every notebook to silence XLA/PJRT C++ chatter.
- README mentions `tests/contract/` (tengri-surface contract tests) — not written yet; tests are flat under `tests/`.
- TSLTE grid open items (design doc §Open items): filter convention, mag zero-point, coverage holes — confirm with Ani before regenerating or mixing grids. The TSLTE zero-point is a convention absorbed by μ, so implied distances from it are meaningless (why nb04 gives brutus no parallax).
- brutus is an optional extra: `pip install -e ".[crossval]"`.

## HANDOFF (auto-updated 2026-07-14)

- **State**: notebooks renumbered 07–10 (Suchetha's `06_band_ablation` holds 06); nb07–nb10 + `branch.py` + both campaigns merged to main and pushed. Grid FITS is a symlink in `data/` → `~/Documents/MIT_Work/Research/magic_scratch/grid/`. Env: `~/miniforge3-arm64/envs/tengri` (jax, jupyter, brutus, dustmaps; kernel "Python (tengri)").
- **Answers so far**: band ranking: CaHK ≫ u ≫ r for [Fe/H]; u+CaHK+g+i ≈ all5. Free-fit log g (nb08): metal-poor regime confused. Binary isochrone test (nb09): works, esp. cool. **The stitch (branch.py + nb10, REAL data): two-hypothesis mixture with χ²+parallax weights beats the current pipeline — branch classification 0.94 vs is_rgb 0.85 on APOGEE spectroscopic log g, [Fe/H] bias −0.09→−0.01 at equal scatter (0.21), g195 scatter 0.38→0.34 (known-DM)**. Mock matrix (12 configs, `stitch_results/summary.md`): mixture ≥ best-of-both at every parallax quality; interior 68% coverage 0.65–0.73.
- **Open question**: nb10 diagnostics (figures in `notebooks/figures/stitch_*.png`) — P(RGB) is over-confident specifically for subgiants in the log g 3.6–4.2 gap between the two hypothesis manifolds (94% of confident failures); fix = third SGB hypothesis or continuous log g family + tempered χ². Interval floor quantified: +0.2 dex → σ_z 1.12 vs APOGEE. 12 Gyr age fixed; [Fe/H] scan bounded [−2.5, −1.0]. TSLTE grid conventions still unverified with Ani.
- **Next step**: extend isochrone set below −2.5 (UMP regime is MAGIC's science case!); add the SGB hypothesis; or draft the method write-up from nb09+nb10.
- **Don't**: (nothing rejected; Ani authorized direct pushes to main 2026-07-14. Coordinate with Suchetha on the overlap between his `06_band_ablation`/Laplace-ESS machinery and the `forecast_campaign` band ranking.)
