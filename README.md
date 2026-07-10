# tengri-stars

Differentiable Bayesian inference of stellar parameters — Teff, log g, [Fe/H], distance,
extinction — from photometry and spectra, built on the
[tengri](https://github.com/suchethac/tengri) SED-fitting stack.

**Status: pre-alpha, under active development.** APIs will change without notice.

## What it does

Fits individual stars against pre-computed synthetic grids (Turbospectrum LTE to start),
using tengri's machinery end to end: N-D differentiable grid interpolation, filter
integration, spectroscopic forward modeling (LSF, flux calibration), and gradient-based
inference (MAP / NUTS / VI). Fit-time photometry is a single smooth lookup on a
pre-integrated (Teff, log g, [Fe/H]) → per-filter-magnitude table — no wavelength
integral in the hot loop, and exact gradients for free.

First science target: the survey footprint of the MAGIC pipeline
(Chiti et al. 2026, [arXiv:2605.26581](https://arxiv.org/abs/2605.26581)) — CaHK +
broadband metallicities — generalized to a full Bayesian fit, with Gaia XP synthetic
photometry and LSST as the next applications.

```
            tengri-stars (this repo)                   tengri (dependency)
 ┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ grids/     synthetic grid → mag LUT, │   │ utils/grid_interp: triweight N-D     │
 │            spectral flux LUT         │──▶│   interpolation, PreintegratedGrid   │
 │ components/ atmosphere, foreground   │──▶│ observation/: filters + integration, │
 │            extinction, dilution      │   │   LSF, flux calibration, noise       │
 │ model.py   StarModel                 │──▶│ forward/: component orchestrator     │
 │ distance/  dilution | parallax |     │──▶│ inference/: MAP / NUTS / VI backends │
 │            color-space | isochrone   │   │ analysis/: posterior diagnostics     │
 └──────────────────────────────────────┘   └──────────────────────────────────────┘
```

## Install (development)

```bash
git clone https://github.com/suchethac/tengri-stars
cd tengri-stars
pip install -e ".[dev]"
```

This pulls `astro-tengri` from GitHub at a pinned commit (see `pyproject.toml`).

Synthetic grids are **not** shipped with the package — point the loaders at a local copy
(e.g. `TSLTE_combined_photometry.fits`).

## Working with tengri

- The tengri dependency is pinned to a commit SHA; bumps are deliberate and go through CI.
- Anything a galaxy fit could also want (filters, likelihoods, interpolation, inference
  backends) lives in tengri and is contributed upstream. Anything that mentions parallax,
  Teff, or an atmosphere grid lives here.
- Contract tests under `tests/contract/` exercise exactly the tengri surfaces this package
  uses, so a failing pin bump names the seam that moved.

## License

BSD 3-Clause. Developed by Suchetha Cooray and Ani Chiti.
