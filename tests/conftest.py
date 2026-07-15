"""Shared fixtures: a synthetic MIST grid and a toy photometry grid.

The real grids live under ``data/``, which is gitignored and absent on CI — so
a suite that only tested against them would silently test nothing. These
fixtures reproduce the structure that matters, including all three coverage
defects of the real MIST grid (an empty metallicity slice, a ragged internal
gap, and tracks that terminate at different EEPs), so the repair path is
exercised on every run.
"""

from __future__ import annotations

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from tengri.utils import edges_for_grid

from tengri_stars import PhotometryGrid, StarModel, load_isochrone_grid

MINI = (0.8, 1.0, 1.2)
EEP = (202.0, 208.0, 214.0, 220.0, 226.0)
FEH = (-1.0, 0.0, 0.45)

#: Terminal EEP of each (mini, feh) track; None = the cell has no track at all.
TERMINAL = {
    (0.8, -1.0): 214.0,  # low mass: runs out of Universe early
    (0.8, 0.0): 220.0,
    (0.8, 0.45): None,  # empty -> forces the [Fe/H]=+0.45 slice to be dropped
    (1.0, -1.0): 226.0,
    (1.0, 0.0): 226.0,
    (1.0, 0.45): 226.0,
    (1.2, -1.0): 226.0,
    (1.2, 0.0): 226.0,
    (1.2, 0.45): 226.0,
}

#: A ragged internal gap, present in the real grid at 34 cells.
RAGGED = (1.0, 214.0, 0.0)

N_FILTERS = 4


def toy_payload(mini, eep, feh):
    """Linear in every axis, so linear gap-filling is exact and checkable."""
    return {
        "logt": 3.70 + 0.01 * mini + 0.0002 * (eep - 202.0) + 0.003 * feh,
        "logg": 4.50 - 0.10 * mini - 0.0100 * (eep - 202.0) + 0.010 * feh,
        "logl": 0.00 + 0.50 * mini + 0.0050 * (eep - 202.0) + 0.020 * feh,
        "feh_surf": feh - 0.05,  # diffusion: photospheric != initial
        "loga": 9.50 + 0.30 * mini + 0.0100 * (eep - 202.0),
        "agewt": 1.0e8 * (1.0 + 0.1 * mini),
    }


@pytest.fixture(scope="session")
def toy_iso_path(tmp_path_factory):
    """Write a synthetic brutus-format MIST grid with realistic coverage holes."""
    rows = []
    for mini in MINI:
        for feh in FEH:
            end = TERMINAL[(mini, feh)]
            if end is None:
                continue
            for eep in EEP:
                if eep > end or (mini, eep, feh) == RAGGED:
                    continue
                rows.append((mini, eep, feh, toy_payload(mini, eep, feh)))

    label_dt = np.dtype([(n, "<f8") for n in ("mini", "eep", "feh", "afe", "smf")])
    param_dt = np.dtype(
        [(n, "<f8") for n in ("loga", "logl", "logt", "logg", "feh_surf", "afe_surf", "agewt")]
    )
    labels = np.zeros(len(rows), dtype=label_dt)
    params = np.zeros(len(rows), dtype=param_dt)
    for i, (mini, eep, feh, p) in enumerate(rows):
        labels[i] = (mini, eep, feh, 0.0, 0.0)
        params[i] = (p["loga"], p["logl"], p["logt"], p["logg"], p["feh_surf"], 0.0, p["agewt"])

    path = tmp_path_factory.mktemp("iso") / "grid_toy.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("labels", data=labels)
        handle.create_dataset("parameters", data=params)
    return str(path)


@pytest.fixture(scope="session")
def iso_grid(toy_iso_path):
    """The synthetic MIST grid, loaded and coverage-repaired."""
    return load_isochrone_grid(toy_iso_path)


@pytest.fixture(scope="session")
def star_model():
    """A toy photometry grid spanning the atmospheres the toy tracks reach."""
    teff = jnp.linspace(4000.0, 7000.0, 7)
    logg = jnp.linspace(0.0, 5.0, 6)
    feh = jnp.linspace(-2.0, 0.5, 6)
    axes = (teff, logg, feh)
    t, g, f = jnp.meshgrid(teff, logg, feh, indexing="ij")
    base = 20.0 - 2.5 * jnp.log10(t / 5000.0) + 0.1 * g - 0.2 * f
    phot = jnp.stack([base + 0.3 * k * jnp.log10(t / 5000.0) for k in range(N_FILTERS)], axis=-1)
    grid = PhotometryGrid(
        axes=axes,
        edges=tuple(edges_for_grid(a) for a in axes),
        phot=phot,
        filter_names=tuple(f"band{k}" for k in range(N_FILTERS)),
        coverage=jnp.ones((7, 6, 6), dtype=bool),
    )
    return StarModel(grid=grid, extinction_coeffs=jnp.linspace(3.5, 2.0, N_FILTERS))
