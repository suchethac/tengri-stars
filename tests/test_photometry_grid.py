"""Tests for the pre-integrated photometry grid loader (TSLTE-style FITS tables)."""

import jax
import numpy as np
import pytest
from astropy.table import Table

from tengri_stars.grids import load_photometry_grid

# Linear payload: mag = c0 + c_teff*teff + c_logg*logg + c_feh*feh (per filter).
# The triweight kernel is symmetric, so interpolation at an interior node must
# reproduce a linear payload exactly (up to float64 noise).
COEFFS = {
    "lsst_g_ab": (30.0, 1e-3, 0.5, 1.0),
    "cahk": (35.0, -2e-3, 0.3, 2.0),
}
TEFF_NODES = [4000.0, 4500.0, 5000.0]
LOGG_NODES = [1.0, 2.0, 3.0]
FEH_NODES = [-2.0, -1.0, 0.0]


def _mag(name, teff, logg, feh):
    c0, ct, cg, cf = COEFFS[name]
    return c0 + ct * teff + cg * logg + cf * feh


def _toy_table(drop_nodes=(), duplicate_first=False):
    """Build a TSLTE-shaped table: one row per grid node + survey junk columns."""
    rows = []
    for teff in TEFF_NODES:
        for logg in LOGG_NODES:
            for feh in FEH_NODES:
                if (teff, logg, feh) in drop_nodes:
                    continue
                rows.append(
                    {
                        "teff": teff,
                        "logg": logg,
                        "feh": feh,
                        "averaged": False,
                        "lsst_g_ab": _mag("lsst_g_ab", teff, logg, feh),
                        "cahk": _mag("cahk", teff, logg, feh),
                        # Junk duplicate-label columns present in the real FITS.
                        "[Fe/H]_1": feh,
                        "logg_1": logg,
                        "Teff_1": teff,
                    }
                )
    if duplicate_first:
        dup = dict(rows[0])
        dup["averaged"] = True
        dup["lsst_g_ab"] += 0.25  # averaged row differs, must win
        rows.append(dup)
    return Table(rows=rows)


def test_load_builds_sorted_axes_and_grid_shape():
    grid = load_photometry_grid(_toy_table())

    np.testing.assert_allclose(grid.axes[0], TEFF_NODES)
    np.testing.assert_allclose(grid.axes[1], LOGG_NODES)
    np.testing.assert_allclose(grid.axes[2], FEH_NODES)
    assert grid.filter_names == ("lsst_g_ab", "cahk")
    assert grid.phot.shape == (3, 3, 3, 2)
    # Junk label columns must not be picked up as filters.
    assert "[Fe/H]_1" not in grid.filter_names


def test_interpolate_reproduces_linear_payload_at_interior_node():
    grid = load_photometry_grid(_toy_table())

    mags = grid.interpolate(teff=4500.0, logg=2.0, feh=-1.0)

    assert mags.shape == (2,)
    np.testing.assert_allclose(mags[0], _mag("lsst_g_ab", 4500.0, 2.0, -1.0), rtol=1e-8)
    np.testing.assert_allclose(mags[1], _mag("cahk", 4500.0, 2.0, -1.0), rtol=1e-8)


def test_interpolate_is_differentiable_with_correct_slope():
    grid = load_photometry_grid(_toy_table())

    dmag_dteff = jax.grad(lambda t: grid.interpolate(teff=t, logg=2.0, feh=-1.0)[0])(4500.0)

    assert np.isfinite(dmag_dteff)
    # The triweight interpolant reproduces linear payloads exactly at interior
    # nodes but its slope carries O(few %) kernel-regression deviation when the
    # neighbors are boundary nodes (3-node toy axes). Assert "close", not exact.
    np.testing.assert_allclose(dmag_dteff, COEFFS["lsst_g_ab"][1], rtol=0.05)


def test_duplicate_nodes_prefer_averaged_row():
    grid = load_photometry_grid(_toy_table(duplicate_first=True))

    node_mag = grid.phot[0, 0, 0, 0]
    expected = _mag("lsst_g_ab", TEFF_NODES[0], LOGG_NODES[0], FEH_NODES[0]) + 0.25
    np.testing.assert_allclose(node_mag, expected)


def test_missing_nodes_raise_by_default():
    table = _toy_table(drop_nodes=((4500.0, 2.0, -1.0),))

    with pytest.raises(ValueError, match="missing"):
        load_photometry_grid(table)


def test_missing_nodes_nearest_fill_carries_coverage_mask():
    table = _toy_table(drop_nodes=((4500.0, 2.0, -1.0),))

    grid = load_photometry_grid(table, fill="nearest")

    assert bool(grid.coverage[1, 1, 1]) is False
    assert bool(grid.coverage[0, 0, 0]) is True
    assert np.all(np.isfinite(np.asarray(grid.phot)))
