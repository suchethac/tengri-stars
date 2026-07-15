"""Tests for the two-hypothesis (RGB/MS) branch machinery."""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from tengri_stars import StarModel
from tengri_stars.branch import binary_scan, combine_mixture, load_dartmouth_branches
from tengri_stars.grids import load_photometry_grid

# Four bands (cahk, g, r, i): g−i carries teff, cahk carries feh, and the u-free
# gravity signal lives in cahk/r so the branches are photometrically separable.
BANDS = ("cahk", "g_band", "r_band", "i_band")
COEFFS = {
    "cahk": (35.0, -2.0e-3, 0.30, 2.00),
    "g_band": (30.0, 1.0e-3, 0.05, 1.20),
    "r_band": (29.5, 0.8e-3, 0.15, 0.80),
    "i_band": (29.0, 0.6e-3, 0.02, 0.40),
}
IG, II = 1, 3
CHI2_BANDS = (0, 1, 2, 3)  # all bands; normalization profiled analytically
SIGMA = np.array([0.03, 0.02, 0.02, 0.02])


def _toy_grid():
    rows = []
    for teff in np.linspace(3800.0, 6400.0, 14):
        for logg in [0.5, 1.5, 2.5, 3.5, 4.5, 5.0]:
            for feh in np.linspace(-3.0, -0.5, 6):
                row = {"teff": teff, "logg": logg, "feh": feh}
                for name, (c0, ct, cg, cf) in COEFFS.items():
                    row[name] = c0 + ct * teff + cg * logg + cf * feh
                rows.append(row)
    return load_photometry_grid(Table(rows=rows))


def _toy_isochrones(tmp_path):
    """Two synthetic Dartmouth-format files: MS rises to a turnoff, RGB descends.

    Columns match the Dartmouth layout used by the loader: index 2 LogTeff,
    3 LogG, 6 M_g (others padded).
    """
    files = {}
    for feh, dt in ((-2.5, 0.0), (-1.0, -120.0)):
        rows = []
        # MS: teff climbs to the turnoff, logg ~ 4.5-5.0, faint M_g
        for i, teff in enumerate(np.linspace(4000.0 + dt, 6300.0 + dt, 30)):
            rows.append(
                [
                    i,
                    0.7,
                    np.log10(teff),
                    5.0 - 0.015 * i,
                    0.0,
                    0.0,
                    7.5 - 0.05 * i,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        # RGB: teff descends from the turnoff, logg 3.5 -> 0.6, bright M_g
        for i, teff in enumerate(np.linspace(6200.0 + dt, 4100.0 + dt, 30)):
            rows.append(
                [
                    30 + i,
                    0.8,
                    np.log10(teff),
                    3.5 - 0.1 * i,
                    0.0,
                    0.0,
                    3.0 - 0.12 * i,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        path = tmp_path / f"iso_{feh}.txt"
        np.savetxt(path, np.array(rows))
        files[feh] = str(path)
    return files


@pytest.fixture
def env(tmp_path):
    grid = _toy_grid()
    model = StarModel(grid=grid, interp_method="pchip")
    tables = load_dartmouth_branches(
        _toy_isochrones(tmp_path),
        model,
        ig=IG,
        ii=II,
        gi_grid=np.arange(0.2, 3.2, 0.005),
        feh_step=0.05,
        rgb_logg_max=3.6,
    )
    return grid, model, tables


def _mock_star(model, tables, branch, k_node, gi0, dm=16.0):
    j = np.searchsorted(tables.gi_grid, gi0)
    feh = float(tables.feh_nodes[k_node])
    # scan row exactly at the node
    krow = int(np.argmin(np.abs(tables.feh_scan - feh)))
    tab = tables.scan[branch]
    teff0, logg0, mg0 = (float(tab[q][krow, j]) for q in ("teff", "logg", "mg"))
    phot = np.asarray(model.predict_mags(teff=teff0, logg=logg0, feh=feh))
    mu = (mg0 - phot[IG]) + dm
    return phot + mu, feh


def test_windows_nonempty(env):
    _, _, tables = env
    for k in range(tables.feh_nodes.size):
        lo, hi = tables.overlap(int(np.argmin(np.abs(tables.feh_scan - tables.feh_nodes[k]))))
        assert lo < hi


def test_noiseless_closure(env):
    _, model, tables = env
    for branch in ("RGB", "MS"):
        obs, feh_true = _mock_star(model, tables, branch, k_node=0, gi0=1.2)
        scans = binary_scan(obs, SIGMA, tables, model, ig=IG, ii=II, chi2_bands=CHI2_BANDS)
        assert scans[branch]["chi2"][0] < 1e-3
        assert abs(scans[branch]["feh"][0] - feh_true) < 0.051


def test_wrong_branch_is_worse(env):
    _, model, tables = env
    obs, _ = _mock_star(model, tables, "RGB", k_node=0, gi0=1.2)
    scans = binary_scan(obs, SIGMA, tables, model, ig=IG, ii=II, chi2_bands=CHI2_BANDS)
    assert scans["MS"]["chi2"][0] > scans["RGB"]["chi2"][0]


def test_known_dm_prefers_true_branch(env):
    _, model, tables = env
    obs, _ = _mock_star(model, tables, "RGB", k_node=0, gi0=1.2, dm=16.0)
    scans = binary_scan(
        obs, SIGMA, tables, model, ig=IG, ii=II, chi2_bands=CHI2_BANDS, dm_known=16.0
    )
    assert scans["RGB"]["chi2_dm"][0] < scans["MS"]["chi2_dm"][0]
    # the luminosity gap makes the known-DM discrimination decisive
    assert scans["MS"]["chi2_dm"][0] - scans["RGB"]["chi2_dm"][0] > 25.0


def test_mixture_equal_curves_gives_prior(env):
    _, _, tables = env
    n_scan = tables.feh_scan.size
    flat = {"chi2_curve": np.zeros((n_scan, 1)), "mg": np.full((n_scan, 1), 2.0)}
    scans = {
        "RGB": dict(flat),
        "MS": {"chi2_curve": np.zeros((n_scan, 1)), "mg": np.full((n_scan, 1), 7.0)},
    }
    out = combine_mixture(scans, np.array([18.0]), tables.feh_scan)
    np.testing.assert_allclose(out["p_rgb"], 0.5, atol=1e-12)


def test_mixture_parallax_breaks_tie(env):
    _, _, tables = env
    n_scan = tables.feh_scan.size
    scans = {
        "RGB": {"chi2_curve": np.zeros((n_scan, 1)), "mg": np.full((n_scan, 1), 0.0)},
        "MS": {"chi2_curve": np.zeros((n_scan, 1)), "mg": np.full((n_scan, 1), 6.0)},
    }
    g_obs = np.array([16.0])  # DM_RGB = 16 -> 0.063 mas; DM_MS = 10 -> 1 mas
    out = combine_mixture(
        scans, g_obs, tables.feh_scan, parallax=np.array([1.0]), parallax_error=np.array([0.1])
    )
    assert out["p_rgb"][0] < 1e-6  # parallax says nearby -> the dwarf hypothesis
    out2 = combine_mixture(
        scans, g_obs, tables.feh_scan, parallax=np.array([0.06]), parallax_error=np.array([0.02])
    )
    assert out2["p_rgb"][0] > 0.99


REAL_GRID = Path(__file__).resolve().parents[1] / "data" / "TSLTE_combined_photometry.fits"
REAL_ISO = Path.home() / "Documents/MIT_Work/Research/magic_scratch/isochrones"


@pytest.mark.skipif(
    not (REAL_GRID.exists() and REAL_ISO.exists()),
    reason="real TSLTE grid / Dartmouth isochrones not available",
)
def test_real_grid_noiseless_closure():
    """The notebook-08 self-check as a regression test on the real assets."""
    bands = ("CaHK_filter_ab", "DECCAM_g_des_ab", "DECCAM_r_des_ab", "DECCAM_i_des_ab")
    grid = load_photometry_grid(REAL_GRID, filter_columns=bands, fill="nearest")
    model = StarModel(grid=grid, interp_method="pchip")
    files = {
        feh: str(REAL_ISO / name)
        for feh, name in (
            (-2.5, "fehm25_12Gyr.txt"),
            (-2.0, "fehm20_12Gyr.txt"),
            (-1.5, "fehm15_12Gyr.txt"),
            (-1.0, "fehm10_12Gyr.txt"),
        )
    }
    tables = load_dartmouth_branches(files, model, ig=1, ii=3)
    sigma = np.array([0.03, 0.02, 0.02, 0.02])
    for branch in ("RGB", "MS"):
        for k_node, gi0 in ((1, 0.9), (3, 1.1)):
            krow = int(np.argmin(np.abs(tables.feh_scan - tables.feh_nodes[k_node])))
            lo, hi = tables.overlap(krow)
            gi0 = float(np.clip(gi0, lo + 0.02, hi - 0.02))
            j = np.searchsorted(tables.gi_grid, gi0)
            tab = tables.scan[branch]
            teff0, logg0, mg0 = (float(tab[q][krow, j]) for q in ("teff", "logg", "mg"))
            feh = float(tables.feh_nodes[k_node])
            phot = np.asarray(model.predict_mags(teff=teff0, logg=logg0, feh=feh))
            obs = phot + (mg0 - phot[1]) + 16.5
            scans = binary_scan(obs, sigma, tables, model, ig=1, ii=3, chi2_bands=(0, 1, 2, 3))
            assert scans[branch]["chi2"][0] < 0.01
            assert abs(scans[branch]["feh"][0] - feh) < 0.021
