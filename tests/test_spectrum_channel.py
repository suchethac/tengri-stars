"""Tests for the spectral grid and the spectrum forward chain (RV → LSF → amplitude)."""

import jax.numpy as jnp
import numpy as np
from tengri.utils.physics_constants import C_KM_S

from tengri_stars import StarModel
from tengri_stars.grids import SpectralGrid

WAVE = np.linspace(4800.0, 5200.0, 801)  # rest-frame vacuum [Å], 0.5 Å sampling
LINE_CENTER = 5000.0
LINE_SIGMA = 1.0  # [Å] intrinsic

TEFF_NODES = np.array([4000.0, 4500.0, 5000.0])
LOGG_NODES = np.array([1.0, 2.0, 3.0])
FEH_NODES = np.array([-3.0, -1.5, 0.0])


def _line_depth(feh):
    """Metal line deepens with metallicity — the [Fe/H] information carrier."""
    return 0.2 + 0.15 * (feh + 3.0)


def _toy_spectral_grid():
    flux = np.empty((3, 3, 3, WAVE.size))
    for i, _teff in enumerate(TEFF_NODES):
        for j, _logg in enumerate(LOGG_NODES):
            for k, feh in enumerate(FEH_NODES):
                profile = np.exp(-0.5 * ((WAVE - LINE_CENTER) / LINE_SIGMA) ** 2)
                flux[i, j, k] = 1.0 - _line_depth(feh) * profile
    return SpectralGrid.from_arrays(
        teff=TEFF_NODES, logg=LOGG_NODES, feh=FEH_NODES, wave=WAVE, flux=flux
    )


def test_spectral_grid_node_spectrum_exact_with_pchip():
    grid = _toy_spectral_grid()

    flux = grid.interpolate(teff=4500.0, logg=2.0, feh=-1.5, method="pchip")

    expected = 1.0 - _line_depth(-1.5) * np.exp(-0.5 * ((WAVE - LINE_CENTER) / LINE_SIGMA) ** 2)
    assert flux.shape == (WAVE.size,)
    np.testing.assert_allclose(flux, expected, rtol=1e-12)


def test_rv_shift_moves_line_center():
    model = StarModel(grid=None, spectral_grid=_toy_spectral_grid())
    rv = 300.0  # [km/s]
    wave_obs = jnp.linspace(4900.0, 5100.0, 4001)  # dense to localize the minimum

    flux = model.predict_spectrum(wave_obs=wave_obs, teff=4500.0, logg=2.0, feh=-1.5, rv_kms=rv)

    observed_center = float(wave_obs[jnp.argmin(flux)])
    expected_center = LINE_CENTER * (1.0 + rv / C_KM_S)
    assert abs(observed_center - expected_center) < 0.1  # within two pixels


def test_lsf_broadens_line_but_preserves_equivalent_width():
    model = StarModel(grid=None, spectral_grid=_toy_spectral_grid())
    wave_obs = jnp.linspace(4900.0, 5100.0, 2001)
    kwargs = dict(wave_obs=wave_obs, teff=4500.0, logg=2.0, feh=-1.5)

    sharp = model.predict_spectrum(**kwargs)
    broad = model.predict_spectrum(**kwargs, resolution=1000.0)

    assert float(jnp.min(broad)) > float(jnp.min(sharp))  # shallower line
    dw = float(wave_obs[1] - wave_obs[0])
    ew_sharp = float(jnp.sum(1.0 - sharp) * dw)
    ew_broad = float(jnp.sum(1.0 - broad) * dw)
    np.testing.assert_allclose(ew_broad, ew_sharp, rtol=0.02)  # convolution conserves EW


def test_missing_grids_raise_clear_errors():
    model = StarModel(spectral_grid=_toy_spectral_grid())
    with np.testing.assert_raises_regex(ValueError, "photometry grid"):
        model.predict_mags(teff=4500.0, logg=2.0, feh=-1.5)
    with np.testing.assert_raises_regex(ValueError, "spectral grid"):
        StarModel().predict_spectrum(wave_obs=WAVE, teff=4500.0, logg=2.0, feh=-1.5)


def test_log_amp_scales_flux_multiplicatively():
    model = StarModel(grid=None, spectral_grid=_toy_spectral_grid())
    wave_obs = jnp.linspace(4900.0, 5100.0, 501)
    kwargs = dict(wave_obs=wave_obs, teff=4500.0, logg=2.0, feh=-1.5)

    base = model.predict_spectrum(**kwargs)
    scaled = model.predict_spectrum(**kwargs, log_amp=-0.4)

    np.testing.assert_allclose(scaled, base * 10.0**-0.4, rtol=1e-10)
