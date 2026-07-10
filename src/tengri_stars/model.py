"""StarModel: the single-star forward model (photometry and spectrum channels).

Photometry chain (magnitudes):

.. math::

    m_X = m_X^{\\rm grid}(T_{\\rm eff}, \\log g, [{\\rm Fe/H}])
          + R_X \\, E(B-V) + \\mu

where :math:`m_X^{\\rm grid}` is the pre-integrated grid magnitude in filter
:math:`X` [AB mag], :math:`R_X = A_X / E(B-V)` is the per-filter reddening
coefficient, and :math:`\\mu` is the dilution term — distance modulus plus any
radius / grid-zero-point offset, one nuisance scalar in magnitude space.
Because the grid's zero-point convention is absorbed by :math:`\\mu`, the fit
is insensitive to it; only distance interpretations (parallax priors,
isochrones) need the convention pinned.

Spectrum chain: grid spectrum at (Teff, log g, [Fe/H]) → radial-velocity shift
:math:`\\lambda_{\\rm obs} = \\lambda_{\\rm rest}(1 + v_r/c)` → resample onto
instrument pixels → instrument LSF convolution (tengri ``apply_lsf``) →
multiplicative amplitude :math:`10^{\\log A}` (absorbs distance dilution and
flux calibration). Foreground extinction on the spectrum shape is not yet
applied — slated to reuse tengri's ``foreground`` laws.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from tengri.observation.spectrum import apply_lsf
from tengri.utils.physics_constants import C_KM_S

from tengri_stars.grids import PhotometryGrid, SpectralGrid


@dataclass(frozen=True)
class StarModel:
    """Differentiable star → observables forward model.

    Parameters
    ----------
    grid : PhotometryGrid, optional
        Pre-integrated (Teff, log g, [Fe/H]) → per-filter magnitude table.
        Required for :meth:`predict_mags`.
    extinction_coeffs : jnp.ndarray, shape (n_filters,), optional
        Per-filter reddening coefficients :math:`R_X = A_X / E(B-V)`
        [mag per mag of E(B-V)], aligned with ``grid.filter_names``.
        When None, ``ebmv`` has no effect.
    spectral_grid : SpectralGrid, optional
        (Teff, log g, [Fe/H]) → spectrum table. Required for
        :meth:`predict_spectrum`.
    interp_method : {'triweight', 'pchip'}, optional
        Override the interpolation method for both channels. When None
        (default), each channel uses its own default: triweight for
        photometry (C² gradients for HMC), pchip for spectra (node-exact).
    """

    grid: PhotometryGrid | None = None
    extinction_coeffs: jnp.ndarray | None = None
    spectral_grid: SpectralGrid | None = None
    interp_method: str | None = None

    def predict_mags(self, *, teff, logg, feh, mu=0.0, ebmv=0.0) -> jnp.ndarray:
        """Predict apparent AB magnitudes for one star.

        Parameters
        ----------
        teff : scalar
            Effective temperature [K].
        logg : scalar
            Surface gravity [dex (cm/s²)].
        feh : scalar
            Metallicity [Fe/H] [dex].
        mu : scalar
            Dilution [mag]: distance modulus plus radius / zero-point offset.
        ebmv : scalar
            Foreground reddening E(B-V) [mag]; requires ``extinction_coeffs``.

        Returns
        -------
        jnp.ndarray, shape (n_filters,)
            Apparent magnitudes [AB mag]. JIT/grad/vmap-safe.
        """
        if self.grid is None:
            raise ValueError("StarModel has no photometry grid; pass grid= to predict mags.")
        method = self.interp_method or "triweight"
        mags = self.grid.interpolate(teff=teff, logg=logg, feh=feh, method=method)
        if self.extinction_coeffs is not None:
            mags = mags + self.extinction_coeffs * ebmv
        return mags + mu

    def predict_spectrum(
        self,
        *,
        wave_obs,
        teff,
        logg,
        feh,
        rv_kms=0.0,
        log_amp=0.0,
        resolution=None,
        sigma_lib_kms=0.0,
    ) -> jnp.ndarray:
        """Predict the observed-frame spectrum on instrument pixels.

        Parameters
        ----------
        wave_obs : array_like, shape (n_pix,)
            Observed-frame vacuum wavelengths of the instrument pixels [Å].
        teff : scalar
            Effective temperature [K].
        logg : scalar
            Surface gravity [dex (cm/s²)].
        feh : scalar
            Metallicity [Fe/H] [dex].
        rv_kms : scalar
            Barycentric radial velocity [km/s]; positive = redshifted.
        log_amp : scalar
            log10 multiplicative amplitude (absorbs distance dilution and
            absolute flux calibration).
        resolution : scalar or array_like, optional
            Instrument resolving power R = λ/Δλ; None skips LSF convolution.
        sigma_lib_kms : scalar
            Intrinsic broadening of the library spectra [km/s], subtracted in
            quadrature inside tengri's ``apply_lsf``.

        Returns
        -------
        jnp.ndarray, shape (n_pix,)
            Model spectrum on ``wave_obs``. JIT/grad/vmap-safe.
        """
        if self.spectral_grid is None:
            raise ValueError(
                "StarModel has no spectral grid; pass spectral_grid= to predict spectra."
            )
        method = self.interp_method or "pchip"
        flux_rest = self.spectral_grid.interpolate(teff=teff, logg=logg, feh=feh, method=method)
        wave_shifted = self.spectral_grid.wave * (1.0 + rv_kms / C_KM_S)
        flux = jnp.interp(jnp.asarray(wave_obs), wave_shifted, flux_rest)
        if resolution is not None:
            flux = apply_lsf(flux, jnp.asarray(wave_obs), resolution, sigma_lib_kms=sigma_lib_kms)
        return flux * 10.0**log_amp
