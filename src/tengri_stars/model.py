"""StarModel: the single-star photometry forward model.

Prediction chain (all magnitudes):

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
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from tengri_stars.grids import PhotometryGrid


@dataclass(frozen=True)
class StarModel:
    """Differentiable star → AB-magnitude forward model.

    Parameters
    ----------
    grid : PhotometryGrid
        Pre-integrated (Teff, log g, [Fe/H]) → per-filter magnitude table.
    extinction_coeffs : jnp.ndarray, shape (n_filters,), optional
        Per-filter reddening coefficients :math:`R_X = A_X / E(B-V)`
        [mag per mag of E(B-V)], aligned with ``grid.filter_names``.
        When None, ``ebmv`` has no effect.
    """

    grid: PhotometryGrid
    extinction_coeffs: jnp.ndarray | None = None

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
        mags = self.grid.interpolate(teff=teff, logg=logg, feh=feh)
        if self.extinction_coeffs is not None:
            mags = mags + self.extinction_coeffs * ebmv
        return mags + mu
