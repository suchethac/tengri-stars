"""Synthetic spectral grids: (Teff, log g, [Fe/H]) → rest-frame flux spectra."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from tengri.utils import edges_for_grid, interp_nd_triweight
from tengri.utils.grid_interp import interp_nd_pchip

_AXIS_NAMES = ("teff", "logg", "feh")


@dataclass(frozen=True)
class SpectralGrid:
    """Dense (Teff, log g, [Fe/H]) → spectrum lookup table.

    Parameters
    ----------
    axes : tuple of jnp.ndarray
        Sorted node values per axis: (teff [K], logg [dex], feh [dex]).
    edges : tuple of jnp.ndarray
        Triweight bin edges per axis.
    wave : jnp.ndarray, shape (n_wave,)
        Rest-frame vacuum wavelength grid [Å].
    flux : jnp.ndarray, shape (n_teff, n_logg, n_feh, n_wave)
        Spectra on the parameter grid; flux units are the source grid's
        (absolute scale is absorbed by the amplitude parameter at fit time).
    """

    axes: tuple[jnp.ndarray, ...]
    edges: tuple[jnp.ndarray, ...]
    wave: jnp.ndarray
    flux: jnp.ndarray

    @classmethod
    def from_arrays(cls, *, teff, logg, feh, wave, flux) -> SpectralGrid:
        """Build a grid from plain arrays, validating shape consistency.

        Parameters
        ----------
        teff, logg, feh : array_like, shape (n_i,)
            Sorted axis node values ([K], [dex], [dex]).
        wave : array_like, shape (n_wave,)
            Rest-frame vacuum wavelengths [Å].
        flux : array_like, shape (n_teff, n_logg, n_feh, n_wave)
            Spectra at every grid node.
        """
        axes_np = tuple(np.asarray(a, dtype=float) for a in (teff, logg, feh))
        wave_np = np.asarray(wave, dtype=float)
        flux_np = np.asarray(flux, dtype=float)
        expected = (*(a.size for a in axes_np), wave_np.size)
        if flux_np.shape != expected:
            raise ValueError(
                f"flux shape {flux_np.shape} != (n_teff, n_logg, n_feh, n_wave) {expected}"
            )
        for name, axis in zip(_AXIS_NAMES, axes_np):
            if axis.size < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(f"{name} axis must be sorted with >= 2 unique nodes")

        axes = tuple(jnp.asarray(a) for a in axes_np)
        return cls(
            axes=axes,
            edges=tuple(edges_for_grid(a) for a in axes),
            wave=jnp.asarray(wave_np),
            flux=jnp.asarray(flux_np),
        )

    def interpolate(self, *, teff, logg, feh, method: str = "pchip") -> jnp.ndarray:
        """Interpolate the spectrum at one point; JIT/grad/vmap-safe.

        Parameters
        ----------
        teff : scalar
            Effective temperature [K].
        logg : scalar
            Surface gravity [dex (cm/s²)].
        feh : scalar
            Metallicity [Fe/H] [dex].
        method : {'pchip', 'triweight'}
            Default ``'pchip'`` (node-exact, monotone): spectra carry sharp
            features where the triweight smoother's node bias is visible.

        Returns
        -------
        jnp.ndarray, shape (n_wave,)
            Rest-frame spectrum at (teff, logg, feh).
        """
        point = (teff, logg, feh)
        if method == "pchip":
            return interp_nd_pchip(self.flux, self.axes, point)
        if method == "triweight":
            return interp_nd_triweight(self.flux, self.axes, self.edges, point)
        raise ValueError(f"method must be 'triweight' or 'pchip', got {method!r}")
