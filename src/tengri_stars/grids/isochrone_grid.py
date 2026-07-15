"""MIST isochrone grid: (M_init, EEP, [Fe/H]) → stellar structure.

Chains *upstream* of :class:`~tengri_stars.grids.PhotometryGrid`. Where the
photometry grid answers "what does an atmosphere with these properties look
like", this grid answers "which atmospheres can a real star actually have" —
restricting (Teff, log g) to the one-dimensional locus a star of a given mass,
age and composition occupies, and supplying the luminosity (hence radius) that
turns the photometry grid's radius-free magnitudes into apparent ones.

Equivalent Evolutionary Point (EEP) is the evolutionary coordinate rather than
age because it resolves fast phases at uniform density: a 1 M_sun star spends
~10 Gyr on the main sequence and ~1 Gyr on the red-giant branch, so in age the
whole giant branch is a sliver a sampler would step over. The age weight
``d(age)/d(EEP)`` converts back — see
:meth:`~tengri_stars.parametrization.MISTTrack.log_prior_extra`.

Grid coverage is not rectangular. Tracks terminate at
``eep_max(mini, feh) = min(808, EEP at 13.8 Gyr)``: most cells hit the EEP
ceiling of the source grid, while low-mass and metal-rich cells run out of
Universe first. Sampling a *fraction* along the track (see
:meth:`IsochroneGrid.eep_from_fraction`) restores a unit cube.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import jax.numpy as jnp
import numpy as np
from tengri.utils import edges_for_grid, interp_nd_triweight
from tengri.utils.grid_interp import interp_nd_pchip

#: Payload columns of :attr:`IsochroneGrid.params`, in order.
PARAM_NAMES = ("logt", "logg", "logl", "feh_surf", "loga", "log_agewt")

#: IAU 2015 nominal solar effective temperature [K].
TEFF_SUN = 5772.0
_LOG10_TEFF_SUN = float(np.log10(TEFF_SUN))


@dataclass(frozen=True)
class IsochroneGrid:
    """Dense (M_init, EEP, [Fe/H]) → stellar-structure lookup table.

    Parameters
    ----------
    axes : tuple of jnp.ndarray
        Sorted node values per axis: (mini [M_sun], eep [-], feh [dex]).
        ``feh`` is the *initial* metallicity; the emergent photospheric value
        is the ``feh_surf`` payload column, which atomic diffusion separates
        from it by up to ~0.9 dex.
    edges : tuple of jnp.ndarray
        Triweight bin edges per axis, from :func:`tengri.utils.edges_for_grid`.
    params : jnp.ndarray, shape (n_mini, n_eep, n_feh, 6)
        Structure payload, columns ordered as :data:`PARAM_NAMES`:
        ``logt`` = log10 Teff [K], ``logg`` = log10 surface gravity [cgs],
        ``logl`` = log10 L/L_sun, ``feh_surf`` [dex], ``loga`` = log10 age [yr],
        ``log_agewt`` = log10 d(age)/d(EEP) [yr per EEP].
    eep_max : jnp.ndarray, shape (n_mini, n_feh)
        Terminal EEP of each track — the surface bounding grid coverage.
    eep_min : float
        First EEP node — the zero point of the track fraction [-].

        A plain Python float, materialized at load, *not* read back out of
        ``axes[1]`` on demand: ``float(axes[1][0])`` raises
        ``ConcretizationTypeError`` under ``jax.jit``, because JAX traces the
        indexing operation into a tracer even when the array itself is a
        closure constant. It works eagerly and under ``grad``, and fails only
        inside a jitted graph — which is to say, only in production.
    coverage : jnp.ndarray, shape (n_mini, n_eep, n_feh)
        True where the source grid provided the node; False where filled.

    Notes
    -----
    Interpolation is JIT/grad/vmap-safe. ``log_agewt`` is tabulated in log
    rather than linear space so that interpolation cannot return a
    non-positive age weight, whose logarithm the prior takes.

    ``eep_max`` is interpolated with the monotone cubic (PCHIP), never the
    triweight kernel. A *smoother* would not return the tabulated terminus at a
    node and could overshoot between nodes, letting a track fraction of 1.0
    land past the end of the track — a domain boundary has to be node-exact.
    """

    axes: tuple[jnp.ndarray, ...]
    edges: tuple[jnp.ndarray, ...]
    params: jnp.ndarray
    eep_max: jnp.ndarray
    eep_min: float
    coverage: jnp.ndarray

    def eep_from_fraction(self, *, mini, feh, x_eep):
        """Map a track fraction onto a native EEP; JIT/grad/vmap-safe.

        Parameters
        ----------
        mini : scalar
            Initial stellar mass [M_sun].
        feh : scalar
            Initial metallicity [dex].
        x_eep : scalar
            Fraction along the track, 0 at ``eep_min`` and 1 at the track's
            terminal EEP. Sampling ``x_eep`` on the unit interval rather than
            EEP on a fixed range keeps every draw inside grid coverage, which
            is the point: the raw (mini, eep, feh) cube is only 76% filled.

        Returns
        -------
        scalar
            EEP, in the native units of ``axes[1]``.
        """
        span = self.eep_span(mini=mini, feh=feh)
        return self.eep_min + x_eep * span

    def eep_span(self, *, mini, feh):
        """Track length ``eep_max(mini, feh) - eep_min``; the EEP Jacobian.

        Parameters
        ----------
        mini : scalar
            Initial stellar mass [M_sun].
        feh : scalar
            Initial metallicity [dex].

        Returns
        -------
        scalar
            ``d(EEP) / d(x_eep)`` [EEP]. Enters the age prior as the second
            half of the chain rule ``d(age)/d(x) = agewt * span``.
        """
        eep_max = interp_nd_pchip(self.eep_max, (self.axes[0], self.axes[2]), (mini, feh))
        return eep_max - self.eep_min

    def interpolate(self, *, mini, eep, feh, method: str = "triweight") -> dict:
        """Interpolate stellar structure at one point; JIT/grad/vmap-safe.

        Parameters
        ----------
        mini : scalar
            Initial stellar mass [M_sun].
        eep : scalar
            Equivalent evolutionary point [-], native units.
        feh : scalar
            Initial metallicity [dex].
        method : {'triweight', 'pchip'}
            ``'triweight'``: C²-smooth kernel smoother (best gradients for
            HMC/NUTS). ``'pchip'``: node-exact monotone cubic, C¹.

        Returns
        -------
        dict
            Keys of :data:`PARAM_NAMES`, each a scalar.
        """
        point = (mini, eep, feh)
        if method == "pchip":
            values = interp_nd_pchip(self.params, self.axes, point)
        elif method == "triweight":
            values = interp_nd_triweight(self.params, self.axes, self.edges, point)
        else:
            raise ValueError(f"method must be 'triweight' or 'pchip', got {method!r}")
        return dict(zip(PARAM_NAMES, values, strict=True))


def log_radius_rsun(*, logt, logl):
    """log10 stellar radius in solar units, from Stefan-Boltzmann.

    .. math::

        \\log_{10}\\frac{R}{R_\\odot} = \\tfrac{1}{2}\\log_{10}\\frac{L}{L_\\odot}
            + 2\\left(\\log_{10} T_{\\rm eff,\\odot} - \\log_{10} T_{\\rm eff}\\right)

    where :math:`T_{\\rm eff,\\odot}` = 5772 K (IAU 2015 nominal) and both
    luminosity and radius are in solar units — so no dimensional constants
    appear, and the whole expression stays in logs (no overflow, C-infinity).

    Parameters
    ----------
    logt : scalar
        log10 effective temperature [log10 K].
    logl : scalar
        log10 bolometric luminosity [log10 L_sun].

    Returns
    -------
    scalar
        log10 R/R_sun [-]. JIT/grad/vmap-safe.

    Notes
    -----
    On the source grid this agrees with the independent route
    :math:`R = \\sqrt{GM/g}` to a median 0.14%, which is the consistency check
    that licenses taking radius from (L, Teff) rather than from (M, g).
    """
    return 0.5 * logl + 2.0 * (_LOG10_TEFF_SUN - logt)


def load_isochrone_grid(path: str) -> IsochroneGrid:
    """Load a brutus-format MIST grid into an :class:`IsochroneGrid`.

    Reads the ``labels`` (mini, eep, feh) and ``parameters`` (loga, logl, logt,
    logg, feh_surf, agewt) tables, discarding the file's own ``mag_coeffs``:
    the atmosphere comes from :class:`~tengri_stars.grids.PhotometryGrid`, not
    from this file's bolometric corrections.

    Parameters
    ----------
    path : str
        Path to a ``grid_mist_*.h5`` file.

    Returns
    -------
    IsochroneGrid

    Raises
    ------
    ValueError
        If any (mini, feh) cell is empty after trimming, which would leave
        ``eep_max`` undefined there.

    Notes
    -----
    Coverage repair, in order: metallicity slices with any empty (mini, feh)
    cell are **dropped** (in ``grid_mist_v9`` this is the [Fe/H] = +0.45 node,
    empty for 27 of 61 masses); internal gaps along EEP are **linearly
    interpolated** (34 ragged cells); and nodes beyond a track's terminal EEP
    are **held at the last real value**, so the interpolation kernel always has
    finite neighbours even when evaluated right at ``eep_max``. Only the third
    of these is ever touched at fit time, and only marginally, because
    :meth:`IsochroneGrid.eep_from_fraction` confines draws to ``eep <= eep_max``.
    """
    with h5py.File(path, "r") as handle:
        labels = handle["labels"][:]
        parameters = handle["parameters"][:]

    mini, eep, feh = labels["mini"], labels["eep"], labels["feh"]
    axes_np = tuple(np.unique(np.asarray(a, dtype=float)) for a in (mini, eep, feh))

    payload = np.stack(
        [
            parameters["logt"],
            parameters["logg"],
            parameters["logl"],
            parameters["feh_surf"],
            parameters["loga"],
            np.log10(np.maximum(parameters["agewt"], 1e-12)),
        ],
        axis=-1,
    )

    shape = tuple(a.size for a in axes_np)
    values = np.full((*shape, len(PARAM_NAMES)), np.nan)
    coverage = np.zeros(shape, dtype=bool)
    idx = tuple(np.searchsorted(axis, col) for axis, col in zip(axes_np, (mini, eep, feh)))
    values[idx] = payload
    coverage[idx] = True

    axes_np, values, coverage = _drop_empty_feh_slices(axes_np, values, coverage)
    eep_max_np = _terminal_eep(axes_np[1], coverage)
    values = _fill_along_eep(values, coverage)

    axes = tuple(jnp.asarray(a) for a in axes_np)
    return IsochroneGrid(
        axes=axes,
        edges=tuple(edges_for_grid(a) for a in axes),
        params=jnp.asarray(values),
        eep_max=jnp.asarray(eep_max_np),
        eep_min=float(axes_np[1][0]),
        coverage=jnp.asarray(coverage),
    )


def _drop_empty_feh_slices(axes_np, values, coverage):
    """Drop [Fe/H] nodes where some mass has no track at all."""
    per_cell = coverage.sum(axis=1)  # (n_mini, n_feh)
    keep = ~(per_cell == 0).any(axis=0)
    if keep.all():
        return axes_np, values, coverage
    axes_np = (axes_np[0], axes_np[1], axes_np[2][keep])
    return axes_np, values[:, :, keep], coverage[:, :, keep]


def _terminal_eep(eep_axis: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Last covered EEP per (mini, feh); the surface bounding the track."""
    any_covered = coverage.any(axis=1)
    if not any_covered.all():
        n_empty = int((~any_covered).sum())
        raise ValueError(
            f"{n_empty} (mini, feh) cells have no track after trimming; "
            "eep_max is undefined there."
        )
    last = coverage.shape[1] - 1 - np.argmax(coverage[:, ::-1, :], axis=1)
    return eep_axis[last]


def _fill_along_eep(values: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Linearly interpolate internal EEP gaps, then hold the last value beyond."""
    filled = values.copy()
    n_eep = coverage.shape[1]
    positions = np.arange(n_eep, dtype=float)
    for i, k in np.ndindex(coverage.shape[0], coverage.shape[2]):
        covered = coverage[i, :, k]
        if covered.all():
            continue
        src = positions[covered]
        for c in range(values.shape[-1]):
            filled[i, :, k, c] = np.interp(positions, src, values[i, covered, k, c])
    return filled
