"""Pre-integrated photometry grids: (Teff, log g, [Fe/H]) → per-filter AB magnitudes.

Loads TSLTE-style FITS tables (one row per grid node, one column per filter) into a
dense lookup table interpolated with tengri's differentiable triweight kernel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from tengri.utils import edges_for_grid, interp_nd_triweight

_AXIS_COLUMNS = ("teff", "logg", "feh")
# Axis columns and their duplicated survey-label variants ("[Fe/H]_1", "Teff_1", ...).
_NON_FILTER = re.compile(r"^(\[fe/h\]|teff|logg|feh)(_\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class PhotometryGrid:
    """Dense (Teff, log g, [Fe/H]) → AB-magnitude lookup table.

    Parameters
    ----------
    axes : tuple of jnp.ndarray
        Sorted unique node values per axis: (teff [K], logg [dex], feh [dex]).
    edges : tuple of jnp.ndarray
        Triweight bin edges per axis, from :func:`tengri.utils.edges_for_grid`.
    phot : jnp.ndarray, shape (n_teff, n_logg, n_feh, n_filters)
        Grid magnitudes [AB mag], in the zero-point convention of the source
        table (absorbed by the dilution parameter at fit time).
    filter_names : tuple of str
        Filter column names, in source-table order.
    coverage : jnp.ndarray, shape (n_teff, n_logg, n_feh)
        True where the source table provided the node; False where it was filled.
    """

    axes: tuple[jnp.ndarray, ...]
    edges: tuple[jnp.ndarray, ...]
    phot: jnp.ndarray
    filter_names: tuple[str, ...]
    coverage: jnp.ndarray

    def interpolate(self, *, teff, logg, feh) -> jnp.ndarray:
        """Interpolate magnitudes at one point; C²-smooth, JIT/grad/vmap-safe.

        Parameters
        ----------
        teff : scalar
            Effective temperature [K].
        logg : scalar
            Surface gravity [dex (cm/s²)].
        feh : scalar
            Metallicity [Fe/H] [dex].

        Returns
        -------
        jnp.ndarray, shape (n_filters,)
            Interpolated magnitudes [AB mag].
        """
        return interp_nd_triweight(self.phot, self.axes, self.edges, (teff, logg, feh))


def load_photometry_grid(
    source: Table | str,
    *,
    filter_columns: tuple[str, ...] | None = None,
    fill: str = "error",
) -> PhotometryGrid:
    """Load a TSLTE-style photometry table into a :class:`PhotometryGrid`.

    Parameters
    ----------
    source : astropy.table.Table or str
        In-memory table, or path to a FITS table. Must carry ``teff``/``logg``/``feh``
        columns; every other float column is treated as a filter unless it matches a
        duplicated label column (``Teff_1``, ``[Fe/H]_1``, ...) or ``filter_columns``
        is given explicitly.
    filter_columns : tuple of str, optional
        Explicit filter column names, overriding auto-detection.
    fill : {'error', 'nearest'}
        Policy for grid nodes absent from the table. ``'error'`` (default) raises;
        ``'nearest'`` copies the nearest covered node (index-space distance) and
        records the hole in ``coverage``.

    Returns
    -------
    PhotometryGrid

    Raises
    ------
    ValueError
        On duplicate nodes with equal ``averaged`` flags, missing nodes under
        ``fill='error'``, or an unknown ``fill`` policy.
    """
    table = source if isinstance(source, Table) else Table.read(source)
    if fill not in ("error", "nearest"):
        raise ValueError(f"fill must be 'error' or 'nearest', got {fill!r}")

    names = filter_columns or _detect_filter_columns(table)
    axes_np = tuple(np.unique(np.asarray(table[c], dtype=float)) for c in _AXIS_COLUMNS)
    node_rows = _node_rows(table)

    shape = tuple(a.size for a in axes_np)
    phot = np.full((*shape, len(names)), np.nan)
    coverage = np.zeros(shape, dtype=bool)
    for (teff, logg, feh), row in node_rows.items():
        idx = tuple(
            int(np.searchsorted(axis, value)) for axis, value in zip(axes_np, (teff, logg, feh))
        )
        phot[idx] = [float(table[name][row]) for name in names]
        coverage[idx] = True

    n_missing = int(coverage.size - coverage.sum())
    if n_missing and fill == "error":
        raise ValueError(
            f"{n_missing}/{coverage.size} grid nodes missing from the table "
            "(non-rectangular coverage). Pass fill='nearest' to fill holes and "
            "track them via PhotometryGrid.coverage."
        )
    if n_missing:
        phot = _nearest_fill(phot, coverage)

    axes = tuple(jnp.asarray(a) for a in axes_np)
    return PhotometryGrid(
        axes=axes,
        edges=tuple(edges_for_grid(a) for a in axes),
        phot=jnp.asarray(phot),
        filter_names=tuple(names),
        coverage=jnp.asarray(coverage),
    )


def _detect_filter_columns(table: Table) -> tuple[str, ...]:
    """Float columns that are neither grid axes, label duplicates, nor flags."""
    names = []
    for name in table.colnames:
        if _NON_FILTER.match(name) or name.lower() == "averaged":
            continue
        if not np.issubdtype(np.asarray(table[name]).dtype, np.floating):
            continue
        names.append(name)
    if not names:
        raise ValueError("No filter columns detected; pass filter_columns explicitly.")
    return tuple(names)


def _node_rows(table: Table) -> dict[tuple[float, float, float], int]:
    """Map each (teff, logg, feh) node to a row index, preferring averaged rows."""
    if "averaged" in table.colnames:
        averaged = np.asarray(table["averaged"], dtype=bool)
    else:
        averaged = np.zeros(len(table), dtype=bool)
    rows: dict[tuple[float, float, float], int] = {}
    for i in range(len(table)):
        key = tuple(float(table[c][i]) for c in _AXIS_COLUMNS)
        if key not in rows:
            rows[key] = i
        elif averaged[i] == averaged[rows[key]]:
            raise ValueError(f"Duplicate grid node {key} with equal 'averaged' flags.")
        elif averaged[i]:
            rows[key] = i
    return rows


def _nearest_fill(phot: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Fill uncovered nodes from the nearest covered node in index space."""
    covered = np.argwhere(coverage)
    filled = phot.copy()
    for hole in np.argwhere(~coverage):
        nearest = covered[np.argmin(((covered - hole) ** 2).sum(axis=1))]
        filled[tuple(hole)] = phot[tuple(nearest)]
    return filled
