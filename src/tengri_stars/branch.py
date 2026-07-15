"""Two-hypothesis (RGB/MS) isochrone-pinned metallicity inference, MAGIC-style.

An observed g−i intersects an old-population isochrone at two points — one on
the main sequence, one on the giant branch — each asserting its own
(Teff, log g, M_g). This module runs *both* hypotheses through a 1-D χ² scan
in [Fe/H] against a :class:`~tengri_stars.PhotometryGrid` forward model and
combines them as a Bayesian model average:

.. math::

    P(b \\mid \\rm data) \\propto
        e^{-\\chi^2_b/2} \\; P(\\varpi_{\\rm obs} \\mid b) \\; P(b)

Because each hypothesis is pinned to the observed g, the isochrone's absolute
magnitude cancels from the photometric χ² (a pure SED-shape test); the
luminosity difference between branches re-enters only through the parallax
term (or a known distance modulus), where the branch-implied
:math:`{\\rm DM}_b = g_{\\rm obs} - M_{g,b}` is *physical* — built from the
isochrone's M_g, independent of the photometry grid's zero-point convention.

Validated in ``notebooks/09_branch_discrimination`` (noiseless closure;
regime maps); the color coordinate of the branch tables is the *grid's* g−i
evaluated at the isochrone's (Teff, log g, [Fe/H]) so that data and hypothesis
chain share one color system — Dartmouth-native colors disagree with the grid
enough to break closure.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tengri_stars.model import StarModel

BRANCHES = ("MS", "RGB")


@dataclass(frozen=True)
class BranchTables:
    """Per-branch isochrone lookups over ([Fe/H] scan × grid-color lattice).

    Parameters
    ----------
    gi_grid : np.ndarray, shape (n_gi,)
        Grid-color (g−i) lattice of the tables.
    feh_nodes : np.ndarray
        [Fe/H] of the source isochrones.
    feh_scan : np.ndarray, shape (n_scan,)
        Dense [Fe/H] axis (the free parameter of each hypothesis).
    scan : dict
        ``scan[branch][q]`` for q in ("teff", "logg", "mg"): (n_scan, n_gi)
        arrays, edge-filled beyond each branch's color range; ``scan[branch]
        ["lo"|"hi"]``: (n_scan,) valid color-window bounds (lerped separately
        from the values so a missing node cannot poison its neighbor).
    """

    gi_grid: np.ndarray
    feh_nodes: np.ndarray
    feh_scan: np.ndarray
    scan: dict

    def overlap(self, k: int) -> tuple[float, float]:
        """(lo, hi) g−i window where both branches exist at scan row k."""
        return (
            max(self.scan["MS"]["lo"][k], self.scan["RGB"]["lo"][k]),
            min(self.scan["MS"]["hi"][k], self.scan["RGB"]["hi"][k]),
        )


def load_dartmouth_branches(
    iso_files: dict[float, str],
    model: StarModel,
    *,
    ig: int,
    ii: int,
    gi_grid: np.ndarray | None = None,
    feh_step: float = 0.02,
    rgb_logg_max: float = 3.6,
) -> BranchTables:
    """Build branch tables from Dartmouth isochrone files.

    Parameters
    ----------
    iso_files : dict of float -> str
        [Fe/H] node -> path of a Dartmouth ``.txt`` isochrone (columns EEP,
        M/Mo, LogTeff, LogG, LogL, then magnitudes with the g band at index 6
        and the i band at index 8, as in the DECam-system files).
    model : StarModel
        Photometry-grid forward model; its band order defines ``ig``/``ii``.
    ig, ii : int
        Indices of the g and i bands in ``model.grid.filter_names``.
    gi_grid : np.ndarray, optional
        Color lattice; default 0.40..2.40 step 0.005.
    feh_step : float
        Step of the dense [Fe/H] scan between the node metallicities.
    rgb_logg_max : float
        Post-turnoff points with log g above this are dropped (SGB elbow).
    """
    if gi_grid is None:
        gi_grid = np.arange(0.40, 2.4001, 0.005)
    feh_nodes = np.array(sorted(iso_files))
    mags_at = jax.jit(
        jax.vmap(lambda t, g, f: model.predict_mags(teff=t, logg=g, feh=f))
    )

    tables = {
        b: {q: np.full((feh_nodes.size, gi_grid.size), np.nan) for q in ("teff", "logg", "mg")}
        for b in BRANCHES
    }
    for k, feh in enumerate(feh_nodes):
        d = np.loadtxt(iso_files[feh])
        logteff, logg, mg = d[:, 2], d[:, 3], d[:, 6]
        teff = 10**logteff
        phot = np.asarray(mags_at(jnp.asarray(teff), jnp.asarray(logg), jnp.full(len(d), feh)))
        gi = phot[:, ig] - phot[:, ii]  # grid-color coordinate
        to = int(np.argmax(logteff))
        branch_masks = {
            "MS": np.arange(len(d)) <= to,
            "RGB": (np.arange(len(d)) > to) & (logg <= rgb_logg_max),
        }
        for b, mask in branch_masks.items():
            s = np.argsort(gi[mask])
            gi_b = gi[mask][s]
            ok = (gi_b[0] <= gi_grid) & (gi_grid <= gi_b[-1])
            for q, col in (("teff", teff[mask][s]), ("logg", logg[mask][s]), ("mg", mg[mask][s])):
                tables[b][q][k, ok] = np.interp(gi_grid[ok], gi_b, col)

    feh_scan = np.arange(feh_nodes[0], feh_nodes[-1] + 1e-9, feh_step)
    idx = np.clip(np.searchsorted(feh_nodes, feh_scan) - 1, 0, feh_nodes.size - 2)
    w = (feh_scan - feh_nodes[idx]) / (feh_nodes[idx + 1] - feh_nodes[idx])

    scan: dict = {}
    for b in BRANCHES:
        ok = ~np.isnan(tables[b]["teff"])
        lo = np.array([gi_grid[row].min() for row in ok])
        hi = np.array([gi_grid[row].max() for row in ok])
        filled = {}
        for q, tab in tables[b].items():
            tab = tab.copy()
            for k in range(feh_nodes.size):
                row_ok = ok[k]
                tab[k] = np.interp(gi_grid, gi_grid[row_ok], tab[k][row_ok])  # edge-extends
            filled[q] = (1 - w[:, None]) * tab[idx] + w[:, None] * tab[idx + 1]
        filled["lo"] = (1 - w) * lo[idx] + w * lo[idx + 1]
        filled["hi"] = (1 - w) * hi[idx] + w * hi[idx + 1]
        scan[b] = filled

    return BranchTables(gi_grid=gi_grid, feh_nodes=feh_nodes, feh_scan=feh_scan, scan=scan)


def binary_scan(
    observed: np.ndarray,
    sigma: np.ndarray,
    tables: BranchTables,
    model: StarModel,
    *,
    ig: int,
    ii: int,
    chi2_bands: tuple[int, ...],
    dm_known: np.ndarray | float | None = None,
    gi_sigma: float | np.ndarray | None = None,
    n_quad: int = 9,
    chunk: int = 200_000,
) -> dict:
    """Both branch hypotheses evaluated on every observed star.

    Parameters
    ----------
    observed : np.ndarray, shape (n_obs, n_bands)
        Observed (dereddened) magnitudes in the model's band order.
    sigma : np.ndarray, shape (n_bands,) or (n_obs, n_bands)
        1σ photometric errors [mag]; per-star errors supported.
    tables : BranchTables
    model : StarModel
    ig, ii : int
        g and i band indices (color + pinning).
    chi2_bands : tuple of int
        Bands entering the shape χ², whose normalization μ is profiled out
        analytically (inverse-variance weighted mean). Include ``ig`` —
        profiling replaces the old hard g-pinning and prices g's noise in.
    dm_known : array_like or float, optional
        Known distance modulus per star (or scalar). When given, a
        ``chi2_dm`` variant is also returned: the model normalized by the
        branch's isochrone luminosity at that distance, all bands counted.
    gi_sigma : float or np.ndarray of shape (n_obs,), optional
        1σ uncertainty of the observed g−i color. When given, the returned
        χ² curves are −2 log of the likelihood marginalized over the color
        error (Gaussian-weighted quadrature, ``n_quad`` nodes over ±3.5σ)
        instead of conditioning on the face-value color — without this,
        downstream credible intervals are overconfident (the hypothesis
        chain's Teff/log g inherit the color noise silently). The
        known-distance likelihood is nearly a delta function in color (the
        isochrone luminosity term is steep), so use a generous ``n_quad``
        (≳ 21) whenever ``dm_known`` is set.
    n_quad : int
        Quadrature nodes for the color marginalization.
    chunk : int
        Grid-lookup batch size.

    Returns
    -------
    dict
        ``out[branch]`` with keys ``chi2_curve`` (n_scan, n_obs; +inf outside
        the branch's color window), ``chi2``/``feh`` (scan minimum), ``mg``
        (n_scan, n_obs isochrone M_g), and when ``dm_known`` is given,
        ``chi2_dm_curve``/``chi2_dm``/``feh_dm``.
    """
    observed = np.atleast_2d(np.asarray(observed, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))  # (1 | n_obs, n_bands)
    gi_center = observed[:, ig] - observed[:, ii]
    gi_grid, feh_scan = tables.gi_grid, tables.feh_scan

    mags_at = jax.jit(
        jax.vmap(lambda t, g, f: model.predict_mags(teff=t, logg=g, feh=f))
    )
    n_obs, n_scan, n_bands = observed.shape[0], feh_scan.size, observed.shape[1]
    cb = np.asarray(chi2_bands, dtype=int)

    if gi_sigma is None:
        gh_x, gh_weights = np.array([0.0]), np.array([1.0])
        gi_sig_arr = np.zeros(n_obs)
    else:
        gh_x = np.linspace(-3.5, 3.5, n_quad)
        gh_weights = np.exp(-0.5 * gh_x**2)
        gh_weights /= gh_weights.sum()
        gi_sig_arr = np.broadcast_to(np.asarray(gi_sigma, dtype=float), (n_obs,))

    def _curves(branch, gi_obs):
        """(chi2_curve, chi2_dm_curve or None, mg) at one color hypothesis."""
        tab = tables.scan[branch]
        j = np.clip(np.searchsorted(gi_grid, gi_obs) - 1, 0, gi_grid.size - 2)
        wgi = ((gi_obs - gi_grid[j]) / (gi_grid[j + 1] - gi_grid[j]))[None, :]
        teff = (1 - wgi) * tab["teff"][:, j] + wgi * tab["teff"][:, j + 1]
        logg = (1 - wgi) * tab["logg"][:, j] + wgi * tab["logg"][:, j + 1]
        mg = (1 - wgi) * tab["mg"][:, j] + wgi * tab["mg"][:, j + 1]
        valid = (gi_obs[None, :] >= tab["lo"][:, None]) & (gi_obs[None, :] <= tab["hi"][:, None])

        feh_flat = np.broadcast_to(feh_scan[:, None], teff.shape).ravel()
        phot = np.empty((teff.size, n_bands))
        t_flat, g_flat = teff.ravel(), logg.ravel()
        for i in range(0, t_flat.size, chunk):
            phot[i : i + chunk] = np.asarray(
                mags_at(
                    jnp.asarray(t_flat[i : i + chunk]),
                    jnp.asarray(g_flat[i : i + chunk]),
                    jnp.asarray(feh_flat[i : i + chunk]),
                )
            )
        phot = phot.reshape(n_scan, n_obs, n_bands)

        # shape test: normalization μ profiled analytically over chi2_bands
        # (inverse-variance weighted mean of the residuals) — prices the noise
        # of every band, including g, into the χ² correctly (a hard g-pinning
        # would leave correlated, underweighted residuals and overconfident
        # intervals)
        delta = (phot - observed[None])[:, :, cb]
        wts = 1.0 / sigma[:, cb] ** 2  # (1 | n_obs, n_cb)
        a = (delta * wts[None]).sum(axis=2)
        chi2_full = (delta**2 * wts[None]).sum(axis=2) - a**2 / wts.sum(axis=1)[None]
        chi2 = np.where(valid, chi2_full, np.inf)
        chi2_dm = None
        if dm_known is not None:
            dm = np.broadcast_to(np.asarray(dm_known, dtype=float), (n_obs,))
            mu_hyp = mg - phot[:, :, ig] + dm[None, :]
            resid_dm = (phot + mu_hyp[:, :, None] - observed[None]) / sigma[None]
            chi2_dm = np.where(valid, (resid_dm**2).sum(axis=2), np.inf)
        return chi2, chi2_dm, mg

    def _gh_average(curves):
        """−2 log of the weighted likelihood average, underflow-safe."""
        stack = np.stack(curves)  # (n_gh, n_scan, n_obs)
        ref = stack.min(axis=0)
        with np.errstate(invalid="ignore"):
            like = np.einsum("k,kij->ij", gh_weights, np.exp(-0.5 * (stack - ref[None])))
        like[~np.isfinite(ref)] = 0.0
        ref = np.where(np.isfinite(ref), ref, np.inf)
        with np.errstate(divide="ignore"):
            return ref - 2.0 * np.log(np.where(like > 0, like, 1.0))

    out: dict = {}
    for b in BRANCHES:
        shape_curves, dm_curves, mg_center = [], [], None
        for x in gh_x:
            chi2, chi2_dm, mg = _curves(b, gi_center + x * gi_sig_arr)
            shape_curves.append(chi2)
            if chi2_dm is not None:
                dm_curves.append(chi2_dm)
            if x == 0.0 or mg_center is None:
                mg_center = mg
        chi2_curve = _gh_average(shape_curves)
        kmin = np.argmin(chi2_curve, axis=0)
        res = {
            "chi2_curve": chi2_curve,
            "chi2": chi2_curve[kmin, np.arange(n_obs)],
            "feh": feh_scan[kmin],
            "mg": mg_center,
        }
        if dm_curves:
            curve_dm = _gh_average(dm_curves)
            kdm = np.argmin(curve_dm, axis=0)
            res["chi2_dm_curve"] = curve_dm
            res["chi2_dm"] = curve_dm[kdm, np.arange(n_obs)]
            res["feh_dm"] = feh_scan[kdm]
        out[b] = res
    return out


def combine_mixture(
    scans: dict,
    g_obs: np.ndarray,
    feh_scan: np.ndarray,
    *,
    parallax: np.ndarray | None = None,
    parallax_error: np.ndarray | None = None,
    prior_rgb: float = 0.5,
    use_dm: bool = False,
) -> dict:
    """Bayesian model average of the two branch hypotheses.

    Per star, the joint posterior over (branch, [Fe/H]) is::

        w_b(f) ∝ exp(-χ²_b(f)/2) · N(ϖ_obs; ϖ_b(f), σ_ϖ) · P(b)

    with ϖ_b(f) [mas] from the branch-implied DM_b(f) = g_obs − M_g,b(f)
    (physical — isochrone luminosity, no grid zero-point involved).

    Parameters
    ----------
    scans : dict
        Output of :func:`binary_scan`.
    g_obs : np.ndarray, shape (n_obs,)
        Observed (dereddened) g magnitudes.
    feh_scan : np.ndarray
        The scan axis (``tables.feh_scan``).
    parallax, parallax_error : np.ndarray, optional
        Gaia parallaxes [mas]; None → photometric-shape weights only.
    prior_rgb : float
        Prior probability of the RGB branch.
    use_dm : bool
        Use the ``chi2_dm_curve`` (known-distance) χ² instead of the shape χ².

    Returns
    -------
    dict
        ``p_rgb`` (n_obs,), ``feh`` (posterior median), ``feh_lo``/``feh_hi``
        (16th/84th percentiles of the mixture), per-branch weights.
    """
    key = "chi2_dm_curve" if use_dm else "chi2_curve"
    g_obs = np.asarray(g_obs, dtype=float)
    n_obs = g_obs.size
    priors = {"RGB": prior_rgb, "MS": 1.0 - prior_rgb}

    logw = {}
    for b in BRANCHES:
        curve = scans[b][key]  # (n_scan, n_obs)
        lw = -0.5 * curve
        if parallax is not None:
            dm_b = g_obs[None, :] - scans[b]["mg"]  # (n_scan, n_obs)
            plx_pred = 10.0 ** (2.0 - dm_b / 5.0)  # [mas]
            lw = lw - 0.5 * ((np.asarray(parallax)[None, :] - plx_pred)
                             / np.asarray(parallax_error)[None, :]) ** 2
        logw[b] = lw + np.log(priors[b])

    # normalize jointly per star; a star outside both branch windows has no
    # posterior at all — flagged with p_rgb = 0.5 and NaN [Fe/H]
    lmax = np.maximum(logw["RGB"].max(axis=0), logw["MS"].max(axis=0))
    dead = ~np.isfinite(lmax)
    lmax = np.where(dead, 0.0, lmax)
    w = {b: np.exp(np.clip(logw[b] - lmax[None, :], -745.0, 0.0)) for b in BRANCHES}
    tot = w["RGB"].sum(axis=0) + w["MS"].sum(axis=0)
    tot_safe = np.where(tot > 0, tot, 1.0)
    p_rgb = np.where(dead, 0.5, w["RGB"].sum(axis=0) / tot_safe)

    density = (w["RGB"] + w["MS"]) / tot_safe[None, :]  # (n_scan, n_obs), sums to 1
    cdf = np.cumsum(density, axis=0)
    feh_med = np.empty(n_obs)
    feh_lo = np.empty(n_obs)
    feh_hi = np.empty(n_obs)
    for q, dest in ((0.5, feh_med), (0.16, feh_lo), (0.84, feh_hi)):
        idx = np.clip(np.argmax(cdf >= q, axis=0), 0, feh_scan.size - 1)
        dest[:] = np.where(dead, np.nan, feh_scan[idx])

    return {"p_rgb": p_rgb, "feh": feh_med, "feh_lo": feh_lo, "feh_hi": feh_hi,
            "weights": w, "invalid": dead}
