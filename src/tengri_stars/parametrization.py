"""Parametrizations: which coordinates you sample, and how they become a star.

Both paths in this module end at the *same* four numbers — Teff, log g,
photospheric [Fe/H], and a magnitude offset — which is all
:meth:`~tengri_stars.model.StarModel.predict_mags` ever needed. They differ
only in what you sample and what the sampled coordinates are allowed to mean:

:class:`FreeAtmosphere`
    Sample ``(teff, logg, feh, mu, ebmv)`` directly, with independent priors
    over a box. Any point in that box is a legal star, including combinations
    of temperature and gravity no real star of any mass or age occupies, and
    ``mu`` is an unidentified nuisance absorbing distance, radius, and the
    grid's zero-point convention together.

:class:`MISTTrack`
    Sample ``(mini, x_eep, feh, dist_pc, ebmv)``, and let MIST decide the
    atmosphere. The isochrone is a hard constraint, not a penalty: off-track
    (Teff, log g) is unreachable rather than merely improbable, which is what
    breaks the dwarf/giant degeneracy in colour space. Radius comes out of the
    track, so ``mu`` dissolves into a real distance and mass, age and radius
    become inferred quantities instead of unavailable ones.

Switching is one line, because every sampler in :mod:`tengri_stars.inference`
takes ``(loglikelihood_fn, priors)`` and reads its parameter names off the
``priors`` dict::

    param = FreeAtmosphere()  # or MISTTrack(iso)
    priors = param.default_priors()


    def loglikelihood(p, mags):
        pred = param.predict_mags(model, p)[fidx]
        return -0.5 * jnp.sum(((pred - mags) / sigma) ** 2)


    result = fit_nss(loglikelihood, priors, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from tengri import LogUniform, Uniform

from tengri_stars.grids.isochrone_grid import IsochroneGrid, log_radius_rsun

#: Zero point of the TSLTE photometry grid [mag], calibrated on the Sun.
#:
#: The grid is radius-free (a log g = 1.5 giant and a log g = 4.5 dwarf at the
#: same Teff differ by 0.025 +/- 0.091 mag, not the ~7.4 mag their radii imply),
#: so its magnitudes carry an arbitrary overall normalisation. Interpolating at
#: (5772 K, 4.438, 0.0) and differencing against the solar absolute AB
#: magnitudes of Willmer (2018) gives 21.05 +/- 0.056 mag, consistent across 12
#: bands — a single scalar, as a pure normalisation must be.
#:
#: The 0.056 mag residual scatter is a genuine band-dependent systematic (the
#: same quantity brutus calibrates as per-filter photometric offsets), not
#: noise. Pass ``zeropoint=`` to override, or free it as a nuisance.
TSLTE_ZEROPOINT = 21.05

#: High-mass Kroupa (2001) IMF slope; the only branch the MIST grid reaches,
#: since its mass axis starts at 0.5 M_sun, above the 0.5 M_sun break.
KROUPA_ALPHA_HIGH = 2.3


@runtime_checkable
class Parametrization(Protocol):
    """What a sampler needs to know to turn coordinates into magnitudes."""

    def default_priors(self) -> dict:
        """Priors keyed by parameter name; defines the sampled coordinates."""
        ...

    def predict_mags(self, model, params: dict):
        """Apparent magnitudes in every filter of ``model.grid`` [AB mag]."""
        ...

    def log_prior_extra(self, params: dict):
        """Log-prior terms that do not factorize over single parameters."""
        ...

    def derived(self, params: dict) -> dict:
        """Physical quantities implied by, but not sampled in, ``params``."""
        ...


@dataclass(frozen=True)
class FreeAtmosphere:
    """Sample the atmosphere directly: the path that predates the isochrone.

    Parameters
    ----------
    teff_range, logg_range, feh_range, ebmv_range : tuple of float
        Uniform prior bounds. Defaults span the TSLTE grid.
    mu_range : tuple of float
        Uniform bounds on the dilution term [mag]. **Grid-dependent** — there
        is no universal default, because ``mu`` absorbs the photometry grid's
        zero-point convention along with distance and radius. On TSLTE, where
        the Sun sits at m_grid ~ 26, a physical star needs
        ``mu ~ -21 + 5log10(d/10pc) - 5log10(R/Rsun)``: about -21 for a solar
        twin at 10 pc, -16 for one at 100 pc, -13 for a red giant at 10 kpc.
        The default is deliberately wide rather than tight-and-wrong; a prior
        that excludes the truth fails silently, converging confidently on the
        boundary.

    Notes
    -----
    Reproduces the hand-rolled closure the notebooks used before this module
    existed, exactly: ``model.predict_mags(teff=..., logg=..., feh=..., mu=...,
    ebmv=...)``. :meth:`log_prior_extra` is identically zero, so the posterior
    is unchanged — this is a refactor of the old path, not a revision of it.
    """

    teff_range: tuple[float, float] = (2500.0, 8000.0)
    logg_range: tuple[float, float] = (-0.5, 5.5)
    feh_range: tuple[float, float] = (-5.0, 1.0)
    mu_range: tuple[float, float] = (-30.0, 10.0)
    ebmv_range: tuple[float, float] = (0.0, 1.0)

    def default_priors(self) -> dict:
        """Independent uniform priors over the grid box."""
        return {
            "teff": Uniform(*self.teff_range),
            "logg": Uniform(*self.logg_range),
            "feh": Uniform(*self.feh_range),
            "mu": Uniform(*self.mu_range),
            "ebmv": Uniform(*self.ebmv_range),
        }

    def atmosphere(self, params: dict) -> dict:
        """Identity: the sampled coordinates already *are* the atmosphere."""
        return {
            "teff": params["teff"],
            "logg": params["logg"],
            "feh": params["feh"],
            "mu": params.get("mu", 0.0),
            "ebmv": params.get("ebmv", 0.0),
        }

    def predict_mags(self, model, params: dict):
        """Apparent magnitudes [AB mag]. JIT/grad/vmap-safe."""
        return model.predict_mags(**self.atmosphere(params))

    def log_prior_extra(self, params: dict):
        """Zero — every prior in this path factorizes over one parameter."""
        return jnp.zeros(())

    def derived(self, params: dict) -> dict:
        """Nothing: without a radius there is no mass, age, or distance."""
        return {}


@dataclass(frozen=True)
class MISTTrack:
    """Sample an evolutionary track; let MIST supply the atmosphere.

    Parameters
    ----------
    isochrone : IsochroneGrid
        The (M_init, EEP, [Fe/H]) → structure table.
    zeropoint : float
        Photometry-grid normalisation [mag]; see :data:`TSLTE_ZEROPOINT`.
    imf_alpha : float
        Kroupa high-mass IMF slope; see :data:`KROUPA_ALPHA_HIGH`.
    distance_prior : {'volume', 'none'}
        ``'volume'``: constant space density, pi(d) ∝ d², the agnostic default.
        ``'none'``: whatever ``dist_pc``'s entry in the priors dict says, with
        no extra term — the hook for a Gaia parallax likelihood or a Galactic
        density prior.
    mini_range, feh_range, dist_range, ebmv_range : tuple of float
        Prior bounds. Mass and metallicity default to the grid's own extent.

    Notes
    -----
    The photometry chain is

    .. math::

        m_X = m_X^{\\rm grid}(T_{\\rm eff}, \\log g, [{\\rm Fe/H}]_{\\rm surf})
              + R_X E(B-V) - c - 5\\log_{10}\\frac{R}{R_\\odot}
              + 5\\log_{10}\\frac{d}{10\\,{\\rm pc}}

    with :math:`c` the grid zero point [mag], :math:`R` from
    :func:`~tengri_stars.grids.isochrone_grid.log_radius_rsun`, and :math:`d`
    the distance [pc]. The atmosphere is evaluated at the *photospheric*
    metallicity ``feh_surf``, not the initial ``feh`` the priors act on —
    atomic diffusion separates the two by up to 0.9 dex, worst at the
    metal-poor end.

    JIT/grad/vmap-safe throughout.
    """

    isochrone: IsochroneGrid
    zeropoint: float = TSLTE_ZEROPOINT
    imf_alpha: float = KROUPA_ALPHA_HIGH
    distance_prior: str = "volume"
    mini_range: tuple[float, float] | None = None
    feh_range: tuple[float, float] | None = None
    dist_range: tuple[float, float] = (10.0, 1.0e5)
    ebmv_range: tuple[float, float] = (0.0, 1.0)
    interp_method: str = field(default="triweight")

    def __post_init__(self):
        if self.distance_prior not in ("volume", "none"):
            raise ValueError(
                f"distance_prior must be 'volume' or 'none', got {self.distance_prior!r}"
            )

    def default_priors(self) -> dict:
        """Uniform in mass, track fraction and reddening; log-uniform in distance.

        ``x_eep`` is uniform on the unit interval by construction — that is the
        whole reason for sampling a track *fraction*. It is not a uniform prior
        in age; :meth:`log_prior_extra` supplies the Jacobian that makes it one.
        """
        axes = self.isochrone.axes
        mini_lo, mini_hi = self.mini_range or (float(axes[0][0]), float(axes[0][-1]))
        feh_lo, feh_hi = self.feh_range or (float(axes[2][0]), float(axes[2][-1]))
        return {
            "mini": Uniform(mini_lo, mini_hi),
            "x_eep": Uniform(0.0, 1.0),
            "feh": Uniform(feh_lo, feh_hi),
            "dist_pc": LogUniform(*self.dist_range),
            "ebmv": Uniform(*self.ebmv_range),
        }

    def structure(self, params: dict) -> dict:
        """Interpolate the track; adds ``eep``, ``eep_span`` and ``log_r``."""
        mini, feh = params["mini"], params["feh"]
        eep = self.isochrone.eep_from_fraction(mini=mini, feh=feh, x_eep=params["x_eep"])
        state = self.isochrone.interpolate(mini=mini, eep=eep, feh=feh, method=self.interp_method)
        state["eep"] = eep
        state["eep_span"] = self.isochrone.eep_span(mini=mini, feh=feh)
        state["log_r"] = log_radius_rsun(logt=state["logt"], logl=state["logl"])
        return state

    def atmosphere(self, params: dict) -> dict:
        """Track coordinates → (Teff, log g, photospheric [Fe/H], offset)."""
        state = self.structure(params)
        dist_pc = params["dist_pc"]
        mu = -self.zeropoint - 5.0 * state["log_r"] + 5.0 * (jnp.log10(dist_pc) - 1.0)
        return {
            "teff": 10.0 ** state["logt"],
            "logg": state["logg"],
            "feh": state["feh_surf"],
            "mu": mu,
            "ebmv": params.get("ebmv", 0.0),
        }

    def predict_mags(self, model, params: dict):
        """Apparent magnitudes [AB mag]. JIT/grad/vmap-safe."""
        return model.predict_mags(**self.atmosphere(params))

    def log_prior_extra(self, params: dict):
        """Joint log-prior: IMF, uniform-in-age, and the space-density volume.

        These three terms couple parameters, so they cannot live in the
        ``priors`` dict, whose per-name ``log_prob`` values ``inference.py``
        simply sums. They ride in the likelihood instead. That is exact, not a
        fudge: writing :math:`L' = L J` and :math:`\\pi' = \\pi_{\\rm sep}` gives
        :math:`\\int L' \\pi' = \\int L (J \\pi_{\\rm sep}) = \\int L \\pi_{\\rm true}`,
        so nested sampling's evidence is unchanged. The only cost is that live
        points are drawn from the separable prior — an efficiency hit, never a
        bias.

        The terms:

        * **IMF** — ``-alpha * log(mini)``, the Kroupa (2001) high-mass slope.
        * **Age** — uniform in age, *not* in EEP. Sampling ``x_eep`` uniformly
          induces a density in age of :math:`1/|{\\rm d}a/{\\rm d}x|`, so the
          Jacobian :math:`{\\rm d}a/{\\rm d}x = {\\rm agewt} \\times {\\rm span}`
          multiplies back out. Without it the posterior would pile up wherever
          MIST happens to space its EEPs densely, which is an artefact of the
          grid, not of stars.
        * **Distance** — ``2 log(d)`` for constant space density, if enabled.
        """
        state = self.structure(params)
        log_imf = -self.imf_alpha * jnp.log(params["mini"])
        log_agewt = jnp.log(10.0) * state["log_agewt"]
        log_age_jacobian = log_agewt + jnp.log(state["eep_span"])
        total = log_imf + log_age_jacobian
        if self.distance_prior == "volume":
            total = total + 2.0 * jnp.log(params["dist_pc"])
        return total

    def derived(self, params: dict) -> dict:
        """Mass, age, radius and atmosphere implied by the sampled track.

        Returns
        -------
        dict
            ``mass`` [M_sun] (initial; this grid tabulates no current mass, so
            it neglects RGB mass loss), ``age_gyr`` [Gyr], ``radius_rsun``
            [R_sun], ``logl`` [log10 L_sun], ``teff`` [K], ``logg`` [cgs dex],
            ``feh_surf`` [dex], ``eep`` [-], and ``dist_pc`` [pc].

            None of these exist on the :class:`FreeAtmosphere` path. They are
            the isochrone's dividend.
        """
        state = self.structure(params)
        return {
            "mass": params["mini"],
            "age_gyr": 10.0 ** (state["loga"] - 9.0),
            "radius_rsun": 10.0 ** state["log_r"],
            "logl": state["logl"],
            "teff": 10.0 ** state["logt"],
            "logg": state["logg"],
            "feh_surf": state["feh_surf"],
            "eep": state["eep"],
            "dist_pc": params["dist_pc"],
        }
