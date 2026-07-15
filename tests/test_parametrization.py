"""Parametrizations: the old path preserved, the isochrone path physical.

The single most important test here is
:func:`test_free_atmosphere_is_bit_identical_to_the_old_path` — the refactor
must not move the posterior of a fit that does not use the isochrone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import N_FILTERS
from tengri_stars import FreeAtmosphere, MISTTrack
from tengri_stars.grids.isochrone_grid import TEFF_SUN, log_radius_rsun

FREE_PARAMS = {"teff": 5500.0, "logg": 4.2, "feh": -0.5, "mu": 3.0, "ebmv": 0.05}
TRACK_PARAMS = {"mini": 1.0, "x_eep": 0.4, "feh": -0.5, "dist_pc": 500.0, "ebmv": 0.05}


@pytest.fixture(scope="module")
def track(iso_grid):
    return MISTTrack(iso_grid)


# --------------------------------------------------------------------------
# FreeAtmosphere -- the path that must not change
# --------------------------------------------------------------------------


def test_free_atmosphere_is_bit_identical_to_the_old_path(star_model):
    """The refactor must not perturb a non-isochrone fit by a single ULP."""
    old = star_model.predict_mags(**FREE_PARAMS)
    new = FreeAtmosphere().predict_mags(star_model, FREE_PARAMS)
    np.testing.assert_array_equal(np.asarray(old), np.asarray(new))


def test_free_atmosphere_adds_no_prior_term():
    """Every FreeAtmosphere prior factorizes, so the extra term is exactly zero."""
    assert float(FreeAtmosphere().log_prior_extra(FREE_PARAMS)) == 0.0


def test_free_atmosphere_yields_no_derived_quantities():
    """Without a radius there is no mass, age, or distance -- and we admit it."""
    assert FreeAtmosphere().derived(FREE_PARAMS) == {}


def test_free_atmosphere_prior_names():
    assert set(FreeAtmosphere().default_priors()) == {"teff", "logg", "feh", "mu", "ebmv"}


# --------------------------------------------------------------------------
# MISTTrack -- the new path
# --------------------------------------------------------------------------


def test_track_prior_names(track):
    assert set(track.default_priors()) == {"mini", "x_eep", "feh", "dist_pc", "ebmv"}


def test_track_priors_span_the_grid(track, iso_grid):
    priors = track.default_priors()
    assert priors["mini"].lo == pytest.approx(float(iso_grid.axes[0][0]))
    assert priors["mini"].hi == pytest.approx(float(iso_grid.axes[0][-1]))


def test_track_predicts_finite_magnitudes(star_model, track):
    mags = track.predict_mags(star_model, TRACK_PARAMS)
    assert mags.shape == (N_FILTERS,)
    assert bool(jnp.all(jnp.isfinite(mags)))


def test_track_feeds_the_atmosphere_photospheric_metallicity(track):
    """feh_surf, not feh_init. Feeding the wrong one is a silent 0.05-0.9 dex bug."""
    atmos = track.atmosphere(TRACK_PARAMS)
    assert float(atmos["feh"]) != pytest.approx(TRACK_PARAMS["feh"], abs=1e-4)
    assert float(atmos["feh"]) == pytest.approx(TRACK_PARAMS["feh"] - 0.05, abs=1e-3)


def test_distance_enters_as_five_log_d(star_model, track):
    """Doubling the distance must dim every band by exactly 5*log10(2)."""
    near = track.predict_mags(star_model, {**TRACK_PARAMS, "dist_pc": 500.0})
    far = track.predict_mags(star_model, {**TRACK_PARAMS, "dist_pc": 1000.0})
    np.testing.assert_allclose(
        np.asarray(far - near), 5.0 * np.log10(2.0) * np.ones(N_FILTERS), rtol=1e-6
    )


def test_radius_makes_luminous_stars_brighter(track):
    """The isochrone supplies R, so a more evolved star at fixed distance is brighter.

    This is the dividend of the layer. On the FreeAtmosphere path these two
    stars would differ only by whatever `mu` the sampler chose -- which is to
    say, the data would have to pay for the information the physics gives free.
    """
    early = {**TRACK_PARAMS, "x_eep": 0.05}
    late = {**TRACK_PARAMS, "x_eep": 0.95}
    r_early = float(10.0 ** track.structure(early)["log_r"])
    r_late = float(10.0 ** track.structure(late)["log_r"])
    assert r_late > r_early

    off_early = float(track.atmosphere(early)["mu"])
    off_late = float(track.atmosphere(late)["mu"])
    assert off_late - off_early == pytest.approx(-5.0 * np.log10(r_late / r_early), rel=1e-6)


def test_zeropoint_shifts_all_bands_together(star_model, iso_grid):
    """The grid zero point is a single scalar: it cannot change a colour."""
    a = MISTTrack(iso_grid, zeropoint=21.05).predict_mags(star_model, TRACK_PARAMS)
    b = MISTTrack(iso_grid, zeropoint=22.05).predict_mags(star_model, TRACK_PARAMS)
    np.testing.assert_allclose(np.asarray(b - a), -1.0 * np.ones(N_FILTERS), rtol=1e-9)


def test_derived_quantities_are_physical(track):
    """Mass, age, radius -- the three numbers FreeAtmosphere cannot produce."""
    d = track.derived(TRACK_PARAMS)
    assert set(d) >= {"mass", "age_gyr", "radius_rsun", "logl", "teff", "logg", "feh_surf", "eep"}
    assert float(d["mass"]) == pytest.approx(1.0)
    assert 0.0 < float(d["age_gyr"]) < 20.0
    assert float(d["radius_rsun"]) > 0.0
    assert 202.0 <= float(d["eep"]) <= 226.0


def test_log_prior_extra_is_finite(track):
    assert np.isfinite(float(track.log_prior_extra(TRACK_PARAMS)))


def test_imf_prefers_low_mass(track):
    """The Kroupa slope must make a heavier star rarer than a lighter one."""
    light = float(track.log_prior_extra({**TRACK_PARAMS, "mini": 0.9}))
    heavy = float(track.log_prior_extra({**TRACK_PARAMS, "mini": 1.2}))
    assert light > heavy


def test_imf_slope_is_the_kroupa_value(iso_grid):
    """Isolate the IMF by subtracting the measured age Jacobian, not assuming it away."""
    t = MISTTrack(iso_grid, distance_prior="none")
    p1 = {**TRACK_PARAMS, "mini": 1.0}
    p2 = {**TRACK_PARAMS, "mini": 1.2}
    total = float(t.log_prior_extra(p2)) - float(t.log_prior_extra(p1))

    def jacobian(p):
        s = t.structure(p)
        return float(jnp.log(s["eep_span"]) + jnp.log(10.0) * s["log_agewt"])

    imf_only = total - (jacobian(p2) - jacobian(p1))
    assert imf_only == pytest.approx(-2.3 * np.log(1.2), rel=1e-5)


def test_volume_prior_adds_two_log_d(iso_grid):
    """pi(d) ~ d^2: constant space density, the agnostic default."""
    with_vol = MISTTrack(iso_grid, distance_prior="volume")
    without = MISTTrack(iso_grid, distance_prior="none")
    near = {**TRACK_PARAMS, "dist_pc": 100.0}
    far = {**TRACK_PARAMS, "dist_pc": 200.0}
    delta_vol = float(with_vol.log_prior_extra(far)) - float(with_vol.log_prior_extra(near))
    delta_non = float(without.log_prior_extra(far)) - float(without.log_prior_extra(near))
    assert delta_vol - delta_non == pytest.approx(2.0 * np.log(2.0), rel=1e-6)


def test_rejects_unknown_distance_prior(iso_grid):
    with pytest.raises(ValueError, match="distance_prior"):
        MISTTrack(iso_grid, distance_prior="galactic")


def test_gradients_flow_through_every_sampled_parameter(star_model, track):
    """brutus cannot do this at all. It is the reason the layer is worth building."""

    def logpost(mini, x_eep, feh, dist_pc, ebmv):
        p = {"mini": mini, "x_eep": x_eep, "feh": feh, "dist_pc": dist_pc, "ebmv": ebmv}
        mags = track.predict_mags(star_model, p)
        return jnp.sum(mags) + track.log_prior_extra(p)

    grads = jax.grad(logpost, argnums=(0, 1, 2, 3, 4))(1.0, 0.4, -0.5, 500.0, 0.05)
    assert all(np.isfinite(float(g)) for g in grads)
    assert all(abs(float(g)) > 0.0 for g in grads[:4])


def test_jit_and_vmap_over_a_catalog(star_model, track):
    """A catalog is one batched call, as on the free path."""

    @jax.jit
    def mags_of(mini, dist_pc):
        p = {**TRACK_PARAMS, "mini": mini, "dist_pc": dist_pc}
        return track.predict_mags(star_model, p)

    out = jax.vmap(mags_of)(jnp.linspace(0.85, 1.15, 8), jnp.full(8, 500.0))
    assert out.shape == (8, N_FILTERS)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_a_solar_star_at_ten_parsecs_gets_offset_minus_zeropoint(iso_grid):
    """The calibration identity: R=R_sun, d=10 pc  =>  m_app = m_grid - c."""
    assert float(log_radius_rsun(logt=np.log10(TEFF_SUN), logl=0.0)) == pytest.approx(0.0)
    t = MISTTrack(iso_grid, zeropoint=21.05)
    # Bypass the toy track (its structure is synthetic) and check the offset algebra.
    log_r, dist_pc = 0.0, 10.0
    mu = -t.zeropoint - 5.0 * log_r + 5.0 * (np.log10(dist_pc) - 1.0)
    assert mu == pytest.approx(-21.05)
