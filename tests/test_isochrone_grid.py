"""IsochroneGrid: loading, coverage repair, track fractions, gradients.

Runs against the synthetic MIST grid from ``conftest`` rather than
``data/grid_mist_v9.h5``, which is gitignored and absent on CI — a data-gated
test is a test that never runs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import EEP, MINI, toy_payload
from tengri_stars.grids import IsochroneGrid, log_radius_rsun
from tengri_stars.grids.isochrone_grid import TEFF_SUN


def test_loads_and_drops_empty_feh_slice(iso_grid):
    """[Fe/H]=+0.45 has a mass with no track, so the whole slice must go."""
    assert isinstance(iso_grid, IsochroneGrid)
    np.testing.assert_allclose(np.asarray(iso_grid.axes[0]), MINI)
    np.testing.assert_allclose(np.asarray(iso_grid.axes[2]), [-1.0, 0.0])
    assert iso_grid.params.shape == (3, 5, 2, 6)


def test_payload_is_finite_everywhere_after_repair(iso_grid):
    """Ragged gaps and beyond-terminus nodes are filled; a NaN would poison HMC."""
    assert bool(jnp.all(jnp.isfinite(iso_grid.params)))


def test_ragged_gap_is_linearly_interpolated(iso_grid):
    """The hole at (1.0, 214, 0.0) refills to the mean of its EEP neighbours."""
    i, k = MINI.index(1.0), 1  # feh = 0.0
    j = EEP.index(214.0)
    assert not bool(iso_grid.coverage[i, j, k])  # it really was a hole
    below, above = iso_grid.params[i, j - 1, k, :], iso_grid.params[i, j + 1, k, :]
    np.testing.assert_allclose(
        np.asarray(iso_grid.params[i, j, k, :]), np.asarray(0.5 * (below + above)), rtol=1e-10
    )


def test_terminal_eep_surface(iso_grid):
    """eep_max recovers each track's last real node, not the axis maximum."""
    expected = np.array([[214.0, 220.0], [226.0, 226.0], [226.0, 226.0]])
    np.testing.assert_allclose(np.asarray(iso_grid.eep_max), expected)


def test_eep_min_is_a_python_float_not_a_jax_scalar(iso_grid):
    """Regression: ``float(axes[1][0])`` raises ConcretizationTypeError under jit.

    JAX traces the *indexing* into a tracer even when the array is a closure
    constant, so a lazily-derived ``eep_min`` works eagerly and under ``grad``
    and blows up only inside a jitted graph -- i.e. only in production. It must
    be materialized at load time.
    """
    assert type(iso_grid.eep_min) is float
    assert iso_grid.eep_min == 202.0


def test_eep_from_fraction_survives_jit(iso_grid):
    """The path that caught the bug above: jit over the track fraction."""

    @jax.jit
    def eep_of(mini, x_eep):
        return iso_grid.eep_from_fraction(mini=mini, feh=-1.0, x_eep=x_eep)

    got = jax.vmap(eep_of)(jnp.array([0.8, 1.0, 1.2]), jnp.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(got), [214.0, 226.0, 226.0], atol=1e-6)


@pytest.mark.parametrize(
    "mini,feh,eep_max",
    [(0.8, -1.0, 214.0), (0.8, 0.0, 220.0), (1.0, 0.0, 226.0), (1.2, -1.0, 226.0)],
)
def test_fraction_endpoints_stay_inside_coverage(iso_grid, mini, feh, eep_max):
    """x=0 lands on eep_min, x=1 lands exactly on that track's eep_max.

    Node-exactness here is load-bearing, not cosmetic: if x=1 overshot the
    terminus, the sampler would read filled nodes and believe in stars past the
    end of their own evolutionary track.
    """
    at_zero = iso_grid.eep_from_fraction(mini=mini, feh=feh, x_eep=0.0)
    at_one = iso_grid.eep_from_fraction(mini=mini, feh=feh, x_eep=1.0)
    assert float(at_zero) == pytest.approx(202.0, abs=1e-6)
    assert float(at_one) == pytest.approx(eep_max, abs=1e-6)


def test_fraction_never_exceeds_the_terminus_between_nodes(iso_grid):
    """PCHIP is monotone, so off-node track fractions cannot overshoot either."""
    for mini in np.linspace(0.8, 1.2, 9):
        for feh in np.linspace(-1.0, 0.0, 5):
            eep = float(iso_grid.eep_from_fraction(mini=float(mini), feh=float(feh), x_eep=1.0))
            assert 202.0 <= eep <= 226.0 + 1e-6


def test_fraction_is_monotone_in_x(iso_grid):
    """A larger track fraction is always a later evolutionary point."""
    eeps = [
        float(iso_grid.eep_from_fraction(mini=1.0, feh=0.0, x_eep=x)) for x in (0.0, 0.3, 0.7, 1.0)
    ]
    assert eeps == sorted(eeps)


def test_interpolate_is_node_exact_under_pchip(iso_grid):
    """pchip must reproduce a tabulated node; a smoother silently would not."""
    truth = toy_payload(1.2, 220.0, 0.0)
    got = iso_grid.interpolate(mini=1.2, eep=220.0, feh=0.0, method="pchip")
    for key in ("logt", "logg", "logl", "feh_surf", "loga"):
        assert float(got[key]) == pytest.approx(truth[key], rel=1e-6)
    assert float(got["log_agewt"]) == pytest.approx(np.log10(truth["agewt"]), rel=1e-6)


def test_surface_metallicity_differs_from_initial(iso_grid):
    """Diffusion is real: feeding the atmosphere feh_init would be a silent bug."""
    state = iso_grid.interpolate(mini=1.0, eep=214.0, feh=0.0, method="pchip")
    assert float(state["feh_surf"]) == pytest.approx(-0.05, abs=1e-6)


def test_gradients_are_finite(iso_grid):
    """The whole point of the layer: d(structure)/d(mass, EEP, feh) exists."""

    def logt(mini, eep, feh):
        return iso_grid.interpolate(mini=mini, eep=eep, feh=feh)["logt"]

    grads = jax.grad(logt, argnums=(0, 1, 2))(1.0, 214.0, 0.0)
    assert all(np.isfinite(float(g)) for g in grads)
    assert abs(float(grads[1])) > 0.0  # EEP genuinely moves the temperature


def test_gradient_through_track_fraction(iso_grid):
    """Gradients survive the eep_max(mini, feh) reparametrization, not just the LUT."""

    def logt_of_x(mini, x_eep, feh):
        eep = iso_grid.eep_from_fraction(mini=mini, feh=feh, x_eep=x_eep)
        return iso_grid.interpolate(mini=mini, eep=eep, feh=feh)["logt"]

    grads = jax.grad(logt_of_x, argnums=(0, 1, 2))(1.0, 0.5, 0.0)
    assert all(np.isfinite(float(g)) for g in grads)


def test_vmap_over_a_catalog(iso_grid):
    """Interpolation vectorizes, so catalogs run as one batched call."""
    minis = jnp.linspace(0.85, 1.15, 8)

    def logg(mini):
        return iso_grid.interpolate(mini=mini, eep=214.0, feh=0.0)["logg"]

    out = jax.vmap(logg)(minis)
    assert out.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_interpolate_rejects_unknown_method(iso_grid):
    with pytest.raises(ValueError, match="triweight"):
        iso_grid.interpolate(mini=1.0, eep=214.0, feh=0.0, method="linear")


def test_log_radius_of_the_sun_is_zero():
    """R/R_sun = 1 at (L, Teff) = solar -- the anchor the zero point relies on."""
    got = log_radius_rsun(logt=np.log10(TEFF_SUN), logl=0.0)
    assert float(got) == pytest.approx(0.0, abs=1e-12)


def test_log_radius_scales_as_sqrt_luminosity():
    """At fixed Teff, quadrupling L doubles R."""
    lo = log_radius_rsun(logt=np.log10(TEFF_SUN), logl=0.0)
    hi = log_radius_rsun(logt=np.log10(TEFF_SUN), logl=np.log10(4.0))
    assert float(hi - lo) == pytest.approx(np.log10(2.0), rel=1e-10)


def test_log_radius_scales_as_inverse_square_temperature():
    """At fixed L, doubling Teff quarters R."""
    lo = log_radius_rsun(logt=np.log10(TEFF_SUN), logl=0.0)
    hi = log_radius_rsun(logt=np.log10(2.0 * TEFF_SUN), logl=0.0)
    assert float(hi - lo) == pytest.approx(np.log10(0.25), rel=1e-10)
