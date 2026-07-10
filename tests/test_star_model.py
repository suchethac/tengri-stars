"""Tests for the star photometry forward model: grid lookup + reddening + dilution."""

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np
from astropy.table import Table

from tengri_stars import StarModel
from tengri_stars.grids import load_photometry_grid

# Five filters with linearly independent responses to (teff, logg, feh) so a
# mock star is identifiable from photometry alone.
COEFFS = {
    "lsst_g_ab": (30.0, 1.0e-3, 0.50, 1.20),
    "lsst_i_ab": (29.0, 0.6e-3, 0.20, 0.40),
    "skymapper_v_ab": (31.0, 1.4e-3, 0.70, 0.90),
    "cahk": (35.0, -2.0e-3, 0.30, 2.00),
    "i_des": (29.5, 0.5e-3, 0.10, 0.30),
}
# A_X / E(B-V) reddening coefficients, one per filter (MAGIC-style).
RED_COEFFS = jnp.array([3.2, 1.7, 2.8, 4.1, 1.6])


def _toy_grid():
    rows = []
    for teff in np.linspace(4000.0, 5200.0, 5):
        for logg in [1.0, 2.0, 3.0]:
            for feh in np.linspace(-3.0, 0.0, 4):
                row = {"teff": teff, "logg": logg, "feh": feh}
                for name, (c0, ct, cg, cf) in COEFFS.items():
                    row[name] = c0 + ct * teff + cg * logg + cf * feh
                rows.append(row)
    return load_photometry_grid(Table(rows=rows))


def test_dilution_shifts_all_filters_equally():
    grid = _toy_grid()
    model = StarModel(grid=grid)

    base = model.predict_mags(teff=4600.0, logg=2.0, feh=-1.5)
    shifted = model.predict_mags(teff=4600.0, logg=2.0, feh=-1.5, mu=14.5)

    np.testing.assert_allclose(shifted - base, 14.5)


def test_extinction_adds_per_filter_reddening():
    grid = _toy_grid()
    model = StarModel(grid=grid, extinction_coeffs=RED_COEFFS)

    base = model.predict_mags(teff=4600.0, logg=2.0, feh=-1.5)
    reddened = model.predict_mags(teff=4600.0, logg=2.0, feh=-1.5, ebmv=0.1)

    np.testing.assert_allclose(reddened - base, 0.1 * np.asarray(RED_COEFFS), rtol=1e-12)


def test_model_interp_method_pchip_is_node_exact():
    grid = _toy_grid()
    model = StarModel(grid=grid, interp_method="pchip")

    mags = model.predict_mags(teff=4000.0, logg=1.0, feh=-3.0)  # corner node

    expected = [c0 + ct * 4000.0 + cg * 1.0 + cf * -3.0 for (c0, ct, cg, cf) in COEFFS.values()]
    np.testing.assert_allclose(mags, expected, rtol=1e-12)


def test_mock_star_recovery_via_gradient_descent():
    """End-to-end physics test: a mock star is recovered by gradient-based MAP.

    Proves the full forward chain (LUT interpolation → reddening → dilution)
    is differentiable and informative enough to invert. Mirrors the objective
    structure of tengri's ``build_loss_fn`` (H = χ²/2 + ξᵀξ/2 over unbounded
    parameters with bounded transforms): the LUT clamps outside the grid hull,
    where gradients vanish — an unbounded optimizer that overshoots strands on
    the plateau, so the bounded reparameterization is load-bearing, not style.
    """
    grid = _toy_grid()
    model = StarModel(grid=grid, extinction_coeffs=RED_COEFFS)
    truth = {"teff": 4600.0, "logg": 2.2, "feh": -1.3, "mu": 14.5}
    sigma = 0.02

    key = jax.random.PRNGKey(0)
    mags_obs = model.predict_mags(ebmv=0.05, **truth) + sigma * jax.random.normal(key, (5,))

    lo = jnp.array([4000.0, 1.0, -3.0, 10.0])  # grid support + a mu range
    hi = jnp.array([5200.0, 3.0, 0.0, 20.0])

    def to_physical(xi):
        return lo + (hi - lo) * jax.nn.sigmoid(xi)

    def hamiltonian(xi):
        p = to_physical(xi)
        pred = model.predict_mags(teff=p[0], logg=p[1], feh=p[2], mu=p[3], ebmv=0.05)
        return 0.5 * jnp.sum((pred - mags_obs) ** 2) / sigma**2 + 0.5 * jnp.sum(xi**2)

    xi0 = jnp.array([-1.0, -1.0, -1.0, -1.0])  # deliberately off-truth start
    result = jax.scipy.optimize.minimize(hamiltonian, xi0, method="BFGS")
    p = to_physical(result.x)

    assert abs(p[0] - truth["teff"]) < 50.0
    assert abs(p[2] - truth["feh"]) < 0.1
    assert abs(p[3] - truth["mu"]) < 0.1
