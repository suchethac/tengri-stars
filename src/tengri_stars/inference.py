"""Posterior sampling for star models via tengri's nested slice sampler.

Reuses tengri's NSS engine (Yallup, Kroupa & Handley 2026, arXiv:2601.23252;
``tengri.inference.backends.nested``) below the Fitter seam: tengri's
``Fitter``/``Parameters`` currently validate against the galaxy parameter
registry, so star fits drive the sampler directly with plain prior and
likelihood callables. The loop mirrors ``tengri.inference.backends.evidence.run_nss``.

NSS samples in *physical* (bounded) space and needs no gradients, which suits
grid-lookup forward models: the clamped LUT's zero gradient outside the hull —
fatal for unbounded gradient descent — is simply never visited, because the
prior bounds the live points, and multimodal stellar-parameter degeneracies
(Teff–[Fe/H]–extinction, RGB/MS) are handled natively. The evidence (log Z)
also enables model comparison, e.g. dwarf-vs-giant classification.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from tengri.inference.backends.nested.base import NSInfo
from tengri.inference.backends.nested.nss import as_top_level_api
from tengri.inference.backends.nested.utils import ess as ns_ess, sample as ns_sample


class _PosteriorSummary:
    """Shared posterior-summary methods; expects a ``samples`` dict attribute."""

    def median(self) -> dict[str, float]:
        """Posterior medians per parameter."""
        return {k: float(jnp.median(v)) for k, v in self.samples.items()}

    def interval(self, q: float = 0.68) -> dict[str, tuple[float, float]]:
        """Central credible interval per parameter.

        Parameters
        ----------
        q : float
            Central probability mass, e.g. 0.68.
        """
        lo, hi = 50.0 * (1.0 - q), 50.0 * (1.0 + q)
        return {
            k: (float(jnp.percentile(v, lo)), float(jnp.percentile(v, hi)))
            for k, v in self.samples.items()
        }


@dataclass(frozen=True)
class NSSResult(_PosteriorSummary):
    """Nested-sampling posterior for one star.

    Parameters
    ----------
    samples : dict of str -> jnp.ndarray, shape (n_posterior_samples,)
        Equal-weight posterior samples per parameter, physical units.
    logz : float
        Log Bayesian evidence.
    ess : float
        Effective sample size of the nested-sampling run.
    n_iterations : int
        NS iterations to convergence.
    wall_time : float
        Sampling wall-clock time [s], including JIT compilation.
    """

    samples: dict[str, jnp.ndarray]
    logz: float
    ess: float
    n_iterations: int
    wall_time: float


@dataclass(frozen=True)
class NUTSResult(_PosteriorSummary):
    """NUTS (blackjax) posterior for one star.

    Parameters
    ----------
    samples : dict of str -> jnp.ndarray, shape (num_samples,)
        Posterior samples per parameter, physical units.
    acceptance_rate : float
        Mean warmup-tuned acceptance rate.
    num_divergent : int
        Number of divergent transitions in the sampling phase.
    wall_time : float
        Sampling wall-clock time [s], including JIT compilation.
    """

    samples: dict[str, jnp.ndarray]
    acceptance_rate: float
    num_divergent: int
    wall_time: float


def fit_nss(
    loglikelihood_fn: Callable,
    priors: dict,
    *,
    key,
    n_live: int = 500,
    num_delete: int = 50,
    num_inner_steps: int | None = None,
    log_evidence_tol: float = -3.0,
    max_iterations: int = 10000,
    n_posterior_samples: int = 2000,
    max_steps: int = 10,
    max_shrinkage: int = 20,
    verbose: bool = False,
) -> NSSResult:
    """Sample a star posterior with tengri's Nested Slice Sampling engine.

    Parameters
    ----------
    loglikelihood_fn : callable
        ``(params: dict[str, scalar]) -> scalar`` log-likelihood, JAX-traceable.
    priors : dict of str -> tengri Distribution
        Per-parameter priors; each must expose ``sample(key)`` and
        ``log_prob(x)`` (e.g. :class:`tengri.Uniform`).
    key : jax.random.PRNGKey
        Random key for the whole run.
    n_live : int
        Live points. More = better evidence and multimodal coverage.
    num_delete : int
        Points replaced per iteration (vmapped — the parallelism knob).
    num_inner_steps : int, optional
        Slice-sampling walk length per replacement; defaults to the
        parameter-space dimension.
    log_evidence_tol : float
        Terminate when log(Z_remaining / Z_accumulated) < this.
    max_iterations : int
        Safety limit on NS iterations.
    n_posterior_samples : int
        Equal-weight posterior draws returned.
    max_steps, max_shrinkage : int
        Slice stepping-out / shrinkage limits (XLA graph size grows with
        these; see tengri ``run_nss`` notes).
    verbose : bool
        Print progress every 10 iterations.

    Returns
    -------
    NSSResult
    """
    names = tuple(priors)
    dim = len(names)
    if num_inner_steps is None:
        num_inner_steps = dim

    def logprior_fn(params):
        parts = [priors[name].log_prob(params[name]) for name in names]
        return jnp.sum(jnp.stack(parts))

    algo = as_top_level_api(
        logprior_fn,
        loglikelihood_fn,
        num_inner_steps,
        num_delete=num_delete,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )
    init_jit = jax.jit(algo.init)
    step_jit = jax.jit(algo.step)

    key, init_key = jax.random.split(key)
    particles = {
        name: jax.vmap(priors[name].sample)(
            jax.random.split(jax.random.fold_in(init_key, i), n_live)
        )
        for i, name in enumerate(names)
    }

    t0 = time.time()
    live = init_jit(particles)
    dead_particles = []
    n_iter = 0
    while True:
        key, step_key = jax.random.split(key)
        live, dead = step_jit(step_key, live)
        dead_particles.append(dead.particles)
        n_iter += 1

        remaining = float(live.integrator.logZ_live - live.integrator.logZ)
        if verbose and n_iter % 10 == 0:
            logz_est = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))
            print(
                f"  NSS iter {n_iter}: log Z ≈ {logz_est:.2f}, "
                f"elapsed={time.time() - t0:.1f}s"
            )
        if remaining < log_evidence_tol or n_iter >= max_iterations:
            break

    wall_time = time.time() - t0
    logz = float(jnp.logaddexp(live.integrator.logZ, live.integrator.logZ_live))

    all_particles = [*dead_particles, live.particles]
    final = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_particles)
    ns_run = NSInfo(final, None)

    key, sample_key, ess_key = jax.random.split(key, 3)
    resampled = ns_sample(sample_key, ns_run, n_posterior_samples)
    ess_val = float(ns_ess(ess_key, ns_run))

    return NSSResult(
        samples={name: resampled.position[name] for name in names},
        logz=logz,
        ess=ess_val,
        n_iterations=n_iter,
        wall_time=wall_time,
    )


def fit_nuts(
    loglikelihood_fn: Callable,
    priors: dict,
    *,
    key,
    num_warmup: int = 1000,
    num_samples: int = 1000,
) -> NUTSResult:
    """Sample a star posterior with blackjax NUTS in unbounded ξ-space.

    Each parameter is mapped through its prior's ``unstandardize`` (standard
    normal ξ → bounded physical value), and the target is the information
    Hamiltonian ``log L(h(ξ)) - ξᵀξ/2`` — tengri's convention. The bounded
    transform is load-bearing for grid-lookup models: the clamped LUT has zero
    gradient outside the grid hull, where an unbounded sampler would strand.

    Prefer :func:`fit_nss` when the posterior may be multimodal (dwarf/giant
    degeneracies) or the evidence is needed; prefer NUTS for higher-dimensional
    unimodal joint fits, with the C²-smooth triweight interpolation
    (``interp_method='triweight'``) for well-behaved gradients.

    Parameters
    ----------
    loglikelihood_fn : callable
        ``(params: dict[str, scalar]) -> scalar`` log-likelihood, JAX-traceable
        and differentiable.
    priors : dict of str -> tengri Distribution
        Per-parameter priors; each must expose ``unstandardize(xi)``
        (e.g. :class:`tengri.Uniform`).
    key : jax.random.PRNGKey
        Random key for warmup and sampling.
    num_warmup : int
        Window-adaptation steps (step size + mass matrix).
    num_samples : int
        Posterior draws after warmup.

    Returns
    -------
    NUTSResult
    """
    import blackjax

    names = tuple(priors)

    def to_physical(xi):
        return {name: priors[name].unstandardize(xi[name]) for name in names}

    def logdensity(xi):
        prior_term = -0.5 * jnp.sum(jnp.stack([xi[name] ** 2 for name in names]))
        return loglikelihood_fn(to_physical(xi)) + prior_term

    t0 = time.time()
    key, warmup_key, sample_key = jax.random.split(key, 3)
    xi0 = {name: jnp.asarray(0.0) for name in names}
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, tuned), _ = warmup.run(warmup_key, xi0, num_steps=num_warmup)
    kernel = blackjax.nuts(logdensity, **tuned).step

    def one_step(state, step_key):
        state, info = kernel(step_key, state)
        return state, (state.position, info.acceptance_rate, info.is_divergent)

    _, (positions, acceptance, divergent) = jax.lax.scan(
        one_step, state, jax.random.split(sample_key, num_samples)
    )
    wall_time = time.time() - t0

    return NUTSResult(
        samples={name: priors[name].unstandardize(positions[name]) for name in names},
        acceptance_rate=float(jnp.mean(acceptance)),
        num_divergent=int(jnp.sum(divergent)),
        wall_time=wall_time,
    )
