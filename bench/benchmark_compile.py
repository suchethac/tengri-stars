"""Where does compile time actually go, and can a catalog compile be cheaper?

Two questions, measured rather than assumed:

1. **Is the persistent cache working?** tengri auto-enables JAX's on-disk
   compilation cache at import. But that cache stores the *XLA compile*
   (HLO → executable) — it cannot cache Python-level **tracing and lowering**,
   which for a big graph (a nested-sampling while_loop, a 100-star vmap) is
   often the dominant cost. This splits the two.

2. **Does vmapping the jitted pipeline compile faster?** ``vmap`` builds a
   *batch-sized* graph: bigger HLO, longer compile, but vectorized execution.
   ``lax.map`` (scan-based) compiles a *single-star* body once and walks the
   catalog on-device. Chunked vmap inside lax.map trades between them.

Run twice back-to-back to see the cache effect on the second process:
    PYTHONPATH=src JAX_PLATFORMS=cpu python bench/benchmark_compile.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time

import jax
import jax.numpy as jnp
import numpy as np
from tengri import Uniform

from tengri_stars import StarModel, load_photometry_grid, make_hmc_pipeline

jax.config.update("jax_enable_x64", True)

N_STARS = 100
CHUNK = 10
SIG_MAG = 0.02

print(f"persistent cache dir : {jax.config.jax_compilation_cache_dir}")
print(f"min compile time [s] : {jax.config.jax_persistent_cache_min_compile_time_secs}")

grid = load_photometry_grid("data/TSLTE_combined_photometry.fits", fill="nearest")
model = StarModel(grid=grid, interp_method="pchip")
priors = {
    "teff": Uniform(float(grid.axes[0][0]), float(grid.axes[0][-1])),
    "logg": Uniform(float(grid.axes[1][0]), float(grid.axes[1][-1])),
    "feh": Uniform(float(grid.axes[2][0]), float(grid.axes[2][-1])),
    "mu": Uniform(-28.0, -8.0),
}


def loglikelihood(p, mags):
    pred = model.predict_mags(teff=p["teff"], logg=p["logg"], feh=p["feh"], mu=p["mu"])
    return -0.5 * jnp.sum(((pred - mags) / SIG_MAG) ** 2)


rng = np.random.default_rng(4)
clean = np.stack(
    [
        np.asarray(
            model.predict_mags(
                teff=rng.uniform(4200, 6200),
                logg=rng.uniform(1, 4.5),
                feh=rng.uniform(-3, -0.5),
                mu=-18.0,
            )
        )
        for _ in range(N_STARS)
    ]
)
mags_batch = jnp.asarray(clean + rng.normal(0.0, SIG_MAG, clean.shape))
keys = jax.random.split(jax.random.PRNGKey(0), N_STARS)

pipeline = make_hmc_pipeline(loglikelihood, priors, num_warmup=500, num_samples=1000)


def split_compile(label, fn, *args):
    """Time lowering (trace, uncacheable) and compilation (XLA, cacheable)."""
    t0 = time.time()
    lowered = jax.jit(fn).lower(*args)
    t_lower = time.time() - t0

    t0 = time.time()
    compiled = lowered.compile()
    t_compile = time.time() - t0

    t0 = time.time()
    jax.block_until_ready(compiled(*args))
    t_run = time.time() - t0
    print(f"{label:26s} lower {t_lower:6.2f} s | compile {t_compile:6.2f} s | run {t_run:7.2f} s")
    return t_lower, t_compile, t_run


print("\n--- where the time goes (lower = tracing, NOT cacheable) ---")
split_compile("single star", pipeline.__wrapped__, keys[0], mags_batch[0])
v_lower, v_compile, v_run = split_compile(
    f"vmap × {N_STARS}", jax.vmap(pipeline.__wrapped__), keys, mags_batch
)


def map_catalog(ks, ms):
    """lax.map: compiles a single-star body, walks the catalog on-device."""
    return jax.lax.map(lambda km: pipeline.__wrapped__(km[0], km[1]), (ks, ms))


m_lower, m_compile, m_run = split_compile(f"lax.map × {N_STARS}", map_catalog, keys, mags_batch)


def chunked_catalog(ks, ms):
    """vmap within chunks, lax.map across them: chunk-sized graph, vectorized."""
    ks = ks.reshape(N_STARS // CHUNK, CHUNK, *ks.shape[1:])
    ms = ms.reshape(N_STARS // CHUNK, CHUNK, *ms.shape[1:])
    per_chunk = jax.vmap(pipeline.__wrapped__)
    return jax.lax.map(lambda km: per_chunk(km[0], km[1]), (ks, ms))


c_lower, c_compile, c_run = split_compile(
    f"chunked vmap ({CHUNK})", chunked_catalog, keys, mags_batch
)

print("\n--- catalog totals (compile + run), 100 stars ---")
for label, lo, co, ru in [
    (f"vmap × {N_STARS}", v_lower, v_compile, v_run),
    (f"lax.map × {N_STARS}", m_lower, m_compile, m_run),
    (f"chunked vmap ({CHUNK})", c_lower, c_compile, c_run),
]:
    total = lo + co + ru
    print(f"{label:26s} total {total:7.1f} s  ({ru / N_STARS * 1000:6.0f} ms/star steady)")
