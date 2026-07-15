"""tengri-stars: differentiable stellar-parameter inference on the tengri stack."""

from tengri_stars.diagnostics import effective_sample_size, ess_summary
from tengri_stars.grids import (
    IsochroneGrid,
    PhotometryGrid,
    SpectralGrid,
    load_isochrone_grid,
    load_photometry_grid,
)
from tengri_stars.inference import (
    LaplaceResult,
    MAPResult,
    NSSResult,
    NUTSResult,
    fit_hmc,
    fit_laplace,
    fit_map,
    fit_nss,
    fit_nuts,
    make_hmc_pipeline,
    make_laplace_pipeline,
    make_nss_pipeline,
)
from tengri_stars.model import StarModel
from tengri_stars.parametrization import (
    TSLTE_ZEROPOINT,
    FreeAtmosphere,
    MISTTrack,
    Parametrization,
)
from tengri_stars.plotting import overlay_corner

__version__ = "0.0.1"
__all__ = [
    "TSLTE_ZEROPOINT",
    "FreeAtmosphere",
    "IsochroneGrid",
    "LaplaceResult",
    "MAPResult",
    "MISTTrack",
    "NSSResult",
    "NUTSResult",
    "Parametrization",
    "PhotometryGrid",
    "SpectralGrid",
    "StarModel",
    "effective_sample_size",
    "ess_summary",
    "fit_hmc",
    "fit_laplace",
    "fit_map",
    "fit_nss",
    "fit_nuts",
    "load_isochrone_grid",
    "load_photometry_grid",
    "make_hmc_pipeline",
    "make_laplace_pipeline",
    "make_nss_pipeline",
    "overlay_corner",
]
