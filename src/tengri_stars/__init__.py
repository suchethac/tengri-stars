"""tengri-stars: differentiable stellar-parameter inference on the tengri stack."""

from tengri_stars.grids import PhotometryGrid, SpectralGrid, load_photometry_grid
from tengri_stars.inference import MAPResult, NSSResult, NUTSResult, fit_map, fit_nss, fit_nuts
from tengri_stars.model import StarModel

__version__ = "0.0.1"
__all__ = [
    "MAPResult",
    "NSSResult",
    "NUTSResult",
    "PhotometryGrid",
    "SpectralGrid",
    "StarModel",
    "fit_map",
    "fit_nss",
    "fit_nuts",
    "load_photometry_grid",
]
