"""tengri-stars: differentiable stellar-parameter inference on the tengri stack."""

from tengri_stars.grids import PhotometryGrid, SpectralGrid, load_photometry_grid
from tengri_stars.inference import NSSResult, fit_nss
from tengri_stars.model import StarModel

__version__ = "0.0.1"
__all__ = [
    "NSSResult",
    "PhotometryGrid",
    "SpectralGrid",
    "StarModel",
    "fit_nss",
    "load_photometry_grid",
]
