"""Synthetic stellar grids: loaders and differentiable lookup tables."""

from tengri_stars.grids.photometry_grid import PhotometryGrid, load_photometry_grid
from tengri_stars.grids.spectral_grid import SpectralGrid

__all__ = ["PhotometryGrid", "SpectralGrid", "load_photometry_grid"]
