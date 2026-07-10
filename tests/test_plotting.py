"""Tests for the overlay corner-plot helper."""

import matplotlib

matplotlib.use("Agg")
import numpy as np

from tengri_stars import overlay_corner


def test_overlay_corner_covers_all_sample_sets_and_truths():
    rng = np.random.default_rng(0)
    narrow = {"a": rng.normal(0.0, 0.1, 800), "b": rng.normal(5.0, 0.2, 800)}
    wide = {"a": rng.normal(0.5, 1.0, 800), "b": rng.normal(4.0, 1.5, 800)}
    truths = {"a": -3.0, "b": 9.0}  # deliberately outside both clouds

    fig = overlay_corner(
        [narrow, wide],
        names=["a", "b"],
        labels=["A", "B"],
        colors=["C0", "C1"],
        legend_labels=["narrow", "wide"],
        truths=truths,
    )

    axes = np.array(fig.axes).reshape(2, 2)
    # Diagonal a-histogram must span both clouds AND the truth marker.
    lo, hi = axes[0, 0].get_xlim()
    assert lo < -3.0 and lo < np.min(wide["a"])
    assert hi > np.max(wide["a"])
    # Panels enlarged: at least 2.5 inches per parameter.
    w, h = fig.get_size_inches()
    assert w >= 5.0 and h >= 5.0
