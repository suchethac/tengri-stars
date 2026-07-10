"""Posterior plotting helpers."""

from __future__ import annotations

import numpy as np


def overlay_corner(
    sample_sets: list[dict],
    *,
    names: list[str],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    legend_labels: list[str] | None = None,
    truths: dict | None = None,
    panel_size: float = 2.8,
    pad_frac: float = 0.08,
    **corner_kwargs,
):
    """Overlay several posteriors on one corner plot with shared, full ranges.

    ``corner.corner`` freezes axis ranges from the first dataset plotted, so a
    wider second posterior gets its contours clipped. This helper computes a
    joint range per parameter across *all* sample sets (and the truth values),
    pads it, and passes it to every layer — every contour is fully visible.

    Parameters
    ----------
    sample_sets : list of dict
        Posterior sample dicts (name → 1-D array), e.g. ``result.samples``.
    names : list of str
        Parameter order for the corner axes.
    labels : list of str, optional
        Axis labels; defaults to ``names``.
    colors : list of str, optional
        One matplotlib color per sample set (default: C0, C1, ...).
    legend_labels : list of str, optional
        Legend entry per sample set; no legend when omitted.
    truths : dict, optional
        True values (name → float), drawn on the first layer and included in
        the range computation so an off-cloud truth marker is never clipped.
    panel_size : float
        Panel edge in inches; figure is ``panel_size × n_params`` square.
    pad_frac : float
        Fractional padding added on each side of the joint range.
    **corner_kwargs
        Forwarded to every ``corner.corner`` call.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import corner
    import matplotlib.pyplot as plt

    if colors is None:
        colors = [f"C{i}" for i in range(len(sample_sets))]

    ranges = []
    for name in names:
        lo = min(float(np.min(np.asarray(s[name]))) for s in sample_sets)
        hi = max(float(np.max(np.asarray(s[name]))) for s in sample_sets)
        if truths is not None and name in truths:
            lo, hi = min(lo, float(truths[name])), max(hi, float(truths[name]))
        pad = pad_frac * (hi - lo) or 1.0
        ranges.append((lo - pad, hi + pad))

    defaults = {"hist_kwargs": {"density": True}, "plot_datapoints": False, "smooth": 1.0}
    defaults.update(corner_kwargs)

    fig = None
    for i, (samples, color) in enumerate(zip(sample_sets, colors)):
        stack = np.column_stack([np.asarray(samples[n]) for n in names])
        fig = corner.corner(
            stack,
            fig=fig,
            range=ranges,
            color=color,
            labels=labels or names,
            truths=[truths[n] for n in names] if (truths and i == 0) else None,
            **defaults,
        )
    fig.set_size_inches(panel_size * len(names), panel_size * len(names))

    if legend_labels is not None:
        fig.legend(
            handles=[
                plt.Line2D([], [], color=c, label=text) for c, text in zip(colors, legend_labels)
            ],
            loc="upper right",
            frameon=False,
        )
    return fig
