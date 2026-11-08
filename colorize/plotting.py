"""Results plotting utilities for domain-adaptive colorization (Person C).

All functions degrade gracefully when matplotlib is not installed — they print
a warning and return ``None`` instead of raising.

Available plots
---------------
* plot_metric_bar              — bar chart for one metric across conditions
* plot_metric_comparison       — grouped bar chart for multiple metrics
* plot_confusion_matrix        — routing confusion-matrix heatmap
* plot_colorfulness_distribution — histogram of colorfulness per condition
* plot_delta_e_map             — per-pixel ΔE heatmap image
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib  # type: ignore[import]
    matplotlib.use("Agg")  # non-interactive backend; works on headless servers
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _get_cmap(name: str, n: int):
    """Return a colourmap, compatible with both old and new matplotlib."""
    try:
        # matplotlib >= 3.7: use colormaps registry
        return plt.colormaps[name].resampled(n)
    except AttributeError:
        # older matplotlib: plt.cm.get_cmap still exists
        return plt.cm.get_cmap(name, n)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_mpl(func_name: str):
    """Warn and return False when matplotlib is unavailable."""
    if not _HAS_MPL:
        warnings.warn(
            f"matplotlib is required for {func_name}. "
            "Install it with: pip install matplotlib",
            stacklevel=3,
        )
        return False
    return True


def _save_or_show(fig, output_path: Optional[Union[str, Path]]) -> None:
    """Save the figure to *output_path* if provided."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _filter_records(records, metric: str, conditions=None):
    """Filter records to those matching *metric* (and optionally *conditions*)."""
    filtered = [r for r in records if r.metric == metric]
    if conditions is not None:
        conditions = set(conditions)
        filtered = [r for r in filtered if r.condition in conditions]
    return filtered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_metric_bar(
    records,
    metric: str,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
):
    """Bar chart comparing conditions on a single metric.

    Parameters
    ----------
    records:
        List of :class:`colorize.results.ResultsRecord` objects.
    metric:
        Metric name to plot (e.g. ``"psnr"``).
    title:
        Plot title.  Defaults to the metric name.
    output_path:
        If provided, the figure is saved here (PNG/PDF/etc.).

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object, or ``None`` if matplotlib is unavailable.
    """
    if not _require_mpl("plot_metric_bar"):
        return None

    filtered = _filter_records(records, metric)
    if not filtered:
        warnings.warn(f"No records found for metric='{metric}'.", stacklevel=2)
        fig, ax = plt.subplots()
        ax.set_title(title or metric)
        _save_or_show(fig, output_path)
        return fig

    # Group by condition
    from collections import defaultdict
    cond_values = defaultdict(list)
    for r in filtered:
        cond_values[r.condition].append(r.value)

    conditions = list(cond_values.keys())
    means = [float(np.mean(cond_values[c])) for c in conditions]
    stds = [float(np.std(cond_values[c])) for c in conditions]

    fig, ax = plt.subplots(figsize=(max(5, len(conditions) * 1.5), 4))
    x = np.arange(len(conditions))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Metric: {metric}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


def plot_metric_comparison(
    records,
    metrics: Sequence[str],
    conditions: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
):
    """Grouped bar chart comparing multiple metrics across conditions.

    Parameters
    ----------
    records:
        List of :class:`colorize.results.ResultsRecord` objects.
    metrics:
        Metric names to include.
    conditions:
        Subset of conditions to plot.  If ``None``, all conditions are used.
    title:
        Plot title.
    output_path:
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _require_mpl("plot_metric_comparison"):
        return None

    from collections import defaultdict

    # Collect mean values per (condition, metric)
    cond_metric_vals: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.metric in metrics:
            if conditions is None or r.condition in conditions:
                cond_metric_vals[r.condition][r.metric].append(r.value)

    if not cond_metric_vals:
        warnings.warn("No matching records found.", stacklevel=2)
        fig, ax = plt.subplots()
        ax.set_title(title or "Metric comparison")
        _save_or_show(fig, output_path)
        return fig

    all_conditions = list(cond_metric_vals.keys())
    n_cond = len(all_conditions)
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    x = np.arange(n_cond)

    fig, ax = plt.subplots(figsize=(max(6, n_cond * 2), 5))
    cmap = _get_cmap("tab10", n_metrics)

    for m_idx, metric in enumerate(metrics):
        means = [
            float(np.mean(cond_metric_vals[c][metric])) if cond_metric_vals[c][metric] else 0.0
            for c in all_conditions
        ]
        offsets = x + (m_idx - n_metrics / 2.0 + 0.5) * width
        ax.bar(offsets, means, width=width * 0.9, label=metric, color=cmap(m_idx))

    ax.set_xticks(x)
    ax.set_xticklabels(all_conditions, rotation=15, ha="right")
    ax.set_title(title or "Metric comparison")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    domain_names: Sequence[str],
    title: str = "Routing Confusion Matrix",
    output_path: Optional[Union[str, Path]] = None,
):
    """Heatmap of a routing confusion matrix.

    Parameters
    ----------
    cm:
        Square int64 array of shape ``(D, D)``.  ``cm[i, j]`` is the number of
        pixels with GT domain *i* predicted as domain *j*.
    domain_names:
        Ordered list of D domain names.
    title:
        Figure title.
    output_path:
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _require_mpl("plot_confusion_matrix"):
        return None

    cm = np.asarray(cm, dtype=np.float64)
    D = cm.shape[0]

    fig, ax = plt.subplots(figsize=(max(4, D * 1.2), max(4, D * 1.2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(domain_names, rotation=45, ha="right")
    ax.set_yticklabels(domain_names)
    ax.set_xlabel("Predicted domain")
    ax.set_ylabel("GT domain")
    ax.set_title(title)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(D):
        for j in range(D):
            ax.text(
                j, i, f"{int(cm[i, j])}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    _save_or_show(fig, output_path)
    return fig


def plot_colorfulness_distribution(
    records,
    output_path: Optional[Union[str, Path]] = None,
):
    """Histogram of colourfulness values per condition.

    Parameters
    ----------
    records:
        List of :class:`colorize.results.ResultsRecord` objects.
    output_path:
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _require_mpl("plot_colorfulness_distribution"):
        return None

    from collections import defaultdict

    cond_values: dict = defaultdict(list)
    for r in records:
        if r.metric == "colorfulness":
            cond_values[r.condition].append(r.value)

    if not cond_values:
        warnings.warn("No colorfulness records found.", stacklevel=2)
        fig, ax = plt.subplots()
        ax.set_title("Colourfulness distribution")
        _save_or_show(fig, output_path)
        return fig

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = _get_cmap("tab10", len(cond_values))
    for idx, (cond, vals) in enumerate(cond_values.items()):
        ax.hist(vals, bins=20, alpha=0.6, label=cond, color=cmap(idx), edgecolor="white")

    ax.set_xlabel("Colourfulness")
    ax.set_ylabel("Count")
    ax.set_title("Colourfulness distribution per condition")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


def plot_delta_e_map(
    img_true: np.ndarray,
    img_pred: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
):
    """Visualise per-pixel ΔE as a heatmap.

    Parameters
    ----------
    img_true:
        Ground-truth image, shape ``(H, W, 3)``, dtype uint8 RGB.
    img_pred:
        Predicted/colorized image, same shape.
    output_path:
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _require_mpl("plot_delta_e_map"):
        return None

    try:
        from skimage import color as skcolor  # type: ignore[import]
    except ImportError:
        warnings.warn(
            "scikit-image is required for plot_delta_e_map. "
            "Install with: pip install scikit-image",
            stacklevel=2,
        )
        return None

    img_true = np.asarray(img_true, dtype=np.uint8)
    img_pred = np.asarray(img_pred, dtype=np.uint8)

    lab_true = skcolor.rgb2lab(img_true.astype(np.float32) / 255.0)
    lab_pred = skcolor.rgb2lab(img_pred.astype(np.float32) / 255.0)
    delta_e = skcolor.deltaE_ciede2000(lab_true, lab_pred)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(img_true)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")

    axes[1].imshow(img_pred)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    im = axes[2].imshow(delta_e, cmap="hot", vmin=0)
    axes[2].set_title("ΔE map")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Mean ΔE = {float(np.mean(delta_e)):.2f}", y=1.01)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig

