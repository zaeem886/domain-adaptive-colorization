"""Unit tests for colorize.plotting (Person C).

Tests verify that plotting functions run without crashing and return a Figure
(or None if matplotlib is unavailable).  No display is required — the Agg
backend is used so tests are headless-safe.

All tests are automatically skipped if matplotlib is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module gracefully if matplotlib is missing
matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")

from colorize.plotting import (
    plot_metric_bar,
    plot_metric_comparison,
    plot_confusion_matrix,
    plot_colorfulness_distribution,
    plot_delta_e_map,
)
from colorize.results import ResultsRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_records(n_per_cond: int = 5):
    """Create synthetic ResultsRecord objects for two conditions."""
    rng = np.random.default_rng(0)
    records = []
    for cond in ("ddcolor_t_whole", "ddcolor_l_whole"):
        for i in range(n_per_cond):
            records.append(
                ResultsRecord(
                    experiment="E3",
                    condition=cond,
                    image_id=str(i),
                    metric="psnr",
                    value=float(rng.uniform(20, 40)),
                )
            )
            records.append(
                ResultsRecord(
                    experiment="E3",
                    condition=cond,
                    image_id=str(i),
                    metric="ssim",
                    value=float(rng.uniform(0.5, 1.0)),
                )
            )
            records.append(
                ResultsRecord(
                    experiment="E3",
                    condition=cond,
                    image_id=str(i),
                    metric="colorfulness",
                    value=float(rng.uniform(10, 80)),
                )
            )
    return records


def _rand_rgb(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlotMetricBar:
    def test_returns_figure(self):
        records = _make_records()
        fig = plot_metric_bar(records, metric="psnr")
        assert fig is not None

    def test_with_output_path(self, tmp_path):
        records = _make_records()
        output = tmp_path / "bar.png"
        fig = plot_metric_bar(records, metric="psnr", title="Test bar", output_path=output)
        assert fig is not None
        # File should have been written
        assert output.exists() and output.stat().st_size > 0

    def test_missing_metric_no_crash(self):
        records = _make_records()
        # Should not raise, just warn
        fig = plot_metric_bar(records, metric="nonexistent_metric")
        assert fig is not None  # returns an empty figure

    def test_empty_records_no_crash(self):
        fig = plot_metric_bar([], metric="psnr")
        assert fig is not None


class TestPlotMetricComparison:
    def test_returns_figure(self):
        records = _make_records()
        fig = plot_metric_comparison(records, metrics=["psnr", "ssim"])
        assert fig is not None

    def test_with_conditions_filter(self):
        records = _make_records()
        fig = plot_metric_comparison(
            records,
            metrics=["psnr"],
            conditions=["ddcolor_t_whole"],
        )
        assert fig is not None

    def test_with_output_path(self, tmp_path):
        records = _make_records()
        output = tmp_path / "comparison.png"
        fig = plot_metric_comparison(
            records,
            metrics=["psnr", "ssim", "colorfulness"],
            output_path=output,
        )
        assert output.exists()


class TestPlotConfusionMatrix:
    def test_4x4_no_crash(self):
        cm = np.array(
            [[50, 5, 2, 1],
             [3, 40, 4, 2],
             [1, 2, 45, 3],
             [0, 1, 2, 48]],
            dtype=np.int64,
        )
        domain_names = ["people", "nature", "urban", "animals"]
        fig = plot_confusion_matrix(cm, domain_names)
        assert fig is not None

    def test_2x2_no_crash(self):
        cm = np.array([[10, 2], [3, 15]], dtype=np.int64)
        fig = plot_confusion_matrix(cm, ["people", "nature"], title="Test CM")
        assert fig is not None

    def test_with_output_path(self, tmp_path):
        cm = np.eye(3, dtype=np.int64) * 20
        output = tmp_path / "cm.png"
        fig = plot_confusion_matrix(cm, ["a", "b", "c"], output_path=output)
        assert output.exists()

    def test_all_zeros_no_crash(self):
        cm = np.zeros((4, 4), dtype=np.int64)
        fig = plot_confusion_matrix(cm, ["p", "n", "u", "a"])
        assert fig is not None


class TestPlotColorfulnessDistribution:
    def test_returns_figure(self):
        records = _make_records()
        fig = plot_colorfulness_distribution(records)
        assert fig is not None

    def test_no_colorfulness_records_no_crash(self):
        psnr_only = [
            ResultsRecord("E3", "cond", "1", "psnr", 30.0),
        ]
        fig = plot_colorfulness_distribution(psnr_only)
        assert fig is not None

    def test_with_output_path(self, tmp_path):
        records = _make_records()
        output = tmp_path / "cf_dist.png"
        fig = plot_colorfulness_distribution(records, output_path=output)
        assert output.exists()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("skimage"),
    reason="scikit-image not installed",
)
class TestPlotDeltaEMap:
    def test_returns_figure(self):
        img_true = _rand_rgb(seed=1)
        img_pred = _rand_rgb(seed=2)
        fig = plot_delta_e_map(img_true, img_pred)
        assert fig is not None

    def test_identical_images_no_crash(self):
        img = _rand_rgb()
        fig = plot_delta_e_map(img, img.copy())
        assert fig is not None

    def test_with_output_path(self, tmp_path):
        img_true = _rand_rgb(seed=3)
        img_pred = _rand_rgb(seed=4)
        output = tmp_path / "delta_e.png"
        fig = plot_delta_e_map(img_true, img_pred, output_path=output)
        assert output.exists() and output.stat().st_size > 0

    def test_pure_color_no_crash(self):
        img_true = np.zeros((16, 16, 3), dtype=np.uint8)
        img_true[..., 0] = 255  # red
        img_pred = np.zeros((16, 16, 3), dtype=np.uint8)
        img_pred[..., 2] = 255  # blue
        fig = plot_delta_e_map(img_true, img_pred)
        assert fig is not None

