"""Unit tests for Experiment E4: Pipeline Control and Component Ablation.

Run with:  pytest tests/test_e4.py -v
"""
from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from colorize.colorization_config import ColorizationConfig
from colorize.extraction import CropBox
from colorize.regions import Decomposition, Region
from colorize.results import ResultsRecord, append_result, load_results
from experiments.e4 import (
    DEFAULT_MARGINS,
    DEFAULT_SMOOTHING_WIDTHS,
    _get_existing_pairs,
    build_default_config,
    build_expert_registry,
    build_parser,
    evaluate_colorization_metrics,
    get_e4_conditions,
    load_active_e4_ids,
    load_ids_json,
    preflight_image_files,
    run_pipeline_for_image_ablations,
)


# ---------------------------------------------------------------------------
# Test: Default Config & Condition Generation
# ---------------------------------------------------------------------------

def test_build_default_config():
    cfg = build_default_config()
    assert cfg.margin == 0.2
    assert cfg.smoothing_width == 10.0
    assert cfg.blending_on is True
    assert cfg.region_shape == "mask"
    assert cfg.input_size == 256


def test_get_e4_conditions_coverage():
    conds = get_e4_conditions(include_baseline_e3=True)

    # 1. Default
    assert "default" in conds
    assert conds["default"].config.margin == 0.2
    assert conds["default"].config.blending_on is True
    assert conds["default"].config.region_shape == "mask"

    # 2. Margins 0.0, 0.1, 0.2, 0.3
    for m in (0.0, 0.1, 0.2, 0.3):
        name = f"margin_{m:.1f}"
        assert name in conds
        assert conds[name].config.margin == pytest.approx(m)
        assert conds[name].config.blending_on is True
        assert conds[name].config.region_shape == "mask"

    # 3. Blending off (binary paste)
    assert "blending_off" in conds
    assert conds["blending_off"].config.blending_on is False
    assert conds["blending_off"].config.margin == 0.2

    # 4. Smoothing widths
    for sw in DEFAULT_SMOOTHING_WIDTHS:
        name = f"smooth_{sw:.1f}"
        assert name in conds
        assert conds[name].config.smoothing_width == pytest.approx(sw)
        assert conds[name].config.blending_on is True

    # 5. Region shapes (mask vs box)
    assert "shape_box" in conds
    assert conds["shape_box"].config.region_shape == "box"
    assert conds["shape_box"].config.blending_on is True

    # 6. E3 Baseline comparison
    assert "e3_ddcolor_t_whole" in conds


def test_build_expert_registry():
    ckpt_path = "checkpoints/ddcolor_paper_tiny.pth"
    registry = build_expert_registry(ckpt_path)

    # Verify all 4 expert slots plus fallback are present
    for slot in ("people", "nature", "urban", "animals", "fallback"):
        assert slot in registry
        assert registry[slot] == str(Path(ckpt_path).resolve())


# ---------------------------------------------------------------------------
# Test: Metrics Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_colorization_metrics_identical_images():
    H, W = 64, 64
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    metrics = evaluate_colorization_metrics(img, img, compute_perceptual=False)

    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["lab_delta_e"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["boundary_delta_e"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["colorfulness"] >= 0.0


def test_evaluate_colorization_metrics_different_images():
    H, W = 64, 64
    gt = np.zeros((H, W, 3), dtype=np.uint8)
    pred = np.full((H, W, 3), 50, dtype=np.uint8)
    metrics = evaluate_colorization_metrics(gt, pred, compute_perceptual=False)

    assert math.isfinite(metrics["psnr"])
    assert metrics["psnr"] > 0
    assert 0 <= metrics["ssim"] <= 1.0
    assert metrics["lab_delta_e"] > 0


# ---------------------------------------------------------------------------
# Test: Mocked Pipeline Execution
# ---------------------------------------------------------------------------

class MockExpertManager:
    """Mock expert manager that returns constant distinct chrominance."""

    def __init__(self):
        self.call_count = 0

    def colorize(self, domain: str, L_crop: np.ndarray) -> np.ndarray:
        self.call_count += 1
        S = L_crop.shape[0]
        # Return distinctive small ab chrominance
        ab = np.zeros((S, S, 2), dtype=np.float32)
        ab[:, :, 0] = 10.0
        ab[:, :, 1] = -10.0
        return ab


def test_run_pipeline_for_image_ablations():
    H, W = 80, 80
    L_full = np.full((H, W), 50.0, dtype=np.float32)
    original_shape = (H, W)

    # 1 region + uncovered fallback
    mask = np.zeros((H, W), dtype=bool)
    mask[20:60, 20:60] = True
    region = Region(
        label="person",
        score=0.9,
        box=np.array([20, 20, 60, 60], dtype=float),
        mask=mask,
        domain="people",
    )

    owner = np.full((H, W), -1, dtype=np.int32)
    owner[mask] = 0
    domain_map = np.full((H, W), 4, dtype=np.int8)  # 4 = fallback
    domain_map[mask] = 0                           # 0 = people

    decomp = Decomposition(
        shape=(H, W),
        regions=[region],
        domains=("people", "nature", "urban", "animals", "fallback"),
        owner=owner,
        domain_map=domain_map,
        coverage=float(mask.mean()),
        domain_share={"people": float(mask.mean()), "fallback": float((~mask).mean())},
    )

    conds = {
        "default": get_e4_conditions()["default"],
        "margin_0.0": get_e4_conditions()["margin_0.0"],
        "margin_0.1": get_e4_conditions()["margin_0.1"],
        "blending_off": get_e4_conditions()["blending_off"],
        "smooth_5.0": get_e4_conditions()["smooth_5.0"],
        "shape_box": get_e4_conditions()["shape_box"],
    }

    mock_mgr = MockExpertManager()
    results = run_pipeline_for_image_ablations(
        L_full=L_full,
        decomp=decomp,
        original_shape=original_shape,
        conditions_to_run=conds,
        manager=mock_mgr,
        checkpoint="fake_ckpt.pth",
        ddcolor_path=".",
        device="cpu",
    )

    # Verify all requested conditions returned an RGB image
    for name in conds:
        assert name in results
        rgb = results[name]
        assert rgb.shape == (H, W, 3)
        assert rgb.dtype == np.uint8

    # Verify expert was called efficiently (2 regions × 3 margins = 6 calls)
    assert mock_mgr.call_count == 6


# ---------------------------------------------------------------------------
# Test: Resumability and CSV Schema
# ---------------------------------------------------------------------------

def test_results_csv_schema_and_resumability():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "results.csv"

        r1 = ResultsRecord(
            experiment="E4",
            condition="default",
            image_id="000000000139",
            metric="psnr",
            value=28.5,
        )
        r2 = ResultsRecord(
            experiment="E4",
            condition="margin_0.0",
            image_id="000000000139",
            metric="psnr",
            value=26.2,
        )
        append_result(csv_path, r1)
        append_result(csv_path, r2)

        # Verify load_results loads exact records
        loaded = load_results(csv_path)
        assert len(loaded) == 2
        assert loaded[0].experiment == "E4"
        assert loaded[0].condition == "default"
        assert loaded[0].image_id == "000000000139"
        assert loaded[0].metric == "psnr"
        assert loaded[0].value == pytest.approx(28.5)

        # Verify _get_existing_pairs returns the recorded pairs
        pairs = _get_existing_pairs(csv_path)
        assert ("000000000139", "default") in pairs
        assert ("000000000139", "margin_0.0") in pairs
        assert ("000000000139", "margin_0.1") not in pairs


# ---------------------------------------------------------------------------
# Test: CLI Parser
# ---------------------------------------------------------------------------

def test_cli_parser_subcommands():
    parser = build_parser()

    # Test make-lists args
    args_lists = parser.parse_args(["make-lists", "--limit", "10"])
    assert args_lists.command == "make-lists"
    assert args_lists.limit == 10

    # Test run args
    args_run = parser.parse_args([
        "run",
        "--checkpoint", "ckpt.pth",
        "--device", "cpu",
        "--margins", "0.0,0.2",
        "--limit", "5",
    ])
    assert args_run.command == "run"
    assert args_run.checkpoint == "ckpt.pth"
    assert args_run.device == "cpu"
    assert args_run.margins == "0.0,0.2"
    assert args_run.limit == 5

    # Test figures args
    args_fig = parser.parse_args(["figures", "--results-dir", "results/test_e4"])
    assert args_fig.command == "figures"
    assert args_fig.results_dir == "results/test_e4"


# ---------------------------------------------------------------------------
# Test: Image-ID resolution and disk preflight
# ---------------------------------------------------------------------------

def _write_ids_json(path: Path, per_set: int = 20, start: int = 1) -> dict:
    payload = {}
    image_id = start
    for key in (
        "domain_people",
        "domain_nature",
        "domain_urban",
        "domain_animals",
        "mixed_scene",
    ):
        payload[key] = list(range(image_id, image_id + per_set))
        image_id += per_set
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_load_ids_json_and_validation(tmp_path):
    path = tmp_path / "e3_ids.json"
    payload = _write_ids_json(path)

    ids = load_ids_json(path)
    assert len(ids) == 100
    assert len(set(ids)) == 100

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing set keys"):
        load_ids_json(empty)

    duplicate = tmp_path / "duplicate.json"
    dup_payload = dict(payload)
    dup_payload["mixed_scene"] = [dup_payload["mixed_scene"][0]] * 20
    duplicate.write_text(json.dumps(dup_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_ids_json(duplicate)

    with pytest.raises(FileNotFoundError):
        load_ids_json(tmp_path / "absent.json")


def test_load_active_e4_ids_priority(tmp_path):
    json_path = tmp_path / "e3_ids.json"
    _write_ids_json(json_path, start=1)

    list_path = tmp_path / "ids.txt"
    list_path.write_text("000000000900\n000000000901\n", encoding="utf-8")

    assert load_active_e4_ids(id_list=list_path, ids_json=json_path) == [900, 901]
    assert len(load_active_e4_ids(ids_json=json_path)) == 100

    # An explicit but missing list must not silently fall back.
    with pytest.raises(FileNotFoundError, match="image-ID list not found"):
        load_active_e4_ids(id_list=tmp_path / "absent.txt", ids_json=json_path)


def test_preflight_image_files(tmp_path):
    ids = [11, 22, 33]
    for image_id in ids[:2]:
        (tmp_path / f"{image_id:012d}.jpg").write_bytes(b"x")

    missing = preflight_image_files(ids, tmp_path, "val2017")
    assert missing == [33]
    assert preflight_image_files(ids[:2], tmp_path, "val2017") == []
