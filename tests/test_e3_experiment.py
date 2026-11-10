"""Tests for the E3 T/L orchestration layer."""
from __future__ import annotations

import csv

from colorize.results import ResultsRecord, append_result
from experiments import e3


def _make_coco_files(tmp_path, set_name="mixed_scene"):
    image_dir = tmp_path / "val2017"
    image_dir.mkdir()
    for image_id in e3.E3_SET_IDS[set_name]:
        (image_dir / f"{image_id:012d}.jpg").write_bytes(b"placeholder")
    return tmp_path


def test_build_e3_image_manifest_covers_selected_set(tmp_path):
    coco_root = _make_coco_files(tmp_path)

    image_paths, image_sets, missing = e3.build_e3_image_manifest(
        coco_root, set_names=["mixed_scene"]
    )

    assert len(image_paths) == 20
    assert not missing
    assert set(image_sets) == set(e3.E3_SET_IDS["mixed_scene"])
    assert all(image_sets[image_id] == ("mixed_scene",) for image_id in image_sets)


def test_build_e3_image_manifest_fails_before_partial_run(tmp_path):
    coco_root = _make_coco_files(tmp_path)
    missing_path = tmp_path / "val2017" / f"{e3.E3_SET_IDS['mixed_scene'][0]:012d}.jpg"
    missing_path.unlink()

    try:
        e3.build_e3_image_manifest(coco_root, set_names=["mixed_scene"])
    except FileNotFoundError as exc:
        assert "mixed_scene" in str(exc)
    else:
        raise AssertionError("missing E3 images must fail unless allow_missing=True")


def test_run_e3_experiment_runs_both_variants_and_writes_summary(tmp_path, monkeypatch):
    coco_root = _make_coco_files(tmp_path)

    def fake_runner(*, image_paths, output_csv, condition, **kwargs):
        assert len(image_paths) == 20
        assert kwargs["input_size"] in (256, 512)
        records = []
        for image_id, _ in image_paths:
            record = ResultsRecord("E3", condition, str(image_id), "psnr", 30.0)
            append_result(output_csv, record)
            records.append(record)
        return records

    monkeypatch.setattr(e3, "run_e3_baseline", fake_runner)
    output_csv = tmp_path / "outputs" / "e3_results.csv"
    summary_csv = tmp_path / "outputs" / "e3_summary.csv"
    manifest_csv = tmp_path / "outputs" / "e3_manifest.csv"

    records = e3.run_e3_experiment(
        coco_root=coco_root,
        ddcolor_path="unused-ddcolor",
        checkpoint_t="unused-t",
        checkpoint_l="unused-l",
        output_csv=output_csv,
        summary_csv=summary_csv,
        manifest_csv=manifest_csv,
        set_names=["mixed_scene"],
        gt_root=coco_root,
        device="cpu",
    )

    assert len(records) == 40
    assert {record.condition for record in records} == {
        "ddcolor_t_whole",
        "ddcolor_l_whole",
    }
    assert output_csv.exists()
    assert summary_csv.exists()
    assert manifest_csv.exists()

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["condition"], row["set"], row["metric"]) for row in rows} == {
        ("ddcolor_t_whole", "mixed_scene", "psnr"),
        ("ddcolor_l_whole", "mixed_scene", "psnr"),
    }
    assert all(row["count"] == "20" for row in rows)
