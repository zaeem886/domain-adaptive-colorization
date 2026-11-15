"""Tests for scripts/generate_test_ids.py.

Run with:  pytest tests/test_generate_test_ids.py -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts.generate_test_ids import (
    SET_KEYS,
    _sample_sets,
    main,
    rewrite_test_sets,
)


def _build_dataset(root: Path, per_set: int = 2) -> None:
    """Create a tiny COCO-shaped mount with domain-identifiable annotations."""
    val_dir = root / "val2017"
    val_dir.mkdir(parents=True)
    images = []
    annotations = []
    next_id = 1

    def add(count: int, areas: dict) -> None:
        nonlocal next_id
        for _ in range(count):
            images.append({
                "id": next_id,
                "width": 1000,
                "height": 1000,
                "file_name": f"{next_id:012d}.jpg",
            })
            for category_id, area in areas.items():
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": next_id,
                    "category_id": category_id,
                    "area": area,
                })
                annotations[-1]["bbox"] = [0, 0, 10, 10]
                annotations[-1]["iscrowd"] = 0
            (val_dir / f"{next_id:012d}.jpg").write_bytes(b"x")
            next_id += 1

    add(per_set + 1, {1: 300000})  # people
    add(per_set + 1, {2: 300000})  # animals
    add(per_set + 1, {})  # nature
    add(3 * per_set, {3: 200000})  # urban
    add(per_set, {1: 150000, 2: 150000})  # mixed

    root.joinpath("instances_val2017.json").write_text(
        json.dumps({
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "dog"},
                {"id": 3, "name": "car"},
            ],
        }),
        encoding="utf-8",
    )


def test_generate_ids_end_to_end(tmp_path):
    _build_dataset(tmp_path, per_set=2)
    output = tmp_path / "outputs" / "e3_ids.json"

    assert main([
        "--images-root", str(tmp_path),
        "--annotations", str(tmp_path / "instances_val2017.json"),
        "--output", str(output),
        "--per-set", "2",
        "--seed", "0",
    ]) == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    assert set(data) == set(SET_KEYS)
    all_ids = [image_id for key in SET_KEYS for image_id in data[key]]
    assert len(all_ids) == 2 * len(SET_KEYS)
    assert len(set(all_ids)) == len(all_ids)
    for key in SET_KEYS:
        assert len(data[key]) == 2

    # Deterministic reuse should produce the same digest without regenerating.
    before = output.read_bytes()
    assert main([
        "--images-root", str(tmp_path),
        "--output", str(output),
        "--per-set", "2",
        "--seed", "0",
    ]) == 0
    assert output.read_bytes() == before


def test_generate_ids_rejects_too_small_pool(tmp_path):
    _build_dataset(tmp_path, per_set=2)
    with pytest.raises(SystemExit, match="only .* candidates"):
        main([
            "--images-root", str(tmp_path),
            "--annotations", str(tmp_path / "instances_val2017.json"),
            "--output", str(tmp_path / "ids.json"),
            "--per-set", "20",
        ])


def test_sample_sets_disjoint_and_deterministic():
    pools = {
        "domain_people": list(range(1, 41)),
        "domain_nature": list(range(41, 81)),
        "domain_urban": list(range(81, 121)),
        "domain_animals": list(range(121, 161)),
        "mixed_scene": list(range(161, 201)),
        "_all": list(range(1, 201)),
    }
    first = _sample_sets(pools, per_set=20, seed=0)
    second = _sample_sets(pools, per_set=20, seed=0)
    assert first == second
    for key in SET_KEYS:
        assert first[key] == sorted(first[key])
    assert len({image_id for key in SET_KEYS for image_id in first[key]}) == 20 * len(SET_KEYS)


def test_rewrite_test_sets(tmp_path):
    source = Path(__file__).resolve().parents[1] / "colorize" / "test_sets.py"
    target = tmp_path / "test_sets.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    sets = {
        key: list(range(index * 20 + 1, index * 20 + 21))
        for index, key in enumerate(SET_KEYS)
    }
    rewrite_test_sets(target, sets)

    spec = importlib.util.spec_from_file_location("rewritten_test_sets", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rewritten_test_sets"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.DOMAIN_SET_IDS["people"] == sets["domain_people"]
    assert module.DOMAIN_SET_IDS["nature"] == sets["domain_nature"]
    assert module.DOMAIN_SET_IDS["urban"] == sets["domain_urban"]
    assert module.DOMAIN_SET_IDS["animals"] == sets["domain_animals"]
    assert module.MIXED_SCENE_IDS == sets["mixed_scene"]
    assert module.SEGMENTATION_SET_IDS == sets["segmentation_set"]
    assert module.REGION_CROP_IDS == sets["region_crop"]
    assert module.find_duplicate_ids() == {}
