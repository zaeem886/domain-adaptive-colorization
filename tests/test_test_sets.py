"""Tests for fixed COCO sets and annotation-backed validation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import colorize.test_sets as test_sets


def _annotation_fixture() -> dict:
    """Build a small COCO-shaped fixture covering the checked-in IDs."""
    memberships = test_sets._all_set_memberships()
    domain_category = {
        "domain_people": "person",
        "domain_nature": "potted plant",
        "domain_urban": "car",
        "domain_animals": "dog",
    }
    images = [
        {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 8, "height": 6}
        for image_id in memberships
    ]
    categories = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "potted plant"},
        {"id": 3, "name": "car"},
        {"id": 4, "name": "dog"},
    ]
    annotations = []
    annotation_id = 1
    for domain, ids in test_sets.DOMAIN_SET_IDS.items():
        category_name = domain_category[f"domain_{domain}"]
        category_id = next(item["id"] for item in categories if item["name"] == category_name)
        for image_id in ids:
            annotations.append(
                {"id": annotation_id, "image_id": image_id, "category_id": category_id}
            )
            annotation_id += 1
    return {"images": images, "categories": categories, "annotations": annotations}


def test_all_fixed_sets_have_expected_size_and_no_overlap():
    assert all(len(ids) == 20 for ids in test_sets.DOMAIN_SET_IDS.values())
    assert len(test_sets.MIXED_SCENE_IDS) == 20
    assert len(test_sets.SEGMENTATION_SET_IDS) == 20
    assert len(test_sets.REGION_CROP_IDS) == 20
    assert test_sets.find_duplicate_ids() == {}
    assert len(test_sets.ALL_TEST_IDS) == sum(
        len(ids) for ids in test_sets.DOMAIN_SET_IDS.values()
    ) + len(test_sets.MIXED_SCENE_IDS) + len(test_sets.SEGMENTATION_SET_IDS) + len(
        test_sets.REGION_CROP_IDS
    )


def test_coco_path_and_named_loader(tmp_path):
    expected = tmp_path / "val2017" / "000000000139.jpg"
    assert test_sets.get_coco_image_path(139, tmp_path) == str(expected)
    loaded = test_sets.load_test_set("domain_people", tmp_path)
    assert loaded[0] == (139, str(expected))

    with pytest.raises(ValueError, match="Unknown test set"):
        test_sets.load_test_set("not-a-set", tmp_path)


def test_validate_coco_ids_and_write_manifest(tmp_path):
    annotations_path = tmp_path / "instances_val2017.json"
    annotations_path.write_text(json.dumps(_annotation_fixture()), encoding="utf-8")
    manifest_path = tmp_path / "metadata" / "coco_id_manifest.json"

    report = test_sets.write_coco_metadata_manifest(annotations_path, manifest_path)

    assert report["valid"] is True
    assert report["missing_image_ids"] == []
    assert report["duplicate_ids"] == {}
    assert all(item["valid"] for item in report["domains"].values())
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["valid"] is True
    assert len(saved["images"]) == len(test_sets.ALL_TEST_IDS)


def test_validate_coco_ids_reports_missing_and_wrong_domain(tmp_path):
    fixture = _annotation_fixture()
    fixture["images"] = [
        image for image in fixture["images"] if image["id"] != test_sets.DOMAIN_SET_IDS["people"][0]
    ]
    fixture["annotations"] = [
        annotation
        for annotation in fixture["annotations"]
        if annotation["image_id"] != test_sets.DOMAIN_SET_IDS["animals"][0]
    ]
    annotations_path = tmp_path / "instances.json"
    annotations_path.write_text(json.dumps(fixture), encoding="utf-8")

    report = test_sets.validate_coco_ids(annotations_path)

    assert report["valid"] is False
    assert test_sets.DOMAIN_SET_IDS["people"][0] in report["missing_image_ids"]
    assert test_sets.DOMAIN_SET_IDS["people"][0] in report["domains"]["people"]["mismatches"]
    assert test_sets.DOMAIN_SET_IDS["animals"][0] in report["domains"]["animals"]["mismatches"]


def test_annotation_file_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="annotation file not found"):
        test_sets.validate_coco_ids(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid COCO annotation JSON"):
        test_sets.validate_coco_ids(malformed)


# ---------------------------------------------------------------------------
# Real-data guard: only runs when COCO_ANN env-var points at the real file.
# This check is what prevents bad ID lists from being committed again.
# On Kaggle: COCO_ANN=/kaggle/input/.../annotations/instances_val2017.json
# Locally:   COCO_ANN=/data/coco/annotations/instances_val2017.json
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("COCO_ANN"),
    reason=(
        "Set COCO_ANN=/path/to/instances_val2017.json to enable real-data checks. "
        "This guard prevents IDs that do not exist in val2017 from being committed."
    ),
)
def test_all_e3_ids_exist_in_val2017_and_no_duplicates():
    """Every fixed E3 ID must exist as a val2017 JPEG; no ID may appear in two sets."""
    ann_path = Path(os.environ["COCO_ANN"])
    coco_root = ann_path.parent.parent  # …/annotations/ → parent → coco root
    val2017 = coco_root / "val2017"

    all_e3_ids = [
        img_id
        for ids in test_sets.DOMAIN_SET_IDS.values()
        for img_id in ids
    ] + list(test_sets.MIXED_SCENE_IDS)

    missing = [
        img_id for img_id in all_e3_ids
        if not (val2017 / f"{img_id:012d}.jpg").exists()
    ]
    assert missing == [], (
        f"{len(missing)} E3 IDs are not in val2017: {missing[:10]}{'...' if len(missing) > 10 else ''}"
    )
    assert test_sets.find_duplicate_ids() == {}, "Duplicate IDs found across test sets"
