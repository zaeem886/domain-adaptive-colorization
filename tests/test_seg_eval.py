"""Tests for E1/E2 evaluation, class mapping, decomposition caching, and loading.

Everything here is synthetic: no COCO-Stuff download, no torch, no SAM, and no
GPU are required.  The model-facing ``CachedDecomposer`` test uses a stub
``GroundedSAM`` that returns fixed detections and box masks, so it can check
both numerical equivalence and the cache call counts.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from colorize.cocostuff import choose_label_offset, load_label_names
from colorize.config import DecompConfig
from colorize.decomposition import GroundedSAM
from colorize.regions import Decomposition, Region, load_decomposition, save_decomposition
from colorize.seg_eval import (
    IGNORE,
    UNCOVERED,
    accumulate_group_confusion,
    aggregate_domain_confusion,
    coverage_stats,
    domain_ious,
    dominant_gt_class,
    evaluate_e1_image,
    evaluate_region_routing,
    group_ious,
    load_class_mapping,
    mean_iou,
    pixel_accuracy,
    predicted_group_map,
    summarize_e1,
    summarize_routing,
)
from experiments.e1_e2 import CachedDecomposer

H, W = 2, 4
IGNORE_VALUE = 255


# ---------------------------------------------------------------------------
# Synthetic class mapping
# ---------------------------------------------------------------------------

# Index 0 is the conventional unlabeled entry.  Pixel values map to names with
# offset +1 (the Caffe-compatible COCO-Stuff convention).
LABEL_NAMES = [
    "unlabeled",
    "person",
    "sky-other",
    "tree",
    "building-other",
    "cat",
    "car",
]

# Pixel values after applying offset +1.
PIXEL_PERSON = 0
PIXEL_SKY = 1
PIXEL_TREE = 2
PIXEL_BUILDING = 3
PIXEL_CAT = 4
PIXEL_CAR = 5
PIXEL_IGNORE = 255

GROUPS = {
    "person": {"domain": "people", "classes": ["person"]},
    "sky": {"domain": "nature", "classes": ["sky-other"]},
    "vegetation": {"domain": "nature", "classes": ["tree"]},
    "building": {"domain": "urban", "classes": ["building-other"]},
    "animal": {"domain": "animals", "classes": ["cat"]},
}
PROMPT_TO_GROUP = {
    "person": "person",
    "face": "person",
    "sky": "sky",
    "vegetation": "vegetation",
    "tree": "vegetation",
    "building": "building",
    "animal": "animal",
}


def write_class_config(tmp_path, groups=None, prompt_to_group=None):
    """Write a class-group JSON file and return its path."""
    path = tmp_path / "class_groups.json"
    path.write_text(
        json.dumps(
            {
                "groups": GROUPS if groups is None else groups,
                "prompt_to_group": (
                    PROMPT_TO_GROUP if prompt_to_group is None else prompt_to_group
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def mapping(tmp_path):
    """Validated mapping for the synthetic groups above."""
    return load_class_mapping(
        write_class_config(tmp_path),
        LABEL_NAMES,
        DecompConfig().prompts,
        label_offset=1,
        ignore_value=IGNORE_VALUE,
    )


def make_region(label, domain=None, y0=0, x0=0, y1=H, x1=W, score=0.5):
    """Build a synthetic region whose mask is a filled rectangle."""
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    box = np.array([x0, y0, x1, y1], dtype=float)
    return Region(label=label, score=score, box=box, mask=mask, domain=domain)


def make_decomposition(owner, regions=(), discarded=(), domains=None):
    """Build a Decomposition with a plausible fallback-filled domain map."""
    owner = np.asarray(owner, dtype=np.int32)
    domains = tuple(domains) if domains is not None else DecompConfig().domain_order
    fallback_index = domains.index("fallback") if "fallback" in domains else len(domains) - 1
    domain_map = np.full(owner.shape, fallback_index, dtype=np.int8)
    regions = list(regions)
    for index, region in enumerate(regions):
        if region.domain in domains:
            domain_map[owner == index] = domains.index(region.domain)
    covered = int((owner >= 0).sum())
    total = int(owner.size)
    share = {
        name: float((domain_map == index).sum()) / total if total else float("nan")
        for index, name in enumerate(domains)
    }
    return Decomposition(
        shape=tuple(int(v) for v in owner.shape),
        regions=regions,
        domains=domains,
        owner=owner,
        domain_map=domain_map,
        coverage=float(covered / total) if total else float("nan"),
        domain_share=share,
        discarded=list(discarded),
    )


# ---------------------------------------------------------------------------
# Label-name loading and offset selection
# ---------------------------------------------------------------------------

def test_load_label_names_accepts_numbered_and_plain_files(tmp_path):
    """Load label names accepts numbered and plain files."""
    numbered = tmp_path / "numbered.txt"
    numbered.write_text("0: unlabeled\n1: person\n\n# comment\n2: sky-other\n")
    assert load_label_names(numbered) == ["unlabeled", "person", "sky-other"]

    plain = tmp_path / "plain.txt"
    plain.write_text("unlabeled\nperson\nsky-other\n")
    assert load_label_names(plain) == ["unlabeled", "person", "sky-other"]


def test_choose_label_offset_prefers_caffe_shift_but_reports_ambiguity():
    """Choose label offset prefers caffe shift but reports ambiguity."""
    names = ["unlabeled"] + [f"class-{index}" for index in range(1, 183)]
    values = np.array([0, 1, 181, IGNORE_VALUE], dtype=np.uint8)
    offset, reason, candidates = choose_label_offset(values, names, "auto")
    assert offset == 1
    assert 0 in candidates and 1 in candidates
    assert "unlabeled" in reason


def test_choose_label_offset_rejects_forced_out_of_range_offset():
    """Choose label offset rejects forced out of range offset."""
    names = ["unlabeled"] + [f"class-{index}" for index in range(1, 183)]
    values = np.array([181], dtype=np.uint8)
    with pytest.raises(ValueError, match="outside"):
        choose_label_offset(values, names, mode=2)


# ---------------------------------------------------------------------------
# Class mapping validation
# ---------------------------------------------------------------------------

def test_class_mapping_encodes_names_and_groups(mapping):
    """Class mapping encodes names and groups."""
    assert mapping.group_names == ("person", "sky", "vegetation", "building", "animal")
    assert mapping.domain_order == ("people", "nature", "urban", "animals")
    assert mapping.class_name(PIXEL_PERSON) == "person"
    assert mapping.class_name(PIXEL_CAR) == "car"

    label_map = np.array(
        [[PIXEL_PERSON, PIXEL_SKY, PIXEL_TREE, PIXEL_BUILDING],
         [PIXEL_CAT, PIXEL_CAR, PIXEL_IGNORE, 254]],
        dtype=np.int32,
    )
    encoded = mapping.encode_label_map(label_map)
    assert encoded.tolist() == [[0, 1, 2, 3], [4, IGNORE, IGNORE, IGNORE]]


def test_class_mapping_rejects_missing_class(tmp_path):
    """Class mapping rejects missing class."""
    groups = dict(GROUPS)
    groups["animal"] = {"domain": "animals", "classes": ["cat", "unicorn"]}
    with pytest.raises(ValueError, match="missing from the label-name list"):
        load_class_mapping(
            write_class_config(tmp_path, groups=groups),
            LABEL_NAMES,
            DecompConfig().prompts,
            label_offset=1,
        )


def test_class_mapping_rejects_duplicate_class(tmp_path):
    """Class mapping rejects duplicate class."""
    groups = dict(GROUPS)
    groups["animal"] = {"domain": "animals", "classes": ["cat", "person"]}
    with pytest.raises(ValueError, match="more than one group"):
        load_class_mapping(
            write_class_config(tmp_path, groups=groups),
            LABEL_NAMES,
            DecompConfig().prompts,
            label_offset=1,
        )


def test_class_mapping_rejects_unknown_domain(tmp_path):
    """Class mapping rejects unknown domain."""
    groups = dict(GROUPS)
    groups["person"] = {"domain": "robots", "classes": ["person"]}
    with pytest.raises(ValueError, match="unknown domains"):
        load_class_mapping(
            write_class_config(tmp_path, groups=groups),
            LABEL_NAMES,
            DecompConfig().prompts,
            label_offset=1,
        )


def test_class_mapping_rejects_missing_prompt(tmp_path):
    """Class mapping rejects missing prompt."""
    prompts = dict(PROMPT_TO_GROUP)
    prompts.pop("face")
    with pytest.raises(ValueError, match="prompts without a group"):
        load_class_mapping(
            write_class_config(tmp_path, prompt_to_group=prompts),
            LABEL_NAMES,
            DecompConfig().prompts,
            label_offset=1,
        )


def test_class_mapping_rejects_prompt_pointing_at_unknown_group(tmp_path):
    """Class mapping rejects prompt pointing at unknown group."""
    prompts = dict(PROMPT_TO_GROUP)
    prompts["car"] = "vehicle"
    with pytest.raises(ValueError, match="unknown groups"):
        load_class_mapping(
            write_class_config(tmp_path, prompt_to_group=prompts),
            LABEL_NAMES,
            DecompConfig().prompts,
            label_offset=1,
        )


def test_class_mapping_rejects_prompt_group_domain_mismatch(tmp_path):
    # "tree" is a nature prompt, but the vegetation class is in a nature group;
    # pointing the prompt at the urban building group must fail.
    """Class mapping rejects prompt group domain mismatch."""
    prompts = dict(PROMPT_TO_GROUP)
    prompts["tree"] = "building"
    with pytest.raises(ValueError, match="domain mismatch"):
        load_class_mapping(
            write_class_config(tmp_path, prompt_to_group=prompts),
            LABEL_NAMES,
            DecompConfig().prompts,
            label_offset=1,
        )


def test_class_mapping_rejects_offset_that_hits_ignore_value(tmp_path):
    # A class at name index 255 would map to pixel 255 with offset 0.
    """Class mapping rejects offset that hits ignore value."""
    names = [f"class-{index}" for index in range(256)]
    names[0] = "unlabeled"
    names[255] = "void-class"
    groups = {"person": {"domain": "people", "classes": ["void-class"]}}
    prompts = {"person": "person"}
    with pytest.raises(ValueError, match="ignore value"):
        load_class_mapping(
            write_class_config(tmp_path, groups=groups, prompt_to_group=prompts),
            names,
            {"people": ["person"]},
            label_offset=0,
        )


# ---------------------------------------------------------------------------
# E1 confusion matrix, IoU, and domain aggregation
# ---------------------------------------------------------------------------

def test_confusion_excludes_ignore_and_counts_uncovered_as_false_negative():
    """Confusion excludes ignore and counts uncovered as false negative."""
    ground_truth = np.array(
        [[0, 0, 0, IGNORE],
         [1, 1, 1, 1]],
        dtype=np.int16,
    )
    predicted = np.array(
        [[0, 1, UNCOVERED, 0],
         [1, 1, UNCOVERED, IGNORE]],
        dtype=np.int16,
    )
    confusion = accumulate_group_confusion(predicted, ground_truth, n_groups=2)
    expected = np.array(
        [[1, 1, 1],
         [0, 2, 2]],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(confusion, expected)
    # The ignored GT pixel must not contribute; the other seven do.
    assert confusion.sum() == 7


def test_confusion_rejects_out_of_range_ground_truth():
    """Confusion rejects out of range ground truth."""
    with pytest.raises(ValueError, match="outside the group range"):
        accumulate_group_confusion(
            np.array([[0]], dtype=np.int16),
            np.array([[3]], dtype=np.int16),
            n_groups=2,
        )


def test_group_iou_and_miou_use_uncovered_only_as_false_negative():
    """Group iou and miou use uncovered only as false negative."""
    confusion = np.array(
        [[8, 1, 1],
         [0, 6, 4]],
        dtype=np.int64,
    )
    ious = group_ious(confusion)
    np.testing.assert_allclose(ious, [0.8, 6.0 / 11.0])
    assert mean_iou(ious) == pytest.approx((0.8 + 6.0 / 11.0) / 2.0)
    assert pixel_accuracy(confusion) == pytest.approx(14.0 / 20.0)


def test_group_iou_is_nan_for_absent_group_and_miou_ignores_it():
    """Group iou is nan for absent group and miou ignores it."""
    confusion = np.zeros((2, 3), dtype=np.int64)
    ious = group_ious(confusion)
    assert np.all(np.isnan(ious))
    assert np.isnan(mean_iou(ious))

    confusion[0, 0] = 5
    ious = group_ious(confusion)
    assert ious[0] == 1.0
    assert np.isnan(ious[1])
    assert mean_iou(ious) == 1.0


def test_domain_iou_merges_groups_from_the_same_domain(mapping):
    # GT sky (group 1) predicted as vegetation (group 2) is wrong at group
    # level but correct after merging the nature domain.
    """Domain iou merges groups from the same domain."""
    confusion = np.zeros((5, 6), dtype=np.int64)
    confusion[1, 2] = 4
    ious = domain_ious(confusion, mapping)
    assert ious["nature"] == 1.0
    assert np.isnan(ious["people"])
    assert np.isnan(ious["urban"])
    assert np.isnan(ious["animals"])


def test_aggregate_domain_confusion_keeps_uncovered_column(mapping):
    """Aggregate domain confusion keeps uncovered column."""
    confusion = np.zeros((5, 6), dtype=np.int64)
    confusion[1, 2] = 4      # GT sky -> predicted vegetation, both nature
    confusion[0, 4] = 3      # GT person -> predicted animal
    confusion[3, 5] = 2      # GT building -> uncovered

    merged, rows, columns = aggregate_domain_confusion(confusion, mapping)
    assert rows == ("people", "nature", "urban", "animals")
    assert columns == ("people", "nature", "urban", "animals", "uncovered")
    people, nature, urban, animals = range(4)
    assert merged[people, animals] == 3
    assert merged[nature, nature] == 4
    assert merged[urban, 4] == 2
    assert merged.sum() == confusion.sum()

    without, _, columns_without = aggregate_domain_confusion(
        confusion, mapping, include_uncovered=False
    )
    assert columns_without == ("people", "nature", "urban", "animals")
    assert without.shape == (4, 4)
    assert without.sum() == confusion.sum() - 2


# ---------------------------------------------------------------------------
# E1 image evaluation and summary
# ---------------------------------------------------------------------------

def _two_region_decomposition():
    """Build a two-region synthetic decomposition."""
    owner = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    regions = [
        make_region("person", domain="people", y0=0, y1=1),
        make_region("tree", domain="nature", y0=1, y1=2),
    ]
    discarded = [make_region("car", score=float("nan"))]
    return make_decomposition(owner, regions=regions, discarded=discarded)


def test_evaluate_e1_image_reports_group_accuracy_but_domain_correctness(mapping):
    """Evaluate e1 image reports group accuracy but domain correctness."""
    decomposition = _two_region_decomposition()
    label_map = np.array(
        [[PIXEL_PERSON] * 4,
         [PIXEL_SKY] * 4],
        dtype=np.int32,
    )
    result = evaluate_e1_image(decomposition, label_map, mapping)

    assert result.total_pixels == 8
    assert result.covered_pixels == 8
    assert result.labeled_pixels == 8
    assert result.correct_pixels == 4
    assert result.accuracy == pytest.approx(0.5)
    assert result.coverage == 1.0
    assert result.fallback_fraction == 0.0
    assert result.n_regions == 2
    assert result.n_discarded == 1
    np.testing.assert_array_equal(
        result.confusion,
        np.array([[4, 0, 0, 0, 0, 0],
                  [0, 0, 4, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]], dtype=np.int64),
    )

    summary = summarize_e1([result], mapping)
    assert summary["n_images"] == 1
    # Group-level: person is correct, sky/vegetation are both wrong.
    assert summary["miou"] == pytest.approx(1.0 / 3.0)
    # Domain-level: sky predicted as vegetation is still nature.
    assert summary["domain_miou"] == pytest.approx(1.0)
    assert summary["coverage_mean"] == 1.0
    assert summary["coverage_pooled"] == 1.0
    assert summary["fallback_fraction"] == 0.0
    assert summary["fallback_images"] == 0
    assert summary["mean_regions"] == 2.0
    assert summary["mean_discarded"] == 1.0


def test_evaluate_e1_image_requires_resized_label_map(mapping):
    """Evaluate e1 image requires resized label map."""
    decomposition = _two_region_decomposition()
    wrong_shape = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="resize with NEAREST"):
        evaluate_e1_image(decomposition, wrong_shape, mapping)


def test_predicted_group_map_rejects_unknown_routed_label(mapping):
    """Predicted group map rejects unknown routed label."""
    owner = np.zeros((H, W), dtype=np.int32)
    decomposition = make_decomposition(
        owner, regions=[make_region("unicorn", domain="people")]
    )
    with pytest.raises(ValueError, match="has no group"):
        predicted_group_map(decomposition, mapping)


# ---------------------------------------------------------------------------
# Coverage / fallback statistics
# ---------------------------------------------------------------------------

def test_coverage_stats_pools_pixels_and_counts_fallback_images():
    """Coverage stats pools pixels and counts fallback images."""
    d1 = make_decomposition(
        np.array([[0, 0], [-1, -1]], dtype=np.int32),
        regions=[make_region("person", domain="people")],
    )
    d2 = make_decomposition(
        np.array([[0, 1], [1, 0]], dtype=np.int32),
        regions=[
            make_region("tree", domain="nature"),
            make_region("person", domain="people"),
        ],
        discarded=[make_region("car", score=float("nan"))],
    )
    stats = coverage_stats([d1, d2])
    assert stats.n_images == 2
    assert stats.total_pixels == 8
    assert stats.covered_pixels == 6
    assert stats.pooled_coverage == pytest.approx(0.75)
    assert stats.mean_coverage == pytest.approx(0.75)
    assert stats.fallback_fraction == pytest.approx(0.25)
    assert stats.fallback_images == 1
    assert stats.mean_regions == pytest.approx(1.5)
    assert stats.mean_discarded == pytest.approx(0.5)


def test_coverage_stats_rejects_empty_input():
    """Coverage stats rejects empty input."""
    with pytest.raises(ValueError):
        coverage_stats([])


# ---------------------------------------------------------------------------
# E2 dominant class and routing
# ---------------------------------------------------------------------------

def test_dominant_gt_class_breaks_ties_by_lowest_class_id():
    """Dominant gt class breaks ties by lowest class id."""
    label_map = np.array([[PIXEL_CAT, PIXEL_PERSON, PIXEL_IGNORE]])
    mask = np.ones(label_map.shape, dtype=bool)
    assert dominant_gt_class(label_map, mask, IGNORE_VALUE) == PIXEL_PERSON


def test_dominant_gt_class_ignores_void_and_unmasked_pixels():
    """Dominant gt class ignores void and unmasked pixels."""
    label_map = np.array([[PIXEL_IGNORE, PIXEL_IGNORE, PIXEL_CAT]])
    mask = np.array([[True, True, False]])
    assert dominant_gt_class(label_map, mask, IGNORE_VALUE) is None
    mask[0, 2] = True
    assert dominant_gt_class(label_map, mask, IGNORE_VALUE) == PIXEL_CAT


def _routing_decomposition():
    # One region per row: three included, one known-class exclusion, one void.
    """Build a decomposition covering included and excluded regions."""
    owner = np.repeat(np.arange(5, dtype=np.int32)[:, None], 4, axis=1)
    regions = [
        make_region("person", domain="people", y0=0, y1=1),
        make_region("tree", domain="nature", y0=1, y1=2),
        make_region("building", domain="urban", y0=2, y1=3),
        make_region("sky", domain="nature", y0=3, y1=4),
        make_region("sky", domain="nature", y0=4, y1=5),
    ]
    return make_decomposition(
        owner,
        regions=regions,
        discarded=[make_region("car", score=float("nan"))],
    )


def test_evaluate_region_routing_orientation_and_exclusions(mapping):
    """Evaluate region routing orientation and exclusions."""
    decomposition = _routing_decomposition()
    label_map = np.array(
        [[PIXEL_PERSON] * 4,
         [PIXEL_SKY] * 4,
         [PIXEL_CAT] * 4,
         [PIXEL_CAR] * 4,
         [PIXEL_IGNORE] * 4],
        dtype=np.int32,
    )
    report = evaluate_region_routing(decomposition, label_map, mapping, "img-1")

    assert report.n_regions == 5
    assert report.n_included == 3
    assert report.n_excluded == 2
    assert report.n_no_label == 1
    assert report.n_discarded == 1

    rows = report.rows
    assert [row.assigned_domain for row in rows] == [
        "people", "nature", "urban", "nature", "nature"
    ]
    assert [row.dominant_gt_class for row in rows] == [
        "person", "sky-other", "cat", "car", ""
    ]
    assert [row.dominant_gt_domain for row in rows] == [
        "people", "nature", "animals", "", ""
    ]
    assert [row.correct for row in rows] == [True, True, False, False, False]
    assert [row.excluded for row in rows] == [False, False, False, True, True]

    # Rows are GT domains; columns are assigned domains.
    people, nature, urban, animals = range(4)
    assert report.confusion[people, people] == 1
    assert report.confusion[nature, nature] == 1
    assert report.confusion[animals, urban] == 1
    assert report.confusion.sum() == 3


def test_summarize_routing_accuracy_and_per_domain(mapping):
    """Summarize routing accuracy and per domain."""
    report = evaluate_region_routing(
        _routing_decomposition(),
        np.array(
            [[PIXEL_PERSON] * 4,
             [PIXEL_SKY] * 4,
             [PIXEL_CAT] * 4,
             [PIXEL_CAR] * 4,
             [PIXEL_IGNORE] * 4],
            dtype=np.int32,
        ),
        mapping,
        "img-1",
    )
    summary = summarize_routing([report])
    assert summary["accuracy"] == pytest.approx(2.0 / 3.0)
    assert summary["n_included"] == 3
    assert summary["n_excluded"] == 2
    assert summary["per_domain_accuracy"]["people"] == 1.0
    assert summary["per_domain_accuracy"]["nature"] == 1.0
    assert summary["per_domain_accuracy"]["animals"] == 0.0
    assert np.isnan(summary["per_domain_accuracy"]["urban"])
    assert summary["confusion"].sum() == 3


def test_evaluate_region_routing_rejects_raw_masks(mapping):
    """Evaluate region routing rejects raw masks."""
    with pytest.raises(ValueError, match="only --region-mask resolved"):
        evaluate_region_routing(
            _routing_decomposition(),
            np.zeros((5, 4), dtype=np.int32),
            mapping,
            "img-1",
            region_mask="raw",
        )


# ---------------------------------------------------------------------------
# Decomposition save/load contract
# ---------------------------------------------------------------------------

def test_save_and_load_decomposition_preserves_evaluation_fields(tmp_path):
    """Save and load decomposition preserves evaluation fields."""
    owner = np.array([[0, 0], [1, -1]], dtype=np.int32)
    regions = [
        make_region("person", domain="people", y0=0, x0=0, y1=1, x1=2, score=0.9),
        make_region("tree", domain="nature", y0=1, x0=0, y1=2, x1=1, score=0.8),
    ]
    discarded = [make_region("car", score=float("nan"))]
    original = make_decomposition(owner, regions=regions, discarded=discarded)

    path = tmp_path / "decomposition.npz"
    save_decomposition(original, str(path))
    loaded = load_decomposition(str(path))

    assert loaded.shape == original.shape
    assert loaded.domains == original.domains
    np.testing.assert_array_equal(loaded.owner, original.owner)
    np.testing.assert_array_equal(loaded.domain_map, original.domain_map)
    assert [r.label for r in loaded.regions] == ["person", "tree"]
    assert [r.domain for r in loaded.regions] == ["people", "nature"]
    assert [r.score for r in loaded.regions] == pytest.approx([0.9, 0.8])
    # Raw SAM masks are not stored; the resolved mask is reconstructed.
    for index, region in enumerate(loaded.regions):
        np.testing.assert_array_equal(region.mask, original.owner == index)
    assert loaded.coverage == pytest.approx(0.75)
    assert [r.label for r in loaded.discarded] == ["car"]
    assert len(loaded.discarded) == 1
    assert loaded.domain_share["fallback"] == pytest.approx(0.25)


def test_load_decomposition_tolerates_older_files_without_discarded_metadata(tmp_path):
    """Load decomposition tolerates older files without discarded metadata."""
    path = tmp_path / "old.npz"
    owner = np.array([[0, -1]], dtype=np.int32)
    np.savez_compressed(
        path,
        owner=owner,
        domain_map=np.zeros_like(owner, dtype=np.int8),
        domains=np.array(["people", "fallback"], dtype=str),
        labels=np.array(["person"], dtype=str),
        region_domains=np.array(["people"], dtype=str),
        scores=np.array([0.5]),
        boxes=np.array([[0.0, 0.0, 1.0, 1.0]]),
    )
    loaded = load_decomposition(str(path))
    assert loaded.discarded == []
    assert loaded.regions[0].label == "person"
    assert loaded.coverage == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CachedDecomposer equivalence and call-count reduction
# ---------------------------------------------------------------------------

class StubGroundedSAM(GroundedSAM):
    """GroundedSAM with deterministic model calls and call counters."""

    def __init__(self, cfg):
        """Initialize the stub with call counters."""
        super().__init__(cfg)
        self.dino_calls = 0
        self.sam_calls = 0
        self.sam_boxes = 0

    def _run_dino(self, rgb):
        """Return fixed DINO detections and increment the call counter."""
        self.dino_calls += 1
        boxes = np.array(
            [[0, 0, 10, 10], [5, 5, 20, 20], [12, 2, 18, 9]], dtype=float
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=float)
        return boxes, scores, ["person", "person", "sky"]

    def _run_sam(self, rgb_array, boxes):
        """Return box-shaped masks and increment the call counter."""
        self.sam_calls += 1
        height, width = rgb_array.shape[:2]
        masks = []
        for box in boxes:
            x0, y0, x1, y1 = (int(round(float(value))) for value in box)
            mask = np.zeros((height, width), dtype=bool)
            mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
            masks.append(mask)
        self.sam_boxes += len(boxes)
        return masks


def _write_test_image(tmp_path):
    """Write a deterministic 30x24 grayscale PNG and return its path."""
    array = np.arange(30 * 24, dtype=np.uint8).reshape(24, 30) * 3
    path = tmp_path / "image.png"
    Image.fromarray(array, mode="L").save(path)
    return path


def _assert_same_decomposition(expected, actual):
    """Assert that two decompositions match field by field."""
    assert expected.shape == actual.shape
    assert expected.domains == actual.domains
    np.testing.assert_array_equal(expected.owner, actual.owner)
    np.testing.assert_array_equal(expected.domain_map, actual.domain_map)
    assert expected.coverage == pytest.approx(actual.coverage)
    assert [r.label for r in expected.regions] == [r.label for r in actual.regions]
    assert [r.domain for r in expected.regions] == [r.domain for r in actual.regions]
    assert [r.score for r in expected.regions] == pytest.approx(
        [r.score for r in actual.regions]
    )
    for expected_region, actual_region in zip(expected.regions, actual.regions):
        np.testing.assert_array_equal(expected_region.box, actual_region.box)
        np.testing.assert_array_equal(expected_region.mask, actual_region.mask)
    assert [r.label for r in expected.discarded] == [r.label for r in actual.discarded]


def test_cached_decomposer_matches_grounded_sam(tmp_path):
    """Cached decomposer matches grounded sam."""
    image_path = _write_test_image(tmp_path)

    reference = StubGroundedSAM(DecompConfig())
    expected = reference(str(image_path))

    stub = StubGroundedSAM(DecompConfig())
    cached = CachedDecomposer(stub, tmp_path / "cache")
    actual = cached.decompose(str(image_path))

    _assert_same_decomposition(expected, actual)
    assert stub.dino_calls == 1
    assert stub.sam_calls == 1
    assert stub.sam_boxes == 3

    # A second call must be served entirely from the in-memory/disk cache.
    cached.decompose(str(image_path))
    assert stub.dino_calls == 1
    assert stub.sam_calls == 1


def test_cached_decomposer_reuses_model_calls_across_parameter_changes(tmp_path):
    """Cached decomposer reuses model calls across parameter changes."""
    image_path = _write_test_image(tmp_path)
    stub = StubGroundedSAM(DecompConfig())
    cached = CachedDecomposer(stub, tmp_path / "cache")

    cached.decompose(str(image_path))
    assert (stub.dino_calls, stub.sam_calls) == (1, 1)

    # Post-DINO filtering and merging parameters do not touch the models.
    stub.cfg.min_score = 0.85
    stub.cfg.nms_iou = 0.2
    stub.cfg.merge_containment = 0.9
    cached.decompose(str(image_path))
    assert (stub.dino_calls, stub.sam_calls) == (1, 1)

    # New DINO thresholds require another DINO call, but all box masks are
    # already cached, so SAM must not run again.
    stub.cfg.box_threshold = 0.40
    stub.cfg.text_threshold = 0.30
    cached.decompose(str(image_path))
    assert (stub.dino_calls, stub.sam_calls) == (2, 1)
    assert stub.sam_boxes == 3


def test_cached_decomposer_reuses_disk_cache_in_a_new_instance(tmp_path):
    """Cached decomposer reuses disk cache in a new instance."""
    image_path = _write_test_image(tmp_path)
    cache_dir = tmp_path / "cache"

    first = StubGroundedSAM(DecompConfig())
    CachedDecomposer(first, cache_dir).decompose(str(image_path))
    assert (first.dino_calls, first.sam_calls) == (1, 1)

    second = StubGroundedSAM(DecompConfig())
    CachedDecomposer(second, cache_dir).decompose(str(image_path))
    assert (second.dino_calls, second.sam_calls) == (0, 0)
