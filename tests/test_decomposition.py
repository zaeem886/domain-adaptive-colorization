"""Unit tests for decomposition: filtering, merging, resizing, routing output,
and the end-to-end stage with stub DINO/SAM models.

Everything is synthetic (32x32 masks, in-memory images). No torch, no GPU, no
checkpoint download and no network access is needed: the model-facing calls are
overridden with stand-ins.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from colorize.config import DecompConfig
from colorize.decomposition import GroundedSAM, load_gray
from colorize.filtering import filter_detections, merge_same_label
from colorize.regions import Region
from colorize.routing import route

H, W = 32, 32


def make_region(label, y0, y1, x0, x1, score=0.5):
    """Build a Region with a filled rectangular mask on the 32x32 canvas."""
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    box = np.array([x0, y0, x1, y1], dtype=float)
    return Region(label=label, score=score, box=box, mask=mask)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_low_score_detections_are_removed():
    """Behaviour (a): detections below min_score are removed."""
    boxes = np.array([[0, 0, 4, 4], [8, 8, 12, 12], [16, 16, 20, 20]], dtype=float)
    scores = np.array([0.90, 0.34, 0.35])
    labels = ["person", "sky", "tree"]

    keep = filter_detections(boxes, scores, labels, min_score=0.35, iou_thresh=0.5)

    assert keep == [0, 2]


def test_nms_suppresses_lower_scoring_duplicate_of_same_label():
    """Behaviour (b): per-class NMS removes the lower-scoring duplicate."""
    boxes = np.array([[0, 0, 8, 8], [0, 0, 8, 8]], dtype=float)
    scores = np.array([0.9, 0.6])
    labels = ["person", "person"]

    assert filter_detections(boxes, scores, labels) == [0]


def test_nms_keeps_overlapping_boxes_with_different_labels():
    """Behaviour (c): NMS never suppresses overlapping boxes of different labels."""
    boxes = np.array([[0, 0, 8, 8], [0, 0, 8, 8]], dtype=float)
    scores = np.array([0.9, 0.6])
    labels = ["person", "sky"]

    assert filter_detections(boxes, scores, labels) == [0, 1]


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def test_same_label_fragments_merge_when_containment_is_high():
    """Behaviour (d): same-label fragments merge when containment exceeds the threshold."""
    big = make_region("person", 0, 8, 0, 8, score=0.8)
    small = make_region("person", 2, 6, 2, 6, score=0.4)

    merged = merge_same_label([big, small], containment_thresh=0.6)

    assert len(merged) == 1
    np.testing.assert_array_equal(merged[0].mask, big.mask)
    assert merged[0].score == pytest.approx(0.8)


def test_same_label_fragments_below_threshold_stay_separate():
    """Behaviour (e): fragments that do not exceed the threshold stay separate."""
    first = make_region("person", 0, 4, 0, 4)
    second = make_region("person", 3, 7, 3, 7)  # containment = 1 / 16 = 0.0625

    merged = merge_same_label([first, second], containment_thresh=0.6)

    assert len(merged) == 2


def test_containment_equal_to_threshold_does_not_merge():
    """Behaviour (d2): pins the strict-inequality rule: containment must exceed the threshold."""
    group = make_region("person", 0, 4, 0, 2)       # 8 pixels
    candidate = make_region("person", 2, 4, 1, 3)   # 4 pixels, 2 of them inside group
    # containment = |intersection| / |candidate| = 2 / 4 = 0.5 exactly.

    merged = merge_same_label([group, candidate], containment_thresh=0.5)

    assert len(merged) == 2


def test_different_labels_never_merge():
    """Behaviour (f): fragments with different labels never merge."""
    person = make_region("person", 2, 6, 2, 6, score=0.8)
    sky = make_region("sky", 2, 6, 2, 6, score=0.7)

    merged = merge_same_label([person, sky], containment_thresh=0.6)

    assert len(merged) == 2
    assert {r.label for r in merged} == {"person", "sky"}


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def test_load_gray_downscales_to_max_side_keeping_aspect_ratio():
    """Behaviour (g): Lanczos downscale to max_side keeps the aspect ratio."""
    array = (np.arange(2000 * 1000, dtype=np.uint32) % 251).astype(np.uint8)
    image = Image.fromarray(array.reshape(1000, 2000), mode="L")

    out = load_gray(image, max_side=500)

    assert out.size == (500, 250)
    assert out.width / out.height == pytest.approx(2.0)
    expected = image.resize((500, 250), Image.LANCZOS)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


def test_load_gray_leaves_small_images_untouched():
    """Behaviour (h): an image already smaller than max_side is not resized."""
    array = (np.arange(300 * 200, dtype=np.uint32) % 251).astype(np.uint8)
    image = Image.fromarray(array.reshape(200, 300), mode="L")

    out = load_gray(image, max_side=1024)

    assert out.size == (300, 200)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(image))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_merged_label_is_discarded_by_route():
    """Behaviour (j): a merged label such as 'person face' is not routed and is discarded."""
    region = make_region("person face", 0, 8, 0, 8, score=0.9)

    d = route([region], (H, W), DecompConfig())

    assert d.regions == []
    assert [r.label for r in d.discarded] == ["person face"]
    assert d.coverage == 0.0
    assert (d.domain_map == d.domains.index("fallback")).all()


def test_equal_area_tie_goes_to_higher_score():
    """Behaviour (l): for equal-area regions the higher-scoring region wins the pixel."""
    low = make_region("person", 0, 8, 0, 8, score=0.4)    # 64 pixels
    high = make_region("sky", 0, 8, 4, 12, score=0.9)     # 64 pixels

    d = route([low, high], (H, W), DecompConfig())

    assert d.regions[0].area == d.regions[1].area == 64
    assert d.owner[2, 6] == 1
    assert d.domain_map[2, 6] == d.domains.index("nature")
    assert d.domain_map[2, 1] == d.domains.index("people")


def test_discarded_region_does_not_steal_pixels_from_routable_region():
    """Behaviour (m): a discarded region never takes pixels from a routable one."""
    person = make_region("person", 0, 8, 0, 8, score=0.9)
    junk = make_region("spaceship", 0, 8, 0, 8, score=0.99)

    d = route([person, junk], (H, W), DecompConfig())

    assert len(d.regions) == 1
    assert [r.label for r in d.discarded] == ["spaceship"]
    np.testing.assert_array_equal(d.owner >= 0, person.mask)
    assert d.coverage == pytest.approx(person.area / (H * W))
    assert np.all(d.domain_map[d.owner >= 0] == d.domains.index("people"))


# ---------------------------------------------------------------------------
# Decomposition output
# ---------------------------------------------------------------------------

def test_decomposition_summary_fields_are_consistent():
    """Behaviour (o): owner, domain_map, coverage, domain_share and discarded agree."""
    regions = [
        make_region("person", 0, 10, 0, 10, score=0.9),
        make_region("sky", 5, 20, 5, 25, score=0.8),
        make_region("building", 12, 18, 12, 18, score=0.7),
        make_region("spaceship", 0, 5, 0, 5, score=0.99),
    ]

    d = route(regions, (H, W), DecompConfig())
    total = H * W

    assert len(d.regions) == 3
    assert [r.label for r in d.discarded] == ["spaceship"]
    assert d.coverage == pytest.approx((d.owner >= 0).sum() / total)
    assert sum(d.domain_share.values()) == pytest.approx(1.0)
    for index, domain in enumerate(d.domains):
        assert d.domain_share[domain] == pytest.approx(
            (d.domain_map == index).sum() / total
        )

    np.testing.assert_array_equal(
        d.domain_map == d.domains.index("fallback"), d.owner == -1
    )
    for i, region in enumerate(d.regions):
        assert np.all(d.domain_map[d.owner == i] == d.domains.index(region.domain))

    disjoint = sum(d.resolved_mask(i).sum() for i in range(len(d.regions)))
    union = np.logical_or.reduce([r.mask for r in d.regions]).sum()
    assert disjoint == union


# ---------------------------------------------------------------------------
# Full stage with stub models
# ---------------------------------------------------------------------------

class StubGroundedSAM(GroundedSAM):
    """GroundedSAM whose model calls are replaced by fixed detections and box masks."""

    def _run_dino(self, rgb):
        """Return two non-overlapping fixed detections."""
        boxes = np.array([[0, 0, 16, 32], [16, 0, 32, 32]], dtype=float)
        scores = np.array([0.9, 0.8], dtype=float)
        return boxes, scores, ["person", "sky"]

    def _run_sam(self, rgb_array, boxes):
        """Return a rectangular mask for each requested box."""
        height, width = rgb_array.shape[:2]
        masks = []
        for x0, y0, x1, y1 in boxes:
            mask = np.zeros((height, width), dtype=bool)
            mask[int(y0):int(y1), int(x0):int(x1)] = True
            masks.append(mask)
        return masks


def test_grounded_sam_end_to_end_with_stub_models():
    """Behaviour (p): the full decomposer runs with stub DINO/SAM and yields the expected domain map."""
    stub = StubGroundedSAM(DecompConfig())
    image = Image.new("L", (W, H), color=128)

    d = stub(image)

    assert stub._loaded is False  # the lazy model loaders were never touched
    assert d.shape == (H, W)
    assert [r.label for r in d.regions] == ["person", "sky"]
    assert [r.domain for r in d.regions] == ["people", "nature"]
    assert d.coverage == pytest.approx(1.0)
    assert (d.domain_map[:, :16] == d.domains.index("people")).all()
    assert (d.domain_map[:, 16:] == d.domains.index("nature")).all()
    assert (d.owner[:, :16] == 0).all()
    assert (d.owner[:, 16:] == 1).all()
