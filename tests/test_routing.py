"""Unit tests for routing. They use synthetic regions, so no GPU or model is needed."""
import numpy as np
import pytest

from colorize import DecompConfig, Region, route, resolve_overlaps, label_to_domain, build_label_table

H, W = 20, 30


def region(label, y0, y1, x0, x1, score=0.5):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return Region(label=label, score=score, box=np.array([x0, y0, x1, y1], float), mask=m)


@pytest.fixture
def table():
    return build_label_table(DecompConfig().prompts)


def test_exact_label_maps_to_domain(table):
    assert label_to_domain("dog", table) == "animals"
    assert label_to_domain("sky", table) == "nature"
    assert label_to_domain("building", table) == "urban"
    assert label_to_domain("person", table) == "people"


def test_merged_label_is_split_by_phrase_matching(table):
    assert label_to_domain("person face", table) == "people"


def test_unknown_label_returns_none(table):
    assert label_to_domain("spaceship", table) is None


def test_unknown_label_is_routed_to_fallback():
    d = route([region("spaceship", 0, 10, 0, 10)], (H, W))
    assert d.regions[0].domain == "fallback"


def test_uncovered_pixels_go_to_fallback_and_coverage_is_correct():
    d = route([region("person", 0, 10, 0, 10)], (H, W))
    assert d.coverage == pytest.approx(100 / (H * W))
    assert d.domain_map[15, 25] == d.domains.index("fallback")
    assert d.domain_map[5, 5] == d.domains.index("people")


def test_every_pixel_gets_exactly_one_domain():
    regs = [region("person", 0, 12, 0, 12), region("sky", 5, 20, 5, 30), region("dog", 8, 14, 8, 14)]
    d = route(regs, (H, W))
    assert d.domain_map.min() >= 0
    assert d.domain_map.max() < len(d.domains)


def test_domain_share_sums_to_one():
    d = route([region("person", 0, 10, 0, 10), region("sky", 10, 20, 0, 30)], (H, W))
    assert sum(d.domain_share.values()) == pytest.approx(1.0)


def test_smaller_region_wins_overlap_by_default():
    big = region("grass", 0, 20, 0, 30, score=0.9)     # nature, large, high score
    small = region("dog", 5, 10, 5, 10, score=0.4)     # animals, small, low score
    d = route([big, small], (H, W))
    assert d.domain_map[7, 7] == d.domains.index("animals")
    assert d.domain_map[15, 25] == d.domains.index("nature")


def test_score_rule_lets_higher_score_win():
    big = region("grass", 0, 20, 0, 30, score=0.9)
    small = region("dog", 5, 10, 5, 10, score=0.4)
    owner = resolve_overlaps([big, small], (H, W), rule="score")
    assert owner[7, 7] == 0   # the high-scoring big region owns the overlap


def test_resolved_masks_are_disjoint_and_cover_the_union():
    regs = [region("person", 0, 12, 0, 12), region("sky", 5, 20, 5, 30)]
    d = route(regs, (H, W))
    total = sum(d.resolved_mask(i).sum() for i in range(len(regs)))
    union = (regs[0].mask | regs[1].mask).sum()
    assert total == union


def test_no_regions_means_everything_is_fallback():
    d = route([], (H, W))
    assert d.coverage == 0.0
    assert d.domain_share["fallback"] == pytest.approx(1.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        route([region("person", 0, 5, 0, 5)], (H + 1, W))


def test_unknown_overlap_rule_is_rejected():
    with pytest.raises(ValueError):
        resolve_overlaps([region("person", 0, 5, 0, 5)], (H, W), rule="nonsense")
