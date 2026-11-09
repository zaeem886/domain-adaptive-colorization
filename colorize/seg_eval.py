"""Segmentation and routing evaluation for experiments E1 and E2.

Why this module exists separately from :mod:`colorize.metrics`
--------------------------------------------------------------
``metrics.py`` contains general-purpose image/segmentation metrics.  E1 and E2
need two specific, dataset-level protocols:

* E1 accumulates a **GT-group by (predicted-group + UNCOVERED)** confusion
  matrix over all pixels, excluding IGNORE pixels.  IoU is then computed per
  prompt group and per domain, and coverage/fallback statistics are summarised.
* E2 judges each **routed region** by the dominant labelled GT class inside its
  resolved mask, maps that class to a domain, and compares it with the domain
  routing assigned.  Regions whose dominant class has no domain are excluded.

Those semantics do not match ``compute_miou`` / ``compute_confusion_matrix`` in
``metrics.py``, so this module implements them without modifying Person C's
code.  Everything here is pure NumPy and JSON: no torch, no models, no GPU.

Encoding conventions
--------------------
* Groups and domains are represented by small non-negative integer indices.
* ``IGNORE`` (-1) means the pixel has no scoring class (unlabeled / unmapped).
* ``UNCOVERED`` (-2) means no routed region owns the pixel.
* E1 confusion rows are GT groups; columns are predicted groups followed by one
  ``UNCOVERED`` column.  UNCOVERED therefore contributes false negatives only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .config import DOMAINS

IGNORE = -1
UNCOVERED = -2


# ---------------------------------------------------------------------------
# Editable class/group/domain mapping
# ---------------------------------------------------------------------------

@dataclass
class ClassMapping:
    """Validated mapping from COCO-Stuff classes to E1 groups and E2 domains.

    Attributes
    ----------
    group_names:
        Ordered prompt-group names, e.g. ``("person", "animal", ...)``.
    group_domains:
        Domain of each group, aligned with ``group_names``.
    class_name_to_group:
        COCO-Stuff class name -> group index.  Classes not listed here are
        IGNORED by E1 and have no domain in E2.
    class_name_to_domain:
        COCO-Stuff class name -> domain, derived from the group table.
    prompt_to_group:
        Detector prompt label (lower-case) -> group index.
    label_names:
        The original COCO-Stuff name list, retained for reporting class names.
    label_offset:
        ``name_index = pixel_value + label_offset`` for label-map values.
    ignore_value:
        Pixel value reserved for unlabeled/void (normally 255).
    pixel_to_group:
        256-entry lookup table used to encode label maps quickly.
    """

    group_names: Tuple[str, ...]
    group_domains: Tuple[str, ...]
    class_name_to_group: Dict[str, int]
    class_name_to_domain: Dict[str, str]
    prompt_to_group: Dict[str, int]
    label_names: Tuple[str, ...]
    label_offset: int
    ignore_value: int
    pixel_to_group: np.ndarray = field(repr=False, compare=False)

    @property
    def n_groups(self) -> int:
        """Number of prompt groups."""
        return len(self.group_names)

    @property
    def domain_order(self) -> Tuple[str, ...]:
        """Domains in first-appearance order, used by E2's confusion matrix."""
        seen: List[str] = []
        for domain in self.group_domains:
            if domain not in seen:
                seen.append(domain)
        return tuple(seen)

    def class_name(self, class_id: int) -> Optional[str]:
        """Return the COCO-Stuff name for a raw label-map value."""
        index = int(class_id) + self.label_offset
        if 0 <= index < len(self.label_names):
            return self.label_names[index]
        return None

    def encode_label_map(self, label_map: np.ndarray) -> np.ndarray:
        """Convert a raw label map to group indices with IGNORE for everything else."""
        array = np.asarray(label_map)
        out = np.full(array.shape, IGNORE, dtype=np.int16)
        valid = (array >= 0) & (array < len(self.pixel_to_group))
        out[valid] = self.pixel_to_group[array[valid]]
        return out


def load_class_mapping(
    path: Union[str, Path],
    label_names: Sequence[str],
    prompts: Mapping[str, Sequence[str]],
    label_offset: int,
    ignore_value: int = 255,
) -> ClassMapping:
    """Load and validate ``configs/class_groups.json``.

    Validation is intentionally strict because this file controls every E1/E2
    number.  It checks that:

    * every class listed in a group exists in the supplied COCO-Stuff names;
    * no class appears in two groups;
    * every group has one of the four pipeline domains;
    * every prompt in ``DecompConfig.prompts`` has a ``prompt_to_group`` entry;
    * every prompt points at an existing group;
    * the prompt's routing domain agrees with the mapped group's domain.

    A single error lists all missing names/prompts so they can be fixed in one
    edit.  The name list is usually ``labels.txt``; the offset maps PNG values
    to indices in it (see :mod:`colorize.cocostuff`).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"class-group config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid class-group JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"class-group JSON must be an object: {path}")

    groups = data.get("groups")
    prompt_map_raw = data.get("prompt_to_group")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("class-group config must contain a non-empty 'groups' object")
    if not isinstance(prompt_map_raw, dict):
        raise ValueError("class-group config must contain a 'prompt_to_group' object")

    name_to_index: Dict[str, int] = {}
    for index, name in enumerate(label_names):
        name_to_index.setdefault(str(name), index)

    group_names: List[str] = []
    group_domains: List[str] = []
    class_to_group: Dict[str, int] = {}
    duplicate_classes: List[str] = []
    missing_classes: List[str] = []
    bad_domains: List[str] = []

    for group_name, spec in groups.items():
        if not isinstance(spec, dict):
            raise ValueError(f"group {group_name!r} must be an object")
        domain = spec.get("domain")
        classes = spec.get("classes")
        if domain not in DOMAINS:
            bad_domains.append(f"{group_name}: {domain!r}")
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"group {group_name!r} must list at least one class")
        group_index = len(group_names)
        group_names.append(str(group_name))
        group_domains.append(str(domain))
        for class_name in classes:
            class_name = str(class_name)
            if class_name not in name_to_index:
                missing_classes.append(class_name)
                continue
            if class_name in class_to_group:
                duplicate_classes.append(class_name)
                continue
            class_to_group[class_name] = group_index

    prompt_to_group: Dict[str, int] = {}
    bad_prompt_groups: List[str] = []
    for prompt, group_name in prompt_map_raw.items():
        key = str(prompt).strip().lower()
        if group_name not in group_names:
            bad_prompt_groups.append(f"{key} -> {group_name!r}")
            continue
        prompt_to_group[key] = group_names.index(group_name)

    all_prompts = [
        str(prompt).strip().lower()
        for phrases in prompts.values()
        for prompt in phrases
    ]
    missing_prompts = sorted({p for p in all_prompts if p not in prompt_to_group})

    # E1 groups must agree with the domain that routing will assign to the same
    # prompt; otherwise the same region is scored as one domain and routed as
    # another.
    prompt_domains = {
        str(prompt).strip().lower(): str(domain)
        for domain, phrases in prompts.items()
        for prompt in phrases
    }
    bad_prompt_domains: List[str] = []
    for prompt, group_index in prompt_to_group.items():
        expected = prompt_domains.get(prompt)
        if expected is not None and group_domains[group_index] != expected:
            bad_prompt_domains.append(
                f"{prompt!r}: routing domain {expected!r} != "
                f"group {group_names[group_index]!r} domain {group_domains[group_index]!r}"
            )

    problems: List[str] = []
    if missing_classes:
        problems.append(f"classes missing from the label-name list: {sorted(set(missing_classes))}")
    if duplicate_classes:
        problems.append(f"classes assigned to more than one group: {sorted(set(duplicate_classes))}")
    if bad_domains:
        problems.append(f"unknown domains: {bad_domains}")
    if bad_prompt_groups:
        problems.append(f"prompts point at unknown groups: {bad_prompt_groups}")
    if missing_prompts:
        problems.append(f"prompts without a group in prompt_to_group: {missing_prompts}")
    if bad_prompt_domains:
        problems.append(f"prompt/group domain mismatch: {bad_prompt_domains}")
    if problems:
        raise ValueError("invalid class-group config:\n  - " + "\n  - ".join(problems))

    # Build a direct pixel-value -> group lookup.  Values whose index falls
    # outside the name list stay IGNORE, which makes offset mistakes visible in
    # check-data rather than silently scoring them as a real class.
    lut = np.full(256, IGNORE, dtype=np.int16)
    for class_name, group_index in class_to_group.items():
        name_index = name_to_index[class_name]
        pixel_value = name_index - int(label_offset)
        if not (0 <= pixel_value < 256):
            raise ValueError(
                f"label offset {label_offset} maps class {class_name!r} to "
                f"pixel value {pixel_value}, outside 0..255"
            )
        if pixel_value == ignore_value:
            raise ValueError(
                f"class {class_name!r} maps to the ignore value {ignore_value}"
            )
        if lut[pixel_value] != IGNORE:
            raise ValueError(
                f"two classes map to the same pixel value {pixel_value}; "
                "check the label-name file and offset"
            )
        lut[pixel_value] = group_index
    lut[int(ignore_value)] = IGNORE

    class_to_domain = {
        name: group_domains[group_index]
        for name, group_index in class_to_group.items()
    }
    return ClassMapping(
        group_names=tuple(group_names),
        group_domains=tuple(group_domains),
        class_name_to_group=class_to_group,
        class_name_to_domain=class_to_domain,
        prompt_to_group=prompt_to_group,
        label_names=tuple(str(name) for name in label_names),
        label_offset=int(label_offset),
        ignore_value=int(ignore_value),
        pixel_to_group=lut,
    )


# ---------------------------------------------------------------------------
# E1: pixel-level group evaluation
# ---------------------------------------------------------------------------

def predicted_group_map(decomposition, mapping: ClassMapping) -> np.ndarray:
    """Build a per-pixel predicted group map from a decomposition.

    Each routed region gets the group of its detector label; pixels owned by no
    region become ``UNCOVERED``.  An unknown routed label is an error because
    the mapping validation guarantees that every configured prompt has a group.
    """
    shape = tuple(int(v) for v in decomposition.shape)
    owner = np.asarray(decomposition.owner)
    if owner.shape != shape:
        raise ValueError(f"owner shape {owner.shape} does not match decomposition shape {shape}")
    out = np.full(shape, UNCOVERED, dtype=np.int16)
    for index, region in enumerate(decomposition.regions):
        key = str(region.label).strip().lower()
        if key not in mapping.prompt_to_group:
            raise ValueError(
                f"routed region label {region.label!r} has no group in the class mapping"
            )
        out[owner == index] = mapping.prompt_to_group[key]
    return out


def accumulate_group_confusion(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Accumulate a ``(n_groups, n_groups + 1)`` pixel confusion matrix.

    Rows are GT groups.  The first ``n_groups`` columns are predicted groups;
    the last column is UNCOVERED.  Pixels with GT ``IGNORE`` are excluded from
    the matrix entirely.  An invalid or missing prediction is folded into the
    UNCOVERED column so it can never be counted as a true positive.
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"prediction shape {predicted.shape} does not match GT shape {ground_truth.shape}"
        )
    cm = np.zeros((n_groups, n_groups + 1), dtype=np.int64)
    valid = ground_truth != IGNORE
    if not np.any(valid):
        return cm
    gt = ground_truth[valid].astype(np.int64, copy=False)
    pred = predicted[valid].astype(np.int64, copy=False)
    if np.any((gt < 0) | (gt >= n_groups)):
        raise ValueError("GT group map contains values outside the group range")
    pred = np.where((pred >= 0) & (pred < n_groups), pred, n_groups)
    np.add.at(cm, (gt, pred), 1)
    return cm


def group_ious(confusion: np.ndarray) -> np.ndarray:
    """Compute per-group IoU from an E1 confusion matrix.

    ``TP = cm[i, i]``, ``FP = predicted i from another GT group``, and
    ``FN = GT i predicted as another group or UNCOVERED``.  A group absent from
    both GT and predictions has a zero denominator and returns ``NaN``; callers
    should exclude it from the mean.
    """
    cm = np.asarray(confusion, dtype=np.int64)
    n_groups = cm.shape[0]
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0)[:n_groups] - tp
    fn = cm.sum(axis=1)[:n_groups] - tp
    denom = tp + fp + fn
    return np.divide(tp, denom, out=np.full(n_groups, np.nan), where=denom > 0)


def mean_iou(ious: Sequence[float]) -> float:
    """Mean of defined IoUs, ignoring ``NaN`` entries; ``NaN`` if none are defined."""
    values = np.asarray(list(ious), dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def pixel_accuracy(confusion: np.ndarray) -> float:
    """Fraction of non-ignored pixels on the confusion-matrix diagonal."""
    cm = np.asarray(confusion, dtype=np.int64)
    total = int(cm.sum())
    if total == 0:
        return float("nan")
    return float(np.trace(cm) / total)


def domain_ious(confusion: np.ndarray, mapping: ClassMapping) -> Dict[str, float]:
    """IoU per domain after merging all groups belonging to that domain.

    For nature this merges sky and vegetation before computing IoU, exactly as
    requested by the E1 protocol.  Domains with no GT and no prediction yield
    ``NaN`` and are excluded from ``mean_domain_iou``.
    """
    cm = np.asarray(confusion, dtype=np.int64)
    out: Dict[str, float] = {}
    for domain in mapping.domain_order:
        indices = [i for i, d in enumerate(mapping.group_domains) if d == domain]
        if not indices:
            out[domain] = float("nan")
            continue
        block = cm[np.ix_(indices, indices)]
        tp = float(block.sum())
        fp = float(cm[:, indices].sum() - tp)
        fn = float(cm[indices, :].sum() - tp)
        denom = tp + fp + fn
        out[domain] = float(tp / denom) if denom > 0 else float("nan")
    return out


def aggregate_domain_confusion(
    confusion: np.ndarray,
    mapping: ClassMapping,
    include_uncovered: bool = True,
) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """Merge an E1 group matrix into a domain-by-domain matrix.

    Returns ``(matrix, row_names, column_names)``.  With
    ``include_uncovered=True`` the last column is ``"uncovered"``; it is kept
    separate rather than being merged into a fallback domain so the figure shows
    the same false-negative-only convention as E1.
    """
    cm = np.asarray(confusion, dtype=np.int64)
    domains = mapping.domain_order
    domain_index = {name: i for i, name in enumerate(domains)}
    n_cols = len(domains) + (1 if include_uncovered else 0)
    out = np.zeros((len(domains), n_cols), dtype=np.int64)
    for gt_group in range(cm.shape[0]):
        gt_domain = domain_index[mapping.group_domains[gt_group]]
        for pred_group in range(cm.shape[1] - 1):
            pred_domain = domain_index[mapping.group_domains[pred_group]]
            out[gt_domain, pred_domain] += cm[gt_group, pred_group]
        if include_uncovered:
            out[gt_domain, len(domains)] += cm[gt_group, cm.shape[1] - 1]
    columns = tuple(domains) + (("uncovered",) if include_uncovered else ())
    return out, tuple(domains), columns


@dataclass
class CoverageStats:
    """Dataset-level coverage/fallback statistics for a list of decompositions."""

    n_images: int
    total_pixels: int
    covered_pixels: int
    pooled_coverage: float
    mean_coverage: float
    fallback_fraction: float
    fallback_images: int
    mean_regions: float
    mean_discarded: float


def coverage_stats(decompositions: Sequence[object]) -> CoverageStats:
    """Summarise coverage, fallback usage, and region counts across images.

    ``pooled_coverage`` weights every pixel equally, while ``mean_coverage`` is
    the average of per-image fractions (so a small image counts as much as a
    large one).  ``fallback_images`` counts images with at least one uncovered
    pixel, i.e. images where the fallback expert would run.
    """
    decompositions = list(decompositions)
    if not decompositions:
        raise ValueError("cannot compute coverage statistics without decompositions")
    total = 0
    covered = 0
    coverages: List[float] = []
    fallback_images = 0
    n_regions = 0
    n_discarded = 0
    for decomposition in decompositions:
        owner = np.asarray(decomposition.owner)
        image_total = int(owner.size)
        image_covered = int((owner >= 0).sum())
        total += image_total
        covered += image_covered
        coverages.append(float(image_covered / image_total) if image_total else float("nan"))
        if image_covered < image_total:
            fallback_images += 1
        n_regions += len(decomposition.regions)
        n_discarded += len(decomposition.discarded)
    pooled = float(covered / total) if total else float("nan")
    return CoverageStats(
        n_images=len(decompositions),
        total_pixels=total,
        covered_pixels=covered,
        pooled_coverage=pooled,
        mean_coverage=float(np.mean(coverages)),
        fallback_fraction=float(1.0 - pooled) if total else float("nan"),
        fallback_images=fallback_images,
        mean_regions=float(n_regions / len(decompositions)),
        mean_discarded=float(n_discarded / len(decompositions)),
    )


@dataclass
class E1ImageResult:
    """Per-image E1 counts and confusion matrix, before dataset aggregation."""

    confusion: np.ndarray
    total_pixels: int
    covered_pixels: int
    labeled_pixels: int
    correct_pixels: int
    n_regions: int
    n_discarded: int

    @property
    def coverage(self) -> float:
        """Fraction of pixels owned by a routed region."""
        return float(self.covered_pixels / self.total_pixels) if self.total_pixels else float("nan")

    @property
    def fallback_fraction(self) -> float:
        """Fraction of pixels that would use the fallback expert."""
        return 1.0 - self.coverage if self.total_pixels else float("nan")

    @property
    def accuracy(self) -> float:
        """Pixel accuracy over non-ignored pixels."""
        return float(self.correct_pixels / self.labeled_pixels) if self.labeled_pixels else float("nan")


def evaluate_e1_image(decomposition, label_map: np.ndarray, mapping: ClassMapping) -> E1ImageResult:
    """Evaluate one decomposition against a full-resolution GT label map.

    The caller must already have resized the label map to ``decomposition.shape``
    with nearest-neighbour (see :func:`colorize.cocostuff.resize_label_map`).
    This keeps the experiment code explicit about resolution and avoids hiding
    an accidental resize inside the metric.
    """
    gt_shape = tuple(int(v) for v in decomposition.shape)
    if tuple(label_map.shape) != gt_shape:
        raise ValueError(
            f"GT label map shape {tuple(label_map.shape)} does not match "
            f"decomposition shape {gt_shape}; resize with NEAREST first"
        )
    gt_groups = mapping.encode_label_map(label_map)
    pred_groups = predicted_group_map(decomposition, mapping)
    cm = accumulate_group_confusion(pred_groups, gt_groups, mapping.n_groups)
    labeled = gt_groups != IGNORE
    correct = int(((pred_groups == gt_groups) & labeled).sum())
    owner = np.asarray(decomposition.owner)
    return E1ImageResult(
        confusion=cm,
        total_pixels=int(owner.size),
        covered_pixels=int((owner >= 0).sum()),
        labeled_pixels=int(labeled.sum()),
        correct_pixels=correct,
        n_regions=len(decomposition.regions),
        n_discarded=len(decomposition.discarded),
    )


def summarize_e1(results: Sequence[E1ImageResult], mapping: ClassMapping) -> Dict[str, object]:
    """Aggregate per-image E1 results into the dataset-level summary.

    Coverage is reported both as the mean of per-image coverage and as a pooled
    pixel fraction.  The fallback fraction is the pooled uncovered fraction;
    ``fallback_images`` counts images with at least one uncovered pixel.
    """
    if not results:
        raise ValueError("cannot summarise E1 without any image results")
    confusion = np.sum([r.confusion for r in results], axis=0)
    ious = group_ious(confusion)
    dom_ious = domain_ious(confusion, mapping)
    total_pixels = sum(r.total_pixels for r in results)
    covered_pixels = sum(r.covered_pixels for r in results)
    labeled_pixels = sum(r.labeled_pixels for r in results)
    correct_pixels = sum(r.correct_pixels for r in results)
    return {
        "n_images": len(results),
        "group_names": list(mapping.group_names),
        "domain_names": list(mapping.domain_order),
        "confusion": confusion,
        "ious": {name: float(value) for name, value in zip(mapping.group_names, ious)},
        "miou": mean_iou(ious),
        "domain_ious": dom_ious,
        "domain_miou": mean_iou(list(dom_ious.values())),
        "pixel_accuracy": float(correct_pixels / labeled_pixels) if labeled_pixels else float("nan"),
        "coverage_mean": float(np.mean([r.coverage for r in results])),
        "coverage_pooled": float(covered_pixels / total_pixels) if total_pixels else float("nan"),
        "fallback_fraction": float((total_pixels - covered_pixels) / total_pixels) if total_pixels else float("nan"),
        "fallback_images": int(sum(1 for r in results if r.covered_pixels < r.total_pixels)),
        "mean_regions": float(np.mean([r.n_regions for r in results])),
        "mean_discarded": float(np.mean([r.n_discarded for r in results])),
        "total_pixels": int(total_pixels),
        "covered_pixels": int(covered_pixels),
        "labeled_pixels": int(labeled_pixels),
    }


# ---------------------------------------------------------------------------
# E2: region-level routing evaluation
# ---------------------------------------------------------------------------

def dominant_gt_class(
    label_map: np.ndarray,
    mask: np.ndarray,
    ignore_value: int = 255,
) -> Optional[int]:
    """Most frequent non-ignore class value inside *mask*.

    Ties are broken by the lowest class id, which makes repeated runs
    deterministic.  Returns ``None`` when the mask contains no labelled pixel.
    """
    values = np.asarray(label_map)[np.asarray(mask, dtype=bool)]
    values = values[values != ignore_value]
    if values.size == 0:
        return None
    counts = np.bincount(values.astype(np.int64))
    return int(np.argmax(counts))


@dataclass
class RegionRoutingRow:
    """One routed-region row written to ``results/e2/regions.csv``."""

    image_id: str
    region_index: int
    label: str
    score: float
    assigned_domain: str
    dominant_gt_class: str
    dominant_gt_domain: str
    correct: bool
    excluded: bool

    def as_csv_row(self) -> Dict[str, object]:
        """Return the exact column order requested by the E2 protocol."""
        return {
            "image_id": self.image_id,
            "region_index": self.region_index,
            "label": self.label,
            "score": self.score,
            "assigned_domain": self.assigned_domain,
            "dominant_gt_class": self.dominant_gt_class,
            "dominant_gt_domain": self.dominant_gt_domain,
            "correct": int(self.correct),
            "excluded": int(self.excluded),
        }


@dataclass
class RoutingReport:
    """Per-image E2 rows and counts."""

    image_id: str
    rows: List[RegionRoutingRow]
    confusion: np.ndarray
    domain_names: Tuple[str, ...]
    n_included: int
    n_excluded: int
    n_no_label: int
    n_discarded: int
    n_regions: int


def evaluate_region_routing(
    decomposition,
    label_map: np.ndarray,
    mapping: ClassMapping,
    image_id: str,
    region_mask: str = "resolved",
) -> RoutingReport:
    """Evaluate E2 routing for one cached decomposition.

    Region pixels are the resolved mask ``owner == i`` by default.  Raw SAM
    masks are not stored in the cache, so ``region_mask='raw'`` raises a clear
    error instead of silently using different pixels.

    For each region the dominant non-255 GT class is found.  Its domain (if any)
    is the GT domain.  Regions with no labelled pixel or with a class outside
    the configured groups are EXCLUDED from accuracy and counted separately.
    """
    if region_mask != "resolved":
        raise ValueError(
            "only --region-mask resolved is supported because cached "
            "decompositions do not store raw SAM masks"
        )
    gt_shape = tuple(int(v) for v in decomposition.shape)
    if tuple(label_map.shape) != gt_shape:
        raise ValueError(
            f"GT label map shape {tuple(label_map.shape)} does not match "
            f"decomposition shape {gt_shape}; resize with NEAREST first"
        )

    domain_names = mapping.domain_order
    domain_index = {name: i for i, name in enumerate(domain_names)}
    confusion = np.zeros((len(domain_names), len(domain_names)), dtype=np.int64)
    rows: List[RegionRoutingRow] = []
    n_included = n_excluded = n_no_label = 0

    for index, region in enumerate(decomposition.regions):
        mask = np.asarray(decomposition.owner) == index
        class_id = dominant_gt_class(label_map, mask, mapping.ignore_value)
        class_name = mapping.class_name(class_id) if class_id is not None else None
        gt_domain = mapping.class_name_to_domain.get(class_name or "")
        assigned = region.domain or ""
        excluded = gt_domain is None
        correct = bool((not excluded) and assigned == gt_domain)

        if excluded:
            n_excluded += 1
            if class_name is None:
                n_no_label += 1
        else:
            n_included += 1
            if gt_domain not in domain_index:
                raise ValueError(f"GT domain {gt_domain!r} is not in the mapping domain order")
            if assigned not in domain_index:
                raise ValueError(f"assigned domain {assigned!r} is not in the mapping domain order")
            confusion[domain_index[gt_domain], domain_index[assigned]] += 1

        rows.append(
            RegionRoutingRow(
                image_id=str(image_id),
                region_index=index,
                label=str(region.label),
                score=float(region.score),
                assigned_domain=assigned,
                dominant_gt_class=class_name or "",
                dominant_gt_domain=gt_domain or "",
                correct=correct,
                excluded=excluded,
            )
        )

    return RoutingReport(
        image_id=str(image_id),
        rows=rows,
        confusion=confusion,
        domain_names=domain_names,
        n_included=n_included,
        n_excluded=n_excluded,
        n_no_label=n_no_label,
        n_discarded=len(decomposition.discarded),
        n_regions=len(decomposition.regions),
    )


def summarize_routing(reports: Sequence[RoutingReport]) -> Dict[str, object]:
    """Aggregate per-image E2 reports into overall accuracy and confusion."""
    if not reports:
        raise ValueError("cannot summarise E2 without any image reports")
    domain_names = reports[0].domain_names
    for report in reports[1:]:
        if report.domain_names != domain_names:
            raise ValueError("routing reports use different domain orders")
    confusion = np.sum([r.confusion for r in reports], axis=0)
    n_included = sum(r.n_included for r in reports)
    n_excluded = sum(r.n_excluded for r in reports)
    n_no_label = sum(r.n_no_label for r in reports)
    n_discarded = sum(r.n_discarded for r in reports)
    n_regions = sum(r.n_regions for r in reports)
    per_domain: Dict[str, float] = {}
    for index, domain in enumerate(domain_names):
        row_total = int(confusion[index].sum())
        per_domain[domain] = (
            float(confusion[index, index] / row_total) if row_total else float("nan")
        )
    correct = int(np.trace(confusion))
    return {
        "n_images": len(reports),
        "n_regions": int(n_regions),
        "n_included": int(n_included),
        "n_excluded": int(n_excluded),
        "n_no_label": int(n_no_label),
        "n_discarded": int(n_discarded),
        "domain_names": list(domain_names),
        "confusion": confusion,
        "accuracy": float(correct / n_included) if n_included else float("nan"),
        "per_domain_accuracy": per_domain,
    }
