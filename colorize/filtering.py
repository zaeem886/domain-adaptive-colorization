"""Detection filtering and fragment merging (report section 5.2.3, steps 4 and 6).
Ported from the Grounded-SAM notebook. Pure numpy, so it is unit-testable."""
from typing import List, Sequence
import numpy as np

from .regions import Region


def box_iou(b1, b2) -> float:
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def filter_detections(boxes: np.ndarray, scores: np.ndarray, labels: Sequence[str],
                      min_score: float = 0.35, iou_thresh: float = 0.5) -> List[int]:
    """Drop low scores, then per-class NMS. Returns the indices to keep.

    Boxes are visited by decreasing score; a later box with the SAME label whose
    IoU with a kept box exceeds `iou_thresh` is removed. Boxes with different
    labels never suppress each other.
    """
    order = np.argsort(-np.asarray(scores))
    used = np.zeros(len(boxes), dtype=bool)
    keep: List[int] = []
    for i in order:
        if used[i] or scores[i] < min_score:
            continue
        keep.append(int(i))
        used[i] = True
        for j in order:
            if used[j] or labels[j] != labels[i]:
                continue
            if box_iou(boxes[i], boxes[j]) > iou_thresh:
                used[j] = True
    return keep


def merge_same_label(regions: Sequence[Region], containment_thresh: float = 0.6) -> List[Region]:
    """Merge fragments of the same object.

    Masks are visited by decreasing area. Another unmerged mask with the same label
    is absorbed when containment = |intersection| / |candidate| exceeds the threshold.
    The group keeps the highest score and its box is recomputed from the merged mask.
    Two separate instances (two people) overlap little, so they stay separate.
    """
    merged: List[Region] = []
    used = [False] * len(regions)
    order = sorted(range(len(regions)), key=lambda i: regions[i].area, reverse=True)
    for i in order:
        if used[i]:
            continue
        r = regions[i]
        group_mask = r.mask.copy()
        group_score = r.score
        used[i] = True
        for j in order:
            if used[j] or regions[j].label != r.label:
                continue
            cand = regions[j]
            cand_area = cand.area
            containment = np.logical_and(group_mask, cand.mask).sum() / cand_area if cand_area > 0 else 0.0
            if containment > containment_thresh:
                group_mask = np.logical_or(group_mask, cand.mask)
                group_score = max(group_score, cand.score)
                used[j] = True
        ys, xs = np.where(group_mask)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float) if len(xs) else r.box
        merged.append(Region(label=r.label, score=float(group_score), box=box, mask=group_mask))
    return merged
