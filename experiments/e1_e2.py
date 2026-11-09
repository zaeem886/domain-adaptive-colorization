"""E1/E2 experiment runner: decomposition quality and routing accuracy.

This is the single command-line entry point required by the work plan:

    python experiments/e1_e2.py check-data ...
    python experiments/e1_e2.py make-lists ...
    python experiments/e1_e2.py tune ...
    python experiments/e1_e2.py test ...
    python experiments/e1_e2.py routing ...
    python experiments/e1_e2.py figures ...

The E1 and E2 computations are separate code paths and separate output folders
(``results/e1`` and ``results/e2``), but they intentionally share one runner:

* E2 reads the test decompositions cached by the E1 ``test`` command and makes
  no model calls.
* Both use the same COCO-Stuff paths, class mapping, label offset, ID lists,
  Grounded-SAM instance, and notes file.

``CachedDecomposer`` is the only model-facing wrapper here.  It reproduces
``GroundedSAM.__call__`` exactly while caching DINO outputs per
``(image, box_threshold, text_threshold)`` and SAM masks per ``(image, box)``.
That is what makes the one-pass threshold sweep practical: only the first
setting pays for detection/SAM; later settings re-run filtering, merging and
routing from the cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# Allow `python experiments/e1_e2.py ...` from the repository root.  Python
# normally puts the script's own directory (experiments/) on sys.path, not the
# current directory, so the package would not be importable without this.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colorize import viz
from colorize.cocostuff import (
    IGNORE_VALUE,
    choose_label_offset,
    colorize_label_map,
    describe_values,
    discover_ids,
    find_image_path,
    find_label_map_path,
    ids_in_label_map,
    load_label_map,
    load_label_names,
    resize_label_map,
)
from colorize.config import DecompConfig
from colorize.decomposition import GroundedSAM, load_gray
from colorize.filtering import filter_detections, merge_same_label
from colorize.regions import Region, load_decomposition, save_decomposition
from colorize.results import ResultsRecord, save_results
from colorize.routing import route
from colorize.seg_eval import (
    E1ImageResult,
    RoutingReport,
    aggregate_domain_confusion,
    evaluate_e1_image,
    evaluate_region_routing,
    load_class_mapping,
    summarize_e1,
    summarize_routing,
)

PARAMETER_ORDER = (
    "box_threshold",
    "text_threshold",
    "min_score",
    "nms_iou",
    "merge_containment",
)

DEFAULT_GRIDS: Dict[str, List[float]] = {
    "box_threshold": [0.25, 0.30, 0.35, 0.40, 0.45],
    "text_threshold": [0.20, 0.25, 0.30],
    "min_score": [0.30, 0.35, 0.45],
    "nms_iou": [0.4, 0.5, 0.6],
    "merge_containment": [0.5, 0.6, 0.7],
}

E1_METRICS = (
    "coverage",
    "fallback_fraction",
    "n_regions",
    "n_discarded",
    "pixel_accuracy",
)


# ---------------------------------------------------------------------------
# Small shared I/O helpers
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Convert NumPy scalars/arrays and NaN to strict-JSON-friendly values."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Union[str, Path], payload) -> None:
    """Write a JSON file with parent creation and NaN converted to ``null``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: Union[str, Path]):
    """Read a JSON file, returning ``None`` when it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_text(path: Union[str, Path], text: str) -> None:
    """Write UTF-8 text, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_rows(path: Union[str, Path], rows: Sequence[Mapping[str, object]],
                fieldnames: Sequence[str]) -> None:
    """Write a list of dicts to CSV with the requested column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_ids(path: Union[str, Path]) -> List[int]:
    """Read one decimal image ID per line; blank lines are ignored."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"image-ID list not found: {path}. Run the make-lists command first."
        )
    ids: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(int(line))
    if not ids:
        raise ValueError(f"image-ID list is empty: {path}")
    return ids


def _write_ids(path: Union[str, Path], ids: Sequence[int]) -> None:
    """Write plain decimal IDs, one per line, in the given order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(image_id)}\n" for image_id in ids), encoding="utf-8")


def _limit(ids: Sequence[int], args) -> List[int]:
    """Apply ``--limit`` for smoke runs without changing on-disk lists."""
    limit = getattr(args, "limit", None)
    if limit is None:
        return list(ids)
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    return list(ids)[:limit]


def _float_or_blank(value) -> object:
    """CSV-friendly float: blank for undefined values."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(value) else value


def _file_sha1(path: Union[str, Path]) -> str:
    """SHA1 of a file's bytes, used to invalidate evaluation caches."""
    path = Path(path)
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels_sha1(label_names: Sequence[str]) -> str:
    """Stable hash of the class-name list used to build class mappings."""
    return hashlib.sha1("\n".join(str(name) for name in label_names).encode("utf-8")).hexdigest()


def _settings_hash(settings: Mapping[str, float], extra: str = "") -> str:
    """Stable short hash for a parameter setting plus arbitrary metadata."""
    payload = json.dumps(
        {"settings": {k: float(settings[k]) for k in sorted(settings)}, "extra": extra},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _eval_cache_hash(
    settings: Mapping[str, float],
    args,
    offset: int,
    label_names: Sequence[str],
) -> str:
    """Hash every input that can change an E1 metric, not just thresholds.

    Paths and non-swept config values are included so a cache cannot silently
    survive a dataset move, a changed prompt set, or a different SAM checkpoint.
    """
    cfg = DecompConfig()
    sam_checkpoint = str(getattr(args, "sam_checkpoint", None) or cfg.sam_checkpoint)
    extra = "|".join(
        [
            f"offset={offset}",
            f"labels={_labels_sha1(label_names)}",
            f"groups={_file_sha1(args.class_groups)}",
            f"ignore={IGNORE_VALUE}",
            f"images={Path(args.images_root).resolve()}",
            f"labels_root={Path(args.labels_root).resolve()}",
            f"train_split={args.train_split}",
            f"val_split={args.val_split}",
            f"max_side={cfg.max_side}",
            f"prompts={json.dumps(cfg.prompts, sort_keys=True)}",
            f"overlap_rule={cfg.overlap_rule}",
            f"split_merged_labels={cfg.split_merged_labels}",
            f"fallback={cfg.fallback}",
            f"dino_id={cfg.dino_id}",
            f"sam_type={cfg.sam_type}",
            f"sam_checkpoint={Path(sam_checkpoint).resolve()}",
        ]
    )
    return _settings_hash(settings, extra)


def _default_params() -> Dict[str, float]:
    """Parameter values from the shared :class:`DecompConfig`."""
    cfg = DecompConfig()
    return {name: float(getattr(cfg, name)) for name in PARAMETER_ORDER}


def _apply_params(gsam: GroundedSAM, params: Mapping[str, float]) -> None:
    """Set the five swept decomposition parameters on an existing model wrapper."""
    for name, value in params.items():
        if name not in PARAMETER_ORDER:
            raise KeyError(f"unknown decomposition parameter: {name}")
        setattr(gsam.cfg, name, float(value))


def _build_gsam(args, params: Optional[Mapping[str, float]] = None) -> GroundedSAM:
    """Create one Grounded-SAM wrapper and apply optional selected parameters."""
    cfg = DecompConfig()
    if getattr(args, "sam_checkpoint", None):
        cfg.sam_checkpoint = str(args.sam_checkpoint)
    if params:
        _apply_params_from_cfg(cfg, params)
    return GroundedSAM(cfg, device=getattr(args, "device", None))


def _apply_params_from_cfg(cfg: DecompConfig, params: Mapping[str, float]) -> None:
    """Set swept parameters on a config object before model construction."""
    for name, value in params.items():
        if name not in PARAMETER_ORDER:
            raise KeyError(f"unknown decomposition parameter: {name}")
        setattr(cfg, name, float(value))


def _resolve_offset(
    args,
    label_names: Sequence[str],
    first_label_map: np.ndarray,
    cache_dir: Union[str, Path],
    preferred: Optional[int] = None,
) -> Tuple[int, str]:
    """Resolve and persist the COCO-Stuff pixel-to-name offset.

    Explicit ``--label-offset`` always wins.  In auto mode this reuses a cached
    decision when the class-name list is unchanged, which keeps tune/test/E2
    consistent.  A ``preferred`` offset from ``selected_params.json``/the E1
    summary is used next, and only then is the PNG range inspected.
    """
    cache_dir = Path(cache_dir)
    state_path = cache_dir / "label_offset.json"
    names_hash = _labels_sha1(label_names)
    mode = str(getattr(args, "label_offset", "auto"))

    if mode != "auto":
        offset, reason, candidates = choose_label_offset(first_label_map, label_names, mode)
        print(f"[offset] {reason}; candidates={candidates}")
        _write_json(
            state_path,
            {"label_offset": offset, "reason": reason, "candidates": candidates,
             "label_names_sha1": names_hash, "forced": True},
        )
        return offset, reason

    if preferred is not None:
        offset = int(preferred)
        bad = [
            int(v) for v in np.unique(first_label_map).tolist()
            if int(v) != IGNORE_VALUE and not (0 <= int(v) + offset < len(label_names))
        ]
        if bad:
            raise ValueError(
                f"preferred label offset {offset} does not fit observed values {bad[:5]}"
            )
        reason = "reused offset recorded by the E1 run"
        _write_json(
            state_path,
            {"label_offset": offset, "reason": reason, "candidates": [offset],
             "label_names_sha1": names_hash, "forced": False},
        )
        return offset, reason

    state = _read_json(state_path)
    if isinstance(state, dict) and state.get("label_names_sha1") == names_hash:
        offset = int(state["label_offset"])
        reason = str(state.get("reason", "cached offset decision"))
        print(f"[offset] reusing cached decision: {reason}")
        return offset, reason

    offset, reason, candidates = choose_label_offset(first_label_map, label_names, "auto")
    print(f"[offset] {reason}; candidates={candidates}")
    if len(candidates) > 1:
        print(
            "[offset] WARNING: the observed value range is ambiguous. "
            "Inspect results/check_data before trusting the chosen offset."
        )
    _write_json(
        state_path,
        {"label_offset": offset, "reason": reason, "candidates": candidates,
         "label_names_sha1": names_hash, "forced": False},
    )
    return offset, reason


def _preferred_offset_from_outputs(results_dir: Union[str, Path]) -> Optional[int]:
    """Read an offset recorded by a previous E1 run, if available."""
    results_dir = Path(results_dir)
    summary = _read_json(results_dir / "e1" / "summary.json")
    if isinstance(summary, dict) and "label_offset" in summary:
        try:
            return int(summary["label_offset"])
        except (TypeError, ValueError):
            return None
    selected = _read_json(results_dir / "e1" / "selected_params.json")
    if isinstance(selected, dict) and "label_offset" in selected:
        try:
            return int(selected["label_offset"])
        except (TypeError, ValueError):
            return None
    return None


def _image_and_label(
    args,
    split: str,
    image_id: int,
) -> Tuple[Path, Path]:
    """Find one image and its label map or raise a useful error."""
    image_path = find_image_path(args.images_root, split, image_id)
    label_path = find_label_map_path(args.labels_root, split, image_id)
    if image_path is None:
        raise FileNotFoundError(f"image {image_id} not found under {args.images_root!r}")
    if label_path is None:
        raise FileNotFoundError(f"label map {image_id} not found under {args.labels_root!r}")
    return image_path, label_path


def _load_gt(args, split: str, image_id: int, shape: Tuple[int, int]) -> np.ndarray:
    """Load and nearest-resize one GT label map to a decomposition shape."""
    _, label_path = _image_and_label(args, split, image_id)
    return resize_label_map(load_label_map(label_path), shape)


def _first_label_map(args, split: str, ids: Sequence[int]) -> np.ndarray:
    """Load the label map used to decide the auto offset."""
    if not ids:
        raise ValueError("cannot resolve a label offset without image IDs")
    _, label_path = _image_and_label(args, split, ids[0])
    return load_label_map(label_path)


# ---------------------------------------------------------------------------
# Cached decomposition
# ---------------------------------------------------------------------------

class CachedDecomposer:
    """Grounded-SAM wrapper that caches DINO outputs and SAM masks.

    The decomposition logic mirrors ``GroundedSAM.__call__`` exactly:
    grayscale load/resize -> DINO -> filter -> SAM -> same-label merge -> route.
    The only difference is that DINO results and box-prompt SAM masks are
    memoised in memory and (optionally) on disk.

    Cache keys
    ----------
    DINO:
        image content hash + DINO model id + caption + box/text thresholds.
    SAM:
        image content hash + SAM type/checkpoint + the box rounded to six
        decimals (sub-pixel precision, to avoid accidental cache collisions).

    Because ``min_score``, ``nms_iou`` and ``merge_containment`` do not affect
    DINO or SAM, sweeping those parameters only re-runs the cheap stages.
    """

    def __init__(
        self,
        gsam: GroundedSAM,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Wrap *gsam*; create ``<cache_dir>/dino`` and ``<cache_dir>/sam``."""
        self.gsam = gsam
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._dino_memory: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
        self._sam_memory: Dict[str, np.ndarray] = {}
        if self.cache_dir is not None:
            (self.cache_dir / "dino").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "sam").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _image_key(gray: Image.Image) -> str:
        """Content hash of the working grayscale image."""
        digest = hashlib.sha1()
        digest.update(str(gray.size).encode("ascii"))
        digest.update(gray.tobytes())
        return digest.hexdigest()

    def _path_for(self, kind: str, key: str) -> Optional[Path]:
        """Return the on-disk cache filename for a key, or ``None`` if disabled."""
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / kind / f"{digest}.npz"

    def _dino_key(self, image_key: str) -> str:
        """Build the full DINO cache key for the current config."""
        cfg = self.gsam.cfg
        return "|".join(
            [
                image_key,
                str(cfg.dino_id),
                self.gsam.caption,
                f"box={float(cfg.box_threshold):.6f}",
                f"text={float(cfg.text_threshold):.6f}",
            ]
        )

    def _sam_key(self, image_key: str, box: np.ndarray) -> str:
        """Build the full SAM cache key for one image and box prompt."""
        cfg = self.gsam.cfg
        rounded = ",".join(f"{float(v):.6f}" for v in np.asarray(box).ravel())
        return "|".join(
            [
                image_key,
                str(cfg.sam_type),
                str(Path(cfg.sam_checkpoint).resolve()),
                rounded,
            ]
        )

    def _load_dino(self, key: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Read a DINO result from memory/disk, rejecting stale entries."""
        if key in self._dino_memory:
            return self._dino_memory[key]
        path = self._path_for("dino", key)
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"]) != key:
                    return None
                result = (
                    np.asarray(data["boxes"], dtype=float),
                    np.asarray(data["scores"], dtype=float),
                    [str(v) for v in data["labels"].tolist()],
                )
        except (OSError, KeyError, ValueError):
            return None
        self._dino_memory[key] = result
        return result

    def _save_dino(self, key: str, result: Tuple[np.ndarray, np.ndarray, List[str]]) -> None:
        """Keep a DINO result in memory and, when enabled, on disk."""
        self._dino_memory[key] = result
        path = self._path_for("dino", key)
        if path is None:
            return
        boxes, scores, labels = result
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            key=np.array(key),
            boxes=np.asarray(boxes, dtype=float),
            scores=np.asarray(scores, dtype=float),
            labels=np.asarray(labels, dtype=str),
        )
        tmp.replace(path)

    def _load_sam(self, key: str) -> Optional[np.ndarray]:
        """Read a SAM mask from memory/disk, rejecting stale entries."""
        if key in self._sam_memory:
            return self._sam_memory[key]
        path = self._path_for("sam", key)
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"]) != key:
                    return None
                mask = np.asarray(data["mask"], dtype=bool)
        except (OSError, KeyError, ValueError):
            return None
        self._sam_memory[key] = mask
        return mask

    def _save_sam(self, key: str, mask: np.ndarray) -> None:
        """Keep a SAM mask in memory and, when enabled, on disk."""
        self._sam_memory[key] = mask
        path = self._path_for("sam", key)
        if path is None:
            return
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, key=np.array(key), mask=np.asarray(mask, dtype=bool))
        tmp.replace(path)

    def _get_dino(self, gray: Image.Image, image_key: str):
        """Return cached DINO outputs or run DINO under the current thresholds."""
        key = self._dino_key(image_key)
        cached = self._load_dino(key)
        if cached is not None:
            return cached

        # GroundedSAM._run_dino reads box/text thresholds from self.cfg; those
        # values are already part of the cache key, so no temporary mutation is
        # needed here.
        result = self.gsam._run_dino(gray.convert("RGB"))
        self._save_dino(key, result)
        return result

    def _get_masks(self, gray: Image.Image, image_key: str,
                   boxes: np.ndarray) -> List[np.ndarray]:
        """Return one SAM mask per box, computing only cache misses."""
        boxes = np.asarray(boxes, dtype=float)
        if len(boxes) == 0:
            return []
        keys = [self._sam_key(image_key, box) for box in boxes]
        masks: List[Optional[np.ndarray]] = [self._load_sam(key) for key in keys]
        missing = [i for i, mask in enumerate(masks) if mask is None]
        if missing:
            missing_boxes = boxes[missing]
            new_masks = self.gsam._run_sam(np.array(gray.convert("RGB")), missing_boxes)
            if len(new_masks) != len(missing):
                raise RuntimeError(
                    f"SAM returned {len(new_masks)} masks for {len(missing)} boxes"
                )
            for index, mask in zip(missing, new_masks):
                masks[index] = np.asarray(mask, dtype=bool)
                self._save_sam(keys[index], masks[index])
        return [np.asarray(mask, dtype=bool) for mask in masks]

    def decompose(self, source, image_key: Optional[str] = None):
        """Return the same :class:`Decomposition` as ``GroundedSAM.__call__``."""
        cfg = self.gsam.cfg
        gray = load_gray(source, cfg.max_side)
        image_key = image_key or self._image_key(gray)
        boxes, scores, labels = self._get_dino(gray, image_key)
        if len(boxes) == 0:
            return route([], (gray.height, gray.width), cfg)

        keep = filter_detections(boxes, scores, labels, cfg.min_score, cfg.nms_iou)
        if not keep:
            return route([], (gray.height, gray.width), cfg)

        masks = self._get_masks(gray, image_key, boxes[keep])
        regions = [
            Region(
                label=labels[i],
                score=float(scores[i]),
                box=np.asarray(boxes[i], dtype=float),
                mask=mask,
            )
            for i, mask in zip(keep, masks)
        ]
        merged = merge_same_label(regions, cfg.merge_containment)
        return route(merged, (gray.height, gray.width), cfg)

    def __call__(self, source, image_key: Optional[str] = None):
        """Alias for :meth:`decompose`."""
        return self.decompose(source, image_key=image_key)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def _write_notes(args) -> None:
    """Regenerate ``results/e1_e2_notes.md`` from files that actually exist.

    This records choices and whatever results have been produced so far.  It
    never fills in numbers that were not measured; missing stages are reported
    as not run yet.
    """
    results_dir = Path(args.results_dir)
    data_lists_dir = Path(args.data_lists_dir)
    lines = [
        "# E1/E2 experiment notes",
        "",
        "Generated by `experiments/e1_e2.py`. Values below are read from real",
        "run outputs; stages that have not run are marked accordingly.",
        "",
        "## Reproducibility choices",
        "",
        f"- Class mapping: `{args.class_groups}` (team must confirm it).",
        f"- Label names: `{args.label_names}`.",
        f"- Label offset mode: `{args.label_offset}`.",
        f"- Tuning IDs: `{args.tuning_ids}`.",
        f"- Test IDs: `{args.test_ids}`.",
        f"- Cache directory: `{args.cache_dir}`.",
        f"- Results directory: `{args.results_dir}`.",
        f"- COCO images root: `{args.images_root}`.",
        f"- COCO-Stuff label-map root: `{args.labels_root}`.",
        f"- Splits: tuning=`{args.train_split}`, test=`{args.val_split}`.",
        f"- Random seed: `{args.seed}`.",
        f"- `--limit`: `{getattr(args, 'limit', None)}`.",
        f"- Region mask mode for E2: `{getattr(args, 'region_mask', 'resolved')}`.",
        f"- Working resolution: longer image side capped at "
        f"`max_side={DecompConfig().max_side}`.",
        "",
    ]

    for label, path in (
        ("Tuning list", data_lists_dir / "e1_tuning_ids.txt"),
        ("Test list", data_lists_dir / "e1_test_ids.txt"),
    ):
        if path.is_file():
            count = len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
            lines.append(f"- {label}: {count} IDs in `{path}`.")
    lines.append("")

    offset_state = _read_json(Path(args.cache_dir) / "label_offset.json")
    if isinstance(offset_state, dict):
        lines.extend(
            [
                "## COCO-Stuff label offset",
                "",
                f"- Chosen offset: `{offset_state.get('label_offset')}`.",
                f"- Reason: {offset_state.get('reason', 'not recorded')}.",
                f"- Candidate offsets: `{offset_state.get('candidates')}`.",
                "",
            ]
        )

    selected = _read_json(results_dir / "e1" / "selected_params.json")
    if isinstance(selected, dict):
        lines.extend(
            [
                "## Selected decomposition parameters",
                "",
                "```json",
                json.dumps(_json_safe(selected), indent=2, sort_keys=True),
                "```",
                "",
            ]
        )

    summary = _read_json(results_dir / "e1" / "summary.json")
    if isinstance(summary, dict):
        lines.extend(
            [
                "## E1 test summary",
                "",
                f"- Images: {summary.get('n_images')}",
                f"- mIoU: {summary.get('miou')}",
                f"- Domain mIoU: {summary.get('domain_miou')}",
                f"- Coverage (pooled): {summary.get('coverage_pooled')}",
                f"- Fallback fraction (pooled): {summary.get('fallback_fraction')}",
                f"- Pixel accuracy: {summary.get('pixel_accuracy')}",
                f"- Used defaults: {summary.get('used_defaults')}",
                "",
            ]
        )

    routing = _read_json(results_dir / "e2" / "routing_summary.json")
    if isinstance(routing, dict):
        lines.extend(
            [
                "## E2 routing summary",
                "",
                f"- Images: {routing.get('n_images')}",
                f"- Regions evaluated: {routing.get('n_included')}",
                f"- Excluded regions: {routing.get('n_excluded')}",
                f"- Accuracy: {routing.get('accuracy')}",
                f"- Per-domain accuracy: `{routing.get('per_domain_accuracy')}`",
                "",
            ]
        )

    _write_text(results_dir / "e1_e2_notes.md", "\n".join(lines) + "\n")


def _update_manifest(path: Union[str, Path], image_id: int, settings: str) -> None:
    """Record which decomposition settings produced one cached test image."""
    path = Path(path)
    manifest = _read_json(path) or {}
    manifest[str(int(image_id))] = settings
    _write_json(path, manifest)


def _manifest_matches(path: Union[str, Path], image_id: int, settings: str) -> bool:
    """Return True when a cached decomposition matches current settings."""
    manifest = _read_json(path) or {}
    return manifest.get(str(int(image_id))) == settings


# ---------------------------------------------------------------------------
# check-data
# ---------------------------------------------------------------------------

def cmd_check_data(args) -> int:
    """Audit label-map values, offset, names, and save visual examples."""
    label_names = load_label_names(args.label_names)
    ids = _limit(discover_ids(args.images_root, args.labels_root, args.split), args)
    if not ids:
        raise SystemExit("no image IDs available for check-data")
    count = min(int(args.n_images), len(ids))
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / "check_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    first_id = ids[0]
    first_map = _first_label_map(args, args.split, [first_id])
    offset, reason = _resolve_offset(args, label_names, first_map, args.cache_dir)
    print(f"[check-data] offset={offset}: {reason}")

    report: List[str] = [
        f"label-names file: {args.label_names}",
        f"images root: {args.images_root}",
        f"labels root: {args.labels_root}",
        f"split: {args.split}",
        f"chosen offset: {offset}",
        f"reason: {reason}",
        "",
    ]
    state = _read_json(Path(args.cache_dir) / "label_offset.json")
    if isinstance(state, dict):
        report.append(f"candidate offsets: {state.get('candidates')}")
        report.append("")

    for image_id in ids[:count]:
        image_path, label_path = _image_and_label(args, args.split, image_id)
        label_map = load_label_map(label_path)
        values = np.unique(label_map)
        report.extend(
            [
                f"image {image_id}",
                f"  label map: {label_path}",
                f"  value range: {int(label_map.min())}..{int(label_map.max())}",
                f"  unique values ({len(values)}): {values[:40].tolist()}"
                + (" ..." if len(values) > 40 else ""),
                "  most common values:",
                *describe_values(label_map, label_names, offset, limit=20),
                "  ids in this map:",
                *[f"    {item}" for item in ids_in_label_map(label_map, label_names, offset)[:40]],
                "",
            ]
        )
        stem = f"{int(image_id):012d}"
        with Image.open(image_path) as image:
            image.convert("RGB").save(out_dir / f"{stem}_input.png")
        colorize_label_map(label_map).save(out_dir / f"{stem}_labels.png")

    _write_text(out_dir / "offset_report.txt", "\n".join(report) + "\n")
    _write_notes(args)
    print("\n".join(report))
    print(f"[check-data] wrote visual checks and report to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# make-lists
# ---------------------------------------------------------------------------

def _sample_or_reuse(
    path: Path,
    ids: Sequence[int],
    size: int,
    seed: int,
    force: bool,
    limit: Optional[int],
) -> List[int]:
    """Reuse an existing list unless forced; otherwise sample deterministically.

    ``--limit`` caps a newly written list (useful for smoke runs) but never
    truncates an existing list, so a previously fixed full list is preserved.
    """
    if path.is_file() and not force:
        chosen = _read_ids(path)
        print(f"[make-lists] reusing {path} ({len(chosen)} IDs)")
        return chosen
    target = min(int(size), len(ids))
    limited = False
    if limit is not None:
        limited = target > int(limit)
        target = min(target, int(limit))
    chosen = sorted(random.Random(seed).sample(list(ids), target))
    _write_ids(path, chosen)
    suffix = " (capped by --limit; regenerate with --force for a full list)" if limited else ""
    print(f"[make-lists] wrote {path} ({len(chosen)} IDs, seed={seed}){suffix}")
    return chosen


def cmd_make_lists(args) -> int:
    """Create/reuse fixed tuning and test ID lists and assert disjointness."""
    tuning_path = Path(args.data_lists_dir) / "e1_tuning_ids.txt"
    test_path = Path(args.data_lists_dir) / "e1_test_ids.txt"
    train_ids = discover_ids(args.images_root, args.labels_root, args.train_split)
    val_ids = discover_ids(args.images_root, args.labels_root, args.val_split)

    tuning = _sample_or_reuse(
        tuning_path, train_ids, args.tuning_size, args.seed, args.force, args.limit
    )
    test = _sample_or_reuse(
        test_path, val_ids, args.test_size, args.seed + 1, args.force, args.limit
    )
    overlap = sorted(set(tuning) & set(test))
    if overlap:
        raise SystemExit(
            "tuning and test lists overlap; refusing to continue: "
            f"{overlap[:10]}{' ...' if len(overlap) > 10 else ''}"
        )
    missing_train = sorted(set(tuning) - set(train_ids))
    missing_val = sorted(set(test) - set(val_ids))
    if missing_train or missing_val:
        print(
            "[make-lists] WARNING: existing lists contain IDs absent from the "
            f"official split (train missing: {missing_train[:5]}, "
            f"val missing: {missing_val[:5]})"
        )
    _write_notes(args)
    print(f"[make-lists] tuning={len(tuning)} test={len(test)} disjoint={not overlap}")
    return 0


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------

def _parse_grid(text: Optional[str], default: Sequence[float]) -> List[float]:
    """Parse a comma-separated CLI grid, falling back to the report default."""
    if text is None:
        return list(default)
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("a parameter grid must contain at least one value")
    return values


def _requested_grids(args) -> Dict[str, List[float]]:
    """Return the five grids, honouring optional CLI overrides."""
    return {
        "box_threshold": _parse_grid(args.box_threshold_grid, DEFAULT_GRIDS["box_threshold"]),
        "text_threshold": _parse_grid(args.text_threshold_grid, DEFAULT_GRIDS["text_threshold"]),
        "min_score": _parse_grid(args.min_score_grid, DEFAULT_GRIDS["min_score"]),
        "nms_iou": _parse_grid(args.nms_iou_grid, DEFAULT_GRIDS["nms_iou"]),
        "merge_containment": _parse_grid(
            args.merge_containment_grid, DEFAULT_GRIDS["merge_containment"]
        ),
    }


def _metric_cache_path(
    cache_dir: Union[str, Path], eval_hash: str, image_id: int
) -> Path:
    """Location of one small per-image E1 metric cache entry."""
    return Path(cache_dir) / "e1_tuning" / "sweep" / eval_hash / f"{int(image_id):012d}.npz"


def _load_cached_e1_image(path: Path) -> Optional[E1ImageResult]:
    """Load a cached per-image E1 result without re-running the model."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return E1ImageResult(
                confusion=np.asarray(data["confusion"], dtype=np.int64),
                total_pixels=int(data["total_pixels"]),
                covered_pixels=int(data["covered_pixels"]),
                labeled_pixels=int(data["labeled_pixels"]),
                correct_pixels=int(data["correct_pixels"]),
                n_regions=int(data["n_regions"]),
                n_discarded=int(data["n_discarded"]),
            )
    except (OSError, KeyError, ValueError):
        return None


def _save_cached_e1_image(path: Path, result: E1ImageResult) -> None:
    """Persist the small counts/matrix needed to resume tuning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        confusion=result.confusion,
        total_pixels=np.array(result.total_pixels),
        covered_pixels=np.array(result.covered_pixels),
        labeled_pixels=np.array(result.labeled_pixels),
        correct_pixels=np.array(result.correct_pixels),
        n_regions=np.array(result.n_regions),
        n_discarded=np.array(result.n_discarded),
    )
    tmp.replace(path)


def _evaluate_setting(
    args,
    decomposer: CachedDecomposer,
    settings: Mapping[str, float],
    ids: Sequence[int],
    split: str,
    mapping,
    offset: int,
    label_names: Sequence[str],
    gt_maps: Mapping[int, np.ndarray],
) -> Tuple[List[E1ImageResult], str]:
    """Evaluate one parameter setting on the tuning IDs using metric caching."""
    eval_hash = _eval_cache_hash(settings, args, offset, label_names)
    _apply_params(decomposer.gsam, settings)
    results: List[E1ImageResult] = []
    for image_id in ids:
        cache_path = _metric_cache_path(args.cache_dir, eval_hash, image_id)
        cached = _load_cached_e1_image(cache_path)
        if cached is not None:
            results.append(cached)
            continue
        image_path, _ = _image_and_label(args, split, image_id)
        decomposition = decomposer.decompose(image_path)
        gt_map = resize_label_map(gt_maps[image_id], decomposition.shape)
        result = evaluate_e1_image(decomposition, gt_map, mapping)
        _save_cached_e1_image(cache_path, result)
        results.append(result)
    return results, eval_hash


def cmd_tune(args) -> int:
    """One-pass coordinate sweep on the tuning split, then save final caches."""
    label_names = load_label_names(args.label_names)
    ids = _limit(_read_ids(args.tuning_ids), args)
    if not ids:
        raise SystemExit("tuning ID list is empty")
    # Guard against an accidental tuning/test overlap: reading the test *list*
    # is safe; no test image is opened by this command.
    test_list = Path(args.test_ids)
    if test_list.is_file():
        overlap = sorted(set(ids) & set(_read_ids(test_list)))
        if overlap:
            raise SystemExit(
                "tuning and test ID lists overlap; refusing to tune on test "
                f"images: {overlap[:10]}"
            )
    first_map = _first_label_map(args, args.train_split, ids)
    offset, reason = _resolve_offset(args, label_names, first_map, args.cache_dir)
    print(f"[tune] offset={offset}: {reason}")
    mapping = load_class_mapping(
        args.class_groups, label_names, DecompConfig().prompts, offset, IGNORE_VALUE
    )
    gt_maps = {
        image_id: load_label_map(_image_and_label(args, args.train_split, image_id)[1])
        for image_id in ids
    }

    gsam = _build_gsam(args)
    decomposer = CachedDecomposer(gsam, Path(args.cache_dir))
    grids = _requested_grids(args)
    best = _default_params()
    sweep_rows: List[Dict[str, object]] = []

    for parameter in PARAMETER_ORDER:
        best_value = best[parameter]
        best_key: Optional[Tuple[float, float]] = None
        for value in grids[parameter]:
            candidate = dict(best)
            candidate[parameter] = float(value)
            print(f"[tune] {parameter}={value} on {len(ids)} tuning images")
            results, _ = _evaluate_setting(
                args, decomposer, candidate, ids, args.train_split,
                mapping, offset, label_names, gt_maps,
            )
            summary = summarize_e1(results, mapping)
            key = (
                float(np.nan_to_num(summary["miou"], nan=-1.0)),
                float(np.nan_to_num(summary["coverage_pooled"], nan=-1.0)),
            )
            sweep_rows.append(
                {
                    "parameter": parameter,
                    "value": float(value),
                    "miou": _float_or_blank(summary["miou"]),
                    "coverage": _float_or_blank(summary["coverage_pooled"]),
                    "n_images": len(results),
                }
            )
            print(
                f"[tune]   mIoU={summary['miou']:.4f} "
                f"coverage={summary['coverage_pooled']:.4f}"
            )
            if best_key is None or key > best_key:
                best_key = key
                best_value = float(value)
        best[parameter] = float(best_value)
        print(f"[tune] selected {parameter}={best[parameter]}")

    results_dir = Path(args.results_dir)
    _write_rows(
        results_dir / "e1" / "tuning_sweep.csv",
        sweep_rows,
        ["parameter", "value", "miou", "coverage", "n_images"],
    )

    # Save full decompositions only for the finally selected setting.
    selected_hash = _eval_cache_hash(best, args, offset, label_names)
    (Path(args.cache_dir) / "e1_tuning").mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.cache_dir) / "e1_tuning" / "manifest.json"
    _apply_params(gsam, best)
    for image_id in ids:
        cache_path = Path(args.cache_dir) / "e1_tuning" / f"{int(image_id):012d}.npz"
        if cache_path.is_file() and _manifest_matches(manifest_path, image_id, selected_hash):
            continue
        image_path, _ = _image_and_label(args, args.train_split, image_id)
        decomposition = decomposer.decompose(image_path)
        save_decomposition(decomposition, str(cache_path))
        _update_manifest(manifest_path, image_id, selected_hash)

    # Re-evaluate the selected setting from the metric cache for the recorded
    # summary (no extra model calls when the cache is warm).
    selected_results, _ = _evaluate_setting(
        args, decomposer, best, ids, args.train_split,
        mapping, offset, label_names, gt_maps,
    )
    selected_summary = summarize_e1(selected_results, mapping)
    selected_payload = {
        "params": best,
        "grids": grids,
        "selection_metric": "tuning mIoU, tie-break pooled coverage",
        "miou": selected_summary["miou"],
        "coverage": selected_summary["coverage_pooled"],
        "n_images": len(selected_results),
        "label_offset": offset,
        "label_offset_reason": reason,
        "label_names": str(args.label_names),
        "class_groups": str(args.class_groups),
        "train_split": args.train_split,
        "tuning_ids": str(args.tuning_ids),
        "cache_dir": str(args.cache_dir),
        "used_defaults": False,
    }
    _write_json(results_dir / "e1" / "selected_params.json", selected_payload)
    _write_notes(args)
    print(f"[tune] wrote tuning_sweep.csv and selected_params.json to {results_dir / 'e1'}")
    return 0


# ---------------------------------------------------------------------------
# test (E1)
# ---------------------------------------------------------------------------

def _load_or_make_test_decomposition(
    args,
    decomposer: CachedDecomposer,
    image_id: int,
    settings: Mapping[str, float],
    selected_hash: str,
) -> Tuple[object, bool]:
    """Return a cached test decomposition and whether it came from disk."""
    cache_path = Path(args.cache_dir) / "e1_test" / f"{int(image_id):012d}.npz"
    manifest_path = Path(args.cache_dir) / "e1_test" / "manifest.json"
    if cache_path.is_file() and _manifest_matches(manifest_path, image_id, selected_hash):
        return load_decomposition(str(cache_path)), True
    image_path, _ = _image_and_label(args, args.val_split, image_id)
    _apply_params(decomposer.gsam, settings)
    decomposition = decomposer.decompose(image_path)
    save_decomposition(decomposition, str(cache_path))
    _update_manifest(manifest_path, image_id, selected_hash)
    return decomposition, False


def cmd_test(args) -> int:
    """Run E1 on the test list, cache decompositions, and write summary files."""
    results_dir = Path(args.results_dir)
    selected_path = results_dir / "e1" / "selected_params.json"
    selected = _read_json(selected_path)
    used_defaults = bool(args.use_defaults)
    if selected is None and not used_defaults:
        raise SystemExit(
            f"{selected_path} does not exist. Run the tune command first, "
            "or pass --use-defaults for an explicitly recorded default run."
        )
    params = _default_params() if used_defaults else {
        name: float(selected["params"][name]) for name in PARAMETER_ORDER
    }
    if selected is None:
        selected = {}

    label_names = load_label_names(args.label_names)
    ids = _limit(_read_ids(args.test_ids), args)
    if not ids:
        raise SystemExit("test ID list is empty")
    preferred = None if used_defaults else selected.get("label_offset")
    first_map = _first_label_map(args, args.val_split, ids)
    offset, reason = _resolve_offset(
        args, label_names, first_map, args.cache_dir,
        preferred=int(preferred) if preferred is not None else None,
    )
    print(f"[test] offset={offset}: {reason}")
    mapping = load_class_mapping(
        args.class_groups, label_names, DecompConfig().prompts, offset, IGNORE_VALUE
    )

    gsam = _build_gsam(args, params)
    decomposer = CachedDecomposer(gsam, Path(args.cache_dir))
    (Path(args.cache_dir) / "e1_test").mkdir(parents=True, exist_ok=True)
    selected_hash = _eval_cache_hash(params, args, offset, label_names)
    results: List[E1ImageResult] = []
    per_image_rows: List[ResultsRecord] = []

    for index, image_id in enumerate(ids, start=1):
        print(f"[test] ({index}/{len(ids)}) image {image_id}")
        decomposition, cached = _load_or_make_test_decomposition(
            args, decomposer, image_id, params, selected_hash
        )
        gt_map = _load_gt(args, args.val_split, image_id, decomposition.shape)
        result = evaluate_e1_image(decomposition, gt_map, mapping)
        results.append(result)
        print(
            f"[test]   {'cached' if cached else 'new'} coverage={result.coverage:.4f} "
            f"accuracy={result.accuracy:.4f}"
        )
        values = {
            "coverage": result.coverage,
            "fallback_fraction": result.fallback_fraction,
            "n_regions": float(result.n_regions),
            "n_discarded": float(result.n_discarded),
            "pixel_accuracy": result.accuracy,
        }
        for metric in E1_METRICS:
            per_image_rows.append(
                ResultsRecord("E1", "decomposition", str(image_id), metric, float(values[metric]))
            )

    summary = summarize_e1(results, mapping)
    summary_payload = {
        **summary,
        "params": params,
        "used_defaults": used_defaults,
        "label_offset": offset,
        "label_offset_reason": reason,
        "label_names": str(args.label_names),
        "class_groups": str(args.class_groups),
        "val_split": args.val_split,
        "test_ids": str(args.test_ids),
        "cache_dir": str(args.cache_dir),
    }
    _write_json(results_dir / "e1" / "summary.json", summary_payload)
    save_results(per_image_rows, results_dir / "e1" / "per_image.csv")

    summary_rows = [
        ResultsRecord("E1", "decomposition", "ALL", "miou", float(summary["miou"])),
        ResultsRecord("E1", "decomposition", "ALL", "domain_miou", float(summary["domain_miou"])),
        ResultsRecord("E1", "decomposition", "ALL", "pixel_accuracy", float(summary["pixel_accuracy"])),
        ResultsRecord("E1", "decomposition", "ALL", "coverage_mean", float(summary["coverage_mean"])),
        ResultsRecord("E1", "decomposition", "ALL", "coverage_pooled", float(summary["coverage_pooled"])),
        ResultsRecord("E1", "decomposition", "ALL", "fallback_fraction", float(summary["fallback_fraction"])),
        ResultsRecord("E1", "decomposition", "ALL", "fallback_images", float(summary["fallback_images"])),
        ResultsRecord("E1", "decomposition", "ALL", "mean_regions", float(summary["mean_regions"])),
        ResultsRecord("E1", "decomposition", "ALL", "mean_discarded", float(summary["mean_discarded"])),
        ResultsRecord("E1", "decomposition", "ALL", "n_images", float(summary["n_images"])),
    ]
    for group, value in summary["ious"].items():
        summary_rows.append(
            ResultsRecord("E1", "decomposition", "ALL", f"iou_{group}", float(value))
        )
    for domain, value in summary["domain_ious"].items():
        summary_rows.append(
            ResultsRecord("E1", "decomposition", "ALL", f"iou_domain_{domain}", float(value))
        )
    save_results(summary_rows, results_dir / "e1" / "summary.csv")

    columns = list(mapping.group_names) + ["uncovered"]
    confusion_rows = []
    for index, group in enumerate(mapping.group_names):
        row = {"gt_group": group}
        row.update({column: int(summary["confusion"][index, j]) for j, column in enumerate(columns)})
        confusion_rows.append(row)
    _write_rows(results_dir / "e1" / "confusion_groups.csv", confusion_rows, ["gt_group"] + columns)
    _write_notes(args)
    print(
        f"[test] mIoU={summary['miou']:.4f} coverage={summary['coverage_pooled']:.4f} "
        f"fallback={summary['fallback_fraction']:.4f}"
    )
    return 0


# ---------------------------------------------------------------------------
# routing (E2)
# ---------------------------------------------------------------------------

E2_COLUMNS = [
    "image_id",
    "region_index",
    "label",
    "score",
    "assigned_domain",
    "dominant_gt_class",
    "dominant_gt_domain",
    "correct",
    "excluded",
]


def cmd_routing(args) -> int:
    """Run E2 from cached test decompositions; never call a model."""
    if args.region_mask != "resolved":
        raise SystemExit(
            "only --region-mask resolved is supported: cached decompositions "
            "store the owner map, not raw SAM masks"
        )
    label_names = load_label_names(args.label_names)
    ids = _limit(_read_ids(args.test_ids), args)
    if not ids:
        raise SystemExit("test ID list is empty")
    preferred = _preferred_offset_from_outputs(args.results_dir)
    first_map = _first_label_map(args, args.val_split, ids)
    offset, reason = _resolve_offset(
        args, label_names, first_map, args.cache_dir, preferred=preferred
    )
    print(f"[routing] offset={offset}: {reason}")
    mapping = load_class_mapping(
        args.class_groups, label_names, DecompConfig().prompts, offset, IGNORE_VALUE
    )

    reports: List[RoutingReport] = []
    rows: List[Dict[str, object]] = []
    for index, image_id in enumerate(ids, start=1):
        cache_path = Path(args.cache_dir) / "e1_test" / f"{int(image_id):012d}.npz"
        if not cache_path.is_file():
            raise SystemExit(
                f"cached decomposition missing for image {image_id}: {cache_path}. "
                "Run the E1 test command first."
            )
        decomposition = load_decomposition(str(cache_path))
        gt_map = _load_gt(args, args.val_split, image_id, decomposition.shape)
        report = evaluate_region_routing(
            decomposition, gt_map, mapping, str(image_id), region_mask=args.region_mask
        )
        reports.append(report)
        rows.extend(row.as_csv_row() for row in report.rows)
        print(
            f"[routing] ({index}/{len(ids)}) image {image_id}: "
            f"{report.n_included} included, {report.n_excluded} excluded"
        )

    summary = summarize_routing(reports)
    payload = {
        **summary,
        "region_mask": args.region_mask,
        "label_offset": offset,
        "label_offset_reason": reason,
        "label_names": str(args.label_names),
        "class_groups": str(args.class_groups),
        "val_split": args.val_split,
        "test_ids": str(args.test_ids),
        "cache_dir": str(args.cache_dir),
    }
    results_dir = Path(args.results_dir)
    _write_json(results_dir / "e2" / "routing_summary.json", payload)
    confusion = summary["confusion"]
    domain_names = list(summary["domain_names"])
    confusion_rows = []
    for index, domain in enumerate(domain_names):
        row = {"gt_domain": domain}
        row.update(
            {name: int(confusion[index, j]) for j, name in enumerate(domain_names)}
        )
        confusion_rows.append(row)
    _write_rows(
        results_dir / "e2" / "confusion_domains.csv",
        confusion_rows,
        ["gt_domain"] + domain_names,
    )
    _write_rows(results_dir / "e2" / "regions.csv", rows, E2_COLUMNS)
    _write_notes(args)
    print(
        f"[routing] accuracy={summary['accuracy']:.4f} "
        f"excluded={summary['n_excluded']} discarded={summary['n_discarded']}"
    )
    return 0


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _read_confusion_csv(path: Union[str, Path]):
    """Read a matrix CSV whose first column is the row label."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]
    matrix = np.asarray([[int(float(value)) for value in row[1:]] for row in rows], dtype=np.int64)
    return matrix, tuple(row[0] for row in rows), tuple(header[1:])


def _plot_confusion(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    path: Union[str, Path],
) -> bool:
    """Save a labelled confusion-matrix heatmap; return False without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[figures] matplotlib not installed; skipping {path}")
        return False
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(max(5, matrix.shape[1] * 1.2), max(4, matrix.shape[0] * 1.1)))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(column_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column, row, str(int(matrix[row, column])),
                ha="center", va="center", fontsize=8,
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _gt_group_image(label_map: np.ndarray, mapping) -> Image.Image:
    """Colour a GT group map with the same domain colours as the overlay."""
    groups = mapping.encode_label_map(label_map)
    arr = np.full((groups.shape[0], groups.shape[1], 3), 40, dtype=np.uint8)
    for index, domain in enumerate(mapping.group_domains):
        arr[groups == index] = viz.DOMAIN_COLORS.get(domain, (200, 200, 200))
    return Image.fromarray(arr, mode="RGB")


def _example_tiles(args, image_id: int, mapping) -> List[Image.Image]:
    """Build the three tiles (input, overlay, GT group map) for one image."""
    image_path, _ = _image_and_label(args, args.val_split, image_id)
    cache_path = Path(args.cache_dir) / "e1_test" / f"{int(image_id):012d}.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"cached decomposition missing: {cache_path}")
    decomposition = load_decomposition(str(cache_path))
    # Reproduce the decomposition resolution exactly: when the original longer
    # side was above max_side, the cached shape's longer side equals max_side;
    # otherwise it equals the original size and no resize happens.
    gray = load_gray(image_path, max(decomposition.shape))
    overlay = viz.overlay(gray.convert("RGB"), decomposition)
    gt_map = _load_gt(args, args.val_split, image_id, decomposition.shape)
    return [
        viz.tile(gray, f"{image_id} input", "working grayscale"),
        viz.tile(overlay, f"{image_id} overlay", "colorize.viz.overlay"),
        viz.tile(_gt_group_image(gt_map, mapping), f"{image_id} GT groups", "domain colours"),
    ]


def _pixel_accuracy_rows(path: Union[str, Path]) -> List[Tuple[int, float]]:
    """Read per-image pixel accuracy, returning finite values sorted ascending."""
    path = Path(path)
    values: List[Tuple[int, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "pixel_accuracy":
                continue
            try:
                value = float(row["value"])
                image_id = int(row["image_id"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append((image_id, value))
    return sorted(values, key=lambda item: (item[1], item[0]))


def _save_example_grid(args, mapping, image_ids: Sequence[int], title: str,
                       path: Union[str, Path]) -> bool:
    """Compose and save input/overlay/GT tiles for a list of images."""
    tiles: List[Image.Image] = []
    for image_id in image_ids:
        tiles.extend(_example_tiles(args, image_id, mapping))
    if not tiles:
        return False
    grid = viz.compose_grid(tiles, cols=3, title=title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    return True


def cmd_figures(args) -> int:
    """Create confusion-matrix figures and example/failure image grids."""
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    label_names = load_label_names(args.label_names)
    preferred = _preferred_offset_from_outputs(results_dir)
    ids = _limit(_read_ids(args.test_ids), args)
    first_map = _first_label_map(args, args.val_split, ids)
    offset, _ = _resolve_offset(
        args, label_names, first_map, args.cache_dir, preferred=preferred
    )
    mapping = load_class_mapping(
        args.class_groups, label_names, DecompConfig().prompts, offset, IGNORE_VALUE
    )

    group_path = results_dir / "e1" / "confusion_groups.csv"
    if group_path.is_file():
        group_cm, group_rows, group_cols = _read_confusion_csv(group_path)
        _plot_confusion(
            group_cm, group_rows, group_cols,
            "E1 decomposition: GT group vs predicted group",
            figures_dir / "e1_confusion_groups.png",
        )
        domain_cm, domain_rows, domain_cols = aggregate_domain_confusion(group_cm, mapping)
        _plot_confusion(
            domain_cm, domain_rows, domain_cols,
            "E1 decomposition: GT domain vs predicted domain",
            figures_dir / "e1_confusion_domains.png",
        )
    else:
        print(f"[figures] missing {group_path}; skipping E1 confusion figures")

    routing_path = results_dir / "e2" / "confusion_domains.csv"
    if routing_path.is_file():
        routing_cm, routing_rows, routing_cols = _read_confusion_csv(routing_path)
        _plot_confusion(
            routing_cm, routing_rows, routing_cols,
            "E2 routing: GT domain vs assigned domain",
            figures_dir / "e2_confusion_domains.png",
        )
    else:
        print(f"[figures] missing {routing_path}; skipping E2 confusion figure")

    rng = random.Random(args.seed)
    random_ids = rng.sample(list(ids), min(6, len(ids)))
    _save_example_grid(
        args, mapping, random_ids,
        "E1 examples: six random test images",
        figures_dir / "e1_examples_random.png",
    )

    per_image_path = results_dir / "e1" / "per_image.csv"
    if per_image_path.is_file():
        failure_ids = [image_id for image_id, _ in _pixel_accuracy_rows(per_image_path)[:12]]
        if failure_ids:
            _save_example_grid(
                args, mapping, failure_ids,
                "E1 failure cases: twelve lowest per-image pixel accuracies",
                figures_dir / "e1_examples_failures.png",
            )
    else:
        print(f"[figures] missing {per_image_path}; skipping failure-case grid")

    print(f"[figures] wrote figures to {figures_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(subparser) -> None:
    """Add path, device, limit and class-mapping options shared by all commands."""
    subparser.add_argument("--images-root", required=True, help="COCO image root")
    subparser.add_argument("--labels-root", required=True, help="COCO-Stuff stuffthingmaps root")
    subparser.add_argument("--label-names", required=True, help="COCO-Stuff labels.txt")
    subparser.add_argument("--class-groups", default="configs/class_groups.json")
    subparser.add_argument("--label-offset", default="auto",
                           help="'auto' or a forced integer offset")
    subparser.add_argument("--cache-dir", default="cache")
    subparser.add_argument("--results-dir", default="results")
    subparser.add_argument("--data-lists-dir", default="data_lists")
    subparser.add_argument("--tuning-ids", default="data_lists/e1_tuning_ids.txt")
    subparser.add_argument("--test-ids", default="data_lists/e1_test_ids.txt")
    subparser.add_argument("--train-split", default="train2017")
    subparser.add_argument("--val-split", default="val2017")
    subparser.add_argument("--sam-checkpoint", default=None)
    subparser.add_argument("--device", default=None)
    subparser.add_argument("--limit", type=int, default=None)
    subparser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    """Build the subcommand parser for the E1/E2 runner."""
    parser = argparse.ArgumentParser(
        description="E1 decomposition quality and E2 routing accuracy experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-data", help="audit label maps and save examples")
    _add_common(check)
    check.add_argument("--split", default="val2017")
    check.add_argument("--n-images", type=int, default=4)
    check.add_argument("--out-dir", default=None)
    check.set_defaults(func=cmd_check_data)

    lists = subparsers.add_parser("make-lists", help="create/reuse fixed ID lists")
    _add_common(lists)
    lists.add_argument("--tuning-size", type=int, default=200)
    lists.add_argument("--test-size", type=int, default=500)
    lists.add_argument("--force", action="store_true")
    lists.set_defaults(func=cmd_make_lists)

    tune = subparsers.add_parser("tune", help="one-pass parameter sweep on tuning IDs")
    _add_common(tune)
    tune.add_argument("--box-threshold-grid", default=None)
    tune.add_argument("--text-threshold-grid", default=None)
    tune.add_argument("--min-score-grid", default=None)
    tune.add_argument("--nms-iou-grid", default=None)
    tune.add_argument("--merge-containment-grid", default=None)
    tune.set_defaults(func=cmd_tune)

    test = subparsers.add_parser("test", help="run E1 on the fixed test IDs")
    _add_common(test)
    test.add_argument("--use-defaults", action="store_true",
                      help="run without selected_params.json and record that fact")
    test.set_defaults(func=cmd_test)

    routing = subparsers.add_parser("routing", help="run E2 from cached test decompositions")
    _add_common(routing)
    routing.add_argument("--region-mask", choices=("resolved", "raw"), default="resolved")
    routing.set_defaults(func=cmd_routing)

    figures = subparsers.add_parser("figures", help="write confusion matrices and example grids")
    _add_common(figures)
    figures.set_defaults(func=cmd_figures)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and dispatch to the selected experiment stage."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
