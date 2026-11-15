#!/usr/bin/env python3
"""Generate the canonical 100-image COCO val2017 evaluation set.

The E3 and E4 experiments use twenty images for each of the people, nature,
urban and animals domains, plus twenty mixed-domain scenes.  The original
hardcoded lists were placeholders and most of their identifiers do not exist
in COCO val2017.  This script rebuilds the lists from the files and
annotations that are actually mounted, using the same deterministic sampling
policy as the E3 notebook:

* scan the real ``val2017`` directory;
* keep only identifiers that have an image on disk;
* build domain pools from COCO instance annotations (the default, which
  matches the E3 fallback) or from COCO-Stuff label maps (``--method stuff``);
* sample twenty images per set with seed 0, without overlap between sets.

The result is written to ``outputs/e3_ids.json``.  With
``--rewrite-test-sets`` the module lists in ``colorize/test_sets.py`` are
updated as well, so the checked-in fixed lists match the generated set.

Typical Kaggle invocation::

    python scripts/generate_test_ids.py \
        --images-root /kaggle/working/cocostuff_data/images \
        --annotations /kaggle/input/cocostuf-2017/annotations/instances_val2017.json \
        --output outputs/e3_ids.json \
        --rewrite-test-sets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SET_KEYS: Tuple[str, ...] = (
    "domain_people",
    "domain_nature",
    "domain_urban",
    "domain_animals",
    "mixed_scene",
    "segmentation_set",
    "region_crop",
)
SET_LABELS: Mapping[str, str] = {
    "domain_people": "people",
    "domain_nature": "nature",
    "domain_urban": "urban",
    "domain_animals": "animals",
    "mixed_scene": "mixed scenes",
    "segmentation_set": "segmentation",
    "region_crop": "region crops",
}

# Sets that only need real, unused val2017 images rather than a domain filter.
FILLER_KEYS: Tuple[str, ...] = ("segmentation_set", "region_crop")

ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
}
OUTDOOR = {
    "bird", "boat", "airplane", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "kite", "surfboard", "skis", "snowboard",
    "sports ball", "frisbee", "skateboard",
}
VEHICLES = {"bicycle", "car", "motorcycle", "bus", "train", "truck"}


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def _find_val2017(images_root: Optional[Path]) -> Path:
    """Return the directory that actually contains val2017 JPEGs."""
    if images_root is not None:
        images_root = Path(images_root)
        if (images_root / "val2017").is_dir():
            return images_root / "val2017"
        if images_root.is_dir() and any(images_root.glob("*.jpg")):
            return images_root
        raise FileNotFoundError(
            f"no val2017 images found under {images_root!s}; expected "
            f"{images_root / 'val2017'} or JPEG files directly in {images_root}"
        )

    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        raise FileNotFoundError(
            "no --images-root given and /kaggle/input is not available"
        )

    candidates = [
        path for path in input_root.rglob("val2017")
        if path.is_dir() and any(path.glob("*.jpg"))
    ]
    if not candidates:
        raise FileNotFoundError(
            "no directory named val2017 containing JPEG files was found under "
            "/kaggle/input; pass --images-root explicitly"
        )
    return max(candidates, key=lambda path: sum(1 for _ in path.glob("*.jpg")))


def _find_annotations(val_dir: Path, explicit: Optional[Path]) -> Path:
    """Locate ``instances_val2017.json``."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"annotation file not found: {path}")
        return path

    roots = [Path("/kaggle/input"), val_dir.parent, val_dir.parent.parent, REPO_ROOT]
    for root in roots:
        if not root.is_dir():
            continue
        candidates = sorted(root.rglob("instances_val2017.json"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        "instances_val2017.json was not found; pass --annotations explicitly"
    )


def _find_stuff_dir(explicit: Optional[Path]) -> Optional[Path]:
    """Locate COCO-Stuff ``stuffthingmaps`` for the val2017 split."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"stuff directory not found: {path}")
        return path

    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return None

    candidates: List[Path] = []
    for name in ("stuffthingmaps_val2017", "stuffthingmaps"):
        candidates.extend(path for path in input_root.rglob(name) if path.is_dir())
    for parent in input_root.rglob("stuffthingmaps_trainval2017"):
        inner = parent / "val2017"
        if inner.is_dir():
            candidates.append(inner)
    for path in input_root.rglob("val2017"):
        if path.is_dir() and "stuffthingmaps" in path.parent.name.lower():
            candidates.append(path)

    candidates = [path for path in candidates if any(path.glob("*.png"))]
    if not candidates:
        return None
    return max(candidates, key=lambda path: sum(1 for _ in path.glob("*.png")))


def _find_labels(explicit: Optional[Path]) -> Optional[Path]:
    """Locate the COCO-Stuff class-name file."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"labels file not found: {path}")
        return path

    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return None
    for name in ("cocostuff-labels.txt", "labels.txt"):
        matches = sorted(input_root.rglob(name))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Annotation and image handling
# ---------------------------------------------------------------------------

def _image_ids_on_disk(val_dir: Path) -> set[int]:
    ids: set[int] = set()
    for path in val_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            ids.add(int(path.stem))
        except ValueError:
            continue
    return ids


def _load_instances(annotation_path: Path) -> Tuple[
    Mapping[int, Mapping[str, Any]],
    Mapping[int, str],
    Mapping[int, Mapping[str, float]],
]:
    """Load image metadata, category names and per-image category areas."""
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    categories = {int(item["id"]): str(item["name"]) for item in data["categories"]}
    images = {int(item["id"]): item for item in data["images"]}
    area: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for annotation in data.get("annotations", []):
        try:
            image_id = int(annotation["image_id"])
            category = categories.get(int(annotation["category_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        if category is None:
            continue
        area[image_id][category.lower()] += float(annotation.get("area", 0.0) or 0.0)
    return images, categories, area


def _instance_pools(
    exists_val: Iterable[int],
    images: Mapping[int, Mapping[str, Any]],
    area: Mapping[int, Mapping[str, float]],
) -> Dict[str, List[int]]:
    """Build the same instance-annotation pools as the E3 fallback."""

    def inst_frac(image_id: int, names: Iterable[str]) -> float:
        meta = images.get(image_id, {})
        total = float(meta.get("width", 0) or 0) * float(meta.get("height", 0) or 0)
        if not total:
            return 0.0
        categories = area.get(image_id, {})
        return sum(float(categories.get(name.lower(), 0.0)) for name in names) / total

    exists = sorted(int(image_id) for image_id in exists_val)
    people_pool = [
        image_id for image_id in exists
        if inst_frac(image_id, {"person"}) >= 0.20
        and inst_frac(image_id, ANIMALS) == 0.0
    ]
    animals_pool = [
        image_id for image_id in exists
        if inst_frac(image_id, ANIMALS) >= 0.20
        and inst_frac(image_id, {"person"}) == 0.0
    ]
    nature_pool = [
        image_id for image_id in exists
        if inst_frac(image_id, {"person"}) < 0.05
        and inst_frac(image_id, ANIMALS) < 0.05
        and inst_frac(image_id, OUTDOOR) < 0.10
    ]
    urban_pool = [
        image_id for image_id in exists
        if inst_frac(image_id, VEHICLES) >= 0.10
        and inst_frac(image_id, {"person"}) < 0.05
    ]
    mixed_pool = [
        image_id for image_id in exists
        if inst_frac(image_id, {"person"}) >= 0.10
        and inst_frac(image_id, ANIMALS) >= 0.10
    ]
    return {
        "domain_people": people_pool,
        "domain_nature": nature_pool,
        "domain_urban": urban_pool,
        "domain_animals": animals_pool,
        "mixed_scene": mixed_pool,
    }


def _parse_label_names(path: Path) -> List[str]:
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^\d+\s*:\s*(.+)$", line)
        names.append(match.group(1).strip() if match else line)
    return names


def _stuff_pools(
    exists_val: Iterable[int],
    stuff_dir: Path,
    labels_path: Path,
    class_groups_path: Path,
) -> Dict[str, List[int]]:
    """Build domain pools from COCO-Stuff label maps."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Kaggle has both
        raise SystemExit(
            "--method stuff requires numpy and Pillow; install them or use "
            "--method instances"
        ) from exc

    label_names = _parse_label_names(labels_path)
    with class_groups_path.open("r", encoding="utf-8") as handle:
        groups_cfg = json.load(handle)

    name_to_group: Dict[str, str] = {}
    for group, info in groups_cfg["groups"].items():
        for class_name in info["classes"]:
            name_to_group[class_name.lower()] = group

    group_names = ("person", "animal", "sky", "vegetation", "building")
    group_index = {name: index for index, name in enumerate(group_names)}
    lut = np.full(256, -1, dtype=np.int16)
    for pixel_value in range(182):
        index = pixel_value + 1
        if index >= len(label_names):
            continue
        group = name_to_group.get(label_names[index].lower())
        if group in group_index:
            lut[pixel_value] = group_index[group]

    nature_pool: List[int] = []
    urban_pool: List[int] = []
    mixed_pool: List[int] = []
    for image_id in sorted(int(image_id) for image_id in exists_val):
        label_path = stuff_dir / f"{image_id:012d}.png"
        if not label_path.is_file():
            continue
        with Image.open(label_path) as image:
            pixels = np.asarray(image)
        if pixels.ndim == 3:
            pixels = pixels[..., 0]
        mapped = lut[pixels.astype(np.int32, copy=False)]
        total = int(pixels.size)
        if not total:
            continue
        counts = np.bincount(mapped[mapped >= 0], minlength=len(group_names))
        fractions = {
            name: counts[group_index[name]] / total for name in group_names
        }
        if (
            fractions["sky"] + fractions["vegetation"] >= 0.40
            and fractions["person"] + fractions["animal"] < 0.02
        ):
            nature_pool.append(image_id)
        if fractions["building"] >= 0.25 and fractions["person"] < 0.05:
            urban_pool.append(image_id)
        if sum(1 for value in fractions.values() if value >= 0.10) >= 2:
            mixed_pool.append(image_id)

    return {
        "domain_people": [],  # filled from instance annotations by caller
        "domain_nature": nature_pool,
        "domain_urban": urban_pool,
        "domain_animals": [],  # filled from instance annotations by caller
        "mixed_scene": mixed_pool,
    }


# ---------------------------------------------------------------------------
# Sampling and output
# ---------------------------------------------------------------------------

def _sample_sets(
    pools: Mapping[str, Sequence[int]],
    per_set: int,
    seed: int,
) -> Dict[str, List[int]]:
    """Sample exactly ``per_set`` disjoint IDs from each pool."""
    rng = random.Random(seed)
    sets: Dict[str, List[int]] = {}
    used: set[int] = set()
    filler_pool = pools.get("_all", ())
    for key in SET_KEYS:
        source = filler_pool if key in FILLER_KEYS else pools.get(key, ())
        available = sorted({int(image_id) for image_id in source} - used)
        if len(available) < per_set:
            raise SystemExit(
                f"{key} has only {len(available)} candidates but {per_set} are "
                "required; the val2017 mount or annotation file is incomplete"
            )
        sample = sorted(rng.sample(available, per_set))
        sets[key] = sample
        used.update(sample)
    return sets


def _validate_sets(sets: Mapping[str, Sequence[int]], per_set: int) -> None:
    if set(sets) != set(SET_KEYS):
        raise ValueError(f"expected set keys {SET_KEYS}, found {tuple(sets)}")
    all_ids: List[int] = []
    for key in SET_KEYS:
        ids = [int(image_id) for image_id in sets[key]]
        if len(ids) != per_set:
            raise ValueError(f"{key} has {len(ids)} IDs, expected {per_set}")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("the generated sets contain duplicate image IDs")


def _load_existing(path: Path, per_set: int) -> Dict[str, List[int]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    sets = {key: [int(image_id) for image_id in data[key]] for key in SET_KEYS}
    _validate_sets(sets, per_set)
    return sets


def _write_json(path: Path, sets: Mapping[str, Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({key: list(sets[key]) for key in SET_KEYS}, handle, indent=1)
        handle.write("\n")


def _format_ids(ids: Sequence[int], indent: int) -> str:
    pad = " " * indent
    rows = [ids[index:index + 5] for index in range(0, len(ids), 5)]
    lines = [", ".join(str(int(image_id)) for image_id in row) for row in rows]
    return (
        "[\n"
        + pad
        + (",\n" + pad).join(lines)
        + ",\n"
        + " " * (indent - 4)
        + "]"
    )


def rewrite_test_sets(path: Path, sets: Mapping[str, Sequence[int]]) -> None:
    """Replace the hardcoded domain and mixed-scene blocks in test_sets.py."""
    source = path.read_text(encoding="utf-8")

    domain_block = (
        "DOMAIN_SET_IDS: Dict[str, List[int]] = {\n"
        '    "people": ' + _format_ids(sets["domain_people"], 8) + ",\n"
        '    "nature": ' + _format_ids(sets["domain_nature"], 8) + ",\n"
        '    "urban":  ' + _format_ids(sets["domain_urban"], 8) + ",\n"
        '    "animals":' + _format_ids(sets["domain_animals"], 8) + ",\n"
        "}"
    )
    source, domain_count = re.subn(
        r"DOMAIN_SET_IDS: Dict\[str, List\[int\]\] = \{.*?\n\}",
        domain_block,
        source,
        count=1,
        flags=re.DOTALL,
    )

    mixed_block = "MIXED_SCENE_IDS: List[int] = " + _format_ids(
        sets["mixed_scene"], 4
    )
    source, mixed_count = re.subn(
        r"MIXED_SCENE_IDS: List\[int\] = \[.*?\]",
        mixed_block,
        source,
        count=1,
        flags=re.DOTALL,
    )

    seg_block = "SEGMENTATION_SET_IDS: List[int] = " + _format_ids(
        sets["segmentation_set"], 4
    )
    source, seg_count = re.subn(
        r"SEGMENTATION_SET_IDS: List\[int\] = \[.*?\]",
        seg_block,
        source,
        count=1,
        flags=re.DOTALL,
    )

    crop_block = "REGION_CROP_IDS: List[int] = " + _format_ids(
        sets["region_crop"], 4
    )
    source, crop_count = re.subn(
        r"REGION_CROP_IDS: List\[int\] = \[.*?\]",
        crop_block,
        source,
        count=1,
        flags=re.DOTALL,
    )

    if 1 not in (domain_count, mixed_count, seg_count, crop_count):
        raise ValueError(
            f"could not rewrite {path}: domain_matches={domain_count}, "
            f"mixed_matches={mixed_count}, seg_matches={seg_count}, "
            f"crop_matches={crop_count}"
        )

    # The old placeholder warning no longer applies once real IDs are written.
    source = re.sub(
        r"# !! REGENERATION REQUIRED !!.*?\n# -{10,}\n",
        (
            "# Canonical IDs are generated from the actual val2017 files by\n"
            "# scripts/generate_test_ids.py (seed 0, 20 images per set).\n"
            "# Regenerate with:\n"
            "#   python scripts/generate_test_ids.py --output outputs/e3_ids.json \\\n"
            "#       --rewrite-test-sets\n"
            "# ---------------------------------------------------------------------------\n"
        ),
        source,
        count=1,
        flags=re.DOTALL,
    )
    path.write_text(source, encoding="utf-8")


def _set_digest(sets: Mapping[str, Sequence[int]]) -> str:
    payload = json.dumps({key: list(sets[key]) for key in SET_KEYS}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate canonical COCO val2017 IDs for E3/E4."
    )
    parser.add_argument("--images-root", default=None,
                        help="Directory containing val2017/ or the val2017 JPEGs.")
    parser.add_argument("--annotations", default=None,
                        help="Path to instances_val2017.json.")
    parser.add_argument("--method", choices=("instances", "stuff", "auto"),
                        default="instances",
                        help="Domain-pool policy. 'instances' matches the E3 fallback.")
    parser.add_argument("--stuff-dir", default=None,
                        help="COCO-Stuff label-map directory for --method stuff.")
    parser.add_argument("--labels", default=None,
                        help="COCO-Stuff class-name file for --method stuff.")
    parser.add_argument("--class-groups", default=str(REPO_ROOT / "configs" / "class_groups.json"),
                        help="Class-to-domain mapping used by --method stuff.")
    parser.add_argument("--output", default="outputs/e3_ids.json",
                        help="Destination JSON file.")
    parser.add_argument("--per-set", type=int, default=20,
                        help="Images per set (default: 20).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed (default: 0).")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if --output already exists.")
    parser.add_argument("--rewrite-test-sets", action="store_true",
                        help="Rewrite colorize/test_sets.py with the generated IDs.")
    parser.add_argument("--test-sets-path", default=str(REPO_ROOT / "colorize" / "test_sets.py"),
                        help="Path to colorize/test_sets.py for --rewrite-test-sets.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output_path = Path(args.output)
    per_set = int(args.per_set)

    if output_path.is_file() and not args.force:
        try:
            sets = _load_existing(output_path, per_set)
        except Exception as exc:
            print(f"[ids] Existing {output_path} is invalid ({exc}); regenerating.")
        else:
            print(f"[ids] Reusing {output_path}")
            if args.rewrite_test_sets:
                rewrite_test_sets(Path(args.test_sets_path), sets)
                print(f"[ids] rewrote {args.test_sets_path}")
            _report(sets, output_path, args)
            return 0

    val_dir = _find_val2017(Path(args.images_root) if args.images_root else None)
    annotation_path = _find_annotations(val_dir, Path(args.annotations) if args.annotations else None)
    images, _categories, area = _load_instances(annotation_path)
    exists_val = set(images) & _image_ids_on_disk(val_dir)
    print(f"[ids] val2017 directory: {val_dir}")
    print(f"[ids] annotation file:   {annotation_path}")
    print(f"[ids] images on disk:    {len(exists_val)}")

    method = args.method
    if method == "auto":
        stuff_dir = _find_stuff_dir(Path(args.stuff_dir) if args.stuff_dir else None)
        labels_path = _find_labels(Path(args.labels) if args.labels else None)
        method = "stuff" if stuff_dir is not None and labels_path is not None else "instances"
        print(f"[ids] auto-selected method: {method}")

    pools = _instance_pools(exists_val, images, area)
    if method == "stuff":
        stuff_dir = _find_stuff_dir(Path(args.stuff_dir) if args.stuff_dir else None)
        labels_path = _find_labels(Path(args.labels) if args.labels else None)
        if stuff_dir is None or labels_path is None:
            raise SystemExit(
                "--method stuff requires COCO-Stuff label maps and a class-name "
                "file; pass --stuff-dir and --labels, or use --method instances"
            )
        stuff_pools = _stuff_pools(
            exists_val,
            stuff_dir,
            labels_path,
            Path(args.class_groups),
        )
        pools["domain_nature"] = stuff_pools["domain_nature"]
        pools["domain_urban"] = stuff_pools["domain_urban"]
        pools["mixed_scene"] = stuff_pools["mixed_scene"]
        print(f"[ids] stuff label maps:  {stuff_dir}")
        print(f"[ids] class-name file:   {labels_path}")

    print("[ids] candidate pools:")
    for key in SET_KEYS:
        if key in FILLER_KEYS:
            count = len(exists_val)
        else:
            count = len(pools[key])
        print(f"  {key}: {count}")

    pools["_all"] = sorted(int(image_id) for image_id in exists_val)

    sets = _sample_sets(pools, per_set, int(args.seed))
    _validate_sets(sets, per_set)
    _write_json(output_path, sets)

    if args.rewrite_test_sets:
        rewrite_test_sets(Path(args.test_sets_path), sets)
        print(f"[ids] rewrote {args.test_sets_path}")

    _report(sets, output_path, args)
    return 0


def _report(sets: Mapping[str, Sequence[int]], output_path: Path, args: argparse.Namespace) -> None:
    print(f"[ids] canonical set: {output_path}")
    for key in SET_KEYS:
        print(f"  {key}: {len(sets[key])} IDs")
    all_ids = sorted(int(image_id) for key in SET_KEYS for image_id in sets[key])
    print(f"[ids] total unique IDs: {len(all_ids)}")
    print(f"[ids] sha256: {_set_digest(sets)}")


if __name__ == "__main__":
    sys.exit(main())
