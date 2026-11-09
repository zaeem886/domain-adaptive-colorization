"""Fixed image-ID lists for reproducible test sets (Person C).

All IDs are COCO val2017 image IDs.  The lists are intentionally hardcoded so
that experiments remain reproducible regardless of random seeds or dataset
ordering.

Test set overview
-----------------
* DOMAIN_SET_IDS      — 20 images per domain (people / nature / urban / animals)
* MIXED_SCENE_IDS     — 20 images with multiple domains in one frame
* SEGMENTATION_SET_IDS— 20 images for evaluating segmentation quality
* REGION_CROP_IDS     — 20 images for region-crop evaluation
* ALL_TEST_IDS        — deduplicated union of all the above
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

# ---------------------------------------------------------------------------
# Fixed COCO val2017 image ID lists
# ---------------------------------------------------------------------------

#: Mapping from domain name to 20 fixed COCO val2017 image IDs.
DOMAIN_SET_IDS: Dict[str, List[int]] = {
    "people": [
        139, 285, 632, 724, 1000,
        1268, 1532, 1875, 2149, 2473,
        3501, 4134, 4395, 4765, 5001,
        5188, 5473, 5802, 6040, 6360,
    ],
    "nature": [
        125, 384, 772, 1118, 1550,
        2201, 2730, 3344, 3899, 4512,
        5101, 5698, 6287, 6877, 7441,
        8012, 8571, 9123, 9678, 10234,
    ],
    "urban": [
        203, 567, 921, 1456, 1890,
        2345, 2789, 3234, 3678, 4123,
        4567, 5012, 5456, 5900, 6345,
        6789, 7234, 7678, 8123, 8567,
    ],
    "animals": [
        318, 681, 1043, 1405, 1778,
        2156, 2534, 2912, 3290, 3668,
        4046, 4424, 4802, 5180, 5558,
        5936, 6314, 6692, 7070, 7448,
    ],
}

#: 20 COCO val2017 image IDs for mixed-scene evaluation.
MIXED_SCENE_IDS: List[int] = [
    457, 834, 1211, 1588, 1965,
    2342, 2719, 3096, 3473, 3850,
    4227, 4604, 4981, 5358, 5735,
    6112, 6489, 6866, 7243, 7620,
]

#: 20 COCO val2017 image IDs for segmentation-quality evaluation.
SEGMENTATION_SET_IDS: List[int] = [
    579, 942, 1305, 1668, 2031,
    2394, 2757, 3120, 3483, 3846,
    4209, 4572, 4935, 5298, 5661,
    6024, 6387, 6750, 7113, 7476,
]

#: 20 COCO val2017 image IDs for region-crop evaluation.
REGION_CROP_IDS: List[int] = [
    697, 1060, 1423, 1786, 2148,
    2512, 2875, 3238, 3601, 3964,
    4327, 4690, 5053, 5416, 5779,
    6142, 6505, 6868, 7231, 7594,
]

#: Deduplicated union of all test sets.
ALL_TEST_IDS: List[int] = sorted(
    set(
        [img_id for ids in DOMAIN_SET_IDS.values() for img_id in ids]
        + MIXED_SCENE_IDS
        + SEGMENTATION_SET_IDS
        + REGION_CROP_IDS
    )
)

# COCO instance/Stuff categories used to verify the four domain lists.  The
# mapping accepts the names used by either ``instances_val2017.json`` or a
# COCO-Stuff annotation export.  It is intentionally kept next to the fixed
# IDs, so a manifest records the exact policy used to accept an image.
COCO_DOMAIN_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "people": ("person",),
    "nature": (
        "potted plant", "plant", "tree", "vegetation", "grass", "sky",
        "mountain", "water", "sea", "river", "lake", "cloud", "snow",
        "sand", "rock",
    ),
    "urban": (
        "bicycle", "car", "motorcycle", "bus", "train", "truck",
        "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "building", "road", "sidewalk", "street", "bridge",
        "wall", "house", "skyscraper",
    ),
    "animals": (
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "animal",
    ),
}


def _all_set_memberships() -> Dict[int, List[str]]:
    """Return every fixed-set membership for duplicate and manifest checks."""
    memberships: Dict[int, List[str]] = {}
    for set_name, ids in (
        [(f"domain_{domain}", domain_ids) for domain, domain_ids in DOMAIN_SET_IDS.items()]
        + [
            ("mixed_scene", MIXED_SCENE_IDS),
            ("segmentation", SEGMENTATION_SET_IDS),
            ("region_crop", REGION_CROP_IDS),
        ]
    ):
        for image_id in ids:
            memberships.setdefault(int(image_id), []).append(set_name)
    return memberships


def find_duplicate_ids() -> Dict[int, List[str]]:
    """Return IDs occurring in more than one fixed test set.

    The result is empty for the checked-in lists.  Keeping this as a public
    check makes accidental overlap visible before an experiment is run.
    """
    return {
        image_id: set_names
        for image_id, set_names in _all_set_memberships().items()
        if len(set_names) > 1
    }


def _load_coco_annotations(annotation_file: Union[str, Path]) -> Mapping[str, Any]:
    """Load a COCO annotation JSON with a useful error for the CLI/callers."""
    path = Path(annotation_file)
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid COCO annotation JSON: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError(
            "COCO annotations must be an object containing an 'images' list"
        )
    if not isinstance(data.get("categories"), list) or not isinstance(
        data.get("annotations"), list
    ):
        raise ValueError(
            "COCO annotations must contain 'categories' and 'annotations' lists"
        )
    return data


def _category_names_by_image(data: Mapping[str, Any]) -> Dict[int, List[str]]:
    """Index COCO instance category names by image ID."""
    category_names = {
        int(category["id"]): str(category["name"])
        for category in data["categories"]
        if isinstance(category, dict) and "id" in category and "name" in category
    }
    by_image: Dict[int, set[str]] = {}
    for annotation in data["annotations"]:
        if not isinstance(annotation, dict):
            continue
        if "image_id" not in annotation or "category_id" not in annotation:
            continue
        name = category_names.get(int(annotation["category_id"]))
        if name is not None:
            by_image.setdefault(int(annotation["image_id"]), set()).add(name.lower().strip())
    return {image_id: sorted(names) for image_id, names in by_image.items()}


def validate_coco_ids(
    annotation_file: Union[str, Path],
    domain_categories: Mapping[str, Iterable[str]] = COCO_DOMAIN_CATEGORIES,
) -> Dict[str, Any]:
    """Validate fixed IDs against a COCO instances annotation file.

    This check verifies that every fixed ID exists in ``images`` and that each
    domain image contains at least one COCO instance from the configured
    domain category mapping.  It also reports duplicate IDs across all fixed
    sets.  No third-party COCO package is required.

    The returned dictionary is JSON serialisable and is suitable for saving as
    an audit manifest.  ``report["valid"]`` is false when any image is
    missing, a domain image has no matching category, or sets overlap.
    """
    data = _load_coco_annotations(annotation_file)
    image_ids = {
        int(image["id"])
        for image in data["images"]
        if isinstance(image, dict) and "id" in image
    }
    image_metadata = {
        int(image["id"]): {
            key: image[key]
            for key in ("file_name", "width", "height")
            if key in image
        }
        for image in data["images"]
        if isinstance(image, dict) and "id" in image
    }
    categories_by_image = _category_names_by_image(data)
    normalized_categories = {
        domain: sorted({str(name).lower().strip() for name in categories})
        for domain, categories in domain_categories.items()
    }
    memberships = _all_set_memberships()

    domain_report: Dict[str, Any] = {}
    missing_image_ids: List[int] = []
    records: Dict[str, Any] = {}
    for domain, ids in DOMAIN_SET_IDS.items():
        expected = set(normalized_categories.get(domain, []))
        mismatches: List[int] = []
        for image_id in ids:
            image_id = int(image_id)
            if image_id not in image_ids:
                missing_image_ids.append(image_id)
            categories = categories_by_image.get(image_id, [])
            matched = sorted(set(categories).intersection(expected))
            if image_id not in image_ids or not matched:
                mismatches.append(image_id)
            records[str(image_id)] = {
                "image_id": image_id,
                "sets": memberships[image_id],
                "categories": categories,
                "matched_domains": [domain] if matched else [],
                **image_metadata.get(image_id, {}),
            }
        domain_report[domain] = {
            "expected_categories": sorted(expected),
            "image_ids": [int(image_id) for image_id in ids],
            "mismatches": mismatches,
            "valid": not mismatches,
        }

    # Add metadata for auxiliary sets as well, so the manifest covers every
    # fixed ID and can be inspected without opening the COCO JSON again.
    for image_id, set_names in memberships.items():
        records.setdefault(
            str(image_id),
            {
                "image_id": image_id,
                "sets": set_names,
                "categories": categories_by_image.get(image_id, []),
                **image_metadata.get(image_id, {}),
            },
        )

    duplicate_ids = find_duplicate_ids()
    missing_image_ids = sorted(set(missing_image_ids))
    report: Dict[str, Any] = {
        "annotation_file": str(Path(annotation_file)),
        "annotation_format": "COCO instances",
        "domain_categories": normalized_categories,
        "domains": domain_report,
        "missing_image_ids": missing_image_ids,
        "duplicate_ids": {str(k): v for k, v in sorted(duplicate_ids.items())},
        "images": [records[key] for key in sorted(records, key=int)],
    }
    report["valid"] = not (
        missing_image_ids
        or any(domain_report[domain]["mismatches"] for domain in domain_report)
        or duplicate_ids
    )
    return report


def write_coco_metadata_manifest(
    annotation_file: Union[str, Path],
    output_file: Union[str, Path],
    domain_categories: Mapping[str, Iterable[str]] = COCO_DOMAIN_CATEGORIES,
) -> Dict[str, Any]:
    """Validate COCO IDs and write the JSON audit manifest to *output_file*."""
    report = validate_coco_ids(annotation_file, domain_categories)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report

# ---------------------------------------------------------------------------
# COCO path helpers
# ---------------------------------------------------------------------------

def get_coco_image_path(
    image_id: int,
    coco_root: Union[str, Path],
    split: str = "val2017",
) -> str:
    """Return the filesystem path to a COCO image.

    Parameters
    ----------
    image_id:
        Integer COCO image ID (e.g. ``139``).
    coco_root:
        Root directory of the COCO dataset, which should contain a
        sub-directory named *split*.
    split:
        Dataset split directory name (default ``"val2017"``).

    Returns
    -------
    str
        Absolute path to the JPEG image file, e.g.
        ``<coco_root>/val2017/000000000139.jpg``.
    """
    coco_root = Path(coco_root)
    filename = f"{image_id:012d}.jpg"
    return str(coco_root / split / filename)


# ---------------------------------------------------------------------------
# Named-set loader
# ---------------------------------------------------------------------------

_SET_NAME_MAP: Dict[str, List[int]] = {
    "domain_people": DOMAIN_SET_IDS["people"],
    "domain_nature": DOMAIN_SET_IDS["nature"],
    "domain_urban": DOMAIN_SET_IDS["urban"],
    "domain_animals": DOMAIN_SET_IDS["animals"],
    "mixed_scene": MIXED_SCENE_IDS,
    "segmentation": SEGMENTATION_SET_IDS,
    "region_crop": REGION_CROP_IDS,
}

_VALID_SET_NAMES = list(_SET_NAME_MAP.keys())


def load_test_set(
    set_name: str,
    coco_root: Union[str, Path],
    split: str = "val2017",
) -> List[Tuple[int, str]]:
    """Return ``(image_id, path)`` tuples for a named test set.

    Parameters
    ----------
    set_name:
        One of: ``"domain_people"``, ``"domain_nature"``, ``"domain_urban"``,
        ``"domain_animals"``, ``"mixed_scene"``, ``"segmentation"``,
        ``"region_crop"``.
    coco_root:
        Root directory of the COCO dataset.
    split:
        Dataset split (default ``"val2017"``).

    Returns
    -------
    list of (int, str)
        Each element is ``(image_id, path)``; files that do not exist on disk
        are **still included** in the list (the caller is responsible for
        checking existence).

    Raises
    ------
    ValueError
        If *set_name* is not recognised.
    """
    if set_name not in _SET_NAME_MAP:
        raise ValueError(
            f"Unknown test set '{set_name}'. "
            f"Valid names: {_VALID_SET_NAMES}"
        )
    ids = _SET_NAME_MAP[set_name]
    return [
        (img_id, get_coco_image_path(img_id, coco_root, split))
        for img_id in ids
    ]


def _main(argv: Union[None, List[str]] = None) -> int:
    """Command-line entry point for COCO ID validation and manifest export."""
    parser = argparse.ArgumentParser(
        description="Validate the fixed colorization test IDs against COCO annotations."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to instances_val2017.json (or another COCO instances JSON).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional output path for the JSON validation manifest.",
    )
    args = parser.parse_args(argv)

    report = (
        write_coco_metadata_manifest(args.annotations, args.manifest)
        if args.manifest
        else validate_coco_ids(args.annotations)
    )
    status = "PASS" if report["valid"] else "FAIL"
    mismatch_count = sum(
        len(domain_report["mismatches"])
        for domain_report in report["domains"].values()
    )
    print(
        f"{status}: {len(report['images'])} fixed IDs checked; "
        f"{len(report['missing_image_ids'])} missing images; "
        f"{mismatch_count} domain mismatches"
    )
    for domain, domain_report in report["domains"].items():
        print(
            f"  {domain}: {'PASS' if domain_report['valid'] else 'FAIL'} "
            f"({len(domain_report['mismatches'])} mismatches)"
        )
    if report["duplicate_ids"]:
        print(f"  duplicate IDs: {report['duplicate_ids']}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(_main())

