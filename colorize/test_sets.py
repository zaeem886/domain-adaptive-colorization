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

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
    697, 1060, 1423, 1786, 2149,
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

