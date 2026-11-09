"""COCO-Stuff label-map loading and data sanity checks for E1/E2.

This module owns everything that depends on the *COCO-Stuff* on-disk format:

* reading the class-name list (``labels.txt``),
* locating COCO images and their ``stuffthingmaps`` PNGs,
* deciding how a PNG pixel value maps to an index in the class-name list,
* nearest-neighbour resizing of label maps to the decomposition resolution,
* the ``check-data`` visual/offset audit.

It deliberately does **not** know about Grounded-SAM, routing, or evaluation
metrics.  Keeping it separate means the same loader can be reused by E3/E4 and
test-set validation later.  All paths are supplied by the caller; nothing here
assumes a particular Kaggle mount point.

COCO-Stuff offset note
----------------------
The official ``labels.txt`` starts with ``unlabeled`` at index 0, while the
Caffe-compatible ``stuffthingmaps`` PNGs use valid class values 0..181 and
255 for unlabeled.  For those files the PNG value ``v`` maps to
``labels[v + 1]`` (offset +1).  Other exports store 1..182 with 0 as
unlabeled (offset 0), and a name list without a leading unlabeled entry may be
1-indexed (offset -1).  ``choose_label_offset`` inspects the actual PNG and the
name list instead of assuming one convention; the CLI can also force an offset.
"""
from __future__ import annotations

import colorsys
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

IGNORE_VALUE = 255

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_LABEL_SUFFIXES = (".png",)

# The official labels.txt lines look like ``0: unlabeled``.  Other exports use
# one bare name per line.  This pattern accepts both without changing the
# meaning of a class name that itself contains a colon.
_NUMBERED_LABEL = re.compile(r"^\s*\d+\s*:\s*(.*)$")


# ---------------------------------------------------------------------------
# Class-name list and label values
# ---------------------------------------------------------------------------

def load_label_names(path: Union[str, Path]) -> List[str]:
    """Read a COCO-Stuff class-name file, returning one name per entry.

    The official ``labels.txt`` uses ``"<index>: <name>"`` lines, while several
    mirrors use one bare name per line.  Both formats are accepted; blank lines
    and ``#`` comments are ignored.  The returned list is zero-based, so
    ``labels.txt`` entry 0 is ``"unlabeled"`` and callers must use
    :func:`choose_label_offset` rather than assuming a PNG value is already an
    index into this list.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"COCO-Stuff label-name file not found: {path}")
    names: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _NUMBERED_LABEL.match(line)
        names.append(match.group(1).strip() if match else line)
    if not names:
        raise ValueError(f"COCO-Stuff label-name file is empty: {path}")
    if any(not name for name in names):
        raise ValueError(f"COCO-Stuff label-name file contains an empty name: {path}")
    return names


def _leading_unlabeled(name: str) -> bool:
    """Return True when *name* is a conventional void/unlabeled entry."""
    return name.strip().lower() in {
        "unlabeled", "void", "background", "ignore", "ignore_label", "other",
    }


def choose_label_offset(
    pixel_values: np.ndarray,
    label_names: Sequence[str],
    mode: Union[str, int] = "auto",
) -> Tuple[int, str, List[int]]:
    """Choose the offset that maps a label-map pixel value to a name-list index.

    ``offset = name_index - pixel_value``.  A value of 255 is always reserved
    for unlabeled and is ignored while inspecting *pixel_values*.

    In ``"auto"`` mode the decision uses the observed value range and whether
    the name list starts with an unlabeled entry.  The official name list does,
    and the official Caffe-compatible maps are 0-based, so offset +1 is the
    preferred choice when it fits.  If ``max + 1`` would fall outside the name
    list but ``max`` itself fits, offset 0 is selected instead.

    Returns
    -------
    (offset, reason, candidates)
        ``candidates`` is the list of offsets that fit the observed range; more
        than one means the file is ambiguous and the visual ``check-data`` step
        should be inspected before trusting the chosen value.

    Raises
    ------
    ValueError
        If a forced offset places an observed pixel value outside the name list
        or if no offset can explain the data.
    """
    values = np.asarray(pixel_values)
    unique = sorted(
        int(v) for v in np.unique(values).tolist()
        if int(v) != IGNORE_VALUE
    )
    n_names = len(label_names)
    if n_names == 0:
        raise ValueError("cannot choose a label offset from an empty name list")

    if mode != "auto":
        offset = int(mode)
        bad = [v for v in unique if not (0 <= v + offset < n_names)]
        if bad:
            raise ValueError(
                f"label offset {offset} maps pixel values {bad[:5]} outside "
                f"the name list of length {n_names}"
            )
        return offset, f"forced by --label-offset {offset}", [offset]

    first_is_unlabeled = _leading_unlabeled(label_names[0])
    min_value = unique[0] if unique else 0
    max_value = unique[-1] if unique else 0

    candidates: List[int] = []
    # 0-based names / 1-based names without a leading unlabeled entry.
    if 0 <= max_value < n_names:
        candidates.append(0)
    # Caffe-compatible stuffthingmaps: values 0..181, names[0] == unlabeled.
    if first_is_unlabeled and max_value + 1 < n_names:
        candidates.append(1)
    # A name list without unlabeled, fed 1-based class ids.
    if not first_is_unlabeled and min_value >= 1 and max_value - 1 < n_names:
        candidates.append(-1)

    if not candidates:
        raise ValueError(
            f"no label offset explains pixel values {min_value}..{max_value} "
            f"with {n_names} class names"
        )
    if not first_is_unlabeled:
        offset = 0 if 0 in candidates else candidates[0]
        reason = "name list has no leading unlabeled entry; using zero-based ids"
    elif 1 in candidates:
        offset = 1
        reason = (
            "name list starts with an unlabeled entry and the official "
            "Caffe-compatible maps are 0-based (255 = ignore), so class index "
            "is pixel value + 1"
        )
    else:
        offset = 0
        reason = "observed maximum only fits without the leading unlabeled shift"
    return offset, reason, candidates


def label_index(value: int, offset: int) -> int:
    """Convert one PNG pixel value to an index into the class-name list."""
    return int(value) + int(offset)


def class_name_for_value(value: int, label_names: Sequence[str], offset: int) -> Optional[str]:
    """Return the class name for a label-map value, or ``None`` if out of range."""
    index = label_index(value, offset)
    if 0 <= index < len(label_names):
        return label_names[index]
    return None


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def _candidate_dirs(root: Path, split: Optional[str]) -> List[Path]:
    """Return likely directories, accepting roots that already include a split."""
    dirs: List[Path] = []
    if split:
        dirs.append(root / split)
    dirs.append(root)
    return [d for d in dirs if d.is_dir()]


def _find_with_suffixes(root: Path, split: Optional[str], image_id: Union[int, str],
                        suffixes: Sequence[str]) -> Optional[Path]:
    """Find an image/label file by trying padded and unpadded numeric stems."""
    try:
        text = f"{int(image_id):012d}"
    except (TypeError, ValueError):
        text = str(image_id)
    stems = [text]
    try:
        stems.append(str(int(image_id)))
    except (TypeError, ValueError):
        pass
    for directory in _candidate_dirs(root, split):
        for stem in stems:
            for suffix in suffixes:
                candidate = directory / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate
    return None


def find_image_path(images_root: Union[str, Path], split: Optional[str],
                    image_id: Union[int, str]) -> Optional[Path]:
    """Locate a COCO image, accepting roots that do or do not include the split."""
    return _find_with_suffixes(Path(images_root), split, image_id, _IMAGE_SUFFIXES)


def find_label_map_path(labels_root: Union[str, Path], split: Optional[str],
                        image_id: Union[int, str]) -> Optional[Path]:
    """Locate a COCO-Stuff ``stuffthingmaps`` PNG for an image ID."""
    return _find_with_suffixes(Path(labels_root), split, image_id, _LABEL_SUFFIXES)


def list_numeric_stems(directory: Union[str, Path],
                       suffixes: Sequence[str]) -> set:
    """Return the set of integer IDs whose files have one of *suffixes*."""
    directory = Path(directory)
    if not directory.is_dir():
        return set()
    ids = set()
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            try:
                ids.add(int(path.stem))
            except ValueError:
                continue
    return ids


def discover_ids(images_root: Union[str, Path], labels_root: Union[str, Path],
                 split: str) -> List[int]:
    """Return sorted IDs that have both an image and a label map for *split*."""
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    image_dirs = [p for p in (images_root / split, images_root) if p.is_dir()]
    label_dirs = [p for p in (labels_root / split, labels_root) if p.is_dir()]
    image_ids = set()
    label_ids = set()
    for directory in image_dirs[:1]:
        image_ids |= list_numeric_stems(directory, _IMAGE_SUFFIXES)
    for directory in label_dirs[:1]:
        label_ids |= list_numeric_stems(directory, _LABEL_SUFFIXES)
    ids = sorted(image_ids & label_ids)
    if not ids:
        raise FileNotFoundError(
            "no matching COCO images and label maps found for split "
            f"{split!r} under {images_root} and {labels_root}"
        )
    return ids


# ---------------------------------------------------------------------------
# Label-map I/O and visualization
# ---------------------------------------------------------------------------

def load_label_map(path: Union[str, Path]) -> np.ndarray:
    """Load a single PNG label map as a 2-D integer array.

    COCO-Stuff maps are indexed grayscale PNGs.  Palette mode is read as its
    palette indices, which is what the dataset means by a label map.  RGB(A)
    files are reduced to their first channel and a clear error is raised if a
    file contains values outside the byte range expected by this project.
    """
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    array = np.asarray(array)
    if array.dtype.kind not in "iu":
        raise ValueError(f"label map is not integer-valued: {path}")
    if array.size and int(array.max()) > 255:
        raise ValueError(
            f"label map {path} has values above 255; this loader expects "
            "COCO-Stuff byte label maps"
        )
    return array.astype(np.int32, copy=False)


def resize_label_map(label_map: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a label map to ``(H, W)`` with nearest-neighbour interpolation.

    Nearest is required: any smoothing would invent class ids at boundaries.
    PIL is used so the operation is deterministic and dependency-light.
    """
    height, width = int(shape[0]), int(shape[1])
    if label_map.shape == (height, width):
        return label_map.astype(np.int32, copy=False)
    image = Image.fromarray(label_map.astype(np.uint8), mode="L")
    resized = image.resize((width, height), Image.NEAREST)
    return np.asarray(resized, dtype=np.int32)


def _palette_for_value(value: int) -> Tuple[int, int, int]:
    """Deterministic, readable colour for a class id."""
    if value == IGNORE_VALUE:
        return (55, 55, 55)
    hue = (int(value) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def colorize_label_map(label_map: np.ndarray) -> Image.Image:
    """Render a label map with one deterministic colour per class id.

    This is a *visual* aid for ``check-data``, not an evaluation input.  Class
    255 is dark grey so unlabeled regions are easy to spot.
    """
    array = np.asarray(label_map)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for value in range(256):
        lut[value] = _palette_for_value(value)
    safe = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(lut[safe], mode="RGB")


def describe_values(pixel_values: np.ndarray, label_names: Sequence[str],
                    offset: int, limit: int = 30) -> List[str]:
    """Return printable ``value -> name`` lines for the most common ids."""
    values, counts = np.unique(pixel_values, return_counts=True)
    order = np.argsort(-counts)
    lines: List[str] = []
    for index in order[:limit]:
        value = int(values[index])
        name = "unlabeled" if value == IGNORE_VALUE else class_name_for_value(value, label_names, offset)
        lines.append(f"  {value:3d}  {counts[index]:8d} px  {name or '<out of range>'}")
    return lines


def ids_in_label_map(pixel_values: np.ndarray, label_names: Sequence[str],
                     offset: int) -> List[str]:
    """Return ``value:name`` strings for every distinct class in a label map."""
    values = sorted(int(v) for v in np.unique(pixel_values).tolist())
    out = []
    for value in values:
        name = "unlabeled" if value == IGNORE_VALUE else class_name_for_value(value, label_names, offset)
        out.append(f"{value}: {name or '<out of range>'}")
    return out
