"""Visualisation helpers for decompositions and intermediate pipeline stages.

Everything here is PIL / NumPy only so it works in scripts, notebooks, and
headless Kaggle sessions.  Rendering is kept out of ``pipeline.py``: the
pipeline collects arrays through ``ColorizationPipeline.run_traced`` and this
module turns them into PNGs.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .blending import _lab_to_rgb
from .regions import Decomposition

if TYPE_CHECKING:
    from .pipeline import PipelineTrace, RegionTrace

DOMAIN_COLORS = {
    "people": (229, 86, 138),
    "nature": (93, 178, 107),
    "urban": (76, 141, 240),
    "animals": (242, 177, 52),
    "fallback": (142, 142, 147),
}

BACKGROUND = (12, 12, 12)
TILE_SIZE = 240
CAPTION_HEIGHT = 46
PADDING = 8


# ---------------------------------------------------------------------------
# Existing decomposition overlay
# ---------------------------------------------------------------------------

def overlay(image: Image.Image, d: Decomposition, alpha: float = 0.45, boxes: bool = True) -> Image.Image:
    base = np.asarray(image.convert("RGB")).astype(float)
    layer = base.copy()
    for k, name in enumerate(d.domains):
        layer[d.domain_map == k] = DOMAIN_COLORS.get(name, (200, 200, 200))
    out = Image.fromarray(((1 - alpha) * base + alpha * layer).astype(np.uint8))
    if boxes:
        draw = ImageDraw.Draw(out)
        for r in d.regions:
            color = DOMAIN_COLORS.get(r.domain, (255, 255, 255))
            draw.rectangle(list(map(float, r.box)), outline=color, width=2)
            draw.text((r.box[0] + 3, r.box[1] + 2), r.label, fill=(255, 255, 255))
    return out


def domain_map_image(d: Decomposition) -> Image.Image:
    """Pure colour-coded domain map without the underlying image."""
    arr = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    for k, name in enumerate(d.domains):
        arr[d.domain_map == k] = DOMAIN_COLORS.get(name, (200, 200, 200))
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Tile / grid composition
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _short(text: str, limit: int = 34) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: max(1, limit - 3)] + "..."


def _as_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _fit(image, size: int, background: Tuple[int, int, int] = (32, 32, 32)) -> Image.Image:
    img = _as_rgb(image)
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), background)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def tile(image, title: str, subtitle: str = "", size: int = TILE_SIZE) -> Image.Image:
    """One fixed-size image tile with a two-line caption."""
    base = Image.new("RGB", (size, size + CAPTION_HEIGHT), (18, 18, 18))
    base.paste(_fit(image, size), (0, 0))
    draw = ImageDraw.Draw(base)
    draw.text((6, size + 4), _short(title), fill=(238, 238, 238), font=_font(13))
    if subtitle:
        draw.text((6, size + 22), _short(subtitle, 40), fill=(168, 168, 168), font=_font(11))
    return base


def compose_grid(
    tiles: Sequence[Image.Image],
    cols: int = 4,
    title: Optional[str] = None,
    pad: int = PADDING,
) -> Image.Image:
    tiles = list(tiles)
    if not tiles:
        return tile(Image.new("RGB", (64, 64), (32, 32, 32)), title or "(empty)")

    cols = max(1, min(cols, len(tiles)))
    rows = (len(tiles) + cols - 1) // cols
    tile_w, tile_h = tiles[0].size
    header = 42 if title else 0
    width = cols * tile_w + (cols + 1) * pad
    height = header + rows * tile_h + (rows + 1) * pad

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    if title:
        ImageDraw.Draw(canvas).text((pad, 10), title, fill=(235, 235, 235), font=_font(18))

    for i, item in enumerate(tiles):
        row, col = divmod(i, cols)
        x = pad + col * (tile_w + pad)
        y = header + pad + row * (tile_h + pad)
        canvas.paste(item, (x, y))
    return canvas


def compose_sections(
    sections: Sequence[Tuple[str, Sequence[Image.Image]]],
    cols: int = 4,
    title: Optional[str] = None,
    pad: int = PADDING,
) -> Image.Image:
    """Stack several labelled tile grids vertically."""
    grids: List[Tuple[Image.Image, Image.Image]] = []
    for heading, tiles in sections:
        grid = compose_grid(tiles, cols=cols, title=None, pad=pad)
        header = Image.new("RGB", (grid.width, 36), BACKGROUND)
        ImageDraw.Draw(header).text((pad, 8), heading, fill=(220, 220, 220), font=_font(16))
        grids.append((header, grid))

    if not grids:
        return compose_grid([], cols=cols, title=title)

    width = max(header.width for header, _ in grids)
    top = 42 if title else 0
    height = top + sum(header.height + grid.height + pad for header, grid in grids) + pad
    canvas = Image.new("RGB", (width, height), BACKGROUND)

    y = top
    if title:
        ImageDraw.Draw(canvas).text((pad, 10), title, fill=(235, 235, 235), font=_font(18))
    for header, grid in grids:
        canvas.paste(header, (0, y))
        y += header.height
        canvas.paste(grid, (0, y))
        y += grid.height + pad
    return canvas


# ---------------------------------------------------------------------------
# Mask and segment tiles
# ---------------------------------------------------------------------------

def _mask_contour(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary of a boolean mask."""
    mask = mask.astype(bool)
    edge = np.zeros_like(mask)
    edge[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    edge[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    edge[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    return edge


def _apply_contour(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
) -> np.ndarray:
    out = rgb.copy()
    out[_mask_contour(mask)] = color
    return out


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize(size, Image.NEAREST)
    return np.asarray(img) > 127


def mask_tile(
    mask: np.ndarray,
    title: str,
    subtitle: str = "",
    domain: Optional[str] = None,
    size: int = TILE_SIZE,
) -> Image.Image:
    mask = np.asarray(mask, dtype=bool)
    color = DOMAIN_COLORS.get(domain, (255, 255, 255))
    arr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    arr[mask] = color
    arr = _apply_contour(arr, mask, (255, 255, 255))
    return tile(Image.fromarray(arr), title, subtitle, size)


def segment_tile(
    L_crop: np.ndarray,
    mask: np.ndarray,
    title: str,
    subtitle: str = "",
    domain: Optional[str] = None,
    size: int = TILE_SIZE,
    dim: float = 0.35,
) -> Image.Image:
    """Grayscale crop with the segment highlighted and outlined."""
    mask = np.asarray(mask, dtype=bool)
    color = DOMAIN_COLORS.get(domain, (255, 255, 255))
    gray = np.clip(np.asarray(L_crop, dtype=np.float32) * (255.0 / 100.0), 0, 255).astype(np.uint8)
    arr = np.repeat(gray[:, :, None], 3, axis=2)
    arr[~mask] = (arr[~mask].astype(np.float32) * dim).astype(np.uint8)
    arr = _apply_contour(arr, mask, color)
    return tile(Image.fromarray(arr), title, subtitle, size)


def colorized_segment_tile(
    L_crop: np.ndarray,
    ab_pred: np.ndarray,
    mask: np.ndarray,
    title: str,
    subtitle: str = "",
    domain: Optional[str] = None,
    size: int = TILE_SIZE,
    dim: float = 0.35,
) -> Image.Image:
    """The expert's colourisation of one crop, with the segment outlined."""
    mask = np.asarray(mask, dtype=bool)
    color = DOMAIN_COLORS.get(domain, (255, 255, 255))
    lab = np.concatenate(
        [
            np.asarray(L_crop, dtype=np.float32)[:, :, None],
            np.asarray(ab_pred, dtype=np.float32),
        ],
        axis=-1,
    )
    colorized = _lab_to_rgb(lab)

    gray = np.clip(np.asarray(L_crop, dtype=np.float32) * (255.0 / 100.0), 0, 255).astype(np.uint8)
    arr = np.repeat(gray[:, :, None], 3, axis=2)
    arr[~mask] = (arr[~mask].astype(np.float32) * dim).astype(np.uint8)
    arr[mask] = colorized[mask]
    arr = _apply_contour(arr, mask, color)
    return tile(Image.fromarray(arr), title, subtitle, size)


# ---------------------------------------------------------------------------
# Full stage dump
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).lower())
    parts = [part for part in cleaned.split("_") if part]
    return "_".join(parts) or "region"


def _crop_mask(trace: "RegionTrace") -> np.ndarray:
    """The blender's mask for a region, cropped and resized to the expert input."""
    crop = trace.crop
    mask = trace.binary_mask[crop.y0:crop.y1, crop.x0:crop.x1]
    side = trace.L_crop.shape[0]
    if mask.shape != (side, side):
        mask = _resize_mask(mask, (side, side))
    return mask


def _caption(trace: "RegionTrace") -> Tuple[str, str]:
    domain = trace.region.domain or "fallback"
    return f"#{trace.index} {trace.region.label}", f"{domain} · score {trace.region.score:.2f}"


def save_pipeline_stages(
    trace: "PipelineTrace",
    decomp: Decomposition,
    gray: Image.Image,
    out_dir,
    cols: int = 4,
    per_region: bool = True,
) -> Dict[str, Path]:
    """Write the five numbered stage images (and optional per-region files).

    Returns a mapping ``stage name -> path``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    gray_rgb = gray.convert("RGB")

    # 1. Decomposition ------------------------------------------------------
    decomposition_tiles = [
        tile(gray_rgb, "Input (working)", f"{decomp.shape[1]} x {decomp.shape[0]}"),
        tile(domain_map_image(decomp), "Domain map", f"coverage {decomp.coverage * 100:.1f}%"),
        tile(overlay(gray_rgb, decomp), "Overlay", f"{len(decomp.regions)} routed regions"),
    ]
    path = out / "01_decomposition.png"
    compose_grid(decomposition_tiles, cols=3, title="1. Decomposition").save(path)
    paths["decomposition"] = path

    # 2-4. Per-region data --------------------------------------------------
    raw_tiles: List[Image.Image] = []
    resolved_tiles: List[Image.Image] = []
    segment_tiles: List[Image.Image] = []
    colorized_tiles: List[Image.Image] = []

    for region_trace in trace.regions:
        caption, subtitle = _caption(region_trace)
        domain = region_trace.region.domain or "fallback"
        crop_mask = _crop_mask(region_trace)

        raw_tiles.append(mask_tile(region_trace.region.mask, caption, subtitle, domain))
        resolved_tiles.append(mask_tile(region_trace.owner_mask, caption, f"{domain} · resolved", domain))
        segment_tiles.append(segment_tile(region_trace.L_crop, crop_mask, caption, subtitle, domain))
        colorized_tiles.append(
            colorized_segment_tile(
                region_trace.L_crop,
                region_trace.ab_pred,
                crop_mask,
                caption,
                subtitle,
                domain,
            )
        )

    for index, region in enumerate(decomp.discarded):
        raw_tiles.append(
            mask_tile(region.mask, f"discarded {region.label}", f"score {region.score:.2f}")
        )

    path = out / "02_sam_masks.png"
    compose_sections(
        [
            ("Raw SAM masks (used by the blender)", raw_tiles),
            ("Resolved masks (owner map, disjoint)", resolved_tiles),
        ],
        cols=cols,
        title="2. SAM masks",
    ).save(path)
    paths["sam_masks"] = path

    path = out / "03_segments.png"
    compose_grid(segment_tiles, cols=cols, title="3. Segments (crop + mask)").save(path)
    paths["segments"] = path

    path = out / "04_colorized_segments.png"
    compose_grid(colorized_tiles, cols=cols, title="4. Colorized segments").save(path)
    paths["colorized_segments"] = path

    # 5. Final blend --------------------------------------------------------
    path = out / "05_blended.png"
    Image.fromarray(trace.rgb_final).save(path)
    paths["blended"] = path

    # Optional individual files --------------------------------------------
    if per_region:
        region_dir = out / "regions"
        region_dir.mkdir(exist_ok=True)
        for region_trace in trace.regions:
            caption, subtitle = _caption(region_trace)
            domain = region_trace.region.domain or "fallback"
            stem = f"region_{region_trace.index:02d}_{_slug(region_trace.region.label)}_{_slug(domain)}"
            crop_mask = _crop_mask(region_trace)

            mask_tile(region_trace.region.mask, caption, subtitle, domain).save(
                region_dir / f"{stem}_mask.png"
            )
            segment_tile(region_trace.L_crop, crop_mask, caption, subtitle, domain).save(
                region_dir / f"{stem}_segment.png"
            )
            colorized_segment_tile(
                region_trace.L_crop,
                region_trace.ab_pred,
                crop_mask,
                caption,
                subtitle,
                domain,
            ).save(region_dir / f"{stem}_colorized.png")

        for index, region in enumerate(decomp.discarded):
            stem = f"discarded_{index:02d}_{_slug(region.label)}"
            mask_tile(region.mask, f"discarded {region.label}", f"score {region.score:.2f}").save(
                region_dir / f"{stem}_mask.png"
            )

    return paths
