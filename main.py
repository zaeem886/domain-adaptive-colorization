#!/usr/bin/env python3
"""End-to-end entry point for the domain-adaptive colorization pipeline.

Examples
--------
Detect regions with Grounded-SAM, then colorize::

    python main.py --image pics/dog.jpeg \
        --single-checkpoint /path/to/ddcolor_paper_tiny.pth \
        --ddcolor-path /path/to/DDColor \
        --output outputs/dog_colorized.png

Skip detection and colorize the whole image with the fallback expert::

    python main.py --image pics/dog.jpeg --skip-detection \
        --single-checkpoint /path/to/ddcolor_paper_tiny.pth \
        --ddcolor-path /path/to/DDColor

Per-domain checkpoints::

    python main.py --image pics/dog.jpeg --checkpoints-dir checkpoints/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from colorize.config import DecompConfig
from colorize.colorization_config import ColorizationConfig
from colorize.decomposition import GroundedSAM, load_gray
from colorize.experts import ExpertRegistry
from colorize.pipeline import ColorizationPipeline
from colorize.regions import Decomposition
from colorize.routing import route
from colorize.viz import overlay


DOMAIN_FILENAMES = {
    "people": ("people.pth", "ddcolor_people.pth", "people.pt", "ddcolor_people.pt"),
    "nature": ("nature.pth", "ddcolor_nature.pth", "nature.pt", "ddcolor_nature.pt"),
    "urban": ("urban.pth", "ddcolor_urban.pth", "urban.pt", "ddcolor_urban.pt"),
    "animals": ("animals.pth", "ddcolor_animals.pth", "animals.pt", "ddcolor_animals.pt"),
    "fallback": ("fallback.pth", "ddcolor_fallback.pth", "fallback.pt", "ddcolor_fallback.pt"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the domain-adaptive (MoE) colorization pipeline end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output RGB path (default: outputs/<image-stem>_colorized.png).",
    )
    parser.add_argument(
        "--save-domain-map",
        action="store_true",
        help="Also save the routed domain overlay next to the output.",
    )

    # Checkpoints
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--single-checkpoint",
        help="One DDColor checkpoint used for every domain (demo / general model).",
    )
    group.add_argument(
        "--checkpoints-dir",
        help="Directory containing people/nature/urban/animals/fallback checkpoints.",
    )
    parser.add_argument(
        "--ddcolor-path",
        default=os.environ.get("DDCOLOR_PATH"),
        help="Root of the cloned DDColor repo (also read from DDCOLOR_PATH).",
    )

    # Decomposition
    parser.add_argument(
        "--sam-checkpoint",
        default="sam_vit_b.pth",
        help="Path to the SAM ViT-B checkpoint (downloaded if missing).",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Longer image side after Lanczos downscaling.",
    )
    parser.add_argument(
        "--split-merged-labels",
        action="store_true",
        help="Split merged detector labels such as 'person face' by phrase matching.",
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip Grounded-SAM and colorize the whole image with the fallback expert.",
    )

    # Colorization
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0' or 'cpu'.")
    parser.add_argument("--input-size", type=int, default=256, help="Expert input side length S.")
    parser.add_argument("--margin", type=float, default=0.2, help="Context margin rho around each crop.")
    parser.add_argument(
        "--smoothing-width",
        type=float,
        default=10.0,
        help="Gaussian sigma_0 for soft blend masks.",
    )
    parser.add_argument(
        "--no-blending",
        action="store_true",
        help="Use hard binary masks instead of Gaussian-smoothed blending.",
    )
    parser.add_argument(
        "--region-shape",
        choices=("mask", "box"),
        default="mask",
        help="Use the SAM mask or the filled bounding box as the region shape.",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(name: str) -> str:
    """Resolve ``auto`` to a concrete device without requiring torch at import time."""
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_registry(args: argparse.Namespace) -> ExpertRegistry:
    """Map every domain to a checkpoint path."""
    domains = list(DecompConfig().prompts.keys()) + [DecompConfig().fallback]

    if args.single_checkpoint:
        checkpoint = str(Path(args.single_checkpoint).expanduser())
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return {domain: checkpoint for domain in domains}

    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    if not checkpoints_dir.is_dir():
        raise NotADirectoryError(f"Checkpoints directory not found: {checkpoints_dir}")

    registry: ExpertRegistry = {}
    missing: List[str] = []
    for domain in domains:
        candidates: Tuple[str, ...] = DOMAIN_FILENAMES[domain]
        match: Optional[Path] = next(
            (checkpoints_dir / name for name in candidates if (checkpoints_dir / name).is_file()),
            None,
        )
        if match is None:
            missing.append(f"{domain} ({' or '.join(candidates)})")
        else:
            registry[domain] = str(match)

    if missing:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoints_dir} for: {', '.join(missing)}"
        )
    return registry


def default_output_path(image_path: str) -> Path:
    stem = Path(image_path).stem
    return Path("outputs") / f"{stem}_colorized.png"


def load_input(image_path: str, max_side: int) -> Tuple[Image.Image, np.ndarray, Tuple[int, int]]:
    """Return the working grayscale image, its L channel in [0, 100], and the original shape."""
    gray = load_gray(image_path, max_side=max_side)
    with Image.open(image_path) as src:
        original_shape = (src.height, src.width)
    L_full = np.asarray(gray, dtype=np.float32) * (100.0 / 255.0)
    return gray, L_full, original_shape


def decompose(
    gray: Image.Image,
    decomp_cfg: DecompConfig,
    device: str,
    skip_detection: bool,
) -> Decomposition:
    """Run Person A (Grounded-SAM + routing), or build a full fallback decomposition."""
    if skip_detection:
        return route([], (gray.height, gray.width), decomp_cfg)
    return GroundedSAM(cfg=decomp_cfg, device=device)(gray)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.ddcolor_path:
        ddcolor_path = str(Path(args.ddcolor_path).expanduser().resolve())
        if not Path(ddcolor_path).is_dir():
            raise NotADirectoryError(f"DDColor path not found: {ddcolor_path}")
        sys.path.insert(0, ddcolor_path)

    device = resolve_device(args.device)

    decomp_cfg = DecompConfig(
        sam_checkpoint=args.sam_checkpoint,
        max_side=args.max_side,
        split_merged_labels=args.split_merged_labels,
    )
    color_cfg = ColorizationConfig(
        margin=args.margin,
        smoothing_width=args.smoothing_width,
        blending_on=not args.no_blending,
        region_shape=args.region_shape,
        input_size=args.input_size,
    )
    registry = build_registry(args)

    print(f"[input]  {args.image}")
    print(f"[device] {device}")

    gray, L_full, original_shape = load_input(args.image, args.max_side)
    print(f"[image]  working={gray.size[::-1]}  original={original_shape}")

    decomposition = decompose(gray, decomp_cfg, device, args.skip_detection)
    print(
        f"[routing] regions={len(decomposition.regions)}  "
        f"coverage={decomposition.coverage * 100:.1f}%"
    )
    print(f"[routing] domain shares: {decomposition.domain_share}")

    pipeline = ColorizationPipeline(registry=registry, cfg=color_cfg, device=device)
    rgb = pipeline.run(
        L_full=L_full,
        decomp=decomposition,
        original_shape=original_shape,
    )

    output_path = Path(args.output).expanduser() if args.output else default_output_path(args.image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)
    print(f"[output] {output_path}")

    if args.save_domain_map:
        domain_map_path = output_path.with_name(f"{output_path.stem}_domains.png")
        overlay(gray.convert("RGB"), decomposition).save(domain_map_path)
        print(f"[output] {domain_map_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
