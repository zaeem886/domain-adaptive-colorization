"""Experiment E3: DDColor-T/L whole-image baselines.

This module is the reproducible E3 entry point.  It evaluates both whole-image
DDColor variants on the fixed domain and mixed-scene COCO sets:

* 20 people, nature, urban, and animals images (80 total)
* 20 mixed-scene images

The per-image measurements use the common five-column results format from
``colorize.results``.  A second CSV contains means and standard deviations for
each fixed set, so domain and mixed-scene results can be compared without
reconstructing set membership from image IDs.

Example::

    python experiments/e3.py run \
        --coco-root /data/coco \
        --ddcolor-path /data/DDColor \
        --checkpoint-t /data/ddcolor_paper_tiny.pth \
        --checkpoint-l /data/ddcolor_paper.pth

The runner deliberately does not import torch or DDColor until inference is
started by ``colorize.e3_baseline``.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Allow ``python experiments/e3.py`` from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colorize.e3_baseline import run_e3_baseline
from colorize.metrics import compute_fid
from colorize.results import ResultsRecord, load_results
from colorize.test_sets import DOMAIN_SET_IDS, MIXED_SCENE_IDS, get_coco_image_path

logger = logging.getLogger(__name__)

E3_SET_IDS: Mapping[str, Sequence[int]] = {
    "domain_people": tuple(DOMAIN_SET_IDS["people"]),
    "domain_nature": tuple(DOMAIN_SET_IDS["nature"]),
    "domain_urban": tuple(DOMAIN_SET_IDS["urban"]),
    "domain_animals": tuple(DOMAIN_SET_IDS["animals"]),
    "mixed_scene": tuple(MIXED_SCENE_IDS),
}
E3_SET_NAMES: Tuple[str, ...] = tuple(E3_SET_IDS)
E3_CONDITIONS: Mapping[str, Tuple[str, int]] = {
    "ddcolor_t_whole": ("checkpoint_t", 256),
    "ddcolor_l_whole": ("checkpoint_l", 512),
}


def get_e3_set_ids() -> Dict[str, Sequence[int]]:
    """Return the active E3 set IDs, preferring outputs/e3_ids.json if present."""
    ids_file = Path("outputs/e3_ids.json")
    if ids_file.is_file():
        try:
            with ids_file.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            if all(k in saved for k in E3_SET_NAMES):
                return {k: tuple(saved[k]) for k in E3_SET_NAMES}
        except Exception:
            pass
    from colorize.test_sets import DOMAIN_SET_IDS, MIXED_SCENE_IDS
    return {
        "domain_people": tuple(DOMAIN_SET_IDS["people"]),
        "domain_nature": tuple(DOMAIN_SET_IDS["nature"]),
        "domain_urban": tuple(DOMAIN_SET_IDS["urban"]),
        "domain_animals": tuple(DOMAIN_SET_IDS["animals"]),
        "mixed_scene": tuple(MIXED_SCENE_IDS),
    }


def build_e3_image_manifest(
    coco_root: Union[str, Path],
    split: str = "val2017",
    set_names: Optional[Iterable[str]] = None,
    allow_missing: bool = False,
) -> Tuple[List[Tuple[int, str]], Dict[int, Tuple[str, ...]], Dict[str, Tuple[int, ...]]]:
    """Build the deterministic E3 image list and set-membership mappings.

    Parameters
    ----------
    coco_root:
        COCO root containing ``split`` (normally ``val2017``).
    split:
        COCO image split directory.
    set_names:
        Subset names from :data:`E3_SET_NAMES`; all five are used by default.
    allow_missing:
        If false, fail before inference when a requested image is absent.  If
        true, absent files are omitted and reported through the returned
        ``missing`` mapping.

    Returns
    -------
    image_paths:
        Deduplicated ``(image_id, path)`` pairs in fixed set order.
    image_sets:
        Mapping from image ID to the set(s) containing it.
    missing:
        Mapping from set name to missing image IDs.  This is empty for a
        complete COCO val2017 installation.
    """
    selected = tuple(set_names) if set_names is not None else E3_SET_NAMES
    unknown = [name for name in selected if name not in E3_SET_IDS]
    if unknown:
        raise ValueError(f"Unknown E3 set(s): {unknown}; choose from {list(E3_SET_NAMES)}")
    if not selected:
        raise ValueError("At least one E3 set is required")

    paths: List[Tuple[int, str]] = []
    image_sets: Dict[int, List[str]] = defaultdict(list)
    missing: Dict[str, List[int]] = {name: [] for name in selected}
    seen: set[int] = set()
    set_id_map = get_e3_set_ids()

    for set_name in selected:
        for image_id in set_id_map.get(set_name, ()):
            image_id = int(image_id)
            path = get_coco_image_path(image_id, coco_root, split=split)
            if not Path(path).is_file():
                missing[set_name].append(image_id)
                continue
            image_sets[image_id].append(set_name)
            if image_id not in seen:
                paths.append((image_id, path))
                seen.add(image_id)

    missing = {name: tuple(ids) for name, ids in missing.items() if ids}
    if missing and not allow_missing:
        details = "; ".join(
            f"{name}: {len(ids)} missing (first IDs: {list(ids[:5])})"
            for name, ids in missing.items()
        )
        raise FileNotFoundError(
            f"E3 requires all requested COCO images under "
            f"{Path(coco_root) / split}; {details}. "
            "Use --allow-missing only for a deliberately partial run."
        )

    return (
        paths,
        {image_id: tuple(names) for image_id, names in image_sets.items()},
        missing,
    )


def write_e3_manifest(
    output_csv: Union[str, Path],
    image_paths: Sequence[Tuple[int, str]],
    image_sets: Mapping[int, Sequence[str]],
    missing: Mapping[str, Sequence[int]],
) -> None:
    """Write the exact E3 image/set manifest used by a run."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id, path in image_paths:
        for set_name in image_sets.get(int(image_id), ()):
            rows.append((int(image_id), set_name, str(path), "present"))
    for set_name, image_ids in missing.items():
        for image_id in image_ids:
            rows.append((int(image_id), set_name, "", "missing"))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "set", "path", "status"))
        writer.writerows(sorted(rows, key=lambda row: (row[1], row[0])))


def write_e3_summary(
    output_csv: Union[str, Path],
    records: Sequence[ResultsRecord],
    image_sets: Mapping[int, Sequence[str]],
) -> None:
    """Write per-set count/mean/std summaries for the E3 records."""
    values: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for record in records:
        try:
            image_id = int(record.image_id)
        except ValueError:
            continue
        for set_name in image_sets.get(image_id, ()):
            values[(record.condition, set_name, record.metric)].append(float(record.value))

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("experiment", "condition", "set", "metric", "count", "mean", "std"))
        for condition in E3_CONDITIONS:
            for set_name in E3_SET_NAMES:
                metric_names = sorted(
                    metric
                    for cond, current_set, metric in values
                    if cond == condition and current_set == set_name
                )
                for metric in metric_names:
                    metric_values = values[(condition, set_name, metric)]
                    mean = sum(metric_values) / len(metric_values)
                    variance = sum((value - mean) ** 2 for value in metric_values) / len(metric_values)
                    writer.writerow(
                        (
                            "E3",
                            condition,
                            set_name,
                            metric,
                            len(metric_values),
                            f"{mean:.10g}",
                            f"{variance ** 0.5:.10g}",
                        )
                    )


def write_e3_fid_summary(
    output_csv: Union[str, Path],
    features_by_condition: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]],
    image_sets: Mapping[int, Sequence[str]],
) -> None:
    """Write one Inception-feature FID value per model and fixed image set."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("experiment", "condition", "set", "metric", "count", "value"))
        available_sets = {
            set_name
            for memberships in image_sets.values()
            for set_name in memberships
        }
        for condition, feature_map in features_by_condition.items():
            for set_name in E3_SET_NAMES:
                if set_name not in available_sets:
                    continue
                ids = [
                    str(image_id) for image_id, memberships in image_sets.items()
                    if set_name in memberships and str(image_id) in feature_map
                ]
                if len(ids) < 2:
                    raise RuntimeError(
                        f"FID needs at least two completed images for {condition}/{set_name}; "
                        f"found {len(ids)}"
                    )
                real = np.stack([feature_map[image_id][0] for image_id in ids])
                fake = np.stack([feature_map[image_id][1] for image_id in ids])
                value = compute_fid(real, fake)
                if not np.isfinite(value):
                    raise RuntimeError(f"FID is non-finite for {condition}/{set_name}")
                writer.writerow(("E3", condition, set_name, "fid", len(ids), f"{value:.10g}"))


def run_e3_experiment(
    coco_root: Union[str, Path],
    ddcolor_path: Union[str, Path],
    checkpoint_t: Union[str, Path],
    checkpoint_l: Union[str, Path],
    output_csv: Union[str, Path] = "outputs/e3_results.csv",
    summary_csv: Union[str, Path] = "outputs/e3_summary.csv",
    fid_csv: Union[str, Path] = "outputs/e3_fid.csv",
    manifest_csv: Union[str, Path] = "outputs/e3_manifest.csv",
    split: str = "val2017",
    set_names: Optional[Iterable[str]] = None,
    variants: Optional[Iterable[str]] = None,
    device: str = "cuda",
    gt_root: Optional[Union[str, Path]] = None,
    allow_missing: bool = True,
    overwrite: bool = False,
) -> List[ResultsRecord]:
    """Run DDColor-T and DDColor-L on the complete E3 image set.

    ``overwrite`` only removes the explicitly requested result CSV before the
    run; the manifest and summary are always regenerated after inference.
    """
    selected_variants = tuple(variants) if variants is not None else tuple(E3_CONDITIONS)
    unknown = [variant for variant in selected_variants if variant not in E3_CONDITIONS]
    if unknown:
        raise ValueError(
            f"Unknown E3 variant(s): {unknown}; choose from {list(E3_CONDITIONS)}"
        )
    if not selected_variants:
        raise ValueError("At least one E3 variant is required")

    image_paths, image_sets, missing = build_e3_image_manifest(
        coco_root=coco_root,
        split=split,
        set_names=set_names,
        allow_missing=allow_missing,
    )
    if not image_paths:
        raise RuntimeError("No E3 images were found; refusing to run inference")

    output_path = Path(output_csv)
    if overwrite and output_path.exists():
        output_path.unlink()

    write_e3_manifest(manifest_csv, image_paths, image_sets, missing)
    checkpoints = {
        "checkpoint_t": str(checkpoint_t),
        "checkpoint_l": str(checkpoint_l),
    }
    features_by_condition: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for variant in selected_variants:
        checkpoint_key, input_size = E3_CONDITIONS[variant]
        logger.info(
            "Running %s on %d images (input_size=%d)",
            variant,
            len(image_paths),
            input_size,
        )
        condition_features: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        run_e3_baseline(
            image_paths=image_paths,
            checkpoint=checkpoints[checkpoint_key],
            ddcolor_path=ddcolor_path,
            output_csv=output_path,
            condition=variant,
            device=device,
            input_size=input_size,
            experiment="E3",
            gt_root=gt_root if gt_root is not None else coco_root,
            fid_feature_records=condition_features,
        )
        features_by_condition[variant] = condition_features

    records = load_results(output_path)
    write_e3_summary(summary_csv, records, image_sets)
    write_e3_fid_summary(fid_csv, features_by_condition, image_sets)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E3 DDColor-T/L whole-image baselines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run both whole-image baselines")
    run.add_argument("--coco-root", required=True, help="COCO root containing val2017/")
    run.add_argument("--ddcolor-path", required=True, help="cloned DDColor repository")
    run.add_argument("--checkpoint-t", required=True, help="DDColor-T checkpoint")
    run.add_argument("--checkpoint-l", required=True, help="DDColor-L checkpoint")
    run.add_argument("--split", default="val2017")
    run.add_argument("--device", default="cuda")
    run.add_argument("--output-csv", default="outputs/e3_results.csv")
    run.add_argument("--summary-csv", default="outputs/e3_summary.csv")
    run.add_argument("--fid-csv", default="outputs/e3_fid.csv")
    run.add_argument("--manifest-csv", default="outputs/e3_manifest.csv")
    run.add_argument(
        "--sets",
        nargs="+",
        choices=E3_SET_NAMES,
        default=list(E3_SET_NAMES),
        help="E3 sets to evaluate (default: all domain and mixed-scene sets)",
    )
    run.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(E3_CONDITIONS),
        default=list(E3_CONDITIONS),
        help="DDColor variants to evaluate (default: T and L)",
    )
    run.add_argument(
        "--allow-missing",
        action="store_true",
        help="run on available images instead of failing on missing COCO files",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the existing per-image results CSV before running",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        records = run_e3_experiment(
            coco_root=args.coco_root,
            ddcolor_path=args.ddcolor_path,
            checkpoint_t=args.checkpoint_t,
            checkpoint_l=args.checkpoint_l,
            output_csv=args.output_csv,
            summary_csv=args.summary_csv,
            fid_csv=args.fid_csv,
            manifest_csv=args.manifest_csv,
            split=args.split,
            set_names=args.sets,
            variants=args.variants,
            device=args.device,
            allow_missing=args.allow_missing,
            overwrite=args.overwrite,
        )
        print(f"E3 complete: {len(records)} metric records written to {args.output_csv}")
        print(f"E3 summary written to {args.summary_csv}")
        print(f"E3 FID summary written to {args.fid_csv}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
