"""Results CSV handling for domain-adaptive colorization experiments (Person C).

Provides a simple, pandas-optional record format for collecting per-image
per-metric evaluation results.

CSV column order: experiment, condition, image_id, metric, value
"""
from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass, fields, astuple
from pathlib import Path
from typing import List, Union

try:
    import pandas as pd  # type: ignore[import]
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = ["experiment", "condition", "image_id", "metric", "value"]


@dataclass
class ResultsRecord:
    """A single metric measurement for one image in one experimental condition.

    Fields
    ------
    experiment:
        Experiment identifier, e.g. ``"E3"``.
    condition:
        Condition/variant name, e.g. ``"ddcolor_t_whole"``.
    image_id:
        Unique image identifier (COCO image ID or file stem).
    metric:
        Metric name, e.g. ``"psnr"``, ``"ssim"``, ``"colorfulness"``.
    value:
        Measured metric value as a float.
    """

    experiment: str
    condition: str
    image_id: str
    metric: str
    value: float


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_results(records: List[ResultsRecord], path: Union[str, Path]) -> None:
    """Save a list of :class:`ResultsRecord` objects to a CSV file.

    Parameters
    ----------
    records:
        Records to write.  An empty list produces an empty CSV with headers.
    path:
        Destination file path.  Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "experiment": rec.experiment,
                    "condition": rec.condition,
                    "image_id": rec.image_id,
                    "metric": rec.metric,
                    "value": rec.value,
                }
            )


def load_results(path: Union[str, Path]) -> List[ResultsRecord]:
    """Load a results CSV into a list of :class:`ResultsRecord` objects.

    Parameters
    ----------
    path:
        Path to the CSV file created by :func:`save_results` or
        :func:`append_result`.

    Returns
    -------
    list of ResultsRecord
        Empty list if the file is empty or only contains a header row.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    records: List[ResultsRecord] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                ResultsRecord(
                    experiment=row["experiment"],
                    condition=row["condition"],
                    image_id=row["image_id"],
                    metric=row["metric"],
                    value=float(row["value"]),
                )
            )
    return records


def append_result(path: Union[str, Path], record: ResultsRecord) -> None:
    """Append a single :class:`ResultsRecord` to *path*, creating it if necessary.

    If *path* does not exist, it is created with a header row.  If it already
    exists, the record is appended without re-writing the header.

    Parameters
    ----------
    path:
        Destination CSV file path.
    record:
        The record to append.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "experiment": record.experiment,
                "condition": record.condition,
                "image_id": record.image_id,
                "metric": record.metric,
                "value": record.value,
            }
        )


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------

def records_to_dataframe(records: List[ResultsRecord]):
    """Convert a list of :class:`ResultsRecord` to a pandas DataFrame (if available).

    Parameters
    ----------
    records:
        Input records.

    Returns
    -------
    pandas.DataFrame or list of dict
        A ``DataFrame`` with columns matching the CSV fieldnames when pandas
        is installed, otherwise a list of plain ``dict`` objects.
    """
    rows = [
        {
            "experiment": r.experiment,
            "condition": r.condition,
            "image_id": r.image_id,
            "metric": r.metric,
            "value": r.value,
        }
        for r in records
    ]
    if _HAS_PANDAS:
        return pd.DataFrame(rows, columns=_CSV_FIELDNAMES)
    warnings.warn(
        "pandas is not installed; records_to_dataframe returns a list of dicts. "
        "Install with: pip install pandas",
        stacklevel=2,
    )
    return rows

