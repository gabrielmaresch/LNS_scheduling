from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import statistics
from typing import Any


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _instance_sort_key(name: str, index_hint: int | None) -> tuple[int, str]:
    if index_hint is not None:
        return index_hint, name
    m = re.search(r"(\d+)$", name)
    if m:
        return int(m.group(1)), name
    return 10**9, name


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    return float(statistics.pstdev(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate TABU benchmark CSVs into one summary CSV."
    )
    parser.add_argument(
        "--glob",
        default="baseline_runs_tabu_*.csv",
        help="Input CSV glob under benchmark_baseline.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV path (default: timestamped file in this folder).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_files = sorted(base_dir.glob(args.glob))
    if not input_files:
        raise FileNotFoundError(f"No input files found for pattern: {args.glob}")

    rows: list[dict[str, Any]] = []
    for csv_path in input_files:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row = dict(raw)
                row["__source_file"] = csv_path.name
                row["instance"] = str(raw.get("instance", "")).strip()
                row["instance_index"] = _parse_int(raw.get("instance_index"))
                row["objective"] = _parse_float(raw.get("objective"))
                row["runtime_seconds"] = _parse_float(raw.get("runtime_seconds"))
                row["timeout_hit"] = _parse_bool(raw.get("timeout_hit"))
                row["status"] = str(raw.get("status", "")).strip()
                row["error"] = str(raw.get("error", "")).strip()
                rows.append(row)

    if not rows:
        raise ValueError("No benchmark rows found in input files.")

    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        instance = row["instance"] or "UNKNOWN"
        by_instance.setdefault(instance, []).append(row)

    sorted_instances = sorted(
        by_instance.keys(),
        key=lambda name: _instance_sort_key(
            name,
            _parse_int(by_instance[name][0].get("instance_index")),
        ),
    )

    output_rows: list[dict[str, Any]] = []
    all_objectives: list[float] = []
    all_runtimes: list[float] = []
    total_timeout = 0
    total_errors = 0
    total_solved = 0
    total_runs = 0
    per_instance_means: list[float] = []

    for instance in sorted_instances:
        items = by_instance[instance]
        n_runs = len(items)
        objectives = [x["objective"] for x in items if x["objective"] is not None]
        runtimes = [x["runtime_seconds"] for x in items if x["runtime_seconds"] is not None]
        timeout_count = sum(1 for x in items if bool(x["timeout_hit"]))
        error_count = sum(
            1 for x in items if str(x["status"]).startswith("ERROR:") or bool(x["error"])
        )
        solved_count = sum(
            1
            for x in items
            if x["objective"] is not None and not bool(x["timeout_hit"]) and not str(x["status"]).startswith("ERROR:")
        )

        instance_mean = _safe_mean(objectives)
        if instance_mean is not None:
            per_instance_means.append(instance_mean)

        all_objectives.extend(objectives)
        all_runtimes.extend(runtimes)
        total_timeout += timeout_count
        total_errors += error_count
        total_solved += solved_count
        total_runs += n_runs

        output_rows.append(
            {
                "row_type": "instance",
                "instance": instance,
                "instance_index": _parse_int(items[0].get("instance_index")),
                "runs": n_runs,
                "objective_count": len(objectives),
                "objective_mean": instance_mean,
                "objective_median": _safe_median(objectives),
                "objective_std": _safe_std(objectives),
                "objective_min": min(objectives) if objectives else None,
                "objective_max": max(objectives) if objectives else None,
                "runtime_mean_seconds": _safe_mean(runtimes),
                "runtime_median_seconds": _safe_median(runtimes),
                "timeout_count": timeout_count,
                "timeout_rate": (timeout_count / n_runs) if n_runs > 0 else None,
                "error_count": error_count,
                "error_rate": (error_count / n_runs) if n_runs > 0 else None,
                "solved_count": solved_count,
                "solved_rate": (solved_count / n_runs) if n_runs > 0 else None,
                "input_files_count": len({x["__source_file"] for x in items}),
            }
        )

    output_rows.append(
        {
            "row_type": "overall",
            "instance": "ALL",
            "instance_index": "",
            "runs": total_runs,
            "objective_count": len(all_objectives),
            "objective_mean": _safe_mean(all_objectives),
            "objective_median": _safe_median(all_objectives),
            "objective_std": _safe_std(all_objectives),
            "objective_min": min(all_objectives) if all_objectives else None,
            "objective_max": max(all_objectives) if all_objectives else None,
            "runtime_mean_seconds": _safe_mean(all_runtimes),
            "runtime_median_seconds": _safe_median(all_runtimes),
            "timeout_count": total_timeout,
            "timeout_rate": (total_timeout / total_runs) if total_runs > 0 else None,
            "error_count": total_errors,
            "error_rate": (total_errors / total_runs) if total_runs > 0 else None,
            "solved_count": total_solved,
            "solved_rate": (total_solved / total_runs) if total_runs > 0 else None,
            "input_files_count": len(input_files),
            "instance_count": len(sorted_instances),
            "instance_mean_objective_avg": _safe_mean(per_instance_means),
        }
    )

    if args.output.strip():
        output_path = Path(args.output).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = base_dir / f"baseline_runs_tabu_aggregate_{stamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_type",
        "instance",
        "instance_index",
        "runs",
        "objective_count",
        "objective_mean",
        "objective_median",
        "objective_std",
        "objective_min",
        "objective_max",
        "runtime_mean_seconds",
        "runtime_median_seconds",
        "timeout_count",
        "timeout_rate",
        "error_count",
        "error_rate",
        "solved_count",
        "solved_rate",
        "input_files_count",
        "instance_count",
        "instance_mean_objective_avg",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Loaded TABU CSVs: {len(input_files)}")
    print(f"Total input rows: {len(rows)}")
    print(f"Wrote aggregate CSV: {output_path}")


if __name__ == "__main__":
    main()
