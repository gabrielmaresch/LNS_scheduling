from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rws_instance_loader import load_instance_and_schedule
from run_baseline_benchmark_tabu import _run_tabu_once


DEFAULT_FIELDS = [
    "instance",
    "instance_index",
    "solver_name",
    "timeout_seconds",
    "status",
    "objective",
    "runtime_seconds",
    "timeout_hit",
    "error",
]


def _to_int(value: Any) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _instance_index_from_stem(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None


def _latest_tabu_csv(benchmark_dir: Path) -> Path:
    candidates = sorted(benchmark_dir.glob("baseline_runs_tabu_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no baseline_runs_tabu_*.csv in {benchmark_dir}")
    return candidates[-1]


def _ask_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return default if raw == "" else raw


def _ask_float(prompt: str, default: float, minimum: float = 0.1) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else float(raw)
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _ask_int(prompt: str, default: int, minimum: int = 0) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else int(raw)
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _ask_mode(prompt: str, default: str = "replace") -> str:
    raw = input(f"{prompt} [{default}] (add/replace): ").strip().lower()
    value = default if raw == "" else raw
    if value not in {"add", "replace"}:
        raise ValueError("mode must be add or replace")
    return value


def _read_csv(path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        fieldnames = list(DEFAULT_FIELDS)
    for f in DEFAULT_FIELDS:
        if f not in fieldnames:
            fieldnames.append(f)
    return rows, fieldnames


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(row: Dict[str, Any]) -> tuple[int, str]:
        idx = _to_int(row.get("instance_index"))
        name = str(row.get("instance", ""))
        return (idx if idx is not None else 10**9, name)

    return sorted(rows, key=_key)


def _print_specs(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    timeouts = sorted({round(_to_float(r.get("timeout_seconds")) or 0.0, 6) for r in rows if _to_float(r.get("timeout_seconds")) is not None})
    solver_names = sorted({str(r.get("solver_name", "")).strip() for r in rows if str(r.get("solver_name", "")).strip() != ""})
    finite_obj = sum(1 for r in rows if _to_float(r.get("objective")) is not None)
    timeout_hits = sum(1 for r in rows if _to_bool(r.get("timeout_hit")))
    errors = sum(
        1
        for r in rows
        if str(r.get("status", "")).startswith("ERROR:") or str(r.get("error", "")).strip() != ""
    )
    unique_instances = len({str(r.get("instance", "")).strip() for r in rows if str(r.get("instance", "")).strip() != ""})

    print("\nLoaded benchmark CSV specs:")
    print(f"  path: {csv_path}")
    print(f"  rows: {len(rows)}")
    print(f"  unique instances: {unique_instances}")
    print(f"  timeouts (s): {timeouts if timeouts else 'n/a'}")
    print(f"  solver_name values: {solver_names if solver_names else 'n/a'}")
    print(f"  rows with objective: {finite_obj}")
    print(f"  timeout_hit rows: {timeout_hits}")
    print(f"  error rows: {errors}")


def main() -> None:
    benchmark_dir = Path(__file__).resolve().parent
    default_csv = _latest_tabu_csv(benchmark_dir)
    csv_path = Path(_ask_str("Baseline TABU CSV path", str(default_csv))).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"csv file not found: {csv_path}")

    rows, fieldnames = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"no rows in {csv_path}")
    _print_specs(rows, csv_path)

    default_timeout = _to_float(rows[0].get("timeout_seconds"))
    timeout_seconds = _ask_float("Timeout per solve (seconds)", default_timeout if default_timeout is not None else 60.0)
    mode = _ask_mode("Operation")
    example_number = _ask_int("Instance number (ExampleN)", 0, minimum=0)

    instance_name = f"Example{example_number}"
    instance_path = BASE_DIR / "Instances1-50" / f"{instance_name}.txt"
    if not instance_path.exists():
        raise FileNotFoundError(f"instance file not found: {instance_path}")

    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True,
        initial_schedule="round_robin",
    )
    print(
        f"\nInstance specs: days={instance.num_days} workers={instance.num_workers} "
        f"shifts={len(instance.shift_names) - 1} forbidden={len(instance.forbidden_sequences)}"
    )

    result = _run_tabu_once(
        instance=instance,
        schedule=schedule,
        model_path=BASE_DIR / "rws_instance.mzn",
        timeout_seconds=timeout_seconds,
        solver_name="chuffed",
    )
    new_row: Dict[str, Any] = {
        "instance": instance_name,
        "instance_index": example_number,
        "solver_name": "tabu",
        "timeout_seconds": timeout_seconds,
        "status": result["status"],
        "objective": result["objective"],
        "runtime_seconds": result["runtime_seconds"],
        "timeout_hit": result["timeout_hit"],
        "error": result.get("error", ""),
    }

    if mode == "replace":
        before = len(rows)
        rows = [r for r in rows if str(r.get("instance", "")).strip() != instance_name]
        removed = before - len(rows)
        print(f"Replacing instance={instance_name}: removed {removed} old row(s).")
    else:
        print(f"Adding new row for instance={instance_name}.")

    rows.append(new_row)
    rows = _sort_rows(rows)

    backup_path = csv_path.with_suffix(csv_path.suffix + f".bak_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(csv_path, backup_path)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nUpdated benchmark CSV written.")
    print(f"  updated_file: {csv_path}")
    print(f"  backup_file:  {backup_path}")
    print(
        "  new_row: "
        f"instance={new_row['instance']} objective={new_row['objective']} "
        f"runtime_seconds={new_row['runtime_seconds']:.3f} timeout_hit={new_row['timeout_hit']} "
        f"status={new_row['status']}"
    )


if __name__ == "__main__":
    main()
