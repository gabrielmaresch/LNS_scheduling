from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List
import re
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rws import rws_lns
from rws_instance_loader import load_instance_and_schedule
from rws_mzk_pipeline import build_rws_model_instance
from tabu import tabu


def _instance_index_from_stem(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None


def _instance_sort_key(path: Path) -> tuple[int, str]:
    instance_index = _instance_index_from_stem(path.stem)
    if instance_index is not None:
        return (instance_index, path.stem)
    return (10**9, path.stem)


def _ask_int(prompt: str, default: int, minimum: int = 1) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else int(raw)
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _ask_float(prompt: str, default: float, minimum: float = 0.1) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else float(raw)
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _run_tabu_once(
    instance: Any,
    schedule: Any,
    model_path: Path,
    timeout_seconds: float,
    solver_name: str = "chuffed",
) -> Dict[str, Any]:
    start_total = perf_counter()
    try:
        eval_lns = rws_lns(instance=instance, incumbent=schedule)
        eval_lns._late_phase = False
        eval_lns._late_phase_strict_improvement = False
        model_instance, _ = build_rws_model_instance(
            lns=eval_lns,
            model_path=model_path,
            solver_name=solver_name,
            sloppy=False,
        )

        start_tabu = perf_counter()
        best_schedule = tabu(
            instance=instance,
            schedule=schedule,
            tabu_length=40,
            timeout=timeout_seconds,
            use_objective_tiebreaker=True,
            model_path=model_path,
            model_instance=model_instance,
            solver_name=solver_name,
            objective_tiebreak_timeout_seconds=2,
            objective_tiebreak_max_count=10,
        )
        tabu_runtime_seconds = perf_counter() - start_tabu
        objective = best_schedule.objective_value(
            model_path=model_path,
            model_instance=model_instance,
            solver_name=solver_name,
            timeout_seconds=max(1, min(10, int(timeout_seconds))),
        )
        runtime_seconds = perf_counter() - start_total
        timeout_hit = tabu_runtime_seconds >= timeout_seconds
        return {
            "status": "TABU_DONE",
            "objective": float(objective),
            "runtime_seconds": runtime_seconds,
            "timeout_hit": timeout_hit,
            "error": "",
        }
    except Exception as exc:
        runtime_seconds = perf_counter() - start_total
        return {
            "status": f"ERROR: {exc.__class__.__name__}",
            "objective": None,
            "runtime_seconds": runtime_seconds,
            "timeout_hit": False,
            "error": " ".join(str(exc).split()),
        }


def main() -> None:
    instances_dir = BASE_DIR / "Instances2000"
    instance_paths = sorted(instances_dir.glob("Example*.txt"), key=_instance_sort_key)
    if not instance_paths:
        raise FileNotFoundError(f"No instance files found in {instances_dir}")

    default_n = min(20, len(instance_paths))
    n_instances = _ask_int("Number of instances (first n)", default_n, minimum=1)
    n_instances = min(n_instances, len(instance_paths))
    timeout_seconds = _ask_float("Timeout per solve (seconds)", 60.0, minimum=0.1)

    selected_instances = instance_paths[:n_instances]
    benchmark_dir = Path(__file__).resolve().parent
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_csv = benchmark_dir / f"baseline_runs_tabu_{timestamp}.csv"

    rows: List[Dict[str, Any]] = []
    total_runs = n_instances
    run_counter = 0
    model_path = BASE_DIR / "rws_instance.mzn"

    for instance_path in selected_instances:
        instance, schedule = load_instance_and_schedule(
            file_path=instance_path,
            cyclicity=True,
            initial_schedule="round_robin",
        )
        run_counter += 1
        print(
            f"[run {run_counter}/{total_runs}] start "
            f"running instance={instance_path.stem} solver=tabu timeout={timeout_seconds}s"
        )
        result = _run_tabu_once(
            instance=instance,
            schedule=schedule,
            model_path=model_path,
            timeout_seconds=timeout_seconds,
            solver_name="chuffed",
        )
        row = {
            "instance": instance_path.stem,
            "instance_index": _instance_index_from_stem(instance_path.stem),
            "solver_name": "tabu",
            "timeout_seconds": timeout_seconds,
            "status": result["status"],
            "objective": result["objective"],
            "runtime_seconds": result["runtime_seconds"],
            "timeout_hit": result["timeout_hit"],
            "error": result.get("error", ""),
        }
        rows.append(row)
        print(
            f"[run {run_counter}/{total_runs}] done "
            f"instance={row['instance']} solver=tabu "
            f"runtime={row['runtime_seconds']:.3f}s objective={row['objective']} "
            f"timeout_hit={row['timeout_hit']}"
        )

    fieldnames = [
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
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    solved_count = sum(1 for row in rows if not row["timeout_hit"] and row["objective"] is not None)
    print(f"\nWrote baseline log: {output_csv}")
    print(f"Runs: {len(rows)}")
    print(f"Solved (objective available, not timeout): {solved_count}")
    print(f"Timeouts: {sum(1 for row in rows if row['timeout_hit'])}")
    error_count = sum(1 for row in rows if str(row["status"]).startswith("ERROR:"))
    print(f"Errors: {error_count}")
    print("Data sufficiency note: one tabu run per instance with fixed hyperparameters.")


if __name__ == "__main__":
    main()
