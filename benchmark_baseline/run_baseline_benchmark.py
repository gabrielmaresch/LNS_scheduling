from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List
import sys
import re
import warnings

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rws import rws_lns
from rws_instance_loader import load_instance_and_schedule
from rws_mzk_pipeline import build_rws_model_instance

try:
    from minizinc import MiniZincWarning
except Exception:  # pragma: no cover - keeps script import-safe
    MiniZincWarning = None


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


def _extract_objective(result: Any) -> float | None:
    solver_objective = float(result.objective) if result.objective is not None else None
    if not result.status.has_solution() or result.solution is None:
        return solver_objective
    solution = result.solution.__dict__
    legacy_objective = solution.get("legacy_objective")
    if legacy_objective is not None:
        return float(legacy_objective)
    return solver_objective


def _solve_once(
    model_instance: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    with model_instance.branch() as run_instance:
        run_instance["num_fixed_assignments"] = 0
        run_instance["fixed_assignments"] = []
        start = perf_counter()
        try:
            result = run_instance.solve(timeout=timedelta(seconds=timeout_seconds))
        except Exception as exc:  # MiniZincError or solver-side failure
            runtime_seconds = perf_counter() - start
            status = f"ERROR: {exc.__class__.__name__}"
            error_text = " ".join(str(exc).split())
            return {
                "status": status,
                "has_solution": False,
                "objective": None,
                "runtime_seconds": runtime_seconds,
                "timeout_hit": False,
                "error": error_text,
            }
        runtime_seconds = perf_counter() - start

    status = str(result.status)
    status_upper = status.upper()
    status_indicates_timeout = ("TIMEOUT" in status_upper) or ("UNKNOWN" in status_upper)
    runtime_exceeded_budget = runtime_seconds >= timeout_seconds
    has_proven_optimality = "OPTIMAL_SOLUTION" in status_upper

    # For optimization problems, status can be SATISFIED when a time limit is hit
    # after finding an incumbent. Treat over-budget runs as timeout unless optimal.
    timeout_hit = status_indicates_timeout or (
        runtime_exceeded_budget and not has_proven_optimality
    )
    return {
        "status": status,
        "has_solution": result.status.has_solution(),
        "objective": _extract_objective(result),
        "runtime_seconds": runtime_seconds,
        "timeout_hit": timeout_hit,
        "error": "",
    }


def main() -> None:
    if MiniZincWarning is not None:
        warnings.filterwarnings("ignore", category=MiniZincWarning)

    instances_dir = BASE_DIR / "Instances1-50"
    instance_paths = sorted(instances_dir.glob("Example*.txt"), key=_instance_sort_key)
    if not instance_paths:
        raise FileNotFoundError(f"No instance files found in {instances_dir}")

    default_n = min(20, len(instance_paths))
    n_instances = _ask_int("Number of instances (first n)", default_n, minimum=1)
    n_instances = min(n_instances, len(instance_paths))
    timeout_seconds = _ask_float("Timeout per solve (seconds)", 60.0, minimum=0.1)
    solver_names = ["chuffed", "gecode"]

    selected_instances = instance_paths[:n_instances]
    benchmark_dir = Path(__file__).resolve().parent
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_csv = benchmark_dir / f"baseline_runs_{timestamp}.csv"

    rows: List[Dict[str, Any]] = []
    total_runs = n_instances * len(solver_names)
    run_counter = 0

    for instance_path in selected_instances:
        instance, schedule = load_instance_and_schedule(
            file_path=instance_path,
            cyclicity=True,
            initial_schedule="round_robin",
        )
        for solver_name in solver_names:
            run_counter += 1
            print(
                f"[run {run_counter}/{total_runs}] start "
                f"running instance={instance_path.stem} solver={solver_name} timeout={timeout_seconds}s"
            )
            lns = rws_lns(instance=instance, incumbent=schedule)
            # Baseline must stay in non-late-phase mode regardless of other experiments.
            lns._late_phase = False
            lns._late_phase_strict_improvement = False
            model_instance, _ = build_rws_model_instance(
                lns=lns,
                model_path=BASE_DIR / "rws_instance.mzn",
                solver_name=solver_name,
                sloppy=False,
            )
            result = _solve_once(
                model_instance=model_instance,
                timeout_seconds=timeout_seconds,
            )
            row = {
                "instance": instance_path.stem,
                "instance_index": _instance_index_from_stem(instance_path.stem),
                "solver_name": solver_name,
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
                f"instance={row['instance']} solver={solver_name} "
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
    print("Data sufficiency note: deterministic baseline with one run per solver per instance.")


if __name__ == "__main__":
    main()
