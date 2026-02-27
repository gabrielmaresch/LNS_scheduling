from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path
import random
import re
from time import perf_counter
from typing import Any, Callable, Dict, List
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from multiarm_bandit import (
    MBandit,
    _make_destroy_forbidden_sequences,
    _make_destroy_max_border_operators,
    _make_destroy_random_days,
    _make_destroy_random_window,
    _make_destroy_random_workers,
    _make_destroy_streak_around_worst_worker,
    _make_destroy_worst_days,
    _make_destroy_worst_workers,
    _make_repair_operator,
)
from rws import rws_lns
from rws_instance_loader import load_instance_and_schedule


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


def _ask_variant_mode() -> list[str]:
    raw = input("ALNS variant [both/plain/late_phase] [both]: ").strip().lower()
    mode = "both" if raw == "" else raw
    if mode in {"both", "all"}:
        return ["alns_plain", "alns_late_phase"]
    if mode in {"plain", "alns_plain"}:
        return ["alns_plain"]
    if mode in {"late_phase", "late", "alns_late_phase"}:
        return ["alns_late_phase"]
    raise ValueError("variant must be one of: both, plain, late_phase")


def _build_operators(
    instance: Any,
    model_path: Path,
) -> tuple[
    Dict[str, Callable[[rws_lns], list[tuple[int, int]]]],
    Dict[str, Callable[[rws_lns], None]],
]:
    repair_ops: Dict[str, Callable[[rws_lns], None]] = {
        "repair_chuffed_fast": _make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": _make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": _make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": _make_repair_operator(model_path, "gecode", 15),
    }
    max_border_ops = _make_destroy_max_border_operators(instance)
    destroy_ops: Dict[str, Callable[[rws_lns], list[tuple[int, int]]]] = {
        "destroy_worst_workers_10pct": _make_destroy_worst_workers(0.10),
        "destroy_worst_workers_30pct": _make_destroy_worst_workers(0.30),
        "destroy_random_workers_20pct": _make_destroy_random_workers(0.20),
        "destroy_random_window_20pct": _make_destroy_random_window(0.20),
        "destroy_all_forbidden_sequences": _make_destroy_forbidden_sequences(1),
        "destroy_streak_around_worst_worker": _make_destroy_streak_around_worst_worker(
            binomial_p=0.3,
        ),
        "destroy_worst_days_10pct": _make_destroy_worst_days(0.10),
        "destroy_worst_days_30pct": _make_destroy_worst_days(0.30),
        "destroy_random_days_20pct": _make_destroy_random_days(0.20),
        **max_border_ops,
    }
    return destroy_ops, repair_ops


def _configure_variant(mab: MBandit, variant: str) -> None:
    if variant == "alns_plain":
        mab.equal_move_allowed_freezeout = 0
        mab.late_phase_strict_improvement = False
        return
    if variant != "alns_late_phase":
        raise ValueError(f"unsupported variant: {variant}")
    for name in (
        "destroy_random_workers_20pct",
        "destroy_random_days_20pct",
        "destroy_random_window_20pct",
        "destroy_worst_workers_30pct",
        "destroy_worst_days_30pct",
    ):
        if name in mab.exclude_in_late_phase_destroy:
            mab.exclude_in_late_phase_destroy[name] = True


def _run_single(
    instance_path: Path,
    seed: int,
    timeout_seconds: float,
    variant: str,
) -> tuple[Dict[str, Any], str]:
    random.seed(seed)
    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True,
        initial_schedule="round_robin",
    )
    model_path = BASE_DIR / "rws_instance.mzn"
    destroy_ops, repair_ops = _build_operators(
        instance=instance,
        model_path=model_path,
    )
    mab = MBandit(
        instance=instance,
        schedule=schedule,
        model_path=model_path,
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
        global_timeout_seconds=timeout_seconds,
        minizinc_timeout_seconds=60,
    )
    _configure_variant(mab, variant)

    for repair_op in mab.weights_repair:
        if "fast" in repair_op:
            mab.weights_repair[repair_op] = 0.4
        elif "long" in repair_op:
            mab.weights_repair[repair_op] = 0.1

    loop_start = perf_counter()
    timed_out = False
    solved = False
    last_iteration = 0
    error = ""
    log_lines: list[str] = []
    instance_name = instance_path.stem
    log_lines.append(f"Loaded instance: {instance_path.name}")
    log_lines.append(f"instance={instance_name}")
    log_lines.append(f"instance_name={instance_name}")
    log_lines.append(f"instance_file={instance_path.name}")
    log_lines.append(f"instance_path={instance_path}")
    log_lines.append(f"variant={variant}")
    log_lines.append(f"seed={seed}")
    log_lines.append("")

    while True:
        elapsed_before = perf_counter() - loop_start
        if elapsed_before >= mab.global_timeout_seconds:
            timed_out = True
            break
        step_start = perf_counter()
        try:
            step = mab._perform_lns_step()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log_lines.append(f"fatal_error={error}")
            break
        step_runtime = perf_counter() - step_start
        elapsed_total = perf_counter() - loop_start
        last_iteration = int(step["iteration"])
        destroyed_label = step["destroyed_display"]
        log_lines.append(
            (
                f"iter={step['iteration']} "
                f"elapsed={elapsed_total:.3f}s "
                f"step_runtime={step_runtime:.3f}s "
                f"destroy={step['selected_destroy_operator']} "
                f"repair={step['selected_repair_operator']} "
                f"objective_metric=minizinc_objective "
                f"incumbent_objective={step['incumbent_objective']} "
                f"contender_objective={step['contender_objective']} "
                f"score={step['contender_score']} "
                f"accepted={step['accepted']} "
                f"repair_failed={step['repair_failed']} "
                f"used_exploration={step['used_exploration']} "
                f"stagnation_rounds={step['stagnation_rounds']} "
                f"weights_updated={step['weights_updated']}"
            )
        )
        if step["repair_error"] is not None:
            log_lines.append(f"  repair_error={step['repair_error']}")
        log_lines.append(f"  destroyed={destroyed_label}")
        if step["weights_updated"]:
            all_ops = list(step["destroy_weight_updates"].keys()) + list(
                step["repair_weight_updates"].keys()
            )
            op_width = max([len("operator")] + [len(name) for name in all_ops])
            log_lines.append("")
            log_lines.append("  weight update (destroy):")
            header = f"    {'operator':<{op_width}} {'before':>8} {'after':>8}"
            log_lines.append(header)
            for name, change in step["destroy_weight_updates"].items():
                log_lines.append(
                    f"    {name:<{op_width}} {change['before']:>8.2f} {change['after']:>8.2f}"
                )
            log_lines.append("")
            log_lines.append("  weight update (repair):")
            log_lines.append(header)
            for name, change in step["repair_weight_updates"].items():
                log_lines.append(
                    f"    {name:<{op_width}} {change['before']:>8.2f} {change['after']:>8.2f}"
                )
        if mab.objective_current_solution <= 0.0:
            solved = True
            break
        if elapsed_total >= mab.global_timeout_seconds:
            timed_out = True
            break

    total_runtime = perf_counter() - loop_start
    status = "solved" if solved else ("timeout" if timed_out else ("error" if error else "stopped"))
    objective: float | None
    if math.isfinite(mab.objective_current_solution):
        objective = float(mab.objective_current_solution)
    else:
        objective = None
    row = {
        "variant": variant,
        "instance": instance_name,
        "instance_index": _instance_index_from_stem(instance_name),
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "status": status,
        "objective": objective,
        "runtime_seconds": total_runtime,
        "iterations": last_iteration,
        "solved": solved,
        "timeout_hit": timed_out,
        "error": error,
        "solver_name": "mixed(chuffed+gecode)",
    }
    return row, "\n".join(log_lines) + "\n"


def main() -> None:
    instances_dir = BASE_DIR / "Instances1-50"
    instance_paths = sorted(instances_dir.glob("Example*.txt"), key=_instance_sort_key)
    if not instance_paths:
        raise FileNotFoundError(f"No instance files found in {instances_dir}")

    default_n = min(20, len(instance_paths))
    n_instances = _ask_int("Number of instances (first n)", default_n, minimum=1)
    n_instances = min(n_instances, len(instance_paths))
    n_seeds = _ask_int("Number of seeds per instance", 5, minimum=1)
    timeout_seconds = _ask_float("Timeout per solve (seconds)", 60.0, minimum=0.1)
    variants = _ask_variant_mode()
    selected_instances = instance_paths[:n_instances]
    known_indexes = [
        idx
        for idx in (_instance_index_from_stem(instance_path.stem) for instance_path in selected_instances)
        if idx is not None
    ]
    index_width = max(2, len(str(max(known_indexes)))) if known_indexes else 2
    benchmark_root = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    total_runs = n_instances * n_seeds * len(variants)
    run_counter = 0

    for variant in variants:
        variant_dir = benchmark_root / variant / run_id
        logs_dir = variant_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []

        config_path = variant_dir / "config.txt"
        config_path.write_text(
            "\n".join(
                [
                    f"run_id={run_id}",
                    f"variant={variant}",
                    f"n_instances={n_instances}",
                    f"n_seeds={n_seeds}",
                    f"timeout_seconds={timeout_seconds}",
                    "solver_setup=mixed(chuffed+gecode) [matches multiarm_bandit main]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        for instance_path in selected_instances:
            instance_index = _instance_index_from_stem(instance_path.stem)
            if instance_index is None:
                instance_tag = "na"
            else:
                instance_tag = f"{instance_index:0{index_width}d}"
            for seed in range(n_seeds):
                run_counter += 1
                print(
                    f"[{run_counter}/{total_runs}] "
                    f"variant={variant} instance={instance_path.stem} seed={seed}"
                )
                row, log_text = _run_single(
                    instance_path=instance_path,
                    seed=seed,
                    timeout_seconds=timeout_seconds,
                    variant=variant,
                )
                rows.append(row)
                log_path = logs_dir / f"{instance_tag}_{instance_path.stem}_seed{seed}.log"
                log_path.write_text(log_text, encoding="utf-8")
                print(
                    f"[{run_counter}/{total_runs}] "
                    f"status={row['status']} runtime={row['runtime_seconds']:.3f}s "
                    f"objective={row['objective']}"
                )

        rows.sort(
            key=lambda row: (
                10**9 if row["instance_index"] is None else int(row["instance_index"]),
                int(row["seed"]),
            )
        )

        summary_path = variant_dir / "summary.csv"
        fieldnames = [
            "variant",
            "instance",
            "instance_index",
            "seed",
            "timeout_seconds",
            "status",
            "objective",
            "runtime_seconds",
            "iterations",
            "solved",
            "timeout_hit",
            "error",
            "solver_name",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        solved_count = sum(1 for row in rows if row["solved"])
        timeout_count = sum(1 for row in rows if row["timeout_hit"])
        error_count = sum(1 for row in rows if row["status"] == "error")
        print(f"\nVariant {variant}: wrote {summary_path}")
        print(
            f"runs={len(rows)} solved={solved_count} timeouts={timeout_count} errors={error_count}"
        )


if __name__ == "__main__":
    main()
