from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path
import random
import re
import shutil
from time import perf_counter
from typing import Any
import sys

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import DRL_PPO_with_late_phase as ppo
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


def _ask_int(
    prompt: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else int(raw)
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"value must be <= {maximum}")
    return value


def _ask_float(
    prompt: str,
    default: float,
    minimum: float,
) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else float(raw)
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_repair_ops(model_path: Path) -> dict[str, Any]:
    return {
        "repair_chuffed_fast": ppo._make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": ppo._make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": ppo._make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": ppo._make_repair_operator(model_path, "gecode", 15),
        "repair_tabu_fast": ppo._make_repair_tabu_operator(model_path, "chuffed", 3),
        "repair_tabu_long": ppo._make_repair_tabu_operator(model_path, "chuffed", 12),
    }


def main() -> None:
    instances_dir = BASE_DIR / "Instances1-50"
    instance_paths = sorted(instances_dir.glob("Example*.txt"), key=_instance_sort_key)
    if len(instance_paths) < 2:
        raise FileNotFoundError(f"Need at least 2 instance files in {instances_dir}")

    default_n = min(15, len(instance_paths))
    n_instances = _ask_int(
        "Number of instances (first n)",
        default=default_n,
        minimum=2,
        maximum=len(instance_paths),
    )
    selected_pool = instance_paths[:n_instances]

    default_train_count = min(n_instances - 1, max(1, int(round(0.8 * n_instances))))
    train_count = _ask_int(
        "Number of training instances",
        default=default_train_count,
        minimum=1,
        maximum=n_instances - 1,
    )

    n_seeds = _ask_int("Number of seeds", default=5, minimum=1)
    timeout_seconds = _ask_float(
        "Timeout max per seed (seconds)",
        default=2500.0,
        minimum=0.1,
    )
    max_iterations = _ask_int(
        "Max iterations per seed",
        default=1000,
        minimum=1,
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    benchmark_root = Path(__file__).resolve().parent
    run_dir = benchmark_root / "ppo_late_phase" / run_id
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.txt"
    config_lines = [
        f"run_id={run_id}",
        "variant=ppo_late_phase",
        f"n_instances={n_instances}",
        f"train_count={train_count}",
        f"test_count={n_instances - train_count}",
        f"n_seeds={n_seeds}",
        f"timeout_seconds={timeout_seconds}",
        f"max_iterations={max_iterations}",
        "pool_instances=" + ",".join(path.stem for path in selected_pool),
    ]
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    model_path = BASE_DIR / "rws_instance.mzn"

    for seed_index in range(n_seeds):
        seed = seed_index
        print(f"[{seed_index + 1}/{n_seeds}] seed={seed}")

        rng = random.Random(seed)
        test_count = n_instances - train_count
        test_positions = set(rng.sample(range(n_instances), k=test_count))
        train_paths = [path for i, path in enumerate(selected_pool) if i not in test_positions]
        test_paths = [path for i, path in enumerate(selected_pool) if i in test_positions]

        if not train_paths:
            raise RuntimeError("No training instances selected; adjust split.")

        _set_seed(seed)

        checkpoint_name = f"checkpoint_seed{seed}.pt"
        checkpoint_path = results_dir / checkpoint_name
        log_path = logs_dir / f"seed{seed}_training.json"

        runtime_seconds = 0.0
        status = "stopped"
        error = ""
        best_objective: float | None = None
        current_objective: float | None = None
        executed_steps = 0

        start_time = perf_counter()
        try:
            instance, schedule = load_instance_and_schedule(
                file_path=train_paths[0],
                cyclicity=True,
                initial_schedule="random",
            )
            solver = ppo.drl_alns(
                instance=instance,
                schedule=schedule,
                destroy_operators=ppo._build_destroy_library(instance=instance),
                repair_operators=_build_repair_ops(model_path=model_path),
            )

            _, log = solver.train(
                total_steps=max_iterations,
                instance_paths=train_paths,
                per_instance_cap=max_iterations,
                timeout_seconds=timeout_seconds,
                checkpoint_path=checkpoint_path,
            )
            ppo.write_training_log(log, log_path)
            ppo.plot_training(
                log,
                show=False,
                output_path=results_dir / f"drl_training_seed{seed}.png",
            )
            ppo.plot_cumulative_reward(
                log,
                show=False,
                output_path=results_dir / f"drl_crwd_seed{seed}.png",
            )
            metrics_src = BASE_DIR / "drl_alns_training_metrics.csv"
            if metrics_src.exists():
                metrics_dst = results_dir / f"drl_alns_training_metrics_seed{seed}.csv"
                shutil.copy2(metrics_src, metrics_dst)

            runtime_seconds = perf_counter() - start_time
            executed_steps = len(log.get("reward", []))
            if runtime_seconds >= timeout_seconds and executed_steps < max_iterations:
                status = "timeout"
            elif executed_steps >= max_iterations:
                status = "max_iterations"
            elif solver.best_objective <= 0.0:
                status = "solved"
            else:
                status = "stopped"

            if math.isfinite(solver.best_objective):
                best_objective = float(solver.best_objective)
            if math.isfinite(solver.current_objective):
                current_objective = float(solver.current_objective)

        except Exception as exc:
            runtime_seconds = perf_counter() - start_time
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        row = {
            "seed_index": seed_index + 1,
            "seed": seed,
            "status": status,
            "runtime_seconds": runtime_seconds,
            "timeout_seconds": timeout_seconds,
            "max_iterations": max_iterations,
            "executed_steps": executed_steps,
            "best_objective": best_objective,
            "current_objective": current_objective,
            "checkpoint": checkpoint_name,
            "train_instances": ",".join(path.stem for path in train_paths),
            "test_instances": ",".join(path.stem for path in test_paths),
            "error": error,
        }
        rows.append(row)

        print(
            f"[{seed_index + 1}/{n_seeds}] "
            f"status={status} runtime={runtime_seconds:.3f}s steps={executed_steps} "
            f"checkpoint={checkpoint_name}"
        )

    summary_path = results_dir / "summary.csv"
    fieldnames = [
        "seed_index",
        "seed",
        "status",
        "runtime_seconds",
        "timeout_seconds",
        "max_iterations",
        "executed_steps",
        "best_objective",
        "current_objective",
        "checkpoint",
        "train_instances",
        "test_instances",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark completed. Summary: {summary_path}")


if __name__ == "__main__":
    main()
