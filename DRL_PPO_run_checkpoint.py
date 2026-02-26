from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict

import torch

from DRL_PPO import (
    _make_destroy_day,
    _make_destroy_streak,
    _make_destroy_worker,
    _make_destroy_random_days,
    _make_destroy_random_window,
    _make_destroy_random_workers,
    _make_destroy_worst_days,
    _make_destroy_worst_workers,
    _make_repair_operator,
    drl_alns,
)
from rws import rws_lns
from rws_instance_loader import load_instance_and_schedule


def _build_repair_library(model_path: Path) -> Dict[str, Callable[[rws_lns], None]]:
    return {
        "repair_chuffed_fast": _make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": _make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": _make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": _make_repair_operator(model_path, "gecode", 15),
    }


def _build_destroy_library() -> Dict[str, Callable[[rws_lns, float], list[tuple[int, int]]]]:
    return {
        "destroy_worker": _make_destroy_worker(),
        "destroy_day": _make_destroy_day(),
        "destroy_worst_workers": _make_destroy_worst_workers(),
        "destroy_random_workers": _make_destroy_random_workers(),
        "destroy_worst_days": _make_destroy_worst_days(),
        "destroy_random_days": _make_destroy_random_days(),
        "destroy_random_window": _make_destroy_random_window(),
        "destroy_streak": _make_destroy_streak(),
    }


def _ordered_ops(
    requested_names: list[str],
    library: Dict[str, Callable],
    kind: str,
) -> Dict[str, Callable]:
    missing = [name for name in requested_names if name not in library]
    if missing:
        raise ValueError(f"Checkpoint requires unknown {kind} operators: {missing}")
    return {name: library[name] for name in requested_names}


def run_checkpoint(
    example_number: int,
    checkpoint_path: Path,
    max_iterations: int = 500,
) -> None:
    base = Path(__file__).resolve().parent
    instance_path = base / "Instances1-50" / f"Example{example_number}.txt"
    if not instance_path.exists():
        raise FileNotFoundError(f"instance file not found: {instance_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_path = base / "rws_instance.mzn"
    repair_lib = _build_repair_library(model_path=model_path)
    destroy_lib = _build_destroy_library()

    checkpoint_repair_names = checkpoint.get("repair_names")
    checkpoint_destroy_names = checkpoint.get("destroy_names")
    if not checkpoint_repair_names or not checkpoint_destroy_names:
        raise ValueError("Checkpoint does not include operator name metadata.")

    repair_ops = _ordered_ops(
        requested_names=list(checkpoint_repair_names),
        library=repair_lib,
        kind="repair",
    )
    destroy_ops = _ordered_ops(
        requested_names=list(checkpoint_destroy_names),
        library=destroy_lib,
        kind="destroy",
    )

    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True,
        initial_schedule="random",
    )

    solver = drl_alns(
        instance=instance,
        schedule=schedule,
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
    )
    solver.model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        solver.optimizer.load_state_dict(optimizer_state)
    solver.model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded instance: {instance_path}")
    print(
        "Checkpoint objective hints:"
        f" best={checkpoint.get('best_objective')} current={checkpoint.get('current_objective')}"
    )

    state = solver._get_state()
    cumulative_reward = 0.0
    start_time = perf_counter()

    for _ in range(max_iterations):
        with torch.no_grad():
            a_d, a_r, a_sev, a_temp, _, _ = solver._select_action(state)
        next_state, reward, info = solver.step(a_d, a_r, a_sev, a_temp, cumulative_reward)
        cumulative_reward += reward
        elapsed = perf_counter() - start_time
        print(
            f"iter={info['iteration']} "
            f"time={elapsed:.3f}s "
            f"destroy={info['destroy_name']} "
            f"repair={info['repair_name']} "
            f"objective={info['incumbent_objective']}->{info['contender_objective']} "
            f"accepted={info['accepted']} "
            f"reward={reward:+.3f}"
        )
        state = next_state

        if solver.best_objective <= 0.0:
            print(f"solved=True stop_iter={info['iteration']} reason=objective_zero")
            break
        if info["stagnation"] >= 20:
            print(f"solved=False stop_iter={info['iteration']} reason=stagnation_limit")
            break

    total_runtime = perf_counter() - start_time
    best_obj = solver.best_objective if math.isfinite(solver.best_objective) else None
    curr_obj = solver.current_objective if math.isfinite(solver.current_objective) else None
    print(f"run_completed=True runtime={total_runtime:.3f}s best_objective={best_obj} current_objective={curr_obj}")
    print("\nFinal schedule:")
    solver.schedule.display_schedule()
    solver.schedule.display_validity()


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    raw_example = input("Example number [1]: ").strip()
    example_number = 1 if raw_example == "" else int(raw_example)
    if example_number < 0:
        raise ValueError("example number must be >= 0")

    raw_max_iterations = input("Max iterations [500]: ").strip()
    max_iterations = 500 if raw_max_iterations == "" else int(raw_max_iterations)
    if max_iterations <= 0:
        raise ValueError("max iterations must be > 0")

    default_checkpoint = base / "drl_ppo_checkpoint_1000.pt"
    raw_checkpoint = input(f"Checkpoint path [{default_checkpoint}]: ").strip()
    checkpoint_path = default_checkpoint if raw_checkpoint == "" else Path(raw_checkpoint)

    run_checkpoint(
        example_number=example_number,
        checkpoint_path=checkpoint_path,
        max_iterations=max_iterations,
    )
