from __future__ import annotations

import math
import os
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Callable, Dict

import numpy as np

# Avoid long torch._dynamo initialization paths in some environments.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import torch
import torch.optim as optim

import DRL_PPO as ppo7
import DRL_PPO_with_late_phase as ppo8
from rws import rws_lns
from rws_instance_loader import load_instance_and_schedule


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "PPO_checkpoints"


def _repair_library(
    make_repair_operator: Callable[[Path, str, int], Callable[[rws_lns], None]],
    model_path: Path,
) -> Dict[str, Callable[[rws_lns], None]]:
    return {
        "repair_chuffed_fast": make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": make_repair_operator(model_path, "gecode", 15),
    }


def _legacy_destroy_library() -> Dict[str, Callable[[rws_lns, float], list[tuple[int, int]]]]:
    return {
        "destroy_worker": ppo7._make_destroy_worker(),
        "destroy_day": ppo7._make_destroy_day(),
        "destroy_worst_workers": ppo7._make_destroy_worst_workers(),
        "destroy_random_workers": ppo7._make_destroy_random_workers(),
        "destroy_worst_days": ppo7._make_destroy_worst_days(),
        "destroy_random_days": ppo7._make_destroy_random_days(),
        "destroy_random_window": ppo7._make_destroy_random_window(),
        "destroy_streak": ppo7._make_destroy_streak(),
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


def _patch_state_getter_6(solver: ppo7.drl_alns) -> None:
    def _get_state_6(self: ppo7.drl_alns) -> np.ndarray:
        best_improved = float(self.prev_best)
        current_accepted = float(self.prev_accepted)
        current_improved = float(self.prev_improved)
        if (not math.isfinite(self.current_objective)) or (not math.isfinite(self.best_objective)):
            is_current_best = 1.0
            cost_diff_best = -1.0
        elif self.current_objective <= self.best_objective:
            is_current_best = 1.0
            cost_diff_best = -1.0
        else:
            is_current_best = 0.0
            cost_diff_best = (self.current_objective - self.best_objective) / max(1.0, self.best_objective)
        return np.array(
            [
                best_improved,
                current_accepted,
                current_improved,
                is_current_best,
                cost_diff_best,
                float(self.stagnation),
            ],
            dtype=np.float32,
        )

    solver.state_dim = 6
    solver._get_state = MethodType(_get_state_6, solver)


def _detect_state_dim(checkpoint: dict) -> int:
    if "state_dim" in checkpoint:
        return int(checkpoint["state_dim"])
    layer_weight = checkpoint.get("model_state_dict", {}).get("shared.0.weight")
    if layer_weight is not None and hasattr(layer_weight, "shape") and len(layer_weight.shape) == 2:
        return int(layer_weight.shape[1])
    return 6


def _load_optimizer_state(solver: object, checkpoint: dict) -> None:
    state = checkpoint.get("optimizer_state_dict")
    if state is None:
        return
    try:
        solver.optimizer.load_state_dict(state)
    except Exception:
        print("warning=optimizer_state_incompatible skipped=True")


def _build_solver(
    checkpoint: dict,
    state_dim: int,
    instance: object,
    schedule: object,
) -> object:
    repair_names = checkpoint.get("repair_names")
    destroy_names = checkpoint.get("destroy_names")
    if not repair_names or not destroy_names:
        raise ValueError("Checkpoint does not include operator name metadata.")

    model_path = BASE_DIR / "rws_instance.mzn"

    if state_dim == 8:
        repair_lib = _repair_library(ppo8._make_repair_operator, model_path)
        destroy_lib = ppo8._build_destroy_library(instance=instance)
        solver = ppo8.drl_alns(
            instance=instance,
            schedule=schedule,
            destroy_operators=_ordered_ops(list(destroy_names), destroy_lib, "destroy"),
            repair_operators=_ordered_ops(list(repair_names), repair_lib, "repair"),
        )
        for key in ("equal_move_allowed_freezeout", "late_phase_weight", "late_phase_strict_improvement"):
            if key in checkpoint:
                setattr(solver, key, checkpoint[key])
    elif state_dim in (6, 7):
        repair_lib = _repair_library(ppo7._make_repair_operator, model_path)
        destroy_lib = _legacy_destroy_library()
        solver = ppo7.drl_alns(
            instance=instance,
            schedule=schedule,
            destroy_operators=_ordered_ops(list(destroy_names), destroy_lib, "destroy"),
            repair_operators=_ordered_ops(list(repair_names), repair_lib, "repair"),
        )
        if state_dim == 6:
            _patch_state_getter_6(solver)
            solver.model = ppo7.ActorCritic(
                state_dim=6,
                n_destroy=len(solver.destroy_names),
                n_repair=len(solver.repair_names),
                n_severity=10,
                n_temp=50,
            )
            solver.optimizer = optim.Adam(solver.model.parameters(), lr=solver.lr)
    else:
        raise ValueError(f"Unsupported checkpoint state_dim={state_dim}. Expected 6, 7, or 8.")

    solver.model.load_state_dict(checkpoint["model_state_dict"])
    _load_optimizer_state(solver, checkpoint)
    solver.model.eval()
    return solver


def _default_checkpoint_filename() -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0].name
    for legacy_name in (
        "drl_ppo_checkpoint_500.pt",
        "drl_ppo_checkpoint_1000_6d.pt",
        "drl_ppo_checkpoint_1000.pt",
        "drl_ppo_checkpoint.pt",
    ):
        if (CHECKPOINT_DIR / legacy_name).exists():
            return legacy_name
        if (BASE_DIR / legacy_name).exists():
            return legacy_name
    return "drl_ppo_checkpoint.pt"


def _resolve_checkpoint_path(raw: str, default_filename: str) -> Path:
    if raw == "":
        filename = default_filename
    else:
        filename = raw
    if "/" in filename or "\\" in filename:
        raise ValueError("Please provide checkpoint filename only (no directory path).")
    return CHECKPOINT_DIR / filename


def run_checkpoint(
    example_number: int,
    checkpoint_path: Path,
    timeout_seconds: float = 120.0,
) -> None:
    instance_path = BASE_DIR / "Instances1-50" / f"Example{example_number}.txt"
    if not instance_path.exists():
        raise FileNotFoundError(f"instance file not found: {instance_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dim = _detect_state_dim(checkpoint)

    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True,
        initial_schedule="random",
    )
    solver = _build_solver(
        checkpoint=checkpoint,
        state_dim=state_dim,
        instance=instance,
        schedule=schedule,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded instance: {instance_path}")
    print(f"Detected state_dim={state_dim}")

    state = solver._get_state()
    cumulative_reward = 0.0
    start_time = perf_counter()

    while True:
        elapsed = perf_counter() - start_time
        if elapsed >= timeout_seconds:
            print(f"solved=False stop_time={elapsed:.3f}s reason=timeout")
            break

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
    raw_example = input("Example number [1]: ").strip()
    example_number = 1 if raw_example == "" else int(raw_example)
    if example_number < 0:
        raise ValueError("example number must be >= 0")

    raw_timeout = input("Timeout seconds [120]: ").strip()
    timeout_seconds = 120.0 if raw_timeout == "" else float(raw_timeout)
    if timeout_seconds <= 0:
        raise ValueError("timeout seconds must be > 0")

    default_checkpoint = _default_checkpoint_filename()
    raw_checkpoint = input(f"Checkpoint filename [{default_checkpoint}]: ").strip()
    checkpoint_path = _resolve_checkpoint_path(raw_checkpoint, default_checkpoint)

    run_checkpoint(
        example_number=example_number,
        checkpoint_path=checkpoint_path,
        timeout_seconds=timeout_seconds,
    )
