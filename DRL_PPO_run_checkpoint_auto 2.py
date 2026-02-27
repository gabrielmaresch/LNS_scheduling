from __future__ import annotations

import math
import os
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Callable, Dict

import numpy as np

# Keep optimizer initialization responsive in environments where dynamo import is slow.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import torch
import torch.optim as optim

import DRL_PPO as ppo7
import DRL_PPO_with_late_phase as ppo8
from rws import rws_lns
from rws_instance_loader import load_instance_and_schedule


def _build_repair_library_legacy(model_path: Path) -> Dict[str, Callable[[rws_lns], None]]:
    return {
        "repair_chuffed_fast": ppo7._make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": ppo7._make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": ppo7._make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": ppo7._make_repair_operator(model_path, "gecode", 15),
    }


def _build_repair_library_late(model_path: Path) -> Dict[str, Callable[[rws_lns], None]]:
    return {
        "repair_chuffed_fast": ppo8._make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": ppo8._make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": ppo8._make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": ppo8._make_repair_operator(model_path, "gecode", 15),
    }


def _build_destroy_library_legacy() -> Dict[str, Callable[[rws_lns, float], list[tuple[int, int]]]]:
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
    """Patch solver state extraction to the legacy 6-feature layout."""

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
        stagnation_count = float(self.stagnation)
        return np.array(
            [
                best_improved,
                current_accepted,
                current_improved,
                is_current_best,
                cost_diff_best,
                stagnation_count,
            ],
            dtype=np.float32,
        )

    solver.state_dim = 6
    solver._get_state = MethodType(_get_state_6, solver)


def _infer_state_dim(checkpoint: dict) -> int:
    if "state_dim" in checkpoint:
        return int(checkpoint["state_dim"])
    model_state = checkpoint.get("model_state_dict", {})
    layer_weight = model_state.get("shared.0.weight")
    if layer_weight is not None and hasattr(layer_weight, "shape") and len(layer_weight.shape) == 2:
        return int(layer_weight.shape[1])
    return 6


def _load_solver_for_state_dim(
    checkpoint: dict,
    state_dim: int,
    instance_path: Path,
) -> object:
    base = Path(__file__).resolve().parent
    model_path = base / "rws_instance.mzn"
    checkpoint_repair_names = checkpoint.get("repair_names")
    checkpoint_destroy_names = checkpoint.get("destroy_names")
    if not checkpoint_repair_names or not checkpoint_destroy_names:
        raise ValueError("Checkpoint does not include operator name metadata.")

    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True,
        initial_schedule="random",
    )

    if state_dim in (6, 7):
        repair_lib = _build_repair_library_legacy(model_path=model_path)
        destroy_lib = _build_destroy_library_legacy()
        solver = ppo7.drl_alns(
            instance=instance,
            schedule=schedule,
            destroy_operators=_ordered_ops(list(checkpoint_destroy_names), destroy_lib, "destroy"),
            repair_operators=_ordered_ops(list(checkpoint_repair_names), repair_lib, "repair"),
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
        solver.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            try:
                solver.optimizer.load_state_dict(optimizer_state)
            except Exception:
                print("warning=optimizer_state_incompatible skipped=True")
        return solver

    if state_dim == 8:
        repair_lib = _build_repair_library_late(model_path=model_path)
        destroy_lib = ppo8._build_destroy_library(instance=instance)
        solver = ppo8.drl_alns(
            instance=instance,
            schedule=schedule,
            destroy_operators=_ordered_ops(list(checkpoint_destroy_names), destroy_lib, "destroy"),
            repair_operators=_ordered_ops(list(checkpoint_repair_names), repair_lib, "repair"),
        )
        for key in (
            "equal_move_allowed_freezeout",
            "late_phase_weight",
            "late_phase_strict_improvement",
        ):
            if key in checkpoint:
                setattr(solver, key, checkpoint[key])
        solver.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            try:
                solver.optimizer.load_state_dict(optimizer_state)
            except Exception:
                print("warning=optimizer_state_incompatible skipped=True")
        return solver

    raise ValueError(f"Unsupported checkpoint state_dim={state_dim}. Expected one of: 6, 7, 8.")


def run_checkpoint(
    example_number: int,
    checkpoint_path: Path,
    timeout_seconds: float = 120.0,
) -> None:
    base = Path(__file__).resolve().parent
    instance_path = base / "Instances1-50" / f"Example{example_number}.txt"
    if not instance_path.exists():
        raise FileNotFoundError(f"instance file not found: {instance_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dim = _infer_state_dim(checkpoint)
    solver = _load_solver_for_state_dim(
        checkpoint=checkpoint,
        state_dim=state_dim,
        instance_path=instance_path,
    )
    solver.model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded instance: {instance_path}")
    print(f"Detected state_dim={state_dim}")
    print(
        "Checkpoint objective hints:"
        f" best={checkpoint.get('best_objective')} current={checkpoint.get('current_objective')}"
    )

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


def _default_checkpoint_path(base: Path) -> Path:
    candidates = [
        base / "drl_ppo_checkpoint_500.pt",
        base / "drl_ppo_checkpoint_1000_6d.pt",
        base / "drl_ppo_checkpoint_1000.pt",
        base / "drl_ppo_checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base / "drl_ppo_checkpoint_500.pt"


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    raw_example = input("Example number [1]: ").strip()
    example_number = 1 if raw_example == "" else int(raw_example)
    if example_number < 0:
        raise ValueError("example number must be >= 0")

    raw_timeout = input("Timeout seconds [120]: ").strip()
    timeout_seconds = 120.0 if raw_timeout == "" else float(raw_timeout)
    if timeout_seconds <= 0:
        raise ValueError("timeout seconds must be > 0")

    default_checkpoint = _default_checkpoint_path(base)
    raw_checkpoint = input(f"Checkpoint path [{default_checkpoint}]: ").strip()
    checkpoint_path = default_checkpoint if raw_checkpoint == "" else Path(raw_checkpoint)

    run_checkpoint(
        example_number=example_number,
        checkpoint_path=checkpoint_path,
        timeout_seconds=timeout_seconds,
    )
