from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Callable, Dict, Optional

from rws import RWS, rws_lns


def _default_score_function(schedule: RWS.Schedule) -> float:
    """Higher is better; fewer violations gives a larger score."""
    totals = schedule.count_total_violations()
    return float(-sum(totals.values()))


@dataclass
class bandit:
    """Configuration and operator container for a multiarm-bandit LNS loop."""

    schedule: RWS.Schedule
    weight_vector: Optional[list[float]] = None
    iterations_till_weight_update: int = 1
    model_path: str | Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "rws_generic.mzn"
    )
    solver_name: str = "chuffed"
    minizinc_timeout_seconds: float = 5
    exploratory_timeout_seconds: float = 30
    good_enough_threshold: float = 0.0
    score_function: Callable[[RWS.Schedule], float] = _default_score_function
    warmstart_instance: Optional[RWS.Instance] = None
    destroy_operators: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    repair_operators: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.warmstart_instance is None:
            self.warmstart_instance = self.schedule.instance

        if self.iterations_till_weight_update <= 0:
            raise ValueError("iterations_till_weight_update must be > 0")

        if self.minizinc_timeout_seconds <= 0:
            raise ValueError("minizinc_timeout_seconds must be > 0")
        if self.exploratory_timeout_seconds <= 0:
            raise ValueError("exploratory_timeout_seconds must be > 0")

        if not self.destroy_operators:
            self.destroy_operators = {
                "destroy_worker": lambda lns, worker: lns.destroy_worker(worker),
                "destroy_day": lambda lns, day: lns.destroy_day(day),
            }
        num_destroy_ops = len(self.destroy_operators)
        if num_destroy_ops == 0:
            raise ValueError("destroy_operators must contain at least one operator")

        if self.weight_vector is None:
            self.weight_vector = [1.0/num_destroy_ops] * num_destroy_ops
        else:
            if len(self.weight_vector) != num_destroy_ops:
                raise ValueError(
                    "weight_vector length must match number of destroy operators "
                    f"({num_destroy_ops})"
                )
            for weight in self.weight_vector:
                if not isinstance(weight, (int, float)):
                    raise TypeError("weight_vector entries must be numeric")
                if weight <= 0:
                    raise ValueError("weight_vector must contain only positive floats")

        # Standard/default repair delegates to rws_lns.repair_exact.
        if not self.repair_operators:
            self.repair_operators = {"repair_exact": rws_lns.repair_exact}

    def build_lns(self) -> rws_lns:
        return rws_lns(instance=self.warmstart_instance, incumbent=self.schedule)

    def score(self, schedule: Optional[RWS.Schedule] = None) -> float:
        return self.score_function(schedule or self.schedule)

    def is_good_enough(self, schedule: Optional[RWS.Schedule] = None) -> bool:
        return self.score(schedule) >= self.good_enough_threshold


def _make_repair_operator(
    model_path: Path, solver_name: str, sloppy: bool, timeout_seconds: int
) -> Callable[[rws_lns], None]:
    """Create a configured repair operator closure."""
    def _op(lns: rws_lns) -> None:
        lns.repair_exact(
            model_path=model_path,
            solver_name=solver_name,
            sloppy=sloppy,
            timeout_seconds=timeout_seconds,
        )
    return _op


def _current_schedule(lns: rws_lns) -> RWS.Schedule:
    return lns.contender if lns.contender is not None else lns.incumbent


def _floor_fraction_count(total: int, fraction: float) -> int:
    return int(total * fraction)


def _take_ranked_ids(ranked: Dict[int, int], k: int) -> list[int]:
    if k <= 0:
        return []
    return list(ranked.keys())[:k]


def _make_destroy_worst_workers(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        schedule = _current_schedule(lns)
        k = _floor_fraction_count(lns.instance.num_workers, fraction)
        worker_ids = _take_ranked_ids(schedule.worker_ranked_by_violations, k)
        return lns.destroy_worker(worker_ids)
    return _op


def _make_destroy_worst_days(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        schedule = _current_schedule(lns)
        k = _floor_fraction_count(lns.instance.num_days, fraction)
        day_ids = _take_ranked_ids(schedule.days_ranked_by_violations, k)
        return lns.destroy_day(day_ids)
    return _op


def _make_destroy_random_workers(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        k = _floor_fraction_count(lns.instance.num_workers, fraction)
        if k <= 0:
            return []
        worker_ids = random.sample(range(lns.instance.num_workers), k)
        return lns.destroy_worker(worker_ids)
    return _op


def _make_destroy_random_days(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        k = _floor_fraction_count(lns.instance.num_days, fraction)
        if k <= 0:
            return []
        day_ids = random.sample(range(lns.instance.num_days), k)
        return lns.destroy_day(day_ids)
    return _op


if __name__ == "__main__":
    from rws_instance_loader import load_instance_and_schedule

    base = Path(__file__).resolve().parent
    instance_path = base / "Instances1-50" / "Example3.txt"
    instance, schedule = load_instance_and_schedule(file_path=instance_path, cyclicity=True)

    model_path = base / "rws_instance.mzn"
    repair_ops: Dict[str, Callable[[rws_lns], None]] = {
        "repair_chuffed_sloppy_1s": _make_repair_operator(model_path, "chuffed", True, 1),
        "repair_chuffed_full_1s": _make_repair_operator(model_path, "chuffed", False, 1),
        "repair_gecode_sloppy_1s": _make_repair_operator(model_path, "gecode", True, 1),
        "repair_gecode_full_1s": _make_repair_operator(model_path, "gecode", False, 1),
        "repair_chuffed_full_20s": _make_repair_operator(model_path, "chuffed", False, 20),
        "repair_gecode_full_20s": _make_repair_operator(model_path, "gecode", False, 20),
    }
    destroy_ops: Dict[str, Callable[[rws_lns], list[tuple[int, int]]]] = {
        "destroy_worst_workers_10pct": _make_destroy_worst_workers(0.10),
        "destroy_worst_workers_30pct": _make_destroy_worst_workers(0.30),
        "destroy_random_workers_20pct": _make_destroy_random_workers(0.20),
        "destroy_worst_days_10pct": _make_destroy_worst_days(0.10),
        "destroy_worst_days_30pct": _make_destroy_worst_days(0.30),
        "destroy_random_days_20pct": _make_destroy_random_days(0.20),
    }

    mab = bandit(
        schedule=schedule,
        warmstart_instance=instance,
        model_path=model_path,
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
    )

    print(f"Loaded instance: {instance_path}")
    print(f"Configured destroy operators: {len(mab.destroy_operators)}")
    for name in mab.destroy_operators:
        print(f"  - {name}")
    print(f"Configured repair operators: {len(mab.repair_operators)}")
    for name in mab.repair_operators:
        print(f"  - {name}")
