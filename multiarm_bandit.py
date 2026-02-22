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
    weights_destroy: Optional[Dict[str, float]] = None
    weights_repair: Optional[Dict[str, float]] = None
    iterations_till_weight_update: int = 1
    model_path: str | Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "rws_generic.mzn"
    )
    solver_name: str = "chuffed"
    minizinc_timeout_seconds: float = 1
    exploratory_timeout_seconds: float = 20
    good_enough_threshold: float = 0.0
    score_function: Callable[rws_lns, float] = _default_score_function
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
        if len(self.destroy_operators) == 0:
            raise ValueError("destroy_operators must contain at least one operator")

        # Standard/default repair delegates to rws_lns.repair_exact.
        if not self.repair_operators:
            self.repair_operators = {"repair_exact": rws_lns.repair_exact}
        if len(self.repair_operators) == 0:
            raise ValueError("repair_operators must contain at least one operator")

        self.weights_destroy = self._init_weights(self.weights_destroy, self.destroy_operators)
        self.weights_repair = self._init_weights(self.weights_repair, self.repair_operators)

    def build_lns(self) -> rws_lns:
        return rws_lns(instance=self.warmstart_instance, incumbent=self.schedule)

    def score(self, schedule: Optional[RWS.Schedule] = None) -> float:
        return self.score_function(schedule or self.schedule)

    def is_good_enough(self, schedule: Optional[RWS.Schedule] = None) -> bool:
        return self.score(schedule) >= self.good_enough_threshold

    def _init_weights(
        self,
        weights: Optional[Dict[str, float]],
        operators: Dict[str, Callable[..., Any]],
    ) -> Dict[str, float]:
        keys = list(operators.keys())
        if weights is None:
            equal = 1.0 / len(keys)
            return {key: equal for key in keys}

        if set(weights.keys()) != set(keys):
            raise ValueError("weight keys must match operator keys")

        for key, value in weights.items():
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"weight for {key!r} must be a positive number")

        total = float(sum(weights.values()))
        return {key: float(weights[key]) / total for key in keys}

    def _choose_repair_operator(self) -> Callable[..., Any]:
        names = list(self.repair_operators.keys())
        probs = [self.weights_repair[name] for name in names]
        chosen = random.choices(names, weights=probs, k=1)[0]
        return self.repair_operators[chosen]

    def _choose_destroy_operator(self) -> Callable[..., Any]:
        names = list(self.destroy_operators.keys())
        probs = [self.weights_destroy[name] for name in names]
        chosen = random.choices(names, weights=probs, k=1)[0]
        return self.destroy_operators[chosen]

    def _perform_lns_step:
    


def _make_repair_operator(
    model_path: Path, solver_name: str, timeout_seconds: int
) -> Callable[[rws_lns], None]:
    """Create a configured repair operator closure."""
    def _op(lns: rws_lns) -> None:
        lns.repair_exact(
            model_path=model_path,
            solver_name=solver_name,
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
        "repair_chuffed_fast": _make_repair_operator(model_path, "chuffed", 1),
        "repair_gecode_fast": _make_repair_operator(model_path, "gecode", 1),
        "repair_chuffed_long": _make_repair_operator(model_path, "chuffed", 20),
        "repair_gecode_long": _make_repair_operator(model_path, "gecode", 20),
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

    #in the beginning favour fast repairs
    for repair_op in mab.weights_repair.keys():
        if 'fast' in repair_op:
            mab.weights_repair[repair_op] = 0.4
        elif 'long' in repair_op:
            mab.weights_repair[repair_op] = 0.1

    #######################################
    print(f"Loaded instance: {instance_path}")
    print(f"Configured destroy operators: {len(mab.destroy_operators)}")
    for name in mab.destroy_operators:
        print(f"  - {name}")
    print(f"Configured repair operators: {len(mab.repair_operators)}")
    for name in mab.repair_operators:
        print(f"  - {name}")

    ######################################
    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True,
    )
    print(f"Loaded: {instance_path}")
    schedule.display_schedule()
    schedule.display_violations()


