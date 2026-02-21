from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
