from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Callable, Dict, Optional

from rws import RWS, rws_lns


def _default_score_function(
    best_objective: float,
    incumbent_objective: float,
    contender_objective: float,
    temperature: float,
    late_phase_threshold: int = 5,
) -> tuple[int, bool]:
    """Return `(score, accepted)` using best/current/contender objective values and SA."""
    early_phase = incumbent_objective > float(late_phase_threshold)

    if contender_objective < best_objective:
        score, accept = 33, True
    elif contender_objective < incumbent_objective:
        score, accept = 9, True
    elif contender_objective == incumbent_objective:
        score, accept = 0, early_phase
    else:
        p = math.exp(
            -(contender_objective - incumbent_objective) / temperature
        )
        score, accept = 0, early_phase and (random.random() < p)

    return score, accept


def compute_softmax(score: float, beta_softmax: float) -> float:
    """Compute unnormalized softmax value."""
    return math.exp(beta_softmax * score)



@dataclass
class MBandit:
    """Configuration and operator container for a multiarm-bandit LNS loop."""

    instance: RWS.Instance
    schedule: RWS.Schedule
    weights_destroy: Optional[Dict[str, float]] = None
    weights_repair: Optional[Dict[str, float]] = None
    iterations_till_weight_update: int = 20
    reaction_factor: float = 0.1
    beta_softmax: float = 0.2
    equal_move_allowed_freezeout: int = 5
    
    annealing_temperature: float = 5
    min_annealing_temperature: float = 0.7
    time_decay_annealing: float = 0.98

    global_timeout_seconds: float = 600.0
    model_path: str | Path | None = None
    solver_name: str = "chuffed"
    minizinc_timeout_seconds: int = 50
    exploratory_timeout_seconds: float = 100
    exploration_after_stagnation: int = 10
    conflicts_best_solution: int = field(init=False)
    conflicts_current_solution: int = field(init=False)
    objective_best_solution: float = field(init=False)
    objective_current_solution: float = field(init=False)
    score_function: Callable[[float, float, float, float, int], tuple[int, bool]] = _default_score_function
    destroy_operators: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    repair_operators: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    destroy_exploration_operator: Callable[[rws_lns], list[tuple[int, int]]] = field(init=False, repr=False)
    repair_exploration_operator: Callable[[rws_lns], None] = field(init=False, repr=False)
    lns: rws_lns = field(init=False)
    lns_loop_counter: int = 0
    operator_score_sums: Dict[str, float] = field(init=False)
    operator_usage_counts: Dict[str, int] = field(init=False)
    destroy_tabu_length: int = 8
    destroy_tabu_history: deque[tuple[frozenset[int], frozenset[int]]] = field(
        init=False, repr=False
    )
    destroy_tabu_counts: Dict[tuple[frozenset[int], frozenset[int]], int] = field(
        init=False, repr=False
    )
    stagnation_rounds: int = 0

    def __post_init__(self) -> None:
        """Validate configuration and initialize operators, weights, and LNS state."""
        warmstart_conflicts = int(sum(self.schedule.count_total_violations().values()))
        self.conflicts_best_solution = warmstart_conflicts
        self.conflicts_current_solution = warmstart_conflicts
        # Warmstart objective is unknown until a repair solve returns a value.
        self.objective_best_solution = float("inf")
        self.objective_current_solution = float("inf")

        if self.iterations_till_weight_update <= 0:
            raise ValueError("iterations_till_weight_update must be > 0")
        if not (0.0 <= self.reaction_factor <= 1.0):
            raise ValueError("reaction_factor must be in [0, 1]")
        if self.global_timeout_seconds <= 0:
            raise ValueError("global_timeout_seconds must be > 0")
        if self.exploration_after_stagnation <= 0:
            raise ValueError("exploration_after_stagnation must be > 0")
        if self.annealing_temperature <= 0:
            raise ValueError("annealing_temperature must be > 0")
        if self.min_annealing_temperature <= 0:
            raise ValueError("min_temperature must be > 0")
        if self.min_annealing_temperature > self.annealing_temperature:
            raise ValueError("min_temperature must be <= annealing_temperature")
        if not (0 < self.time_decay_annealing <= 1):
            raise ValueError("time_decay must be in (0, 1]")

        if self.minizinc_timeout_seconds <= 0:
            raise ValueError("minizinc_timeout_seconds must be > 0")
        if self.exploratory_timeout_seconds <= 0:
            raise ValueError("exploratory_timeout_seconds must be > 0")
        if self.destroy_tabu_length <= 0:
            raise ValueError("destroy_tabu_length must be > 0")
        if self.equal_move_allowed_freezeout < 0:
            raise ValueError("equal_move_allowed_freezeout must be >= 0")

            

        if not self.destroy_operators:
            self.destroy_operators = {
                "destroy_worker": (
                    lambda lns: lns.destroy_worker(random.randrange(lns.instance.num_workers))
                ),
                "destroy_day": (
                    lambda lns: lns.destroy_day(random.randrange(lns.instance.num_days))
                ),
                "destroy_worst_window_20pct": _make_destroy_worst_window(0.20),
            }

        if not self.repair_operators:
            self.repair_operators = {
                "repair_exact": (
                    lambda lns: lns.repair_exact(
                        model_path=self.model_path,
                        solver_name=self.solver_name,
                        timeout_seconds=self.minizinc_timeout_seconds,
                    )
                )
            }

        self.destroy_exploration_operator = _make_destroy_random_workers_and_days(
            workers_fraction=0.2,
            days_fraction=0.1,
        )
        self.repair_exploration_operator = (
            lambda lns: lns.repair_exact(
                model_path=self.model_path,
                solver_name=self.solver_name,
                timeout_seconds=int(self.exploratory_timeout_seconds),
            )
        )

        self.weights_destroy = self._init_weights(self.weights_destroy, self.destroy_operators)
        self.weights_repair = self._init_weights(self.weights_repair, self.repair_operators)
        self._initialize_operator_tracking()
        self.destroy_tabu_history = deque()
        self.destroy_tabu_counts = {}
        self.lns = rws_lns(instance=self.instance, incumbent=self.schedule)

    def _init_weights(
        self,
        weights: Optional[Dict[str, float]],
        operators: Dict[str, Callable[..., Any]],
    ) -> Dict[str, float]:
        """Initialize operator weights, or create equal defaults."""
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

    def _choose_repair_operator(self) -> tuple[str, Callable[..., Any]]:
        """Sample one repair operator according to current repair weights."""
        names = list(self.repair_operators.keys())
        probs = [self.weights_repair[name] for name in names]
        chosen = random.choices(names, weights=probs, k=1)[0]
        return chosen, self.repair_operators[chosen]

    def _choose_destroy_operator(self) -> tuple[str, Callable[..., Any]]:
        """Sample one destroy operator according to current destroy weights."""
        names = list(self.destroy_operators.keys())
        probs = [self.weights_destroy[name] for name in names]
        chosen = random.choices(names, weights=probs, k=1)[0]
        return chosen, self.destroy_operators[chosen]

    def _destroyed_id_sets(
        self,
        destroy_name: str,
        destroyed_pairs: list[tuple[int, int]],
        lns: rws_lns,
        use_exploration: bool,
    ) -> tuple[set[int], set[int]]:
        """Extract targeted IDs for tabu checks from the current destroy move."""
        if use_exploration:
            workers = set(getattr(lns, "_last_destroy_selected_workers", []))
            days = set(getattr(lns, "_last_destroy_selected_days", []))
            return workers, days
        if "worker" in destroy_name:
            return {worker for _, worker in destroyed_pairs}, set()
        if "day" in destroy_name:
            return set(), {day for day, _ in destroyed_pairs}
        workers = {worker for _, worker in destroyed_pairs}
        days = {day for day, _ in destroyed_pairs}
        return workers, days

    def _is_strict_tabu(self, destroyed_workers: set[int], destroyed_days: set[int]) -> bool:
        """Return True when current destroyed IDs match a recent tabu signature."""
        if not destroyed_workers and not destroyed_days:
            return False
        signature = self._destroy_signature(destroyed_workers, destroyed_days)
        return signature in self.destroy_tabu_counts

    def _choose_and_apply_destroy(
        self,
        lns: rws_lns,
        use_exploration: bool,
    ) -> tuple[str, list[tuple[int, int]], set[int], set[int]]:
        """Apply a destroy move while avoiding immediate strict repetition of destroyed IDs."""
        attempts = max(8, len(self.destroy_operators) * 3)
        last_name = "destroy_exploration" if use_exploration else "destroy"
        last_result: list[tuple[int, int]] = []
        last_workers: set[int] = set()
        last_days: set[int] = set()

        for _ in range(attempts):
            lns._initialize_fixed_vars(self.schedule)
            if use_exploration:
                destroy_name = "destroy_exploration"
                destroy_op = self.destroy_exploration_operator
            else:
                destroy_name, destroy_op = self._choose_destroy_operator()

            destroy_result = destroy_op(lns)
            workers, days = self._destroyed_id_sets(
                destroy_name=destroy_name,
                destroyed_pairs=destroy_result,
                lns=lns,
                use_exploration=use_exploration,
            )
            if not self._is_strict_tabu(workers, days):
                self._record_destroy_signature(workers, days)
                return destroy_name, destroy_result, workers, days

            last_name = destroy_name
            last_result = destroy_result
            last_workers = workers
            last_days = days

        # Fallback: keep the last attempt if no non-matching destroy could be found.
        self._record_destroy_signature(last_workers, last_days)
        return last_name, last_result, last_workers, last_days

    def _operator_key(self, kind: str, name: str) -> str:
        """Build the tracking key used for operator score/usage dictionaries."""
        return f"{kind}::{name}"

    def _initialize_operator_tracking(self) -> None:
        """Initialize per-operator score and usage accumulators."""
        keys = [self._operator_key("destroy", name) for name in self.destroy_operators]
        keys.extend(self._operator_key("repair", name) for name in self.repair_operators)
        self.operator_score_sums = {key: 0.0 for key in keys}
        self.operator_usage_counts = {key: 0 for key in keys}

    def _reset_operator_tracking(self) -> None:
        """Reset per-operator score and usage accumulators to zero."""
        for key in self.operator_score_sums:
            self.operator_score_sums[key] = 0.0
        for key in self.operator_usage_counts:
            self.operator_usage_counts[key] = 0

    def _update_operator_weights(self) -> None:
        """Update destroy/repair weights from tracked average scores."""
        targets_by_kind: Dict[str, Dict[str, float]] = {}
        for kind, operators in (
            ("destroy", self.destroy_operators),
            ("repair", self.repair_operators),
        ):
            targets: Dict[str, float] = {}
            total = 0.0
            for name in operators:
                key = self._operator_key(kind, name)
                usage = self.operator_usage_counts.get(key, 0)
                score_sum = self.operator_score_sums.get(key, 0.0)
                avg_score = score_sum / usage if usage > 0 else 0.0
                targets[name] = compute_softmax(avg_score, self.beta_softmax)
                total += targets[name]

            if total <= 0.0:
                equal_weight = 1.0 / len(operators)
                targets = {name: equal_weight for name in operators}
            else:
                for name in targets:
                    targets[name] /= total
            targets_by_kind[kind] = targets

        for weights, targets in (
            (self.weights_destroy, targets_by_kind["destroy"]),
            (self.weights_repair, targets_by_kind["repair"]),
        ):
            for name, old_weight in list(weights.items()):
                weights[name] = (1 - self.reaction_factor) * float(old_weight) + (
                    self.reaction_factor * targets[name]
                )

        self._reset_operator_tracking()

    
    def _perform_lns_step(self) -> Dict[str, Any]:
        """Run one destroy/repair iteration and return metrics for logging/display."""
        self.lns_loop_counter += 1
        lns = self.lns
        lns.incumbent = self.schedule
        lns.contender = None

        incumbent_conflicts = int(sum(self.schedule.count_total_violations().values()))
        incumbent_objective = self.objective_current_solution

        use_exploration = self.stagnation_rounds >= self.exploration_after_stagnation
        if use_exploration:
            repair_name = "repair_exploration"
            repair_op = self.repair_exploration_operator
        else:
            repair_name, repair_op = self._choose_repair_operator()
        update_this_round = (
            self.lns_loop_counter % self.iterations_till_weight_update == 0
        )
        destroy_weights_before_all = (
            {name: float(weight) for name, weight in self.weights_destroy.items()}
            if update_this_round
            else {}
        )
        repair_weights_before_all = (
            {name: float(weight) for name, weight in self.weights_repair.items()}
            if update_this_round
            else {}
        )

        destroy_name, destroy_result, destroyed_workers_set, destroyed_days_set = self._choose_and_apply_destroy(
            lns=lns,
            use_exploration=use_exploration,
        )
        destroyed_workers = sorted(destroyed_workers_set)
        destroyed_days = sorted(destroyed_days_set)
        if use_exploration:
            destroyed_target_type = "window"
            destroyed_target_ids = {
                "workers": destroyed_workers,
                "days": destroyed_days,
            }
        elif "worker" in destroy_name:
            destroyed_target_type = "workers"
            destroyed_target_ids = destroyed_workers
        elif "day" in destroy_name:
            destroyed_target_type = "days"
            destroyed_target_ids = destroyed_days
        else:
            destroyed_target_type = "pairs"
            destroyed_target_ids = sorted(destroy_result)
        repair_failed = False
        repair_error: Optional[str] = None
        try:
            repair_op(lns)
        except Exception as exc:
            repair_failed = True
            repair_error = f"{type(exc).__name__}: {exc}"

        contender_conflicts = incumbent_conflicts
        contender_objective = incumbent_objective
        contender_score = 0
        contender_accepted = False
        if not repair_failed and lns.contender is not None:
            contender_conflicts = int(sum(lns.contender.count_total_violations().values()))
            contender_objective_raw = getattr(lns, "contender_objective", None)
            if contender_objective_raw is None:
                repair_failed = True
                repair_error = "repair operator did not return MiniZinc objective value"
            else:
                contender_objective = float(contender_objective_raw)
                contender_score, contender_accepted = self.score_function(
                    self.objective_best_solution,
                    self.objective_current_solution,
                    contender_objective,
                    self.annealing_temperature,
                    self.equal_move_allowed_freezeout,
                )
        elif repair_error is None:
            repair_failed = True
            repair_error = "repair operator did not produce a contender schedule"
        self.annealing_temperature = max(
            self.annealing_temperature * self.time_decay_annealing,
            self.min_annealing_temperature,
        )
        
        if not use_exploration:
            destroy_key = self._operator_key("destroy", destroy_name)
            repair_key = self._operator_key("repair", repair_name)
            for key in (destroy_key, repair_key):
                self.operator_score_sums[key] = (
                    self.operator_score_sums.get(key, 0.0) + float(contender_score)
                )
                self.operator_usage_counts[key] = (
                    self.operator_usage_counts.get(key, 0) + 1
                )

        accepted = (not repair_failed) and bool(contender_accepted)
        if accepted:
            self.schedule = lns.contender
            self.conflicts_current_solution = contender_conflicts
            if contender_conflicts < self.conflicts_best_solution:
                self.conflicts_best_solution = contender_conflicts
            self.objective_current_solution = contender_objective
            if contender_objective < self.objective_best_solution:
                self.objective_best_solution = contender_objective
        if use_exploration:
            self.stagnation_rounds = 0
        elif self.objective_current_solution < incumbent_objective:
            self.stagnation_rounds = 0
        else:
            self.stagnation_rounds += 1
        weights_updated = False
        destroy_weight_updates: Dict[str, Dict[str, float]] = {}
        repair_weight_updates: Dict[str, Dict[str, float]] = {}
        if update_this_round:
            self._update_operator_weights()
            weights_updated = True
            destroy_weight_updates = {
                name: {
                    "before": destroy_weights_before_all[name],
                    "after": float(self.weights_destroy[name]),
                }
                for name in self.weights_destroy
            }
            repair_weight_updates = {
                name: {
                    "before": repair_weights_before_all[name],
                    "after": float(self.weights_repair[name]),
                }
                for name in self.weights_repair
            }
        return {
            "iteration": self.lns_loop_counter,
            "incumbent_conflicts": incumbent_conflicts,
            "contender_conflicts": contender_conflicts,
            "incumbent_objective": incumbent_objective,
            "contender_objective": contender_objective,
            "contender_score": contender_score,
            "accepted": accepted,
            "selected_destroy_operator": destroy_name,
            "selected_repair_operator": repair_name,
            "repair_failed": repair_failed,
            "repair_error": repair_error,
            "used_exploration": use_exploration,
            "stagnation_rounds": self.stagnation_rounds,
            "weights_updated": weights_updated,
            "destroy_weight_updates": destroy_weight_updates,
            "repair_weight_updates": repair_weight_updates,
            "destroyed_target_type": destroyed_target_type,
            "destroyed_target_ids": destroyed_target_ids,
        }

    def final_push(self, k: int) -> None:
        """TODO: Intensify repair when remaining conflicts drop below `k`."""
        # TODO:
        # - Trigger only if self.conflicts_current_solution < k.
        # - Temporarily switch to stronger/slower repair operators (e.g. long timeout).
        # - Restrict destroy neighborhoods to small focused moves around worst violations.
        # - Accept only improving moves (or stricter SA) and stop on stagnation.
        raise NotImplementedError("TODO: implement final_push")

    def _destroy_signature(
        self,
        destroyed_workers: set[int],
        destroyed_days: set[int],
    ) -> tuple[frozenset[int], frozenset[int]]:
        """Convert current destroy IDs into a hashable tabu signature."""
        return frozenset(destroyed_workers), frozenset(destroyed_days)

    def _record_destroy_signature(
        self,
        destroyed_workers: set[int],
        destroyed_days: set[int],
    ) -> None:
        """Record current destroy signature in bounded FIFO tabu history."""
        if not destroyed_workers and not destroyed_days:
            return

        if len(self.destroy_tabu_history) >= self.destroy_tabu_length:
            evicted = self.destroy_tabu_history.popleft()
            evicted_count = self.destroy_tabu_counts.get(evicted, 0)
            if evicted_count <= 1:
                self.destroy_tabu_counts.pop(evicted, None)
            else:
                self.destroy_tabu_counts[evicted] = evicted_count - 1

        signature = self._destroy_signature(destroyed_workers, destroyed_days)
        self.destroy_tabu_history.append(signature)
        self.destroy_tabu_counts[signature] = self.destroy_tabu_counts.get(signature, 0) + 1



def _make_repair_operator(
    model_path: Path, solver_name: str, timeout_seconds: int
) -> Callable[[rws_lns], None]:
    """Create a configured repair operator closure."""
    def _op(lns: rws_lns) -> None:
        """Run exact MiniZinc repair with fixed solver/model/timeout settings."""
        lns.repair_exact(
            model_path=model_path,
            solver_name=solver_name,
            timeout_seconds=timeout_seconds,
        )
    return _op


def _current_schedule(lns: rws_lns) -> RWS.Schedule:
    """Return contender when available, otherwise incumbent."""
    return lns.contender if lns.contender is not None else lns.incumbent


def _ceil_fraction_count(total: int, fraction: float) -> int:
    """Convert a fraction to a bounded ceiling count."""
    if fraction <= 0:
        return 0
    return min(total, math.ceil(total * fraction))


def _take_ranked_ids(ranked: Dict[int, int], k: int) -> list[int]:
    """Take the first k IDs from a ranked ID->score mapping."""
    if k <= 0:
        return []
    return list(ranked.keys())[:k]


def _smallest_cover_interval(ids: list[int], size: int, cyclic: bool) -> list[int]:
    """Return the smallest contiguous interval covering `ids` on a line or ring."""
    if size <= 0 or not ids:
        return []

    unique_sorted = sorted(set(ids))
    if not cyclic:
        start = unique_sorted[0]
        end = unique_sorted[-1]
        return list(range(start, end + 1))

    if len(unique_sorted) == size:
        return list(range(size))

    max_gap = -1
    max_gap_idx = 0
    count = len(unique_sorted)
    for idx in range(count):
        left = unique_sorted[idx]
        right = unique_sorted[(idx + 1) % count]
        gap = (right - left) % size
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = idx

    start = unique_sorted[(max_gap_idx + 1) % count]
    end = unique_sorted[max_gap_idx]
    interval: list[int] = []
    current = start
    while True:
        interval.append(current)
        if current == end:
            break
        current = (current + 1) % size
    return interval



####### Library of different destroy-operators

def _make_destroy_worst_workers(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Build a destroy op that frees the worst workers by violation ranking."""
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy assignments for the current worst-ranked workers."""
        schedule = _current_schedule(lns)
        k = _ceil_fraction_count(lns.instance.num_workers, fraction)
        worker_ids = _take_ranked_ids(schedule.worker_ranked_by_violations, k)
        return lns.destroy_worker(worker_ids)
    return _op

def _make_destroy_worst_days(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Build a destroy op that frees the worst days by violation ranking."""
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy assignments for the current worst-ranked days."""
        schedule = _current_schedule(lns)
        k = _ceil_fraction_count(lns.instance.num_days, fraction)
        day_ids = _take_ranked_ids(schedule.days_ranked_by_violations, k)
        return lns.destroy_day(day_ids)
    return _op


def _make_destroy_worst_window(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Build a destroy op that frees a worst-ranked workers x days index window."""
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must be in [0, 1]")

    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy entries in the smallest interval window covering top-p worst IDs."""
        schedule = _current_schedule(lns)

        worker_k = _ceil_fraction_count(lns.instance.num_workers, fraction)
        day_k = _ceil_fraction_count(lns.instance.num_days, fraction)
        worker_ids = _take_ranked_ids(schedule.worker_ranked_by_violations, worker_k)
        day_ids = _take_ranked_ids(schedule.days_ranked_by_violations, day_k)

        worker_window = _smallest_cover_interval(
            ids=worker_ids,
            size=lns.instance.num_workers,
            cyclic=True,
        )
        day_window = _smallest_cover_interval(
            ids=day_ids,
            size=lns.instance.num_days,
            cyclic=lns.instance.cyclicity,
        )

        if not worker_window or not day_window:
            return []

        return lns.destroy_window(workers=worker_window, days=day_window)

    return _op


def _make_destroy_random_workers(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Build a destroy op that frees a random subset of workers."""
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy assignments for randomly selected workers."""
        k = _ceil_fraction_count(lns.instance.num_workers, fraction)
        if k <= 0:
            return []
        worker_ids = random.sample(range(lns.instance.num_workers), k)
        return lns.destroy_worker(worker_ids)
    return _op

def _make_destroy_random_days(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Build a destroy op that frees a random subset of days."""
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy assignments for randomly selected days."""
        k = _ceil_fraction_count(lns.instance.num_days, fraction)
        if k <= 0:
            return []
        day_ids = random.sample(range(lns.instance.num_days), k)
        return lns.destroy_day(day_ids)
    return _op

def _make_destroy_random_workers_and_days(
    workers_fraction: float,
    days_fraction: float,
) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Build a destroy op that frees random workers and random days in one move."""
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy assignments from both random worker and random day selections."""
        destroyed: list[tuple[int, int]] = []
        selected_workers: list[int] = []
        selected_days: list[int] = []
        workers_k = _ceil_fraction_count(lns.instance.num_workers, workers_fraction)
        if workers_k > 0:
            selected_workers = random.sample(range(lns.instance.num_workers), workers_k)
            destroyed.extend(lns.destroy_worker(selected_workers))
        days_k = _ceil_fraction_count(lns.instance.num_days, days_fraction)
        if days_k > 0:
            selected_days = random.sample(range(lns.instance.num_days), days_k)
            destroyed.extend(lns.destroy_day(selected_days))
        lns._last_destroy_selected_workers = selected_workers
        lns._last_destroy_selected_days = selected_days
        return destroyed

    return _op

###################################################################

if __name__ == "__main__":
    from rws_instance_loader import load_instance_and_schedule

    base = Path(__file__).resolve().parent
    raw_example = input("Example number [1]: ").strip()
    example_number = 1 if raw_example == "" else int(raw_example)
    if example_number < 1:
        raise ValueError("example number must be >= 1")

    instance_path = base / "Instances1-50" / f"Example{example_number}.txt"
    if not instance_path.exists():
        raise FileNotFoundError(f"instance file not found: {instance_path}")

    instance, schedule = load_instance_and_schedule(file_path=instance_path, cyclicity=True)

    model_path = base / "rws_instance.mzn"
    repair_ops: Dict[str, Callable[[rws_lns], None]] = {
        "repair_chuffed_fast": _make_repair_operator(model_path, "chuffed", 3),
        "repair_gecode_fast": _make_repair_operator(model_path, "gecode", 3),
        "repair_chuffed_long": _make_repair_operator(model_path, "chuffed", 15),
        "repair_gecode_long": _make_repair_operator(model_path, "gecode", 15),
    }


    destroy_ops: Dict[str, Callable[[rws_lns], list[tuple[int, int]]]] = {
        "destroy_worst_workers_10pct": _make_destroy_worst_workers(0.10),
        "destroy_worst_workers_20pct": _make_destroy_worst_workers(0.20),
        "destroy_random_workers_20pct": _make_destroy_random_workers(0.20),
        "destroy_worst_window_05pct": _make_destroy_worst_window(0.05),
        "destroy_worst_window_10pct": _make_destroy_worst_window(0.10),
        "destroy_worst_window_20pct": _make_destroy_worst_window(0.20),
        "destroy_worst_days_10pct": _make_destroy_worst_days(0.10),
        "destroy_worst_days_20pct": _make_destroy_worst_days(0.20),
        "destroy_random_days_20pct": _make_destroy_random_days(0.20),
    }

    mab = MBandit(
        instance=instance,
        schedule=schedule,
        model_path=model_path,
        destroy_operators=destroy_ops,
        repair_operators=repair_ops,
    )

    #in the beginning favour fast repairs
    for repair_op in mab.weights_repair.keys():
        if 'fast' in repair_op:
            mab.weights_repair[repair_op] = 0.35
        elif 'long' in repair_op:
            mab.weights_repair[repair_op] = 0.15

    print(f"Loaded instance: {instance_path}")
    print("Initial schedule:")
    schedule.display_schedule()
    loop_start = perf_counter()
    timed_out = False
    solved = False
    last_iteration = 0
    log_lines: list[str] = []
    ANSI_GREEN = "\033[32m"
    ANSI_PURPLE = "\033[35m"
    ANSI_RESET = "\033[0m"

    while True:
        elapsed_before = perf_counter() - loop_start
        if elapsed_before >= mab.global_timeout_seconds:
            timed_out = True
            break

        step_start = perf_counter()
        step = mab._perform_lns_step()
        step_runtime = perf_counter() - step_start
        elapsed_total = perf_counter() - loop_start
        last_iteration = int(step["iteration"])
        destroyed_label = f"{step['destroyed_target_type']}={step['destroyed_target_ids']}"

        summary_line = (
            f"iter={step['iteration']} "
            f"time={elapsed_total:.3f}s "
            f"violations={step['contender_conflicts']} "
            f"score={step['contender_score']} "
            f"{destroyed_label}"
        )
        if step["repair_failed"]:
            summary_line += " repair_failed"
        is_improvement = step["contender_conflicts"] < step["incumbent_conflicts"]
        if is_improvement:
            print(f"{ANSI_GREEN}{summary_line}{ANSI_RESET}")
        elif step["used_exploration"]:
            print(f"{ANSI_PURPLE}{summary_line}{ANSI_RESET}")
        else:
            print(summary_line)
        log_lines.append(
            (
                f"iter={step['iteration']} "
                f"elapsed={elapsed_total:.3f}s "
                f"step_runtime={step_runtime:.3f}s "
                f"destroy={step['selected_destroy_operator']} "
                f"repair={step['selected_repair_operator']} "
                f"violation_metric=count_total_violations "
                f"incumbent_violations={step['incumbent_conflicts']} "
                f"contender_violations={step['contender_conflicts']} "
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
        log_lines.append(
            (
                f"  destroyed_{step['destroyed_target_type']}={step['destroyed_target_ids']}"
            )
        )
        if step["weights_updated"]:
            print("  weight update (destroy):")
            log_lines.append("  weight update (destroy):")
            header = "    operator\tbefore\tafter"
            print(header)
            log_lines.append(header)
            for name, change in step["destroy_weight_updates"].items():
                line = (
                    f"    {name:<28}\t{change['before']:>5.2f}\t{change['after']:>5.2f}"
                )
                print(line)
                log_lines.append(line)

            print("  weight update (repair):")
            log_lines.append("  weight update (repair):")
            print(header)
            log_lines.append(header)
            for name, change in step["repair_weight_updates"].items():
                line = (
                    f"    {name:<28}\t{change['before']:>5.2f}\t{change['after']:>5.2f}"
                )
                print(line)
                log_lines.append(line)
        if mab.conflicts_current_solution == 0:
            solved = True
            break
        if elapsed_total >= mab.global_timeout_seconds:
            timed_out = True
            break

    total_runtime = perf_counter() - loop_start
    log_path = base / "multiarm_bandit.log"
    with log_path.open("w", encoding="utf-8") as handle:
        if log_lines:
            handle.write("\n".join(log_lines) + "\n")

    if solved:
        print(
            f"Stopped after {last_iteration} iterations in {total_runtime:.3f}s "
            "(zero conflicts)."
        )
        print("Final schedule:")
        mab.schedule.display_schedule()
    elif timed_out:
        print(
            f"Timed out after {total_runtime:.3f}s at iteration {last_iteration}. "
            f"Current conflicts: {mab.conflicts_current_solution}"
        )
        print("Last schedule:")
        mab.schedule.display_schedule()
        mab.schedule.display_violations()

    print(f"Wrote run log: {log_path}")
