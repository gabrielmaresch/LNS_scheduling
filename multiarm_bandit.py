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
        score, accept = 5, True
    elif contender_objective < incumbent_objective:
        score, accept = 3, True
    elif contender_objective == incumbent_objective:
        score, accept = int(early_phase), early_phase
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
    
    ###### Parameters for weight updates
    iterations_till_weight_update: int = 15
    reaction_factor: float = 0.2
    beta_softmax: float = 0.8
    equal_move_allowed_freezeout: int = 5
    
    ##### Simulated annealing for accepting subpar solutions
    annealing_temperature: float = 5
    min_annealing_temperature: float = 0.7
    time_decay_annealing: float = 0.98
    reshuffle_before_exploration: bool = True
    reshuffle_only_in_early_phase: bool = False

    global_timeout_seconds: float = 300.0
    model_path: str | Path | None = None
    minizinc_timeout_seconds: int = 50
   
    objective_best_solution: float = field(init=False)
    objective_current_solution: float = field(init=False)
    score_function: Callable[[float, float, float, float, int], tuple[int, bool]] = _default_score_function
    
    destroy_operators: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    repair_operators: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    
    ######## Exploration
    exploration_after_stagnation: int = 5
    destroy_exploration_operator: Callable[[rws_lns], list[tuple[int, int]]] = field(init=False, repr=False)
    solver_name: str = "chuffed"
    exploratory_timeout_seconds: float = 100
    repair_exploration_operator: Callable[[rws_lns], None] = field(init=False, repr=False)
    
    lns: rws_lns = field(init=False)
    lns_loop_counter: int = 0
    operator_score_sums: Dict[str, float] = field(init=False)
    operator_usage_counts: Dict[str, int] = field(init=False)
    
    ######## Tabu parameters
    destroy_tabu_length: int = 8
    destroy_tabu_history: deque[tuple[frozenset[int], frozenset[int]]] = field(
        init=False, repr=False
    )
    destroy_tabu_counts: Dict[tuple[frozenset[int], frozenset[int]], int] = field(
        init=False, repr=False
    )
    stagnation_rounds: int = 0

    def __post_init__(self) -> None:
        """Initialize operators, weights, and LNS state."""
        # Warmstart objective is unknown until a repair solve returns a value.
        self.objective_best_solution = float("inf")
        self.objective_current_solution = float("inf")

        if min(
            self.iterations_till_weight_update,
            self.global_timeout_seconds,
            self.exploration_after_stagnation,
            self.annealing_temperature,
            self.min_annealing_temperature,
            self.minizinc_timeout_seconds,
            self.exploratory_timeout_seconds,
            self.destroy_tabu_length,
        ) <= 0:
            raise ValueError("positive configuration values required")
        if not (0.0 <= self.reaction_factor <= 1.0 and 0 < self.time_decay_annealing <= 1):
            raise ValueError("reaction_factor/time_decay out of bounds")
        if self.min_annealing_temperature > self.annealing_temperature:
            raise ValueError("min_temperature must be <= annealing_temperature")
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
                "destroy_random_window_20pct": _make_destroy_random_window(0.20),
                "destroy_maxsalvage_streak_holes_0pct": _make_destroy_maxsalvage_streak_with_holes(0.0),
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
        ### Here we set the standard destroy operator for explorations

        self.destroy_exploration_operator = _make_destroy_random_window(0.25)
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

        total = float(sum(float(weights[key]) for key in keys))
        if total <= 0.0:
            raise ValueError("weights must sum to > 0")
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
        if use_exploration or "window" in destroy_name:
            workers = set(getattr(lns, "_last_destroy_selected_workers", []))
            days = set(getattr(lns, "_last_destroy_selected_days", []))
            if workers or days:
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
            destroy_name, destroy_op = (
                ("destroy_exploration", self.destroy_exploration_operator)
                if use_exploration
                else self._choose_destroy_operator()
            )

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

        incumbent_objective = self.objective_current_solution

        use_exploration = self.stagnation_rounds >= self.exploration_after_stagnation
        early_phase = incumbent_objective > float(self.equal_move_allowed_freezeout)
        shuffle_suffix = ""
        reshuffle_allowed = (
            self.reshuffle_before_exploration
            and (early_phase or not self.reshuffle_only_in_early_phase)
        )
        if use_exploration and reshuffle_allowed:
            day_ok = self.instance.num_days > 1
            worker_ok = self.instance.num_workers > 1
            if day_ok and (not worker_ok or random.random() < 0.5):
                shift = random.randrange(1, self.instance.num_days)
                self.schedule.days_shuffle_cyclic(shift=shift)
                shuffle_suffix = f" shuffle=days_left_{shift}"
            elif worker_ok:
                shift = random.randrange(1, self.instance.num_workers)
                self.schedule.workers_shuffle_cyclic(shift=shift)
                shuffle_suffix = f" shuffle=workers_left_{shift}"
        repair_name, repair_op = (
            ("repair_exploration", self.repair_exploration_operator)
            if use_exploration
            else self._choose_repair_operator()
        )
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
        if "maxsalvage" in destroy_name:
            keep_pairs = list(getattr(lns, "_last_destroy_selected_pairs", []))
            hole_count = int(getattr(lns, "_last_destroy_holes_count", 0))
            if keep_pairs:
                start_day, start_worker = keep_pairs[0]
                end_day, end_worker = keep_pairs[-1]
                keep_text = (
                    f"len={len(keep_pairs)} "
                    f"start=(w{start_worker},d{start_day}) "
                    f"end=(w{end_worker},d{end_day})"
                )
            else:
                keep_text = "len=0"
            destroyed_display = (
                f"salvage streak with holes {keep_text}; holes={hole_count}; "
                f"destroyed_outside={len(destroy_result)}"
            )
        elif use_exploration or "window" in destroy_name:
            destroyed_display = f"destroyed window: {destroyed_workers} x {destroyed_days}"
        elif "worker" in destroy_name:
            destroyed_display = f"destroyed workers: {destroyed_workers}"
        elif "day" in destroy_name:
            destroyed_display = f"destroyed days: {destroyed_days}"
        else:
            destroyed_display = f"destroyed pairs: {len(destroy_result)}"
        destroyed_display += shuffle_suffix
        repair_failed = False
        repair_error: Optional[str] = None
        try:
            repair_op(lns)
        except Exception as exc:
            repair_failed = True
            repair_error = f"{type(exc).__name__}: {exc}"

        contender_objective = incumbent_objective
        contender_score = 0
        contender_accepted = False
        if not repair_failed and lns.contender is not None:
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
            "destroyed_display": destroyed_display,
        }

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


def _ceil_fraction_count(total: int, fraction: float) -> int:
    """Convert a fraction to a bounded ceiling count."""
    if fraction <= 0:
        return 0
    return min(total, math.ceil(total * fraction))


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
    """Build a destroy op that frees workers with earliest first violation day."""
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy assignments for worst-ranked workers by first-violation distance."""
        schedule = lns.contender if lns.contender is not None else lns.incumbent
        k = _ceil_fraction_count(lns.instance.num_workers, fraction)
        if k <= 0:
            return []
        scores = schedule.worker_days_until_first_violation()
        ranked = sorted(scores.keys(), key=lambda worker: (scores[worker], worker))
        worker_ids = [worker for worker in ranked if scores[worker] <= lns.instance.num_days][:k]
        if len(worker_ids) < k:
            remaining = [worker for worker in range(lns.instance.num_workers) if worker not in worker_ids]
            worker_ids.extend(random.sample(remaining, k - len(worker_ids)))
        return lns.destroy_worker(worker_ids)
    return _op

def _make_destroy_worst_days(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Build a destroy op that frees days with largest requirement mismatch counts."""
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy assignments for worst-ranked days by staffing-requirement mismatch."""
        schedule = lns.contender if lns.contender is not None else lns.incumbent
        k = _ceil_fraction_count(lns.instance.num_days, fraction)
        if k <= 0:
            return []
        scores = schedule.day_shift_requirement_violation_counts()
        ranked = sorted(scores.keys(), key=lambda day: (-scores[day], day))
        day_ids = [day for day in ranked if scores[day] > 0][:k]
        if len(day_ids) < k:
            remaining = [day for day in range(lns.instance.num_days) if day not in day_ids]
            day_ids.extend(random.sample(remaining, k - len(day_ids)))
        return lns.destroy_day(day_ids)
    return _op


def _make_destroy_random_window(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Build a destroy op that frees a random workers x days index window."""
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        """Destroy entries in the smallest interval window covering sampled IDs."""

        worker_k = _ceil_fraction_count(lns.instance.num_workers, fraction)
        day_k = _ceil_fraction_count(lns.instance.num_days, fraction)
        worker_ids = (
            random.sample(range(lns.instance.num_workers), worker_k) if worker_k > 0 else []
        )
        day_ids = random.sample(range(lns.instance.num_days), day_k) if day_k > 0 else []

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

        lns._last_destroy_selected_workers = list(worker_window)
        lns._last_destroy_selected_days = list(day_window)
        if not worker_window or not day_window:
            return []

        return lns.destroy_window(workers=worker_window, days=day_window)

    return _op


def _make_destroy_maxsalvage_streak_with_holes(
    hole_fraction: float = 0.0,
) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Destroy outside best flattened streak and additionally free a fraction inside it."""
    if hole_fraction < 0.0 or hole_fraction > 1.0:
        raise ValueError("hole_fraction must be in [0, 1]")

    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        schedule = lns.contender if lns.contender is not None else lns.incumbent
        n_workers = lns.instance.num_workers
        n_days = lns.instance.num_days
        total_cells = n_workers * n_days

        # NOTE: This is intentionally simple and called rarely; there is room for
        # performance improvements by caching and incremental updates.
        blocked_days = schedule._max_feasable_blocked_days()
        memo: dict[tuple[int, int, int], int] = {}
        best_worker = 0
        best_day = 0
        best_backward = 1
        best_len = 0
        for worker in range(n_workers):
            for day in range(n_days):
                forward_len = schedule.max_feasable_streak(
                    worker=worker,
                    day=day,
                    forward=True,
                    _blocked_days=blocked_days,
                    _memo=memo,
                )
                if forward_len == 0:
                    continue
                backward_len = schedule.max_feasable_streak(
                    worker=worker,
                    day=day,
                    forward=False,
                    _blocked_days=blocked_days,
                    _memo=memo,
                )
                total_len = min(total_cells, forward_len + backward_len - 1)
                if total_len > best_len:
                    best_worker = worker
                    best_day = day
                    best_backward = backward_len
                    best_len = total_len

        start_idx = (best_worker * n_days + best_day - best_backward + 1) % total_cells
        keep_pairs = []
        for idx in range(start_idx, start_idx + max(1, best_len)):
            worker_idx, day_idx = divmod(idx % total_cells, n_days)
            keep_pairs.append((day_idx, worker_idx))
        keep_pairs_set = set(keep_pairs)
        hole_candidates = [key for key in keep_pairs if key in lns.fixed_vars]
        hole_count = _ceil_fraction_count(len(hole_candidates), hole_fraction)
        hole_pairs = set(random.sample(hole_candidates, hole_count)) if hole_count > 0 else set()
        lns._last_destroy_selected_workers = sorted({worker for _, worker in keep_pairs})
        lns._last_destroy_selected_days = sorted({day for day, _ in keep_pairs})
        lns._last_destroy_selected_pairs = keep_pairs
        lns._last_destroy_holes_count = len(hole_pairs)

        freed: list[tuple[int, int]] = []
        for key in list(lns.fixed_vars):
            if key not in keep_pairs_set or key in hole_pairs:
                freed.append(key)
                del lns.fixed_vars[key]
        return freed

    return _op


def _make_destroy_maxsalvage(
    hole_fraction: float = 0.0,
) -> Callable[[rws_lns], list[tuple[int, int]]]:
    """Backward-compatible alias."""
    return _make_destroy_maxsalvage_streak_with_holes(hole_fraction)


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

###################################################################

if __name__ == "__main__":
    from rws_instance_loader import load_instance_and_schedule

    base = Path(__file__).resolve().parent
    raw_example = input("Example number [1]: ").strip()
    example_number = 1 if raw_example == "" else int(raw_example)

    instance_path = base / "Instances1-50" / f"Example{example_number}.txt"
    if not instance_path.exists():
        raise FileNotFoundError(f"instance file not found: {instance_path}")

    instance, schedule = load_instance_and_schedule(file_path=instance_path, cyclicity=True, initial_schedule='round_robin')

    model_path = base / "rws_instance.mzn"
    repair_ops: Dict[str, Callable[[rws_lns], None]] = {
        "repair_chuffed_fast": _make_repair_operator(model_path, "chuffed", 8),
        "repair_gecode_fast": _make_repair_operator(model_path, "gecode", 8),
        "repair_chuffed_long": _make_repair_operator(model_path, "chuffed", 24),
        "repair_gecode_long": _make_repair_operator(model_path, "gecode", 24),
    }


    destroy_ops: Dict[str, Callable[[rws_lns], list[tuple[int, int]]]] = {
        "destroy_worst_workers_10pct": _make_destroy_worst_workers(0.10),
        "destroy_worst_workers_30pct": _make_destroy_worst_workers(0.30),
        "destroy_random_workers_20pct": _make_destroy_random_workers(0.20),
        "destroy_random_window_20pct": _make_destroy_random_window(0.20),
        "destroy_maxsalvage_streak_holes_0pct": _make_destroy_maxsalvage_streak_with_holes(0.0),
        "destroy_maxsalvage_streak_holes_20pct": _make_destroy_maxsalvage_streak_with_holes(0.20),
        "destroy_worst_days_10pct": _make_destroy_worst_days(0.10),
        "destroy_worst_days_30pct": _make_destroy_worst_days(0.30),
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
        destroyed_label = step["destroyed_display"]

        summary_line = (
            f"iter={step['iteration']} "
            f"time={elapsed_total:.3f}s "
            f"objective={step['contender_objective']} "
            f"score={step['contender_score']} "
            f"{destroyed_label}"
        )
        if step["repair_failed"]:
            summary_line += " repair_failed"
        is_improvement = step["contender_objective"] < step["incumbent_objective"]
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
        if mab.objective_current_solution <= 0.0:
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
            "(objective_current_solution <= 0)."
        )
        print("Final schedule:")
        mab.schedule.display_schedule()
        mab.schedule.display_validity()
    elif timed_out:
        print(
            f"Timed out after {total_runtime:.3f}s at iteration {last_iteration}. "
            f"Schedule valid: {mab.schedule.is_valid()}"
        )
        print("Last schedule:")
        mab.schedule.display_schedule()
        mab.schedule.display_validity()

    print(f"Wrote run log: {log_path}")
