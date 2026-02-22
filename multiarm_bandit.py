from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Callable, Dict, Optional

from rws import RWS, rws_lns


def _default_score_function(
    number_of_conflicts_best: int,
    number_of_conflicts_incumbent: int,
    number_of_conflicts_contender: int,
    temperature: float,
) -> int:
    if number_of_conflicts_contender < number_of_conflicts_best:
        return 33
    elif number_of_conflicts_contender < number_of_conflicts_incumbent:
        return 9
    elif number_of_conflicts_contender == number_of_conflicts_incumbent:
        return 3
    p = math.exp(- (number_of_conflicts_contender - number_of_conflicts_incumbent)/temperature)
    return 3 if random.random() < p else 0

@dataclass
class bandit:
    """Configuration and operator container for a multiarm-bandit LNS loop."""

    instance: RWS.Instance
    schedule: RWS.Schedule
    weights_destroy: Optional[Dict[str, float]] = None
    weights_repair: Optional[Dict[str, float]] = None
    iterations_till_weight_update: int = 10
    reaction_factor: float = 0.2
    annealing_temperature: float = 5
    min_temperature: float = 0.7
    time_decay: float = 0.98
    epsilon: float = 0.05
    global_timeout_seconds: float = 300.0
    model_path: str | Path | None = None
    solver_name: str = "chuffed"
    minizinc_timeout_seconds: int = 10
    exploratory_timeout_seconds: float = 30
    exploration_after_stagnation: int = 5
    conflicts_best_solution: int = field(init=False)
    conflicts_current_solution: int = field(init=False)
    score_function: Callable[[int, int, int, float], int] = _default_score_function
    warmstart_instance: Optional[RWS.Instance] = None  # Not used by the LNS object yet.
    destroy_operators: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    repair_operators: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    repair_exploration_operator: Callable[[rws_lns], None] = field(init=False, repr=False)
    lns: rws_lns = field(init=False)
    lns_loop_counter: int = 0
    operator_score_sums: Dict[str, float] = field(init=False)
    operator_usage_counts: Dict[str, int] = field(init=False)
    stagnation_rounds: int = 0
    tabu: bool = False

    def __post_init__(self) -> None:
        if self.warmstart_instance is None:
            self.warmstart_instance = self.schedule.instance

        warmstart_conflicts = int(sum(self.schedule.count_total_violations().values()))
        self.conflicts_best_solution = warmstart_conflicts
        self.conflicts_current_solution = warmstart_conflicts

        if self.iterations_till_weight_update <= 0:
            raise ValueError("iterations_till_weight_update must be > 0")
        if not (0.0 <= self.reaction_factor <= 1.0):
            raise ValueError("reaction_factor must be in [0, 1]")
        if not (0.0 <= self.epsilon < 1.0):
            raise ValueError("epsilon must be in [0, 1)")
        if self.global_timeout_seconds <= 0:
            raise ValueError("global_timeout_seconds must be > 0")
        if self.exploration_after_stagnation <= 0:
            raise ValueError("exploration_after_stagnation must be > 0")
        if self.annealing_temperature <= 0:
            raise ValueError("annealing_temperature must be > 0")
        if self.min_temperature <= 0:
            raise ValueError("min_temperature must be > 0")
        if self.min_temperature > self.annealing_temperature:
            raise ValueError("min_temperature must be <= annealing_temperature")
        if not (0 < self.time_decay <= 1):
            raise ValueError("time_decay must be in (0, 1]")

        if self.minizinc_timeout_seconds <= 0:
            raise ValueError("minizinc_timeout_seconds must be > 0")
        if self.exploratory_timeout_seconds <= 0:
            raise ValueError("exploratory_timeout_seconds must be > 0")

            

        if not self.destroy_operators:
            self.destroy_operators = {
                "destroy_worker": (
                    lambda lns: lns.destroy_worker(random.randrange(lns.instance.num_workers))
                ),
                "destroy_day": (
                    lambda lns: lns.destroy_day(random.randrange(lns.instance.num_days))
                ),
            }
        if self.epsilon * len(self.destroy_operators) > 1.0:
            raise ValueError("epsilon too large for number of destroy operators")

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
        if self.epsilon * len(self.repair_operators) > 1.0:
            raise ValueError("epsilon too large for number of repair operators")
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
        self.lns = rws_lns(instance=self.instance, incumbent=self.schedule)


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
        normalized = {key: float(weights[key]) / total for key in keys}
        return self._normalize_weights(normalized)

    def _choose_repair_operator(self) -> tuple[str, Callable[..., Any]]:
        names = list(self.repair_operators.keys())
        probs = [self.weights_repair[name] for name in names]
        chosen = random.choices(names, weights=probs, k=1)[0]
        return chosen, self.repair_operators[chosen]

    def _choose_destroy_operator(self) -> tuple[str, Callable[..., Any]]:
        names = list(self.destroy_operators.keys())
        probs = [self.weights_destroy[name] for name in names]
        chosen = random.choices(names, weights=probs, k=1)[0]
        return chosen, self.destroy_operators[chosen]

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        n = len(weights)
        if n == 0:
            return {}
        if self.epsilon * n > 1.0:
            raise ValueError("epsilon too large for number of operators")

        total = float(sum(weights.values()))
        if total <= 0:
            normalized = {name: 1.0 / n for name in weights}
        else:
            normalized = {name: max(float(value), 0.0) / total for name, value in weights.items()}

        floor = self.epsilon
        if floor == 0.0:
            return normalized

        residual_mass = 1.0 - n * floor
        if residual_mass < 0.0:
            raise ValueError("epsilon too large for number of operators")

        extras = {name: max(value - floor, 0.0) for name, value in normalized.items()}
        extras_total = float(sum(extras.values()))
        if extras_total <= 0.0:
            return {name: 1.0 / n for name in weights}

        return {
            name: floor + residual_mass * (extra / extras_total)
            for name, extra in extras.items()
        }

    def _operator_key(self, kind: str, name: str) -> str:
        return f"{kind}::{name}"

    def _initialize_operator_tracking(self) -> None:
        keys = [self._operator_key("destroy", name) for name in self.destroy_operators]
        keys.extend(self._operator_key("repair", name) for name in self.repair_operators)
        self.operator_score_sums = {key: 0.0 for key in keys}
        self.operator_usage_counts = {key: 0 for key in keys}

    def _reset_operator_tracking(self) -> None:
        for key in self.operator_score_sums:
            self.operator_score_sums[key] = 0.0
        for key in self.operator_usage_counts:
            self.operator_usage_counts[key] = 0

    def update_operator_weights(self) -> None:
        for name, old_weight in self.weights_destroy.items():
            key = self._operator_key("destroy", name)
            usage = self.operator_usage_counts.get(key, 0)
            avg_score = (
                self.operator_score_sums.get(key, 0.0) / usage if usage > 0 else 0.0
            )
            self.weights_destroy[name] = (1 - self.reaction_factor) * float(old_weight) + (
                self.reaction_factor * avg_score
            )

        for name, old_weight in self.weights_repair.items():
            key = self._operator_key("repair", name)
            usage = self.operator_usage_counts.get(key, 0)
            avg_score = (
                self.operator_score_sums.get(key, 0.0) / usage if usage > 0 else 0.0
            )
            self.weights_repair[name] = (1 - self.reaction_factor) * float(old_weight) + (
                self.reaction_factor * avg_score
            )

        self.weights_destroy = self._normalize_weights(self.weights_destroy)
        self.weights_repair = self._normalize_weights(self.weights_repair)
        self._reset_operator_tracking()

    
    def _perform_lns_step(self) -> Dict[str, Any]:
        self.lns_loop_counter += 1
        lns = self.lns
        lns.incumbent = self.schedule
        lns.contender = None
        lns._initialize_fixed_vars(self.schedule)

        incumbent_conflicts = int(sum(self.schedule.count_total_violations().values()))

        destroy_name, destroy_op = self._choose_destroy_operator()
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

        destroy_result = destroy_op(lns)
        destroyed_workers = sorted({worker for _, worker in destroy_result})
        destroyed_days = sorted({day for day, _ in destroy_result})
        if "worker" in destroy_name:
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

        if not repair_failed and lns.contender is not None:
            contender_conflicts = int(sum(lns.contender.count_total_violations().values()))
            contender_score = int(
                self.score_function(
                    self.conflicts_best_solution,
                    self.conflicts_current_solution,
                    contender_conflicts,
                    self.annealing_temperature,
                )
            )
        else:
            repair_failed = True
            if repair_error is None:
                repair_error = "repair operator did not produce a contender schedule"
            contender_conflicts = incumbent_conflicts
            contender_score = 0
        self.annealing_temperature = max(
            self.annealing_temperature * self.time_decay,
            self.min_temperature,
        )
        
        destroy_key = self._operator_key("destroy", destroy_name)
        repair_key = self._operator_key("repair", repair_name)
        self.operator_score_sums[destroy_key] = (
            self.operator_score_sums.get(destroy_key, 0.0) + float(contender_score)
        )
        self.operator_usage_counts[destroy_key] = (
            self.operator_usage_counts.get(destroy_key, 0) + 1
        )
        self.operator_score_sums[repair_key] = (
            self.operator_score_sums.get(repair_key, 0.0) + float(contender_score)
        )
        self.operator_usage_counts[repair_key] = (
            self.operator_usage_counts.get(repair_key, 0) + 1
        )

        accepted = (not repair_failed) and contender_score > 2
        if accepted:
            self.schedule = lns.contender
            self.conflicts_current_solution = contender_conflicts
            if contender_conflicts < self.conflicts_best_solution:
                self.conflicts_best_solution = contender_conflicts
        if self.conflicts_current_solution < incumbent_conflicts:
            self.stagnation_rounds = 0
        else:
            self.stagnation_rounds += 1
        weights_updated = False
        destroy_weight_updates: Dict[str, Dict[str, float]] = {}
        repair_weight_updates: Dict[str, Dict[str, float]] = {}
        if update_this_round:
            self.update_operator_weights()
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


def _ceil_fraction_count(total: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return min(total, math.ceil(total * fraction))


def _take_ranked_ids(ranked: Dict[int, int], k: int) -> list[int]:
    if k <= 0:
        return []
    return list(ranked.keys())[:k]


def _make_destroy_worst_workers(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        schedule = _current_schedule(lns)
        k = _ceil_fraction_count(lns.instance.num_workers, fraction)
        worker_ids = _take_ranked_ids(schedule.worker_ranked_by_violations, k)
        return lns.destroy_worker(worker_ids)
    return _op


def _make_destroy_worst_days(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        schedule = _current_schedule(lns)
        k = _ceil_fraction_count(lns.instance.num_days, fraction)
        day_ids = _take_ranked_ids(schedule.days_ranked_by_violations, k)
        return lns.destroy_day(day_ids)
    return _op


def _make_destroy_random_workers(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        k = _ceil_fraction_count(lns.instance.num_workers, fraction)
        if k <= 0:
            return []
        worker_ids = random.sample(range(lns.instance.num_workers), k)
        return lns.destroy_worker(worker_ids)
    return _op


def _make_destroy_random_days(fraction: float) -> Callable[[rws_lns], list[tuple[int, int]]]:
    def _op(lns: rws_lns) -> list[tuple[int, int]]:
        k = _ceil_fraction_count(lns.instance.num_days, fraction)
        if k <= 0:
            return []
        day_ids = random.sample(range(lns.instance.num_days), k)
        return lns.destroy_day(day_ids)
    return _op



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
        "destroy_worst_days_10pct": _make_destroy_worst_days(0.10),
        "destroy_worst_days_20pct": _make_destroy_worst_days(0.20),
        "destroy_random_days_20pct": _make_destroy_random_days(0.20),
    }

    mab = bandit(
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
        print(summary_line)
        log_lines.append(
            (
                f"iter={step['iteration']} "
                f"elapsed={elapsed_total:.3f}s "
                f"step_runtime={step_runtime:.3f}s "
                f"destroy={step['selected_destroy_operator']} "
                f"repair={step['selected_repair_operator']} "
                f"incumbent_violations={step['incumbent_conflicts']} "
                f"contender_violations={step['contender_conflicts']} "
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
        mab.schedule.display_violations()
    elif timed_out:
        print(
            f"Timed out after {total_runtime:.3f}s at iteration {last_iteration}. "
            f"Current conflicts: {mab.conflicts_current_solution}"
        )

    print(f"Wrote run log: {log_path}")
