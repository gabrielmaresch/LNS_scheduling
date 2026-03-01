from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from time import perf_counter

from rws import rws_lns
from rws_instance_loader import load_instance_and_schedule
from rws_mzk_pipeline import build_rws_model_instance
from tabu import tabu


def run_minizinc_full(
    instance,
    schedule,
    model_path: Path,
    timeout_seconds: float,
    late_phase: bool,
    late_phase_weight: int = 100,
    solver_name: str = "chuffed",
) -> dict[str, float | str | None]:
    lns = rws_lns(instance=instance, incumbent=schedule)
    lns.fixed_vars.clear()
    lns._late_phase = bool(late_phase)
    lns._late_phase_weight = int(late_phase_weight)
    lns._late_phase_strict_improvement = False
    lns._incumbent_legacy_objective = 10**9

    model_instance, _ = build_rws_model_instance(
        lns=lns,
        model_path=model_path,
        solver_name=solver_name,
        sloppy=False,
    )

    with model_instance.branch() as run_instance:
        run_instance["num_fixed_assignments"] = 0
        run_instance["fixed_assignments"] = []
        start = perf_counter()
        result = run_instance.solve(timeout=timedelta(seconds=timeout_seconds))
        runtime_seconds = perf_counter() - start

    status = str(result.status)
    solver_objective = float(result.objective) if result.objective is not None else None
    late_phase_objective = None
    late_phase_primary = None
    late_phase_secondary = None
    legacy_objective = None

    if result.status.has_solution() and result.solution is not None:
        solved = result.solution.__dict__
        late_phase_objective = solved.get("late_phase_objective")
        late_phase_primary = solved.get("late_phase_primary_objective")
        late_phase_secondary = solved.get("late_phase_secondary_objective")
        legacy_objective = solved.get("legacy_objective")

    return {
        "status": status,
        "runtime_seconds": runtime_seconds,
        "solver_objective": None if solver_objective is None else float(solver_objective),
        "late_phase_objective": None if late_phase_objective is None else float(late_phase_objective),
        "late_phase_primary_objective": None if late_phase_primary is None else float(late_phase_primary),
        "late_phase_secondary_objective": None if late_phase_secondary is None else float(late_phase_secondary),
        "legacy_objective": None if legacy_objective is None else float(legacy_objective),
    }


def main() -> None:
    base = Path(__file__).resolve().parent

    raw_example = input("Example number [1]: ").strip()
    example_number = 1 if raw_example == "" else int(raw_example)
    if example_number < 0:
        raise ValueError("example number must be >= 0")

    raw_timeout = input("Timeout seconds [120]: ").strip()
    timeout_seconds = 120.0 if raw_timeout == "" else float(raw_timeout)
    if timeout_seconds <= 0:
        raise ValueError("timeout seconds must be > 0")

    instance_path = base / "Instances1-50" / f"Example{example_number}.txt"
    if not instance_path.exists():
        raise FileNotFoundError(f"instance file not found: {instance_path}")

    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True,
        initial_schedule="random",
    )
    print(
        f"instance_specs: days={instance.num_days} workers={instance.num_workers} "
        f"shifts={len(instance.shift_names) - 1} forbidden={len(instance.forbidden_sequences)}"
    )

    model_path = base / "rws_instance.mzn"
    tiebreak_lns = rws_lns(instance=instance, incumbent=schedule)
    tiebreak_lns._late_phase = False
    tiebreak_lns._late_phase_strict_improvement = False
    tiebreak_model_instance, _ = build_rws_model_instance(
        lns=tiebreak_lns,
        model_path=model_path,
        solver_name="chuffed",
        sloppy=False,
    )

    print(f"Loaded instance: {instance_path}")
    print("count trajectory:")
    best_schedule = tabu(
        instance=instance,
        schedule=schedule,
        tabu_length=40,
        timeout=timeout_seconds,
        use_objective_tiebreaker=True,
        model_path=model_path,
        model_instance=tiebreak_model_instance,
        objective_tiebreak_timeout_seconds=2,
        objective_tiebreak_max_count=10,
    )

    final_objective = best_schedule.objective_value(
        model_path=model_path,
        model_instance=tiebreak_model_instance,
        solver_name="chuffed",
        timeout_seconds=max(1, int(timeout_seconds)),
    )
    print(f"method=tabu_with_count_tiebreak legacy_objective={final_objective}")

    late_phase_result = run_minizinc_full(
        instance=instance,
        schedule=schedule,
        model_path=model_path,
        timeout_seconds=timeout_seconds,
        late_phase=True,
        late_phase_weight=1000,
        solver_name="gecode",
    )
    print(f"method=minizinc_late_phase_w1000 legacy_objective={late_phase_result['legacy_objective']}")

    plain_result = run_minizinc_full(
        instance=instance,
        schedule=schedule,
        model_path=model_path,
        timeout_seconds=timeout_seconds,
        late_phase=False,
        late_phase_weight=1000,
        solver_name="gecode",
    )
    print(f"method=minizinc_plain_no_latephase legacy_objective={plain_result['legacy_objective']}")


if __name__ == "__main__":
    main()
