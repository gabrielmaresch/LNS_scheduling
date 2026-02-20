from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from rws import rws_lns


REQUIRED_MZN_PARAMS = (
    "d",
    "n",
    "s",
    "max_min_lengths",
    "min_rest",
    "max_rest",
    "min_work",
    "max_work",
    "required_workers",
)


def _required_workers_from_lns(lns: "rws_lns") -> List[List[int]]:
    """Build required_workers[day][shift] for shifts 1..s."""
    inst = lns.instance
    num_days = inst.num_days
    num_shifts = len(inst.shift_names) - 1

    required_workers = [[0 for _ in range(num_shifts)] for _ in range(num_days)]
    for shift_id in range(1, num_shifts + 1):
        req = inst.required_number_of_shifts.get(shift_id, 0)
        if isinstance(req, int):
            for day in range(num_days):
                required_workers[day][shift_id - 1] = req
        else:
            for day in range(num_days):
                required_workers[day][shift_id - 1] = req[day]
    return required_workers


def _max_min_lengths_from_lns(lns: "rws_lns") -> List[List[int]]:
    """Build max_min_lengths[0..s+1,1..2] from instance bounds.

    Row mapping:
    - 0: off
    - 1..s: exact shift IDs
    - s+1: total work streak
    """
    inst = lns.instance
    num_shifts = len(inst.shift_names) - 1
    num_days = inst.num_days

    rows: List[List[int]] = []
    for shift_id in range(0, num_shifts + 2):
        if shift_id == 0:
            mn = inst.min_consecutive_off
            mx = inst.max_consecutive_off
        elif 1 <= shift_id <= num_shifts:
            mn = inst.min_consecutive_shift.get(shift_id, 0)
            mx = inst.max_consecutive_shift.get(shift_id, num_days)
        else:
            mn = inst.min_consecutive_work
            mx = inst.max_consecutive_work
        rows.append([mn, mx])
    return rows


def _model_params_from_lns(lns: "rws_lns") -> Dict[str, Any]:
    """Build all required MiniZinc input parameters from an rws_lns object."""
    inst = lns.instance
    return {
        "d": inst.num_days,
        "n": inst.num_workers,
        "s": len(inst.shift_names) - 1,
        "max_min_lengths": _max_min_lengths_from_lns(lns),
        "min_rest": inst.min_consecutive_off,
        "max_rest": inst.max_consecutive_off,
        "min_work": inst.min_consecutive_work,
        "max_work": inst.max_consecutive_work,
        "required_workers": _required_workers_from_lns(lns),
    }


def _fixed_vars_constraints_mzn(lns: "rws_lns") -> str:
    """Generate MiniZinc constraints fixing selected (day, worker) shifts."""
    lines: List[str] = []
    for (day, worker), shift in sorted(lns.fixed_vars.items()):
        # MiniZinc arrays are 1-based for worker/day in this model.
        lines.append(f"constraint works[{worker + 1}, {day + 1}, {shift}] = 1;")
    return "\n".join(lines)


def solve_rws_lns(
    lns: "rws_lns",
    model_path: str | Path = "rws.mzn",
    solver_name: str = "chuffed",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Solve `rws.mzn` using data and fixed-variable constraints from an rws_lns object."""
    from minizinc import Instance, Model, Solver

    model_file = Path(model_path)
    model = Model(str(model_file))

    solver = Solver.lookup(solver_name)
    mzn_instance = Instance(solver, model)

    params = _model_params_from_lns(lns)
    for key, value in params.items():
        mzn_instance[key] = value

    missing = [name for name in REQUIRED_MZN_PARAMS if name not in params]
    if missing:
        raise ValueError(f"Missing required MiniZinc parameters: {missing}")

    fixed_constraints = _fixed_vars_constraints_mzn(lns)
    if fixed_constraints:
        mzn_instance.add_string(fixed_constraints)

    start = perf_counter()
    result = mzn_instance.solve(timeout=timedelta(seconds=timeout_seconds))
    elapsed = perf_counter() - start

    out: Dict[str, Any] = {
        "status": str(result.status),
        "solver": solver_name,
        "model": str(model_file),
        "solve_time_sec": elapsed,
        "has_solution": result.status.has_solution(),
    }

    if result.status.has_solution():
        out["solution"] = result.solution.__dict__
    else:
        out["solution"] = None
    return out


def _assignment_from_works(
    works: Any, num_days: int, num_workers: int, num_shifts: int
) -> List[List[int]]:
    """Convert MiniZinc `works[w][d][shift]` to assignment[day][worker]."""
    assignment = [[0 for _ in range(num_workers)] for _ in range(num_days)]
    for worker in range(num_workers):
        for day in range(num_days):
            chosen = None
            for shift in range(num_shifts + 1):
                if works[worker][day][shift] in (1, True):
                    chosen = shift
                    break
            if chosen is None:
                raise ValueError(f"No assigned shift for day={day}, worker={worker}")
            assignment[day][worker] = chosen
    return assignment


def _build_test_lns() -> "rws_lns":
    from rws import RWS, rws_lns

    instance = RWS.Instance(
        num_days=7,
        num_workers=4,
        shift_names=["-", "D", "A", "N"],
        cyclicity=True,
        forbidden_sequences=[(3, 1), (3, 3, 3)],
        min_consecutive_shift={1: 1, 2: 1, 3: 1},
        max_consecutive_shift={1: 5, 2: 5, 3: 3},
        max_consecutive_work=4,
        max_consecutive_off=3,
        required_number_of_shifts={1: 1, 2: 1, 3: 1},
    )

    schedule = RWS.Schedule(
        instance=instance,
        assignment=[
            [3, 1, 0, 0],
            [1, 3, 2, 0],
            [1, 0, 1, 3],
            [0, 3, 0, 2],
            [1, 2, 3, 0],
            [2, 3, 1, 0],
            [3, 1, 2, 0],
        ],
    )

    return rws_lns(instance=instance, incumbent=schedule)


if __name__ == "__main__":
    from rws import RWS, rws_lns

    seed_lns = _build_test_lns()
    current = seed_lns.incumbent
    instance = seed_lns.instance
    model_path = Path(__file__).resolve().parent / "rws.mzn"
    destroyed_workers: set[int] = set()
    round_idx = 1

    while True:
        print(f"\n=== Runde {round_idx} ===")
        current.display_schedule()
        totals = current.violation_totals()
        total_viol = sum(totals.values())
        print(f"Gesamtverletzungen: {total_viol} ({totals})")

        per_worker = current.violation_worker()
        print("Verletzungen pro Worker:")
        for worker in sorted(per_worker):
            stats = per_worker[worker]
            worker_total = sum(stats.values())
            print(f"  Worker {worker}: total={worker_total}, details={stats}")

        if total_viol == 0:
            break

        raw = input(f"Welcher Worker soll zerstört werden? (0..{instance.num_workers - 1}, q=abbrechen): ").strip().lower()
        if raw == "q":
            break
        if not raw.isdigit():
            print("Ungültige Eingabe.")
            continue

        worker = int(raw)
        if not (0 <= worker < instance.num_workers):
            print("Worker-ID außerhalb des Bereichs.")
            continue

        destroyed_workers.add(worker)
        print(f"Aktuelle Destroy-Menge: {sorted(destroyed_workers)}")

        lns = rws_lns(instance=instance, incumbent=current)
        for w in sorted(destroyed_workers):
            lns.destroy_worker(w)

        summary = solve_rws_lns(lns=lns, model_path=model_path, solver_name="chuffed", timeout_seconds=30)
        print(f"Solver status: {summary['status']}")
        if not summary["has_solution"] or summary["solution"] is None:
            print("Keine Lösung gefunden, Schleife wird beendet.")
            break

        works = summary["solution"]["works"]
        num_shifts = len(instance.shift_names) - 1
        assignment = _assignment_from_works(
            works=works,
            num_days=instance.num_days,
            num_workers=instance.num_workers,
            num_shifts=num_shifts,
        )
        current = RWS.Schedule(instance=instance, assignment=assignment)
        round_idx += 1

    print("\n=== Finaler Check ===")
    final_issues = current.check_compatibility()
    if not final_issues:
        print("check_schedule: OK (keine Kompatibilitätsprobleme gefunden)")
    else:
        print("check_schedule: Probleme gefunden")
        for issue in final_issues:
            print(f"  - {issue}")
