from __future__ import annotations

from datetime import timedelta
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# If performance becomes an issue, we can use numpy arrays for the assignment and consecutive counters, 
# but for simplicity and readability we use lists of lists here.

class RWS:
    """Rotating Workforce Scheduling container.

    - `Instance` stores model parameters and static constraints.
    - `Schedule` stores an actual assignment matrix and counters.
    """

    @dataclass
    class Instance:
        num_days: int
        num_workers: int
        shift_names: Sequence[str]
        cyclicity: bool = True
        #for now only 2-shift and 3-shift forbidden patterns are supported, but this can be extended if needed
        forbidden_sequences: Sequence[Union[Tuple[int, int], Tuple[int, int, int]]] = field(default_factory=tuple)
        min_consecutive_shift: Dict[int, int] = field(default_factory=dict)
        max_consecutive_shift: Dict[int, int] = field(default_factory=dict)
        min_consecutive_work: int = 0
        max_consecutive_work: int = 10**9
        min_consecutive_off: int = 0
        max_consecutive_off: int = 10**9
        required_number_of_shifts: Dict[int, Union[int, Sequence[int]]] = field(default_factory=dict)

        # not needed in basic version, but can be used to store additional constraints for schedule validation
        #time_off: Dict[int, Iterable[int]] = field(default_factory=dict)
        #workdays: Dict[int, Iterable[int]] = field(default_factory=dict)

        def __post_init__(self) -> None:
            if self.num_days <= 0:
                raise ValueError("num_days must be > 0")
            if self.num_workers <= 0:
                raise ValueError("num_workers must be > 0")
            if len(self.shift_names) == 0:
                raise ValueError("shift_names must include at least shift 0 (off)")

            # shift 0 is reserved for OFF by definition
            if self.shift_names[0].strip().lower() not in {"off", "0", "x", "-"}:
                raise ValueError("shift_names[0] must represent off (e.g. 'off')")

            max_shift = len(self.shift_names) - 1

            for seq in self.forbidden_sequences:
                if len(seq) < 2:
                    raise ValueError("forbidden_sequences must contain tuples with at least 2 elements")
                for shift_id in seq:
                    self._check_shift_id(shift_id, max_shift)

            for shift_id in self.min_consecutive_shift:
                self._check_shift_id(shift_id, max_shift)
            for shift_id in self.max_consecutive_shift:
                self._check_shift_id(shift_id, max_shift)

            for shift_id, mn in self.min_consecutive_shift.items():
                mx = self.max_consecutive_shift.get(shift_id, 10**9)
                if mn < 0 or mx < 0 or mn > mx:
                    raise ValueError(f"invalid min/max consecutive for shift {shift_id}")

            if self.min_consecutive_work < 0 or self.max_consecutive_work < 0:
                raise ValueError("workday consecutive bounds must be >= 0")
            if self.min_consecutive_work > self.max_consecutive_work:
                raise ValueError("min_consecutive_work > max_consecutive_work")

            if self.min_consecutive_off < 0 or self.max_consecutive_off < 0:
                raise ValueError("offday consecutive bounds must be >= 0")
            if self.min_consecutive_off > self.max_consecutive_off:
                raise ValueError("min_consecutive_off > max_consecutive_off")

            for shift_id, req_count in self.required_number_of_shifts.items():
                self._check_shift_id(shift_id, max_shift)
                if isinstance(req_count, int):
                    if req_count < 0:
                        raise ValueError(f"required_number_of_shifts for shift {shift_id} must be >= 0")
                else:
                    if len(req_count) != self.num_days:
                        raise ValueError(
                            f"required_number_of_shifts for shift {shift_id} has {len(req_count)} days, "
                            f"but instance has {self.num_days} days"
                        )
                    for day, count in enumerate(req_count):
                        if count < 0:
                            raise ValueError(
                                f"required_number_of_shifts for shift {shift_id} on day {day} must be >= 0"
                            )

            # self.time_off = {
            #     w: set(days) for w, days in self.time_off.items()
            # }
            # self.workdays = {
            #     w: set(days) for w, days in self.workdays.items()
            # }

            #for worker, days in self.time_off.items():
            #    self._check_worker(worker)
            #    self._check_days(days)
            #
            #for worker, days in self.workdays.items():
            #    self._check_worker(worker)
            #    self._check_days(days)

        def _check_shift_id(self, shift_id: int, max_shift: int) -> None:
            if not (0 <= shift_id <= max_shift):
                raise ValueError(f"invalid shift id {shift_id}; expected in [0, {max_shift}]")

        def _check_worker(self, worker: int) -> None:
            if not (0 <= worker < self.num_workers):
                raise ValueError(f"invalid worker id {worker}; expected in [0, {self.num_workers - 1}]")

        def _check_days(self, days: Iterable[int]) -> None:
            for d in days:
                if not (0 <= d < self.num_days):
                    raise ValueError(f"invalid day {d}; expected in [0, {self.num_days - 1}]")

    @dataclass
    class Schedule:
        instance: "RWS.Instance"
        assignment: List[List[int]]
        run_compatibility_check: bool = False
        compatibility_issues: List[str] = field(init=False, default_factory=list)

        def __post_init__(self) -> None:
            self._check_admissibility()
            if self.run_compatibility_check:
                self.compatibility_issues = self.check_compatibility()

        def _check_admissibility(self) -> None:
            inst = self.instance
            if len(self.assignment) != inst.num_days:
                raise AssertionError("assignment must contain one row per day")

            max_shift = len(inst.shift_names) - 1
            for day, row in enumerate(self.assignment):
                if len(row) != inst.num_workers:
                    raise AssertionError(f"day {day} does not contain num_workers entries")
                for worker, shift in enumerate(row):
                    if not (0 <= shift <= max_shift):
                        raise AssertionError(
                            f"invalid shift id at day {day}, worker {worker}: {shift}"
                        )

        def check_compatibility(self) -> List[str]:
            """Return empty list if schedule is feasible, otherwise summarize violations."""
            totals = self.count_total_violations()
            total_violations = sum(totals.values())
            if total_violations == 0:
                return []

            issues = [f"total violations: {total_violations}"]
            for key in ("sequence", "min", "max", "required"):
                count = totals[key]
                if count > 0:
                    issues.append(f"{key} violations: {count}")
            return issues

        # Detect all kindas of violations and return by worker and day involved

        def _detect_forbidden_sequences(self) -> Tuple[Dict[int, int], Dict[int, int]]:
            """Detect forbidden sequence violations by worker and by day."""
            inst = self.instance
            forbidden = set(inst.forbidden_sequences)
            by_worker: Dict[int, int] = {worker: 0 for worker in range(inst.num_workers)}
            by_day: Dict[int, int] = {day: 0 for day in range(inst.num_days)}

            for worker in range(inst.num_workers):
                day_range = range(inst.num_days) if inst.cyclicity else range(1, inst.num_days)
                for day in day_range:
                    prev_day = (day - 1) % inst.num_days
                    prev_shift = self.assignment[prev_day][worker]
                    cur_shift = self.assignment[day][worker]
                    
                    # Check 2-tuples
                    if (prev_shift, cur_shift) in forbidden:
                        by_worker[worker] += 1
                        by_day[prev_day] += 1
                        by_day[day] += 1
                    
                    # Check 3-shift patterns
                    if day >= 2 or (inst.cyclicity and day >= 1):
                        prev_prev_day = (day - 2) % inst.num_days
                        prev_prev_shift = self.assignment[prev_prev_day][worker]
                        if (prev_prev_shift, prev_shift, cur_shift) in forbidden:
                            by_worker[worker] += 1
                            by_day[prev_prev_day] += 1
                            by_day[prev_day] += 1
                            by_day[day] += 1

            return by_worker, by_day

        def _detect_min_violations(self) -> Tuple[Dict[int, int], Dict[int, int]]:
            """Detect minimum consecutive violations by worker and by day.
               Violations are not discriminated per workday, off-day and shift"""
            inst = self.instance
            by_worker: Dict[int, int] = {worker: 0 for worker in range(inst.num_workers)}
            by_day: Dict[int, int] = {day: 0 for day in range(inst.num_days)}

            for worker in range(inst.num_workers):
                work_runs = self._runs_for_worker_days(worker, lambda s: s != 0)
                off_runs = self._runs_for_worker_days(worker, lambda s: s == 0)

                for run_days in work_runs:
                    if len(run_days) < inst.min_consecutive_work:
                        by_worker[worker] += 1
                        for day in run_days:
                            by_day[day] += 1
                for run_days in off_runs:
                    if len(run_days) < inst.min_consecutive_off:
                        by_worker[worker] += 1
                        for day in run_days:
                            by_day[day] += 1

                for shift_id in range(len(inst.shift_names)):
                    shift_runs = self._runs_for_worker_days(worker, lambda s, sid=shift_id: s == sid)
                    min_shift = inst.min_consecutive_shift.get(shift_id, 0)
                    for run_days in shift_runs:
                        if len(run_days) < min_shift:
                            by_worker[worker] += 1
                            for day in run_days:
                                by_day[day] += 1

            return by_worker, by_day

        def _detect_max_violations(self) -> Tuple[Dict[int, int], Dict[int, int]]:
            """Detect maximum consecutive violations by worker and by day.
               Violations are not discriminated per workday, off-day and shift"""
            inst = self.instance
            by_worker: Dict[int, int] = {worker: 0 for worker in range(inst.num_workers)}
            by_day: Dict[int, int] = {day: 0 for day in range(inst.num_days)}

            for worker in range(inst.num_workers):
                work_runs = self._runs_for_worker_days(worker, lambda s: s != 0)
                off_runs = self._runs_for_worker_days(worker, lambda s: s == 0)

                for run_days in work_runs:
                    if len(run_days) > inst.max_consecutive_work:
                        by_worker[worker] += 1
                        for day in run_days:
                            by_day[day] += 1
                for run_days in off_runs:
                    if len(run_days) > inst.max_consecutive_off:
                        by_worker[worker] += 1
                        for day in run_days:
                            by_day[day] += 1

                for shift_id in range(len(inst.shift_names)):
                    shift_runs = self._runs_for_worker_days(worker, lambda s, sid=shift_id: s == sid)
                    max_shift = inst.max_consecutive_shift.get(shift_id, 10**9)
                    for run_days in shift_runs:
                        if len(run_days) > max_shift:
                            by_worker[worker] += 1
                            for day in run_days:
                                by_day[day] += 1

            return by_worker, by_day

        def _detect_required_shifts_violations(self) -> Tuple[Dict[int, int], Dict[int, int]]:
            """Detect required-shift violations by day; worker aggregation is kept at 0."""
            inst = self.instance
            by_worker: Dict[int, int] = {worker: 0 for worker in range(inst.num_workers)}
            by_day: Dict[int, int] = {day: 0 for day in range(inst.num_days)}

            for shift_id, req_count in inst.required_number_of_shifts.items():
                # Normalize to per-day requirements
                if isinstance(req_count, int):
                    required_per_day = [req_count] * inst.num_days
                else:
                    required_per_day = list(req_count)
                
                # Check per-day requirements
                for day, required in enumerate(required_per_day):
                    actual = sum(
                        1 for worker in range(inst.num_workers)
                        if self.assignment[day][worker] == shift_id
                    )
                    if actual != required:
                        by_day[day] += 1

            return by_worker, by_day

        
        # Aggregation of different types of constraint violations
        
        def detect_all_violations(self) -> Dict[str, Dict[str, Dict[int, int]]]:
            """Return per-type violation detection with by_worker and by_day counts."""
            sequence_by_worker, sequence_by_day = self._detect_forbidden_sequences()
            min_by_worker, min_by_day = self._detect_min_violations()
            max_by_worker, max_by_day = self._detect_max_violations()
            required_by_worker, required_by_day = self._detect_required_shifts_violations()
            return {
                "sequence": {"by_worker": sequence_by_worker, "by_day": sequence_by_day},
                "min": {"by_worker": min_by_worker, "by_day": min_by_day},
                "max": {"by_worker": max_by_worker, "by_day": max_by_day},
                "required": {"by_worker": required_by_worker, "by_day": required_by_day},
            }

        def count_total_violations(
            self,
            summary: Optional[Dict[str, Dict[str, Dict[int, int]]]] = None,
        ) -> Dict[str, int]:
            """Return per-type totals derived from detection output."""
            s = summary or self.detect_all_violations()
            return {
                "sequence": sum(s["sequence"]["by_worker"].values()),
                "min": sum(s["min"]["by_worker"].values()),
                "max": sum(s["max"]["by_worker"].values()),
                "required": sum(s["required"]["by_day"].values()),
            }


        # Helper methods for calculation of constraint-violations

        def _runs_for_worker_days(self, worker: int, day_condition) -> List[List[int]]:
            """Return day-index runs for one worker under the given condition."""
            inst = self.instance
            values = [day_condition(self.assignment[day][worker]) for day in range(inst.num_days)]
            return self._extract_runs_with_days(values, inst.cyclicity)
        
        @staticmethod
        def _extract_runs_with_days(flags: Sequence[bool], cyclic: bool) -> List[List[int]]:
            """Return runs as lists of day indices, merging cyclic boundary runs."""
            n = len(flags)
            if n == 0:
                return []

            runs: List[List[int]] = []
            i = 0
            while i < n:
                if flags[i]:
                    start = i
                    while i < n and flags[i]:
                        i += 1
                    runs.append(list(range(start, i)))
                else:
                    i += 1

            if cyclic and runs and flags[0] and flags[-1]:
                if len(runs) == 1 and len(runs[0]) == n:
                    return [list(range(n))]
                merged = runs[-1] + runs[0]
                middle = runs[1:-1] if len(runs) > 2 else []
                runs = [merged] + middle

            return runs

        # Display the parameters, the schedule and the violations nicely
        
        def display_schedule(self) -> None:
            """Display the schedule in a readable format."""
            inst = self.instance
            
            # Header
            print("\n" + "="*80)
            print(f"Schedule for {inst.num_workers} workers over {inst.num_days} days")
            print("="*80)

            if inst.forbidden_sequences:
                print("Forbidden sequences:")
                for seq in inst.forbidden_sequences:
                    names = " -> ".join(inst.shift_names[shift_id] for shift_id in seq)
                    print(f"  {names}")
            else:
                print("Forbidden sequences: none")

            print("Min/Max requirements:")
            print(
                f"  Work streak: min={inst.min_consecutive_work}, "
                f"max={inst.max_consecutive_work}"
            )
            print(
                f"  Off streak:  min={inst.min_consecutive_off}, "
                f"max={inst.max_consecutive_off}"
            )
            
            for shift_id in range(1, len(inst.shift_names)):
                shift_name = inst.shift_names[shift_id]
                min_shift = inst.min_consecutive_shift.get(shift_id, 0)
                max_shift = inst.max_consecutive_shift.get(shift_id)
                max_text = "inf" if max_shift is None else str(max_shift)
                print(f"    {shift_name} streak: min={min_shift}, max={max_text}")
            print("-"*80)
            
            # Day header
            print("Day:      ", end="")
            for day in range(inst.num_days):
                print(f"{day:>3}", end=" ")
            print()
            print("-"*80)
            
            # Worker assignments
            for worker in range(inst.num_workers):
                print(f"Worker {worker}: ", end="")
                for day in range(inst.num_days):
                    shift_id = self.assignment[day][worker]
                    shift_name = inst.shift_names[shift_id]
                    print(f"{shift_name:>3}", end=" ")
                print()
            
            print("="*80 + "\n")

        def display_violations(self) -> None:
            """Print compact violation summaries (per-worker, per-day and totals)."""
            summary = self.detect_all_violations()
            totals = self.count_total_violations(summary)
            inst = self.instance

            print("Violation counts per worker:")
            print("=" * 80)
            for worker in range(inst.num_workers):
                print(
                    f"  Worker {worker}: sequence={summary['sequence']['by_worker'][worker]:>2}  "
                    f"min={summary['min']['by_worker'][worker]:>2}  "
                    f"max={summary['max']['by_worker'][worker]:>2}"
                )
            print("=" * 80 + "\n")

            print("Violation counts per day:")
            print("=" * 80)
            for day in range(inst.num_days):
                print(
                    f"  Day {day}: sequence={summary['sequence']['by_day'][day]:>2}  "
                    f"min={summary['min']['by_day'][day]:>2}  "
                    f"max={summary['max']['by_day'][day]:>2}  "
                    f"required shifts={summary['required']['by_day'][day]:>2}"
                )
            print("=" * 80 + "\n")

            print("Total violated clauses:")
            print("=" * 80)
            for key, value in totals.items():
                print(f"  {key:.<40} {value:>3}")
            print("=" * 80)
            print(f"  {'Total violations':.<40} {sum(totals.values()):>3}")
            print("=" * 80 + "\n")


@dataclass
class rws_lns:
    """Minimal LNS context linking an `RWS.Instance` with a current schedule.

    This is a skeleton; replace the method bodies with your own LNS logic.
    """
    instance: "RWS.Instance"
    incumbent: "RWS.Schedule"
    contender: Optional["RWS.Schedule"] = None
    features: Any = None
    fixed_vars: Dict[Tuple[int, int], int] = field(default_factory=dict)


    # in the beginning treat all entries from the given schedule as fixed.
    def __post_init__(self) -> None:
        if not self.fixed_vars:
            self.fixed_vars = {
                (day, worker): self.incumbent.assignment[day][worker]
                for day in range(self.instance.num_days)
                for worker in range(self.instance.num_workers)
            }

    def destroy_worker(self, worker: int) -> List[Tuple[int, int]]:
        """Free all fixed variables for a given worker."""
        if not (0 <= worker < self.instance.num_workers):
            raise ValueError(
                f"invalid worker id {worker}; expected in [0, {self.instance.num_workers - 1}]"
            )
        freed = [key for key in self.fixed_vars if key[1] == worker]
        for key in freed:
            del self.fixed_vars[key]
        return freed

    def destroy_day(self, day: int) -> List[Tuple[int, int]]:
        """Free all fixed variables for a given day."""
        if not (0 <= day < self.instance.num_days):
            raise ValueError(f"invalid day {day}; expected in [0, {self.instance.num_days - 1}]")
        freed = [key for key in self.fixed_vars if key[0] == day]
        for key in freed:
            del self.fixed_vars[key]
        return freed


    def repair(self) -> "RWS.Schedule":
        """Repair by solving the MiniZinc subproblem induced by current fixed_vars."""
        model_path = Path(__file__).resolve().parent / "rws.mzn"
        solve_summary = solve_rws_lns(self, model_path=model_path)
        solution = solve_summary.get("solution")
        if not solve_summary.get("has_solution") or solution is None:
            raise RuntimeError(f"MiniZinc repair failed with status: {solve_summary['status']}")
        if "works" not in solution:
            raise RuntimeError("MiniZinc solution does not expose `works` variable")

        num_shifts = len(self.instance.shift_names) - 1
        assignment = _assignment_from_works(
            works=solution["works"],
            num_days=self.instance.num_days,
            num_workers=self.instance.num_workers,
            num_shifts=num_shifts,
        )
        self.contender = RWS.Schedule(instance=self.instance, assignment=assignment)
        return self.contender


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
    """Build max_min_lengths[0..s+1,1..2] from instance bounds."""
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
    solver_name: str = "gecode",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Solve `rws.mzn` using data and fixed-variable constraints from an rws_lns object."""
    from minizinc import Instance, Model, Solver

    model_file = Path(model_path)
    model = Model(str(model_file))

    solver = Solver.lookup(solver_name)
    mzn_instance = Instance(solver, model)

    num_shifts = len(lns.instance.shift_names) - 1
    mzn_instance["d"] = lns.instance.num_days
    mzn_instance["n"] = lns.instance.num_workers
    mzn_instance["s"] = num_shifts
    mzn_instance["min_rest"] = lns.instance.min_consecutive_off
    mzn_instance["max_rest"] = lns.instance.max_consecutive_off
    mzn_instance["min_work"] = lns.instance.min_consecutive_work
    mzn_instance["max_work"] = lns.instance.max_consecutive_work
    mzn_instance["max_min_lengths"] = _max_min_lengths_from_lns(lns)
    mzn_instance["required_workers"] = _required_workers_from_lns(lns)

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
    """Convert MiniZinc `works[w][d][shift]` solution into assignment[day][worker]."""
    assignment = [[0 for _ in range(num_workers)] for _ in range(num_days)]
    for worker in range(num_workers):
        for day in range(num_days):
            fixed_shift = None
            for shift in range(num_shifts + 1):
                if works[worker][day][shift] in (1, True):
                    fixed_shift = shift
                    break
            if fixed_shift is None:
                raise ValueError(
                    f"no assigned shift in MiniZinc solution for day={day}, worker={worker}"
                )
            assignment[day][worker] = fixed_shift
    return assignment


if __name__ == "__main__":
    # Minimal usage example
    instance = RWS.Instance(
        num_days=7,
        num_workers=4,
        shift_names=["-", "D", "A", "N"],
        cyclicity=True,
        forbidden_sequences=[(3, 1), (3, 3, 3)],  # N->D and N->N->D forbidden
        min_consecutive_shift={1: 1, 2: 1, 3: 1},
        max_consecutive_shift={1: 5, 2: 5, 3: 3},
        max_consecutive_work=4,
        max_consecutive_off=3,
        required_number_of_shifts={1: 1, 2: 1, 3: 1},  # D, A, N: 1 per day each (21 total)
        #time_off={0: {3}},
        #workdays={1: {0}},
    )

    schedule = RWS.Schedule(
        instance=instance,
        assignment=[
            [3, 1, 0, 0],      # Day 0: required shift violation (A=0 instead of 1
            [1, 3, 2, 0],      # Day 1: N->D violation for W0 (shift 3->1)
            [1, 0, 1, 3],      # Day 2: required shift violation (D=2 instead of 1)
            [0, 3, 0, 2],      # Day 3: required shift violation (N=2 instead of 1)
            [1, 2, 3, 0],      # Day 4: valid
            [2, 3, 1, 0],      # Day 5: valid
            [3, 1, 2, 0],      # Day 6: valid
        ],
    )

    schedule.display_schedule()
    schedule.display_violations()
