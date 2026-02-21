from __future__ import annotations

from datetime import timedelta
from dataclasses import dataclass, field
from pathlib import Path
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
    _cached_model_instance: Any = field(default=None, init=False, repr=False)
    _cached_model_path: Optional[Path] = field(default=None, init=False, repr=False)
    _cached_solver_name: Optional[str] = field(default=None, init=False, repr=False)


    def _initialize_fixed_vars(self, schedule: Optional["RWS.Schedule"] = None) -> None:
        """(Re)initialize fixed vars from the provided schedule (default: incumbent)."""
        src = self.incumbent if schedule is None else schedule
        self.fixed_vars = {
            (day, worker): src.assignment[day][worker]
            for day in range(self.instance.num_days)
            for worker in range(self.instance.num_workers)
        }

    def __post_init__(self) -> None:
        if not self.fixed_vars:
            self._initialize_fixed_vars()

    def destroy_worker(self, worker: int | Iterable[int]) -> List[Tuple[int, int]]:
        """Free all fixed variables for one or many workers."""
        workers = {worker} if isinstance(worker, int) else set(worker)
        if not workers:
            return []
        for worker_id in workers:
            if not (0 <= worker_id < self.instance.num_workers):
                raise ValueError(
                    f"invalid worker id {worker_id}; expected in [0, {self.instance.num_workers - 1}]"
                )
        freed = [key for key in self.fixed_vars if key[1] in workers]
        for key in freed:
            del self.fixed_vars[key]
        return freed

    def destroy_day(self, day: int | Iterable[int]) -> List[Tuple[int, int]]:
        """Free all fixed variables for one or many days."""
        days = {day} if isinstance(day, int) else set(day)
        if not days:
            return []
        for day_id in days:
            if not (0 <= day_id < self.instance.num_days):
                raise ValueError(f"invalid day {day_id}; expected in [0, {self.instance.num_days - 1}]")
        freed = [key for key in self.fixed_vars if key[0] in days]
        for key in freed:
            del self.fixed_vars[key]
        return freed

    def repair_exact(
        self,
        model_instance: Any | None = None,
        model_path: str | Path | None = None,
        solver_name: str = "chuffed",
        timeout_seconds: int = 30,
    ) -> None:
        """Run an exact MiniZinc repair and store the result in `self.contender`.

        If `model_instance` is provided, it is reused directly; otherwise a new one
        is built from `model_path` and `solver_name` and then cached for reuse.
        """
        from rws_mzk_pipeline import build_rws_model_instance, solve_rws_lns

        if model_instance is None:
            if model_path is None:
                model_path = Path(__file__).resolve().parent / "rws_instance.mzn"
            resolved_model_path = Path(model_path)
            if not resolved_model_path.is_absolute():
                resolved_model_path = Path(__file__).resolve().parent / resolved_model_path

            if (
                self._cached_model_instance is None
                or self._cached_model_path != resolved_model_path
                or self._cached_solver_name != solver_name
            ):
                self._cached_model_instance, _ = build_rws_model_instance(
                    lns=self,
                    model_path=resolved_model_path,
                    solver_name=solver_name,
                )
                self._cached_model_path = resolved_model_path
                self._cached_solver_name = solver_name

            model_instance = self._cached_model_instance
        else:
            self._cached_model_instance = model_instance
        summary = solve_rws_lns(
            lns=self,
            model_instance=model_instance,
            timeout_seconds=timeout_seconds,
        )
        if not summary.get("has_solution") or self.contender is None:
            raise RuntimeError(f"MiniZinc repair failed with status: {summary['status']}")

        self._initialize_fixed_vars(self.contender)


def _parse_id_list(raw: str) -> List[int]:
    """Parse comma-separated integer IDs."""
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


if __name__ == "__main__":
    from rws_mzk_pipeline import build_rws_model_instance, solve_rws_lns
    from rws_instance_loader import load_instance_and_schedule

    raw_example = input("Example number [1]: ").strip()
    example_number = 1 if raw_example == "" else int(raw_example)
    if example_number < 1:
        raise ValueError("example number must be >= 1")

    instance_path = (
        Path(__file__).resolve().parent / "Instances1-50" / f"Example{example_number}.txt"
    )
    if not instance_path.exists():
        raise FileNotFoundError(f"instance file not found: {instance_path}")

    instance, schedule = load_instance_and_schedule(
        file_path=instance_path,
        cyclicity=True,
    )
    print(f"Loaded instance: {instance_path}")

    lns = rws_lns(instance=instance, incumbent=schedule)
    solver_name = "chuffed"
    timeout_seconds = 30
    model_path = Path(__file__).resolve().parent / "rws_instance.mzn"
    model_instance, _ = build_rws_model_instance(
        lns=lns,
        model_path=model_path,
        solver_name=solver_name,
    )
    run_idx = 1

    while True:
        current_schedule = lns.contender if lns.contender is not None else lns.incumbent
        before_totals = current_schedule.count_total_violations()
        before_total = sum(before_totals.values())

        print(f"\n=== LNS run {run_idx} ===")
        print("Current schedule before destroy/repair:")
        current_schedule.display_schedule()
        current_schedule.display_violations()
        print(f"Total violations before repair: {before_total} ({before_totals})")
        print("Available destroy operators: worker, day")

        selected_raw = input("Which destroy operators to apply? (comma-separated): ").strip().lower()
        selected_ops = {op.strip() for op in selected_raw.split(",") if op.strip()}

        if "worker" in selected_ops:
            raw_workers = input(
                f"Worker ids to destroy (comma-separated, 0..{instance.num_workers - 1}): "
            ).strip()
            worker_ids = _parse_id_list(raw_workers)
            freed = lns.destroy_worker(worker_ids)
            print(f"Destroyed worker vars: {len(freed)}")

        if "day" in selected_ops:
            raw_days = input(
                f"Day ids to destroy (comma-separated, 0..{instance.num_days - 1}): "
            ).strip()
            day_ids = _parse_id_list(raw_days)
            freed = lns.destroy_day(day_ids)
            print(f"Destroyed day vars: {len(freed)}")

        summary = solve_rws_lns(
            lns=lns,
            model_instance=model_instance,
            timeout_seconds=timeout_seconds,
        )
        runtime = summary["solve_time_sec"]
        if not summary.get("has_solution") or lns.contender is None:
            raise RuntimeError(f"MiniZinc repair failed with status: {summary['status']}")
        lns._initialize_fixed_vars(lns.contender)

        after_totals = lns.contender.count_total_violations()
        after_total = sum(after_totals.values())
        delta = after_total - before_total

        print("Schedule after repair:")
        lns.contender.display_schedule()
        lns.contender.display_violations()
        print(f"Total violations after repair:  {after_total} ({after_totals})")
        print(f"Total violation delta (after - before): {delta}")
        print(f"Repair runtime: {runtime:.3f}s")

        abort_raw = input("Abort further runs? [y/N]: ").strip().lower()
        if abort_raw in {"y", "yes"}:
            break

        run_idx += 1
