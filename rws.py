from __future__ import annotations

from datetime import timedelta
from dataclasses import dataclass, field
from pathlib import Path
import random
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
            """Validate instance dimensions, bounds, shift IDs, and input arrays."""
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
            """Validate that a shift ID lies in the allowed [0, max_shift] range."""
            if not (0 <= shift_id <= max_shift):
                raise ValueError(f"invalid shift id {shift_id}; expected in [0, {max_shift}]")

        def _check_worker(self, worker: int) -> None:
            """Validate that a worker ID lies in the allowed worker range."""
            if not (0 <= worker < self.num_workers):
                raise ValueError(f"invalid worker id {worker}; expected in [0, {self.num_workers - 1}]")

        def _check_days(self, days: Iterable[int]) -> None:
            """Validate that each day index lies in the allowed day range."""
            for d in days:
                if not (0 <= d < self.num_days):
                    raise ValueError(f"invalid day {d}; expected in [0, {self.num_days - 1}]")

    @dataclass
    class Schedule:
        instance: "RWS.Instance"
        assignment: List[List[int]]

        def __post_init__(self) -> None:
            """Validate assignment matrix shape and value domain."""
            self._check_admissibility()

        def _check_admissibility(self) -> None:
            """Assert assignment matrix shape and shift values fit the instance."""
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


        def is_valid(self) -> bool:
            """Single schedule validity check across all modeled constraints."""
            inst = self.instance
            worker_first = self.worker_days_until_first_violation()
            if any(day <= inst.num_days for day in worker_first.values()):
                return False
            return not any(
                count > 0 for count in self.day_shift_requirement_violation_counts().values()
            )

        def worker_days_until_first_violation(self) -> Dict[int, int]:
            """Return per-worker days until first worker-level violation (1..d, d+1 if none)."""
            inst = self.instance
            first_day: Dict[int, int] = {worker: inst.num_days + 1 for worker in range(inst.num_workers)}

            def _mark(worker: int, day: int) -> None:
                first_day[worker] = min(first_day[worker], day + 1)

            # Sequence violations can involve rotated boundary slots, so mark all involved workers.
            forbidden = set(inst.forbidden_sequences)
            if forbidden:
                for worker in range(inst.num_workers):
                    for day in range(inst.num_days):
                        cur_shift = self.assignment[day][worker]

                        if inst.cyclicity or day < inst.num_days - 1:
                            next_worker, next_day = self._next_slot(worker, day)
                            next_shift = self.assignment[next_day][next_worker]
                            if (cur_shift, next_shift) in forbidden:
                                _mark(worker, day)
                                _mark(next_worker, next_day)

                        if inst.cyclicity or day < inst.num_days - 2:
                            next_worker, next_day = self._next_slot(worker, day)
                            next2_worker, next2_day = self._next_slot(next_worker, next_day)
                            next_shift = self.assignment[next_day][next_worker]
                            next2_shift = self.assignment[next2_day][next2_worker]
                            if (cur_shift, next_shift, next2_shift) in forbidden:
                                _mark(worker, day)
                                _mark(next_worker, next_day)
                                _mark(next2_worker, next2_day)

            for worker in range(inst.num_workers):
                work_runs = self._runs_for_worker_days(worker, lambda s: s != 0)
                off_runs = self._runs_for_worker_days(worker, lambda s: s == 0)

                for run_days in work_runs:
                    if not run_days:
                        continue
                    run_len = len(run_days)
                    if run_days[0] == 0 and inst.cyclicity:
                        prev_worker = self._prev_worker(worker)
                        carry_len = self._tail_run_length(prev_worker, lambda s: s != 0)
                        run_len = min(inst.num_days, run_len + carry_len)
                    continues = self._run_continues_in_next_worker(worker, run_days, lambda s: s != 0)
                    if (not continues and run_len < inst.min_consecutive_work) or run_len > inst.max_consecutive_work:
                        _mark(worker, run_days[0])

                for run_days in off_runs:
                    if not run_days:
                        continue
                    continues = self._run_continues_in_next_worker(worker, run_days, lambda s: s == 0)
                    run_len = len(run_days)
                    if (not continues and run_len < inst.min_consecutive_off) or run_len > inst.max_consecutive_off:
                        _mark(worker, run_days[0])

                for shift_id in range(len(inst.shift_names)):
                    shift_runs = self._runs_for_worker_days(worker, lambda s, sid=shift_id: s == sid)
                    min_shift = inst.min_consecutive_shift.get(shift_id, 0)
                    max_shift = inst.max_consecutive_shift.get(shift_id, 10**9)
                    for run_days in shift_runs:
                        if not run_days:
                            continue
                        continues = self._run_continues_in_next_worker(
                            worker,
                            run_days,
                            lambda s, sid=shift_id: s == sid,
                        )
                        run_len = len(run_days)
                        if (not continues and run_len < min_shift) or run_len > max_shift:
                            _mark(worker, run_days[0])

            return first_day

        def day_shift_requirement_violation_counts(self) -> Dict[int, int]:
            """Return per-day shift-requirement mismatch counts (sum of absolute deltas)."""
            inst = self.instance
            by_day: Dict[int, int] = {day: 0 for day in range(inst.num_days)}
            for shift_id, req_count in inst.required_number_of_shifts.items():
                if isinstance(req_count, int):
                    required_per_day = [req_count] * inst.num_days
                else:
                    required_per_day = list(req_count)
                for day, required in enumerate(required_per_day):
                    actual = sum(
                        1 for worker in range(inst.num_workers)
                        if self.assignment[day][worker] == shift_id
                    )
                    # Ranking signal: total staffing mismatch magnitude for this day.
                    by_day[day] += abs(actual - required)
            return by_day

        # Helper methods for validity checks

        def _runs_for_worker_days(self, worker: int, day_condition) -> List[List[int]]:
            """Return day-index runs for one worker under the given condition."""
            inst = self.instance
            values = [day_condition(self.assignment[day][worker]) for day in range(inst.num_days)]
            return self._extract_runs_with_days(values)

        def _next_worker(self, worker: int) -> int:
            inst = self.instance
            return (worker + 1) % inst.num_workers

        def _prev_worker(self, worker: int) -> int:
            inst = self.instance
            return (worker - 1) % inst.num_workers

        def _next_slot(self, worker: int, day: int) -> tuple[int, int]:
            inst = self.instance
            if day < inst.num_days - 1:
                return worker, day + 1
            return self._next_worker(worker), 0

        def _tail_run_length(self, worker: int, day_condition) -> int:
            inst = self.instance
            length = 0
            for day in range(inst.num_days - 1, -1, -1):
                if day_condition(self.assignment[day][worker]):
                    length += 1
                else:
                    break
            return length

        def _run_continues_in_next_worker(
            self,
            worker: int,
            run_days: Sequence[int],
            day_condition,
        ) -> bool:
            inst = self.instance
            if not inst.cyclicity or not run_days:
                return False
            if run_days[-1] != inst.num_days - 1:
                return False
            next_worker = self._next_worker(worker)
            return day_condition(self.assignment[0][next_worker])
        
        @staticmethod
        def _extract_runs_with_days(flags: Sequence[bool]) -> List[List[int]]:
            """Return runs as lists of day indices."""
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

            return runs

        @staticmethod
        def _extract_runs_with_days_cyclic(flags: Sequence[bool], cyclic: bool) -> List[List[int]]:
            """Return runs as day-index lists, optionally merged across cyclic boundary."""
            runs = RWS.Schedule._extract_runs_with_days(flags)
            if cyclic and len(runs) > 1 and flags and flags[0] and flags[-1]:
                runs[0] = runs[-1] + runs[0]
                runs.pop()
            return runs

        def _max_feasable_blocked_days(self) -> Dict[int, List[bool]]:
            """Build per-worker/day blocked flags for streak feasibility checks."""
            inst = self.instance
            n_days = inst.num_days
            blocked: Dict[int, List[bool]] = {
                worker: [False] * n_days for worker in range(inst.num_workers)
            }
            first_day = self.worker_days_until_first_violation()
            for worker, day1 in first_day.items():
                if day1 <= n_days:
                    blocked[worker][day1 - 1] = True

            return blocked

        def max_feasable_streak(
            self,
            worker: int,
            day: int,
            forward: bool = True,
            _blocked_days: Optional[Dict[int, List[bool]]] = None,
            _memo: Optional[Dict[Tuple[int, int, int], int]] = None,
        ) -> int:
            """Return streak length in circular flattened (worker, day) order."""
            inst = self.instance
            if not (0 <= worker < inst.num_workers):
                raise ValueError(f"invalid worker {worker}; expected in [0, {inst.num_workers - 1}]")
            if not (0 <= day < inst.num_days):
                raise ValueError(f"invalid day {day}; expected in [0, {inst.num_days - 1}]")

            blocked_days = self._max_feasable_blocked_days() if _blocked_days is None else _blocked_days
            memo = {} if _memo is None else _memo
            step = 1 if forward else -1
            key = (worker, day, step)
            if key in memo:
                return memo[key]

            if blocked_days[worker][day]:
                memo[key] = 0
                return 0

            n_days = inst.num_days
            total_cells = inst.num_workers * n_days
            start_idx = worker * n_days + day
            visited: List[int] = []
            idx = start_idx
            while len(visited) < total_cells:
                w_idx, d_idx = divmod(idx, n_days)
                if blocked_days[w_idx][d_idx]:
                    break
                visited.append(idx)
                next_idx = (idx + step) % total_cells
                if next_idx == start_idx:
                    break
                idx = next_idx

            if not visited:
                memo[key] = 0
                return 0

            if len(visited) == total_cells:
                for i in range(total_cells):
                    w_idx, d_idx = divmod(i, n_days)
                    memo[(w_idx, d_idx, step)] = total_cells
                return total_cells

            run_len = len(visited)
            for pos, flat_idx in enumerate(visited):
                w_idx, d_idx = divmod(flat_idx, n_days)
                memo[(w_idx, d_idx, step)] = run_len - pos
            return memo[key]

        def days_shuffle_cyclic(self, shift: Optional[int] = None) -> None:
            """Cyclically left-shift day assignments in-place."""
            n_days = self.instance.num_days
            if n_days <= 1:
                return
            used_shift = random.randrange(n_days) if shift is None else int(shift) % n_days
            if used_shift == 0:
                return
            self.assignment[:] = self.assignment[used_shift:] + self.assignment[:used_shift]

        def workers_shuffle_cyclic(self, shift: Optional[int] = None) -> None:
            """Cyclically left-shift worker assignments in-place."""
            n_workers = self.instance.num_workers
            if n_workers <= 1:
                return
            used_shift = random.randrange(n_workers) if shift is None else int(shift) % n_workers
            if used_shift == 0:
                return
            for day in range(self.instance.num_days):
                row = self.assignment[day]
                row[:] = row[used_shift:] + row[:used_shift]

        # Display the parameters and schedule nicely
        
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

        def display_validity(self) -> None:
            """Print whether the schedule is valid."""
            print("Schedule valid." if self.is_valid() else "Schedule invalid.")


@dataclass
class rws_lns:
    """Minimal LNS context linking an `RWS.Instance` with a current schedule.

    This is a skeleton; replace the method bodies with your own LNS logic.
    """
    instance: "RWS.Instance"
    incumbent: "RWS.Schedule"
    contender: Optional["RWS.Schedule"] = None
    contender_objective: Optional[float] = None
    features: Any = None
    fixed_vars: Dict[Tuple[int, int], int] = field(default_factory=dict)
    _cached_model_instances: Dict[Tuple[Path, str, bool], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _cached_instance_id: Optional[int] = field(default=None, init=False, repr=False)


    def _initialize_fixed_vars(self, schedule: Optional["RWS.Schedule"] = None) -> None:
        """(Re)initialize fixed vars from the provided schedule (default: incumbent)."""
        src = self.incumbent if schedule is None else schedule
        self.fixed_vars = {
            (day, worker): src.assignment[day][worker]
            for day in range(self.instance.num_days)
            for worker in range(self.instance.num_workers)
        }

    def __post_init__(self) -> None:
        """Initialize fixed variables from incumbent when no fixed set is provided."""
        self._cached_instance_id = id(self.instance)
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

    def destroy_window(
        self,
        workers: int | Iterable[int],
        days: int | Iterable[int],
    ) -> List[Tuple[int, int]]:
        """Free fixed variables for the Cartesian window `days x workers`."""
        worker_ids = {workers} if isinstance(workers, int) else set(workers)
        day_ids = {days} if isinstance(days, int) else set(days)
        if not worker_ids or not day_ids:
            return []

        for worker_id in worker_ids:
            if not (0 <= worker_id < self.instance.num_workers):
                raise ValueError(
                    f"invalid worker id {worker_id}; expected in [0, {self.instance.num_workers - 1}]"
                )
        for day_id in day_ids:
            if not (0 <= day_id < self.instance.num_days):
                raise ValueError(f"invalid day {day_id}; expected in [0, {self.instance.num_days - 1}]")

        freed = [key for key in self.fixed_vars if key[0] in day_ids and key[1] in worker_ids]
        for key in freed:
            del self.fixed_vars[key]
        return freed

    def repair_exact(
        self,
        model_instance: Any | None = None,
        model_path: str | Path | None = None,
        solver_name: str = "chuffed",
        sloppy: bool = False,
        timeout_seconds: int = 10,
    ) -> None:
        """Run an exact MiniZinc repair and store the result in `self.contender`.

        Model instances are cached by `(model_path, solver_name, sloppy)` and the
        cache is cleared automatically when the `instance` object changes.
        If `model_instance` is provided explicitly, it is used directly and also
        stored under the same cache key for future reuse.
        """
        from rws_mzk_pipeline import build_rws_model_instance, solve_rws_lns

        current_instance_id = id(self.instance)
        if self._cached_instance_id != current_instance_id:
            self._cached_model_instances.clear()
            self._cached_instance_id = current_instance_id

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "rws_instance.mzn"
        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            resolved_model_path = Path(__file__).resolve().parent / resolved_model_path
        cache_key = (resolved_model_path, solver_name, sloppy)

        if model_instance is None:
            model_instance = self._cached_model_instances.get(cache_key)
            if model_instance is None:
                model_instance, _ = build_rws_model_instance(
                    lns=self,
                    model_path=resolved_model_path,
                    solver_name=solver_name,
                    sloppy=sloppy,
                )
                self._cached_model_instances[cache_key] = model_instance
        else:
            self._cached_model_instances[cache_key] = model_instance
        self.contender_objective = None
        summary = solve_rws_lns(
            lns=self,
            model_instance=model_instance,
            timeout_seconds=timeout_seconds,
        )
        if not summary.get("has_solution") or self.contender is None:
            raise RuntimeError(f"MiniZinc repair failed with status: {summary['status']}")
        objective = summary.get("objective")
        self.contender_objective = float(objective) if objective is not None else None

        self._initialize_fixed_vars(self.contender)


def _parse_id_list(raw: str) -> List[int]:
    """Parse comma-separated integer IDs for main"""
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


if __name__ == "__main__":
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
    solver_name = "gecode"
    sloppy = True
    timeout_seconds = 1
    model_path = Path(__file__).resolve().parent / "rws_instance.mzn"
    run_idx = 1

    while True:
        current_schedule = lns.contender if lns.contender is not None else lns.incumbent
        before_valid = current_schedule.is_valid()

        print(f"\n=== LNS run {run_idx} ===")
        print("Current schedule before destroy/repair:")
        current_schedule.display_schedule()
        current_schedule.display_validity()
        print(f"Valid before repair: {before_valid}")
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

        repair_start = perf_counter()
        lns.repair_exact(
            model_path=model_path,
            solver_name=solver_name,
            timeout_seconds=timeout_seconds,
            sloppy=False
        )
        runtime = perf_counter() - repair_start

        after_valid = lns.contender.is_valid()

        print("Schedule after repair:")
        lns.contender.display_schedule()
        lns.contender.display_validity()
        print(f"Valid after repair: {after_valid}")
        print(f"Repair runtime: {runtime:.3f}s")

        abort_raw = input("Abort further runs? [y/N]: ").strip().lower()
        if abort_raw in {"y", "yes"}:
            break

        run_idx += 1
