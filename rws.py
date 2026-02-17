from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple, Union

# If performance becomes an issue, we can use numpy arrays for the assignment and consecutive counters, 
# but for simplicity and readability we use lists of lists here.

class RWS:
    """Roster/Workforce Scheduling container.

    - `Instance` stores model parameters and static constraints.
    - `Schedule` stores an actual assignment matrix and derived counters.
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
        cons_workdays: List[List[int]] = field(init=False)
        cons_offdays: List[List[int]] = field(init=False)
        cons_shiftdays: Dict[int, List[List[int]]] = field(init=False)

        def __post_init__(self) -> None:
            self._validate_shape_and_domain()
            # self._assert_compatibility()
            self.cons_workdays = self._compute_consecutive(lambda s: s != 0)
            self.cons_offdays = self._compute_consecutive(lambda s: s == 0)
            self.cons_shiftdays = {
                shift_id: self._compute_consecutive(lambda s, sid=shift_id: s == sid)
                for shift_id in range(len(self.instance.shift_names))
            }

        def _validate_shape_and_domain(self) -> None:
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

        def _assert_compatibility(self) -> None:
            inst = self.instance
            forbidden = set(inst.forbidden_sequences)

            for worker in range(inst.num_workers):
                # hard day requirements
                # off_days = inst.time_off.get(worker, set())
                # for day in off_days:
                #     if self.assignment[day][worker] != 0:
                #         raise AssertionError(
                #             f"worker {worker} must be off on day {day}"
                #         )

                # required_days = inst.workdays.get(worker, set())
                # for day in required_days:
                #     if self.assignment[day][worker] == 0:
                #         raise AssertionError(
                #             f"worker {worker} must work on day {day}"
                #         )

                # forbidden adjacent shift sequences (2-shift and 3-shift patterns)
                day_range = range(inst.num_days) if inst.cyclicity else range(1, inst.num_days)
                for day in day_range:
                    prev_day = (day - 1) % inst.num_days
                    prev_shift = self.assignment[prev_day][worker]
                    cur_shift = self.assignment[day][worker]
                    # Check 2-tuples (global)
                    if (prev_shift, cur_shift) in forbidden:
                        raise AssertionError(
                            f"forbidden sequence for worker {worker}: "
                            f"day {prev_day}->{day}: {prev_shift}->{cur_shift}"
                        )
                    
                    # Check 3-shift patterns (3-tuples)
                    if day >= 2 or (inst.cyclicity and day >= 1):
                        prev_prev_day = (day - 2) % inst.num_days
                        prev_prev_shift = self.assignment[prev_prev_day][worker]
                        if (prev_prev_shift, prev_shift, cur_shift) in forbidden:
                            raise AssertionError(
                                f"forbidden 3-shift sequence for worker {worker}: "
                                f"day {prev_prev_day}->{prev_day}->{day}: {prev_prev_shift}->{prev_shift}->{cur_shift}"
                            )

            # consecutive constraints (global work/off and per shift)
            self._assert_consecutive_constraints()
            
            # required number of shifts
            self._assert_required_shifts()

        def _assert_consecutive_constraints(self) -> None:
            inst = self.instance

            def runs_for_worker(worker: int, pred) -> List[int]:
                values = [pred(self.assignment[d][worker]) for d in range(inst.num_days)]
                return self._extract_run_lengths(values, inst.cyclicity)

            for worker in range(inst.num_workers):
                work_runs = runs_for_worker(worker, lambda s: s != 0)
                off_runs = runs_for_worker(worker, lambda s: s == 0)

                if any(r < inst.min_consecutive_work for r in work_runs):
                    raise AssertionError(f"worker {worker} violates min_consecutive_work")
                if any(r > inst.max_consecutive_work for r in work_runs):
                    raise AssertionError(f"worker {worker} violates max_consecutive_work")
                if any(r < inst.min_consecutive_off for r in off_runs):
                    raise AssertionError(f"worker {worker} violates min_consecutive_off")
                if any(r > inst.max_consecutive_off for r in off_runs):
                    raise AssertionError(f"worker {worker} violates max_consecutive_off")

                for shift_id in range(len(inst.shift_names)):
                    shift_runs = runs_for_worker(worker, lambda s, sid=shift_id: s == sid)
                    min_shift = inst.min_consecutive_shift.get(shift_id, 0)
                    max_shift = inst.max_consecutive_shift.get(shift_id, 10**9)
                    if any(r < min_shift for r in shift_runs):
                        raise AssertionError(
                            f"worker {worker} violates min consecutive for shift {shift_id}"
                        )
                    if any(r > max_shift for r in shift_runs):
                        raise AssertionError(
                            f"worker {worker} violates max consecutive for shift {shift_id}"
                        )

        def _assert_required_shifts(self) -> None:
            """Validate that each shift appears the required number of times per day in the schedule."""
            inst = self.instance
            
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
                        raise AssertionError(
                            f"shift {shift_id} ({inst.shift_names[shift_id]}) requires {required} occurrences on day {day}, "
                            f"but has {actual}"
                        )

        @staticmethod
        def _extract_run_lengths(flags: Sequence[bool], cyclic: bool) -> List[int]:
            n = len(flags)
            if n == 0:
                return []

            runs: List[int] = []
            cur = 0
            for v in flags:
                if v:
                    cur += 1
                elif cur > 0:
                    runs.append(cur)
                    cur = 0
            if cur > 0:
                runs.append(cur)

            # Merge first/last run for cyclic sequences when both ends are True.
            if cyclic and runs and flags[0] and flags[-1]:
                if len(runs) == 1 and all(flags):
                    return [n]
                first_len = 0
                for v in flags:
                    if v:
                        first_len += 1
                    else:
                        break
                last_len = 0
                for i in range(n - 1, -1, -1):
                    if flags[i]:
                        last_len += 1
                    else:
                        break
                middle = runs[1:-1] if len(runs) > 2 else []
                runs = [first_len + last_len] + middle

            return runs

        def _compute_consecutive(self, predicate) -> List[List[int]]:
            """Count consecutive days satisfying predicate up to each day, per worker.

            If cyclic, the beginning of the horizon inherits tail-run length.
            """
            inst = self.instance
            out = [[0 for _ in range(inst.num_workers)] for _ in range(inst.num_days)]

            for worker in range(inst.num_workers):
                carry = 0
                if inst.cyclicity:
                    for day in range(inst.num_days - 1, -1, -1):
                        if predicate(self.assignment[day][worker]):
                            carry += 1
                        else:
                            break
                c = carry
                for day in range(inst.num_days):
                    if predicate(self.assignment[day][worker]):
                        c += 1
                    else:
                        c = 0
                    out[day][worker] = c

            return out

        def display_schedule(self) -> None:
            """Display the schedule in a readable format."""
            inst = self.instance
            
            # Header
            print("\n" + "="*80)
            print(f"Schedule for {inst.num_workers} workers over {inst.num_days} days")
            print("="*80)
            
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

        def display_consecutive_workdays(self) -> None:
            """Display consecutive workdays in a readable format."""
            inst = self.instance
            
            print("\n" + "="*80)
            print(f"Consecutive workdays for {inst.num_workers} workers over {inst.num_days} days")
            print("="*80)
            
            # Day header
            print("Day:      ", end="")
            for day in range(inst.num_days):
                print(f"{day:>3}", end=" ")
            print()
            print("-"*80)
            
            # Worker consecutive workdays
            for worker in range(inst.num_workers):
                print(f"Worker {worker}: ", end="")
                for day in range(inst.num_days):
                    consecutive = self.cons_workdays[day][worker]
                    print(f"{consecutive:>3}", end=" ")
                print()
            
            print("="*80 + "\n")


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
        required_number_of_shifts={1: 1, 2: 1, 3: 1},  # D, A, N: 1 per day each (21 total)
        #time_off={0: {3}},
        #workdays={1: {0}},
    )

    schedule = RWS.Schedule(
        instance=instance,
        assignment=[
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 1, 2, 0],
            [0, 2, 3, 1],
            [2, 0, 1, 3],
            [3, 1, 0, 2],
            [1, 3, 2, 0],
        ],
    )

    schedule.display_schedule()
    schedule.display_consecutive_workdays()
