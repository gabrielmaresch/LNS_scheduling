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

        ########## Basic checks on data consistency ####################################
        
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

        ########## Perform immediate check ##############################################
        
        def __post_init__(self) -> None:
            ''' Perform a check, wheter the given parameters for the given scheduling instance make sense  '''   
            
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
                if len(seq) > 3:
                    raise ValueError("forbidden_sequences must not contain tuples with more than 3 elements")
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

        
    @dataclass
    class Schedule:
        instance: "RWS.Instance"
        assignment: List[List[int]]
        run_compatibility_check: bool = False
        cons_workdays: List[List[int]] = field(init=False)
        cons_offdays: List[List[int]] = field(init=False)
        cons_shiftdays: Dict[int, List[List[int]]] = field(init=False)
        compatibility_issues: List[str] = field(init=False, default_factory=list)

        ################### Check if assignment is actually a complete schedule ##################
        def _check_validity(self) -> None:
            inst = self.instance
            if len(self.assignment) != inst.num_days:
                raise AssertionError("assignment must contain one row per day")

            max_shift = len(inst.shift_names) - 1
            for day, row in enumerate(self.assignment):
                if len(row) != inst.num_workers:
                    raise AssertionError(f"day {day} does not contain num_workers entries")
                for worker, shift in enumerate(row):
                    if not (0 <= shift <= max_shift):
                        raise AssertionError(f"invalid shift id at day {day}, worker {worker}: {shift}")

        def __post_init__(self) -> None:
            self._check_validity()

            ################ Compute streaks respecting cyclicity of the schedule #############
            ###### Condition: day is not off-day
            self.cons_workdays = self._compute_consecutive(lambda s: s != 0)
            ###### Condition: day is off-day
            self.cons_offdays = self._compute_consecutive(lambda s: s == 0)
            ###### Condition: day is workday with specified shift
            self.cons_shiftdays = {
                shift_id: self._compute_consecutive(lambda s, s_id=shift_id: s == s_id)
                for shift_id in range(len(self.instance.shift_names))
            }
  




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

        def _compute_consecutive(self, day_condition) -> List[List[int]]:
            """Count consecutive days satisfying day_condition up to each day, per worker.
            - respects cyclicity of the schedule
            - returns 2d array indexed by (worker, day) """
            inst = self.instance
            out = [[0 for _ in range(inst.num_workers)] for _ in range(inst.num_days)]

            for worker in range(inst.num_workers):
                carry = 0
                if inst.cyclicity:
                    for day in range(inst.num_days - 1, -1, -1):
                        if day_condition(self.assignment[day][worker]):
                            carry += 1
                        else:
                            break
                c = carry
                for day in range(inst.num_days):
                    if day_condition(self.assignment[day][worker]):
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

        
            """Return total number of violated clauses per bucket."""
            return {
                "sequence": self._count_forbidden_sequences(),
                "min": self._count_min_violations(),
                "max": self._count_max_violations(),
                "required": self._count_required_shifts_violations(),
            }



@dataclass
class rws_lns:
    """Minimal LNS context linking an `RWS.Instance` with a current schedule."""

    instance: "RWS.Instance"
    incumbent: "RWS.Schedule"
    contender: Optional["RWS.Schedule"] = None
    features: Any = None
    fixed_vars: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # In the beginning treat all entries from the given schedule as fixed.
        if not self.fixed_vars:
            self.fixed_vars = {
                (day, worker): self.incumbent.assignment[day][worker]
                for day in range(self.instance.num_days)
                for worker in range(self.instance.num_workers)
            }

    def destroy_worker(self, worker: int) -> List[Tuple[int, int]]:
        """Free all fixed variables for a given worker."""
        if not (0 <= worker < self.instance.num_workers):
            raise ValueError(f"invalid worker id {worker}; expected in [0, {self.instance.num_workers - 1}]")
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
    schedule.display_consecutive_workdays()
    schedule.display_violations()
