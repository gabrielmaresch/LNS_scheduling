from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from rws import rws_lns


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
def _fixed_assignments_from_lns(lns: "rws_lns") -> List[List[int]]:
    """Build fixed assignments as rows [worker, day, shift] (1-based worker/day).

    Row width is always 3 because each entry is exactly one worker/day/shift triple.
    """
    rows: List[List[int]] = []
    for (day, worker), shift in sorted(lns.fixed_vars.items()):
        rows.append([worker + 1, day + 1, shift])
    return rows
def _forbidden_sequence_arrays_from_lns(lns: "rws_lns") -> tuple[List[int], List[List[int]]]:
    """Build (lengths, padded_sequences) for forbidden sequences (supported lengths: 2, 3).

    The sequence matrix has fixed width 3 (max supported sequence length). Length-2
    sequences are padded with a trailing 0 and interpreted via the lengths array.
    """
    lengths: List[int] = []
    sequences: List[List[int]] = []
    for seq in lns.instance.forbidden_sequences:
        if len(seq) not in (2, 3):
            raise ValueError("Only forbidden sequences of length 2 or 3 are supported")
        lengths.append(len(seq))
        if len(seq) == 2:
            sequences.append([seq[0], seq[1], 0])
        else:
            sequences.append([seq[0], seq[1], seq[2]])
    return lengths, sequences


def _array2d_literal(
    row_lb: int, row_ub: int, col_lb: int, col_ub: int, values: List[List[int]]
) -> str:
    flat = ", ".join(str(v) for row in values for v in row)
    return f"array2d({row_lb}..{row_ub}, {col_lb}..{col_ub}, [{flat}])"


def _array1d_literal(lb: int, ub: int, values: List[int]) -> str:
    flat = ", ".join(str(v) for v in values)
    return f"array1d({lb}..{ub}, [{flat}])"


def _replace_once(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Could not replace pattern in model: {pattern}")
    return updated


def _generate_rws_instance_mzn(
    lns: "rws_lns",
    base_model_path: str | Path = "rws_generic.mzn",
    output_model_path: str | Path = "rws_instance.mzn",
) -> Path:
    """Generate a fully-parameterized MiniZinc model file for this RWS instance."""
    base_file = Path(base_model_path)
    if not base_file.is_absolute():
        base_file = Path(__file__).resolve().parent / base_file

    out_file = Path(output_model_path)
    if not out_file.is_absolute():
        out_file = base_file.parent / out_file

    text = base_file.read_text(encoding="utf-8")
    params = _model_params_from_lns(lns)
    max_min_lengths = _max_min_lengths_from_lns(lns)
    required_workers = _required_workers_from_lns(lns)
    forbidden_lengths, forbidden_sequences = _forbidden_sequence_arrays_from_lns(lns)
    num_forbidden_sequences = len(forbidden_sequences)

    text = _replace_once(text, r"^int:\s*d\s*;.*$", f"int: d = {params['d']}; %number of days to be scheduled")
    text = _replace_once(text, r"^int:\s*n\s*;.*$", f"int: n = {params['n']}; %number of workers")
    text = _replace_once(text, r"^int:\s*s\s*;.*$", f"int: s = {params['s']}; %number of shifts")
    text = _replace_once(
        text,
        r"^array\[0\.\.s\+1,\s*1\.\.2\]\s+of\s+int:\s+max_min_lengths\s*;.*$",
        "array[0..s+1, 1..2] of int: max_min_lengths = "
        + _array2d_literal(0, params["s"] + 1, 1, 2, max_min_lengths)
        + ";",
    )
    text = _replace_once(text, r"^int:\s*min_rest\s*;.*$", f"int: min_rest = {params['min_rest']};")
    text = _replace_once(text, r"^int:\s*max_rest\s*;.*$", f"int: max_rest = {params['max_rest']};")
    text = _replace_once(text, r"^int:\s*min_work\s*;.*$", f"int: min_work = {params['min_work']};")
    text = _replace_once(text, r"^int:\s*max_work\s*;.*$", f"int: max_work = {params['max_work']};")
    text = _replace_once(text, r"^bool:\s*cyclic\s*;.*$", f"bool: cyclic = {str(bool(lns.instance.cyclicity)).lower()};")
    text = _replace_once(
        text,
        r"^array\[1\.\.d,\s*1\.\.s\]\s+of\s+int:\s+required_workers\s*;.*$",
        "array[1..d, 1..s] of int: required_workers = "
        + _array2d_literal(1, params["d"], 1, params["s"], required_workers)
        + ";",
    )
    text = _replace_once(
        text,
        r"^int:\s*num_forbidden_sequences\s*;.*$",
        f"int: num_forbidden_sequences = {num_forbidden_sequences};",
    )
    text = _replace_once(
        text,
        r"^array\[1\.\.num_forbidden_sequences\]\s+of\s+int:\s+forbidden_sequence_lengths\s*;.*$",
        "array[1..num_forbidden_sequences] of int: forbidden_sequence_lengths = "
        + _array1d_literal(1, num_forbidden_sequences, forbidden_lengths)
        + ";",
    )
    text = _replace_once(
        text,
        r"^array\[1\.\.num_forbidden_sequences,\s*1\.\.3\]\s+of\s+int:\s+forbidden_sequences\s*;.*$",
        "array[1..num_forbidden_sequences, 1..3] of int: forbidden_sequences = "
        + _array2d_literal(1, num_forbidden_sequences, 1, 3, forbidden_sequences)
        + ";",
    )

    out_file.write_text(text, encoding="utf-8")
    return out_file


def _assignment_from_shift_of(
    shift_of: Any, num_days: int, num_workers: int, num_shifts: int
) -> List[List[int]]:
    """Convert MiniZinc `shift_of[w][d]` to assignment[day][worker]."""
    assignment = [[0 for _ in range(num_workers)] for _ in range(num_days)]
    for worker in range(num_workers):
        for day in range(num_days):
            shift = int(shift_of[worker][day])
            if shift < 0 or shift > num_shifts:
                raise ValueError(
                    f"Invalid shift value in solution for day={day}, worker={worker}: {shift}"
                )
            assignment[day][worker] = shift
    return assignment


def build_rws_model_instance(
    lns: "rws_lns",
    model_path: str | Path = "rws_instance.mzn",
    solver_name: str = "chuffed",
) -> tuple[Any, Path]:
    """Build and return a MiniZinc Instance for RWS with injected instance-level data."""
    from minizinc import Instance, Model, Solver

    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = Path(__file__).resolve().parent / model_file

    base_model_file = model_file.with_name("rws_generic.mzn")
    if model_file.name == "rws_generic.mzn":
        base_model_file = model_file
        model_file = model_file.with_name("rws_instance.mzn")

    _generate_rws_instance_mzn(
        lns=lns,
        base_model_path=base_model_file,
        output_model_path=model_file,
    )

    model = Model(str(model_file))
    solver = Solver.lookup(solver_name)
    return Instance(solver, model), model_file


def solve_rws_lns(
    lns: "rws_lns",
    model_instance: Any,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Solve using a prepared MiniZinc model instance by only adding fixed assignments.

    A fresh branched instance is used per call so the same base model_instance can be
    reused across multiple LNS iterations.
    """
    fixed_assignments = _fixed_assignments_from_lns(lns)
    with model_instance.branch() as run_instance:
        run_instance["num_fixed_assignments"] = len(fixed_assignments)
        run_instance["fixed_assignments"] = fixed_assignments

        start = perf_counter()
        result = run_instance.solve(timeout=timedelta(seconds=timeout_seconds))
        elapsed = perf_counter() - start

    out: Dict[str, Any] = {
        "status": str(result.status),
        "solve_time_sec": elapsed,
        "has_solution": result.status.has_solution(),
    }

    lns.contender = None
    if result.status.has_solution():
        solution = result.solution.__dict__
        shift_of = solution.get("shift_of")
        if shift_of is None:
            raise ValueError("MiniZinc solution does not contain `shift_of`")
        assignment = _assignment_from_shift_of(
            shift_of=shift_of,
            num_days=lns.instance.num_days,
            num_workers=lns.instance.num_workers,
            num_shifts=len(lns.instance.shift_names) - 1,
        )
        from rws import RWS

        lns.contender = RWS.Schedule(instance=lns.instance, assignment=assignment)
    return out
