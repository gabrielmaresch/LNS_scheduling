from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rws import RWS


def _find_header_index(lines: Sequence[str], needle: str, start_idx: int = 0) -> int:
    needle_lc = needle.lower()
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        text = line.strip().lower()
        if text.startswith("#") and needle_lc in text:
            return idx
    raise ValueError(f"Could not find header containing: {needle!r}")


def _next_data_line(lines: Sequence[str], start_idx: int) -> Tuple[int, str]:
    for idx in range(start_idx + 1, len(lines)):
        text = lines[idx].strip()
        if text and not text.startswith("#"):
            return idx, text
    raise ValueError(f"Could not find data line after index {start_idx}")


def _parse_int_row(line: str) -> List[int]:
    return [int(token) for token in line.split()]


def parse_instance_file(file_path: str | Path, cyclicity: bool = True) -> RWS.Instance:
    """Parse one `Instances1-50/Example*.txt` file into an `RWS.Instance`."""
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    idx_days = _find_header_index(lines, "length of the schedule")
    _, days_line = _next_data_line(lines, idx_days)
    num_days = int(days_line)

    idx_workers = _find_header_index(lines, "number of employees")
    _, workers_line = _next_data_line(lines, idx_workers)
    num_workers = int(workers_line)

    idx_shifts = _find_header_index(lines, "number of shifts")
    _, shifts_line = _next_data_line(lines, idx_shifts)
    num_shifts = int(shifts_line)

    idx_req = _find_header_index(lines, "temporal requirements matrix")
    requirements_rows: List[List[int]] = []
    cur_idx = idx_req
    for _ in range(num_shifts):
        cur_idx, req_line = _next_data_line(lines, cur_idx)
        row = _parse_int_row(req_line)
        if len(row) != num_days:
            raise ValueError(
                f"Requirements row has {len(row)} entries, expected {num_days}: {req_line!r}"
            )
        requirements_rows.append(row)

    idx_shift_defs = _find_header_index(lines, "shiftname, start, length, name, minlengthofblocks")
    shift_names: List[str] = ["-"]
    min_consecutive_shift: Dict[int, int] = {}
    max_consecutive_shift: Dict[int, int] = {}
    cur_idx = idx_shift_defs
    for shift_id in range(1, num_shifts + 1):
        cur_idx, shift_line = _next_data_line(lines, cur_idx)
        parts = shift_line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid shift definition line: {shift_line!r}")
        shift_name = parts[0]
        min_len = int(parts[-2])
        max_len = int(parts[-1])
        shift_names.append(shift_name)
        min_consecutive_shift[shift_id] = min_len
        max_consecutive_shift[shift_id] = max_len

    idx_off = _find_header_index(lines, "minimum and maximum length of days-off blocks")
    _, off_line = _next_data_line(lines, idx_off)
    min_consecutive_off, max_consecutive_off = _parse_int_row(off_line)

    idx_work = _find_header_index(lines, "minimum and maximum length of work blocks")
    _, work_line = _next_data_line(lines, idx_work)
    min_consecutive_work, max_consecutive_work = _parse_int_row(work_line)

    idx_forbidden_counts = _find_header_index(lines, "number of not allowed shift sequences")
    _, forbidden_count_line = _next_data_line(lines, idx_forbidden_counts)
    count_2, count_3 = _parse_int_row(forbidden_count_line)
    total_forbidden = count_2 + count_3

    idx_forbidden = _find_header_index(lines, "not allowed shift sequences", start_idx=idx_forbidden_counts + 1)
    shift_to_id = {name: idx for idx, name in enumerate(shift_names)}
    forbidden_sequences: List[Tuple[int, ...]] = []
    cur_idx = idx_forbidden
    for _ in range(total_forbidden):
        cur_idx, seq_line = _next_data_line(lines, cur_idx)
        tokens = seq_line.split()
        if len(tokens) not in (2, 3):
            raise ValueError(f"Forbidden sequence must have length 2 or 3: {seq_line!r}")
        try:
            forbidden_sequences.append(tuple(shift_to_id[token] for token in tokens))
        except KeyError as exc:
            raise ValueError(f"Unknown shift name in forbidden sequence line: {seq_line!r}") from exc

    required_number_of_shifts: Dict[int, Sequence[int]] = {
        shift_id: requirements_rows[shift_id - 1] for shift_id in range(1, num_shifts + 1)
    }

    return RWS.Instance(
        num_days=num_days,
        num_workers=num_workers,
        shift_names=shift_names,
        cyclicity=cyclicity,
        forbidden_sequences=forbidden_sequences,
        min_consecutive_shift=min_consecutive_shift,
        max_consecutive_shift=max_consecutive_shift,
        min_consecutive_work=min_consecutive_work,
        max_consecutive_work=max_consecutive_work,
        min_consecutive_off=min_consecutive_off,
        max_consecutive_off=max_consecutive_off,
        required_number_of_shifts=required_number_of_shifts,
    )


def initialize_schedule(instance: RWS.Instance) -> RWS.Schedule:
    """Create a simple initial schedule by filling each day's required shifts in round-robin order."""
    num_days = instance.num_days
    num_workers = instance.num_workers
    num_shifts = len(instance.shift_names) - 1

    assignment = [[0 for _ in range(num_workers)] for _ in range(num_days)]
    base_workers = list(range(num_workers))

    for day in range(num_days):
        # Rotate start worker by day to spread assignments.
        start = day % num_workers
        worker_order = base_workers[start:] + base_workers[:start]
        cursor = 0

        for shift_id in range(1, num_shifts + 1):
            required = instance.required_number_of_shifts.get(shift_id, 0)
            required_on_day = required[day] if not isinstance(required, int) else required
            for _ in range(required_on_day):
                if cursor >= num_workers:
                    break
                worker = worker_order[cursor]
                assignment[day][worker] = shift_id
                cursor += 1

    return RWS.Schedule(instance=instance, assignment=assignment)


def load_instance_and_schedule(file_path: str | Path, cyclicity: bool = True) -> Tuple[RWS.Instance, RWS.Schedule]:
    instance = parse_instance_file(file_path=file_path, cyclicity=cyclicity)
    schedule = initialize_schedule(instance)
    return instance, schedule


if __name__ == "__main__":
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
    print(f"Loaded: {instance_path}")
    schedule.display_schedule()
    schedule.display_violations()
