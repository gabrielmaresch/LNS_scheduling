from __future__ import annotations

from collections import deque
from pathlib import Path
from time import perf_counter
import random
from typing import Deque, Tuple

from rws import RWS


Move = Tuple[str, int, int, int, int]


def _copy_schedule(schedule: RWS.Schedule) -> RWS.Schedule:
    return RWS.Schedule(
        instance=schedule.instance,
        assignment=[list(row) for row in schedule.assignment],
    )


def _score(schedule: RWS.Schedule) -> int:
    return len(schedule.ordered_violation_hits())


def _apply_swap(schedule: RWS.Schedule, worker_a: int, day_a: int, worker_b: int, day_b: int) -> None:
    schedule.assignment[day_a][worker_a], schedule.assignment[day_b][worker_b] = (
        schedule.assignment[day_b][worker_b],
        schedule.assignment[day_a][worker_a],
    )


def _apply_shift_change(schedule: RWS.Schedule, worker: int, day: int, new_shift: int) -> None:
    schedule.assignment[day][worker] = new_shift


def _schedule_key(schedule: RWS.Schedule) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in schedule.assignment)


def _objective_with_cache(
    schedule: RWS.Schedule,
    cache: dict[tuple[tuple[int, ...], ...], float],
    model_path: str | Path | None,
    solver_name: str,
    timeout_seconds: int,
    model_instance: object | None = None,
) -> float:
    key = _schedule_key(schedule)
    if key in cache:
        return cache[key]
    try:
        value = float(
            schedule.objective_value(
                model_path=model_path,
                solver_name=solver_name,
                timeout_seconds=max(1, timeout_seconds),
                model_instance=model_instance,
            )
        )
    except Exception:
        value = float("inf")
    cache[key] = value
    return value


def _canonical_swap_move(
    worker_a: int,
    day_a: int,
    worker_b: int,
    day_b: int,
) -> Move:
    left = (worker_a, day_a)
    right = (worker_b, day_b)
    if right < left:
        left, right = right, left
    return ("swap", left[0], left[1], right[0], right[1])


def tabu(
    instance: RWS.Instance,
    schedule: RWS.Schedule,
    tabu_length: int = 10,
    timeout: float = 10.0,
    max_neighbors: int = 64,
    use_objective_tiebreaker: bool = False,
    model_path: str | Path | None = None,
    model_instance: object | None = None,
    solver_name: str = "chuffed",
    objective_tiebreak_timeout_seconds: int = 2,
    objective_tiebreak_max_count: int = 10,
) -> RWS.Schedule:
    if objective_tiebreak_max_count < 0:
        raise ValueError("objective_tiebreak_max_count must be >= 0")

    best = _copy_schedule(schedule)
    best_score = _score(best)
    if best_score == 0:
        return best

    current = _copy_schedule(best)
    current_score = best_score

    start = perf_counter()
    rng = random.Random()
    next_report_at = 0.0

    tabu_history: Deque[Move] = deque(maxlen=max(1, tabu_length))
    tabu_counts: dict[Move, int] = {}
    objective_cache: dict[tuple[tuple[int, ...], ...], float] = {}
    iteration = 0

    n_workers = instance.num_workers
    n_days = instance.num_days
    n_shifts = len(instance.shift_names)

    while perf_counter() - start < timeout and best_score > 0:
        best_candidate = None
        best_candidate_score = None
        best_candidate_move: Move | None = None
        best_candidate_objective: float | None = None

        for _ in range(max(1, max_neighbors)):
            candidate = _copy_schedule(current)

            if rng.random() < 0.5:
                worker_a = rng.randrange(n_workers)
                day_a = rng.randrange(n_days)
                worker_b = rng.randrange(n_workers)
                day_b = rng.randrange(n_days)
                move = _canonical_swap_move(worker_a, day_a, worker_b, day_b)
                _apply_swap(candidate, worker_a, day_a, worker_b, day_b)
            else:
                worker = rng.randrange(n_workers)
                day = rng.randrange(n_days)
                old_shift = candidate.assignment[day][worker]
                if n_shifts <= 1:
                    continue
                # Shift ID 0 is included here and represents OFF/day-off.
                new_shift = rng.randrange(n_shifts)
                if new_shift == old_shift:
                    new_shift = (new_shift + 1) % n_shifts
                move = ("shift", worker, day, old_shift, new_shift)
                _apply_shift_change(candidate, worker, day, new_shift)

            cand_score = _score(candidate)

            # Aspiration: allow tabu if it improves global best.
            if tabu_counts.get(move, 0) > 0 and cand_score >= best_score:
                continue

            if best_candidate is None or cand_score < int(best_candidate_score):
                best_candidate = candidate
                best_candidate_score = cand_score
                best_candidate_move = move
                best_candidate_objective = None
            elif (
                use_objective_tiebreaker
                and cand_score == int(best_candidate_score)
                and cand_score < int(objective_tiebreak_max_count)
                and best_candidate is not None
            ):
                cand_obj = _objective_with_cache(
                    schedule=candidate,
                    cache=objective_cache,
                    model_path=model_path,
                    solver_name=solver_name,
                    timeout_seconds=objective_tiebreak_timeout_seconds,
                    model_instance=model_instance,
                )
                if best_candidate_objective is None:
                    best_candidate_objective = _objective_with_cache(
                        schedule=best_candidate,
                        cache=objective_cache,
                        model_path=model_path,
                        solver_name=solver_name,
                        timeout_seconds=objective_tiebreak_timeout_seconds,
                        model_instance=model_instance,
                    )
                if cand_obj < best_candidate_objective:
                    best_candidate = candidate
                    best_candidate_score = cand_score
                    best_candidate_move = move
                    best_candidate_objective = cand_obj

        if best_candidate is None or best_candidate_move is None or best_candidate_score is None:
            break

        current = best_candidate
        current_score = int(best_candidate_score)

        if len(tabu_history) == tabu_history.maxlen:
            evicted = tabu_history[0]
            remaining = tabu_counts.get(evicted, 0) - 1
            if remaining <= 0:
                tabu_counts.pop(evicted, None)
            else:
                tabu_counts[evicted] = remaining
        tabu_history.append(best_candidate_move)
        tabu_counts[best_candidate_move] = tabu_counts.get(best_candidate_move, 0) + 1

        if current_score < best_score:
            best = _copy_schedule(current)
            best_score = current_score
        iteration += 1
        elapsed_now = perf_counter() - start
        if elapsed_now >= next_report_at:
            print(
                f"\relapsed={elapsed_now:6.1f}s iter={iteration} count={best_score}",
                end="",
                flush=True,
            )
            next_report_at += 10.0

    if iteration > 0:
        print()

    return best
