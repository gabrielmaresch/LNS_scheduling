from __future__ import annotations

from pathlib import Path

from rws import RWS, rws_lns
from rws_instance_loader import parse_instance_file
from rws_mzk_pipeline import build_rws_model_instance, solve_rws_lns


def main() -> None:
    base = Path(__file__).resolve().parent
    instance_path = base / "Instances2000" / "Example1.txt"
    model_path = base / "rws_instance.mzn"
    instance = parse_instance_file(instance_path, cyclicity=True)

    worker_rows = [
        [0, 2, 2, 2, 2, 2, 3],  # worker 1
        [3, 0, 0, 2, 2, 2, 2],  # worker 2
        [2, 3, 3, 0, 0, 1, 1],  # worker 3
        [1, 1, 1, 3, 3, 0, 0],  # worker 4
        [0, 1, 1, 2, 2, 3, 3],  # worker 5
        [3, 0, 0, 1, 1, 2, 2],  # worker 6
        [0, 0, 0, 1, 1, 1, 1],  # worker 7
        [1, 3, 3, 0, 0, 0, 0],  # worker 8
        [2, 2, 2, 3, 3, 3, 0],  # worker 9
    ]

    assignment = [
        [worker_rows[worker][day] for worker in range(instance.num_workers)]
        for day in range(instance.num_days)
    ]
    schedule = RWS.Schedule(instance=instance, assignment=assignment)

    print(f"Instance: {instance_path}")
    schedule.display_schedule(show_first_violation=True)
    schedule.display_validity()

    print("violation_diagnostic_from_find_first_violation_after:")
    first = schedule.find_first_violation_after(0, 0)
    if first is None:
        print("none")
    else:
        print(f"first_from_w0_d0: w{first[0]}, d{first[1]}, type={first[2]}")
        total_slots = instance.num_workers * instance.num_days
        cursor_worker, cursor_day = 0, 0
        seen: set[tuple[int, int, str]] = set()
        for _ in range(total_slots):
            hit = schedule.find_first_violation_after(cursor_worker, cursor_day)
            if hit is None or hit in seen:
                break
            seen.add(hit)
            print(f"- w{hit[0]}, d{hit[1]}, type={hit[2]}")
            idx = hit[0] * instance.num_days + hit[1]
            next_idx = (idx + 1) % total_slots
            cursor_worker, cursor_day = divmod(next_idx, instance.num_days)

    lns = rws_lns(instance=instance, incumbent=schedule)
    model_instance, used_model_path = build_rws_model_instance(
        lns=lns,
        model_path=model_path,
        solver_name="gecode",
        sloppy=False,
    )
    summary = solve_rws_lns(
        lns=lns,
        model_instance=model_instance,
        timeout_seconds=30,
    )

    print("\nfixed-schedule objective diagnostic:")
    print(f"model={used_model_path}")
    print(f"status={summary['status']}")
    print(f"has_solution={summary['has_solution']}")
    print(f"objective={summary['objective']}")


if __name__ == "__main__":
    main()
