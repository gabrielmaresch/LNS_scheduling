from __future__ import annotations

from rws import RWS, rws_lns


def build_test_lns() -> rws_lns:
    """Build the same test fixture used in rws.py."""
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


def main() -> None:
    lns = build_test_lns()

    print("Generic example schedule (from rws.py):")
    lns.incumbent.display_schedule()
    before_totals = lns.incumbent.violation_totals()
    before_count = sum(before_totals.values())
    print(f"Violations before destroy/repair: {before_count}")

    lns.destroy_worker(0)
    contender = lns.repair()
    after_totals = contender.violation_totals()
    after_count = sum(after_totals.values())
    print(f"Violations after destroy worker 0 + repair: {after_count}")


if __name__ == "__main__":
    main()
