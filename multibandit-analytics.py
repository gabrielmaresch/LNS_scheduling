from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

# Ensure matplotlib/fontconfig caches are writable in sandboxed environments.
_BASE_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _BASE_DIR / ".mpl_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ITER_RE = re.compile(r"(\w+)=([^\s]+)")


@dataclass
class IterationRecord:
    iteration: int
    elapsed: float
    step_runtime: float
    incumbent_violations: int
    contender_violations: int
    incumbent_objective: Optional[float]
    contender_objective: Optional[float]
    accepted: bool
    temperature: Optional[float]


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_log(
    log_path: Path,
) -> Tuple[
    List[IterationRecord],
    Dict[int, Dict[str, Tuple[float, float]]],
    Dict[int, Dict[str, Tuple[float, float]]],
]:
    records: List[IterationRecord] = []
    destroy_updates: Dict[int, Dict[str, Tuple[float, float]]] = {}
    repair_updates: Dict[int, Dict[str, Tuple[float, float]]] = {}

    current_iter: Optional[int] = None
    section: Optional[str] = None

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")

        if line.startswith("iter="):
            section = None
            fields = dict(ITER_RE.findall(line))
            required = {
                "iter",
                "elapsed",
                "step_runtime",
                "incumbent_violations",
                "contender_violations",
                "accepted",
            }
            missing = required - set(fields)
            if missing:
                raise ValueError(f"missing fields in iteration line: {sorted(missing)}")

            record = IterationRecord(
                iteration=int(fields["iter"]),
                elapsed=float(fields["elapsed"].rstrip("s")),
                step_runtime=float(fields["step_runtime"].rstrip("s")),
                incumbent_violations=int(fields["incumbent_violations"]),
                contender_violations=int(fields["contender_violations"]),
                incumbent_objective=_parse_optional_float(fields.get("incumbent_objective")),
                contender_objective=_parse_optional_float(fields.get("contender_objective")),
                accepted=_parse_bool(fields["accepted"]),
                temperature=(
                    float(fields["temperature"])
                    if "temperature" in fields
                    else None
                ),
            )
            records.append(record)
            current_iter = record.iteration
            continue

        if line.startswith("  weight update (destroy):"):
            section = "destroy"
            continue
        if line.startswith("  weight update (repair):"):
            section = "repair"
            continue

        if section and line.strip().startswith("operator"):
            continue

        if section and line.startswith("    "):
            if current_iter is None:
                continue
            parts = [part.strip() for part in line.split("\t") if part.strip()]
            if len(parts) < 3:
                continue
            operator_name = parts[0]
            before = float(parts[1])
            after = float(parts[2])
            target = destroy_updates if section == "destroy" else repair_updates
            target.setdefault(current_iter, {})[operator_name] = (before, after)
            continue

    if not records:
        raise ValueError(f"no iteration lines found in {log_path}")

    return records, destroy_updates, repair_updates


def _build_weight_update_bars(
    updates: Dict[int, Dict[str, Tuple[float, float]]],
) -> Tuple[List[int], List[str], Dict[str, List[float]]]:
    """Build normalized stacked-bar weights at each update iteration."""
    update_iters = sorted(updates.keys())
    if not update_iters:
        return [], [], {}

    operators = sorted({name for by_iter in updates.values() for name in by_iter})
    if not operators:
        return [], [], {}

    current: Dict[str, float] = {name: 0.0 for name in operators}
    bars: Dict[str, List[float]] = {name: [] for name in operators}

    for iteration in update_iters:
        for name, (_before, after) in updates[iteration].items():
            current[name] = max(0.0, float(after))

        total = float(sum(current.values()))
        if total <= 0.0:
            normalized = {name: 1.0 / len(operators) for name in operators}
        else:
            normalized = {name: value / total for name, value in current.items()}

        for name in operators:
            bars[name].append(normalized[name])

    return update_iters, operators, bars


def _bars_have_variation(operators: List[str], bars: Dict[str, List[float]]) -> bool:
    if not operators:
        return False
    num_updates = len(bars[operators[0]])
    snapshots = [
        tuple(round(bars[name][idx], 8) for name in operators)
        for idx in range(num_updates)
    ]
    return len(set(snapshots)) > 1


def _reconstruct_temperature(
    iterations: List[int],
    initial_temp: float,
    decay: float,
    min_temp: float,
) -> List[float]:
    return [
        max(min_temp, initial_temp * (decay ** max(0, iteration - 1)))
        for iteration in iterations
    ]


def plot_analytics(
    records: List[IterationRecord],
    destroy_updates: Dict[int, Dict[str, Tuple[float, float]]],
    repair_updates: Dict[int, Dict[str, Tuple[float, float]]],
    initial_temp: float,
    decay: float,
    min_temp: float,
    output_path: Path,
    show: bool,
) -> None:
    iterations = [record.iteration for record in records]
    runtimes = [record.step_runtime for record in records]
    contender_violations = [record.contender_violations for record in records]
    contender_objectives = [record.contender_objective for record in records]

    accepted_trajectory: List[int] = []
    current = records[0].incumbent_violations
    for record in records:
        if record.accepted:
            current = record.contender_violations
        accepted_trajectory.append(current)

    accepted_objective_trajectory: List[Optional[float]] = []
    current_obj = records[0].incumbent_objective
    for record in records:
        if record.accepted and record.contender_objective is not None:
            current_obj = record.contender_objective
        accepted_objective_trajectory.append(current_obj)

    has_objective = any(value is not None for value in contender_objectives)

    explicit_temps = [record.temperature for record in records]
    if all(temp is not None for temp in explicit_temps):
        temperatures = [float(temp) for temp in explicit_temps]
        temperature_label = "Temperature"
    else:
        temperatures = _reconstruct_temperature(
            iterations=iterations,
            initial_temp=initial_temp,
            decay=decay,
            min_temp=min_temp,
        )
        temperature_label = "Temperature (reconstructed)"

    destroy_update_iters, destroy_operators, destroy_bars = _build_weight_update_bars(
        destroy_updates
    )
    repair_update_iters, repair_operators, repair_bars = _build_weight_update_bars(
        repair_updates
    )

    fig, axes = plt.subplots(
        3,
        1,
        sharex=False,
        figsize=(15, 11),
        constrained_layout=True,
    )

    ax_v = axes[0]
    ax_v.plot(
        iterations,
        accepted_trajectory,
        color="tab:blue",
        linewidth=2.0,
        label="Accepted violations",
    )
    ax_v.plot(
        iterations,
        contender_violations,
        color="tab:blue",
        alpha=0.35,
        linewidth=1.2,
        linestyle="--",
        label="Contender violations",
    )
    if has_objective:
        ax_v.plot(
            iterations,
            accepted_objective_trajectory,
            color="tab:green",
            linewidth=2.0,
            label="Accepted objective",
        )
        ax_v.plot(
            iterations,
            contender_objectives,
            color="tab:green",
            alpha=0.35,
            linewidth=1.2,
            linestyle="--",
            label="Contender objective",
        )
    else:
        ax_v.text(
            0.01,
            0.92,
            "Objective not present in log",
            transform=ax_v.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    ax_v.set_ylabel("Counts")
    ax_v.grid(alpha=0.25)

    ax_rt = ax_v.twinx()
    ax_rt.bar(
        iterations,
        runtimes,
        width=0.85,
        color="tab:gray",
        alpha=0.3,
        label="Step runtime (s)",
    )
    ax_rt.set_ylabel("Runtime (s)")

    ax_temp = ax_v.twinx()
    ax_temp.spines["right"].set_position(("axes", 1.12))
    ax_temp.plot(
        iterations,
        temperatures,
        color="tab:red",
        linewidth=1.6,
        linestyle="-.",
        label=temperature_label,
    )
    ax_temp.set_ylabel("Temperature")

    handles = []
    labels = []
    for ax in (ax_v, ax_rt, ax_temp):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax_v.legend(handles, labels, loc="upper right")
    ax_v.set_title("Violations and Objective with Runtime and Temperature Overlay")

    ax_dw = axes[1]
    if destroy_update_iters:
        positions = list(range(len(destroy_update_iters)))
        bottoms = [0.0] * len(positions)
        for name in destroy_operators:
            values = destroy_bars[name]
            ax_dw.bar(
                positions,
                values,
                width=0.92,
                bottom=bottoms,
                alpha=0.92,
                edgecolor="white",
                linewidth=0.4,
                label=name,
            )
            bottoms = [base + value for base, value in zip(bottoms, values)]
        ax_dw.set_xlim(-0.5, len(positions) - 0.5)
        ax_dw.set_xticks(positions)
        ax_dw.set_xticklabels([str(it) for it in destroy_update_iters])
        ax_dw.legend(ncol=2, fontsize=8, loc="upper right")
        if not _bars_have_variation(destroy_operators, destroy_bars):
            ax_dw.text(
                0.01,
                0.93,
                "No visible change across logged updates",
                transform=ax_dw.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
    else:
        ax_dw.text(
            0.5,
            0.5,
            "No destroy weight updates in log",
            transform=ax_dw.transAxes,
            ha="center",
            va="center",
        )
    ax_dw.set_ylabel("Destroy weight")
    ax_dw.set_xlabel("Update iteration")
    ax_dw.set_ylim(0.0, 1.0)
    ax_dw.grid(alpha=0.25)
    ax_dw.set_title("Destroy Weights per Update (stacked)")

    ax_rw = axes[2]
    if repair_update_iters:
        positions = list(range(len(repair_update_iters)))
        bottoms = [0.0] * len(positions)
        for name in repair_operators:
            values = repair_bars[name]
            ax_rw.bar(
                positions,
                values,
                width=0.92,
                bottom=bottoms,
                alpha=0.92,
                edgecolor="white",
                linewidth=0.4,
                label=name,
            )
            bottoms = [base + value for base, value in zip(bottoms, values)]
        ax_rw.set_xlim(-0.5, len(positions) - 0.5)
        ax_rw.set_xticks(positions)
        ax_rw.set_xticklabels([str(it) for it in repair_update_iters])
        ax_rw.legend(ncol=2, fontsize=8, loc="upper right")
        if not _bars_have_variation(repair_operators, repair_bars):
            ax_rw.text(
                0.01,
                0.93,
                "No visible change across logged updates",
                transform=ax_rw.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
    else:
        ax_rw.text(
            0.5,
            0.5,
            "No repair weight updates in log",
            transform=ax_rw.transAxes,
            ha="center",
            va="center",
        )
    ax_rw.set_ylabel("Repair weight")
    ax_rw.set_xlabel("Update iteration")
    ax_rw.set_ylim(0.0, 1.0)
    ax_rw.grid(alpha=0.25)
    ax_rw.set_title("Repair Weights per Update (stacked)")

    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse multiarm_bandit.log and plot violations/objective/runtime, "
            "weights trajectory, and temperature."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().parent / "multiarm_bandit.log",
        help="Path to multiarm bandit log file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "multibandit-analytics.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--initial-temp",
        type=float,
        default=5.0,
        help="Initial temperature used when reconstructing missing temperatures.",
    )
    parser.add_argument(
        "--temp-decay",
        type=float,
        default=0.98,
        help="Temperature decay factor used when reconstructing missing temperatures.",
    )
    parser.add_argument(
        "--min-temp",
        type=float,
        default=0.7,
        help="Minimum temperature used when reconstructing missing temperatures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot window in addition to saving the figure.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"log file not found: {args.log}")
    if args.initial_temp <= 0:
        raise ValueError("--initial-temp must be > 0")
    if args.min_temp <= 0:
        raise ValueError("--min-temp must be > 0")
    if not (0 < args.temp_decay <= 1):
        raise ValueError("--temp-decay must be in (0, 1]")

    records, destroy_updates, repair_updates = parse_log(args.log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_analytics(
        records=records,
        destroy_updates=destroy_updates,
        repair_updates=repair_updates,
        initial_temp=args.initial_temp,
        decay=args.temp_decay,
        min_temp=args.min_temp,
        output_path=args.out,
        show=args.show,
    )
    print(f"Wrote analytics plot: {args.out}")


if __name__ == "__main__":
    main()
