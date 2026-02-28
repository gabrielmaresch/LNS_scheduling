from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Dict, Iterable, List, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = BASE_DIR / ".mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-9


def _to_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "":
        return math.nan
    if text.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text.lower() in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(text)
    except Exception:
        return math.nan


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except Exception:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _instance_index(instance_name: str) -> int | None:
    match = re.search(r"(\d+)$", instance_name)
    if match:
        return int(match.group(1))
    return None


def _instance_sort_key(instance_name: str) -> Tuple[int, str]:
    idx = _instance_index(instance_name)
    if idx is None:
        return (10**9, instance_name)
    return (idx, instance_name)


def _safe_mean(values: Sequence[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def _safe_median(values: Sequence[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return math.nan
    return float(statistics.median(finite))


def _safe_std(values: Sequence[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if len(finite) <= 1:
        return 0.0 if finite else math.nan
    return float(statistics.stdev(finite))


def _safe_min(values: Sequence[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return min(finite) if finite else math.nan


def _safe_max(values: Sequence[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return max(finite) if finite else math.nan


def _denom_for_gap(cp_value: float) -> float:
    if not math.isfinite(cp_value):
        return math.nan
    if abs(cp_value) > EPS:
        return abs(cp_value)
    return 1.0


def _gap_to_cp(obj_value: float, cp_value: float) -> float:
    if not (math.isfinite(obj_value) and math.isfinite(cp_value)):
        return math.nan
    return (obj_value - cp_value) / _denom_for_gap(cp_value)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _latest_baseline_csv(benchmark_baseline_dir: Path) -> Path:
    candidates = sorted(benchmark_baseline_dir.glob("baseline_runs_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no baseline_runs_*.csv found in {benchmark_baseline_dir}")
    scored: List[Tuple[int, int, int, Path]] = []
    for idx, path in enumerate(candidates):
        try:
            rows = _read_csv(path)
        except Exception:
            continue
        finite_count = 0
        finite_instances: set[str] = set()
        for row in rows:
            obj = _to_float(row.get("objective"))
            if math.isfinite(obj):
                finite_count += 1
                instance = str(row.get("instance", "")).strip()
                if instance:
                    finite_instances.add(instance)
        # Sort by: unique finite instances first, then finite rows, then recency by index.
        scored.append((len(finite_instances), finite_count, idx, path))
    if not scored:
        return candidates[-1]
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return scored[-1][3]


def _latest_run_id(variant_dir: Path) -> str:
    run_dirs = sorted([p for p in variant_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found in {variant_dir}")
    return run_dirs[-1].name


def _resolve_summary_path(alns_root: Path, variant: str, run_id: str | None) -> Path:
    variant_dir = alns_root / variant
    if not variant_dir.exists():
        raise FileNotFoundError(f"variant directory not found: {variant_dir}")
    selected_run_id = run_id if run_id is not None else _latest_run_id(variant_dir)
    summary_path = variant_dir / selected_run_id / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")
    return summary_path


def _find_log_for_run(logs_dir: Path, instance: str, seed: int) -> Path | None:
    matches = sorted(logs_dir.glob(f"*{instance}_seed{seed}.log"))
    if not matches:
        return None
    return matches[0]


@dataclass(frozen=True)
class RunKey:
    algorithm: str
    instance: str
    seed: int


def _read_alns_summary(summary_path: Path, algorithm_label: str) -> Tuple[List[Dict[str, Any]], Dict[RunKey, Path]]:
    rows_raw = _read_csv(summary_path)
    run_rows: List[Dict[str, Any]] = []
    log_map: Dict[RunKey, Path] = {}
    logs_dir = summary_path.parent / "logs"
    for row in rows_raw:
        instance = str(row.get("instance", "")).strip()
        seed = _to_int(row.get("seed"))
        if instance == "" or seed is None:
            continue
        run_row = {
            "algorithm": algorithm_label,
            "variant": str(row.get("variant", "")),
            "instance": instance,
            "instance_index": _to_int(row.get("instance_index")),
            "seed": seed,
            "timeout_seconds": _to_float(row.get("timeout_seconds")),
            "status": str(row.get("status", "")),
            "objective": _to_float(row.get("objective")),
            "runtime_seconds": _to_float(row.get("runtime_seconds")),
            "iterations": _to_int(row.get("iterations")),
            "solved": _to_bool(row.get("solved")),
            "timeout_hit": _to_bool(row.get("timeout_hit")),
            "error": str(row.get("error", "")),
            "solver_name": str(row.get("solver_name", "")),
            "summary_path": str(summary_path),
            "run_id": summary_path.parent.name,
        }
        run_rows.append(run_row)
        log_path = _find_log_for_run(logs_dir, instance=instance, seed=seed)
        if log_path is not None:
            log_map[RunKey(algorithm=algorithm_label, instance=instance, seed=seed)] = log_path
    return run_rows, log_map


def _build_cp_baseline_rows(baseline_csv_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows_raw = _read_csv(baseline_csv_path)
    by_instance: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows_raw:
        instance = str(row.get("instance", "")).strip()
        if instance == "":
            continue
        enriched = {
            "instance": instance,
            "instance_index": _to_int(row.get("instance_index")),
            "solver_name": str(row.get("solver_name", "")),
            "status": str(row.get("status", "")),
            "objective": _to_float(row.get("objective")),
            "runtime_seconds": _to_float(row.get("runtime_seconds")),
            "timeout_hit": _to_bool(row.get("timeout_hit")),
            "error": str(row.get("error", "")),
            "source_csv": str(baseline_csv_path),
        }
        by_instance.setdefault(instance, []).append(enriched)

    cp_rows: List[Dict[str, Any]] = []
    cp_map: Dict[str, float] = {}
    for instance, candidates in by_instance.items():
        valid = [r for r in candidates if math.isfinite(r["objective"])]
        if valid:
            chosen = min(valid, key=lambda r: r["objective"])
            cp_obj = float(chosen["objective"])
        else:
            chosen = candidates[0]
            cp_obj = math.nan
        cp_map[instance] = cp_obj
        cp_rows.append(
            {
                "algorithm": "CP-SAT",
                "variant": "CP-SAT",
                "instance": instance,
                "instance_index": chosen["instance_index"],
                "seed": None,
                "timeout_seconds": math.nan,
                "status": chosen["status"],
                "objective": cp_obj,
                "runtime_seconds": chosen["runtime_seconds"],
                "iterations": None,
                "solved": math.isfinite(cp_obj),
                "timeout_hit": chosen["timeout_hit"],
                "error": chosen["error"],
                "solver_name": chosen["solver_name"],
                "summary_path": str(baseline_csv_path),
                "run_id": baseline_csv_path.stem,
                "baseline_candidates": len(candidates),
            }
        )
    return cp_rows, cp_map


def _build_master_table(
    alns_rows: Sequence[Dict[str, Any]],
    cp_rows: Sequence[Dict[str, Any]],
    cp_map: Dict[str, float],
    log_map: Dict[RunKey, Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in cp_rows:
        cp_obj = row["objective"]
        rows.append(
            {
                "instance": row["instance"],
                "instance_index": row["instance_index"],
                "algorithm": row["algorithm"],
                "seed": "",
                "objective": cp_obj,
                "runtime_seconds": row["runtime_seconds"],
                "gap_to_cp": 0.0 if math.isfinite(cp_obj) else math.nan,
                "status": row["status"],
                "timeout_hit": row["timeout_hit"],
                "solver_name": row["solver_name"],
                "log_path": "",
                "source_summary": row["summary_path"],
            }
        )

    for row in alns_rows:
        instance = row["instance"]
        seed = int(row["seed"])
        cp_obj = cp_map.get(instance, math.nan)
        gap = _gap_to_cp(row["objective"], cp_obj)
        log_path = log_map.get(RunKey(algorithm=row["algorithm"], instance=instance, seed=seed))
        rows.append(
            {
                "instance": instance,
                "instance_index": row["instance_index"],
                "algorithm": row["algorithm"],
                "seed": seed,
                "objective": row["objective"],
                "runtime_seconds": row["runtime_seconds"],
                "gap_to_cp": gap,
                "status": row["status"],
                "timeout_hit": row["timeout_hit"],
                "solver_name": row["solver_name"],
                "log_path": str(log_path) if log_path is not None else "",
                "source_summary": row["summary_path"],
            }
        )

    rows.sort(key=lambda r: (_instance_sort_key(str(r["instance"])), str(r["algorithm"]), str(r["seed"])))
    return rows


def _group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def _per_instance_aggregation(alns_rows: Sequence[Dict[str, Any]], cp_map: Dict[str, float]) -> List[Dict[str, Any]]:
    grouped = _group_rows(alns_rows, ["instance", "algorithm"])
    out: List[Dict[str, Any]] = []
    for (instance, algorithm), rows in grouped.items():
        objectives = [float(r["objective"]) for r in rows]
        gaps = [_gap_to_cp(float(r["objective"]), cp_map.get(str(instance), math.nan)) for r in rows]
        mean_obj = _safe_mean(objectives)
        std_obj = _safe_std(objectives)
        cv = std_obj / abs(mean_obj) if (math.isfinite(std_obj) and math.isfinite(mean_obj) and abs(mean_obj) > EPS) else math.nan
        out.append(
            {
                "instance": instance,
                "instance_index": _instance_index(str(instance)),
                "algorithm": algorithm,
                "n_seeds": len(rows),
                "mean_objective": mean_obj,
                "median_objective": _safe_median(objectives),
                "best_objective": _safe_min(objectives),
                "std_objective": std_obj,
                "min_objective": _safe_min(objectives),
                "max_objective": _safe_max(objectives),
                "mean_gap_to_cp": _safe_mean(gaps),
                "best_gap_to_cp": _safe_min(gaps),
                "cv_objective": cv,
            }
        )
    out.sort(key=lambda r: (_instance_sort_key(str(r["instance"])), str(r["algorithm"])))
    return out


def _win_loss(x: float, y: float) -> str:
    if not (math.isfinite(x) and math.isfinite(y)):
        return "missing"
    if x < y - EPS:
        return "win"
    if y < x - EPS:
        return "loss"
    return "tie"


def _win_loss_summary(per_instance_stats: Sequence[Dict[str, Any]], algorithm_a: str, algorithm_b: str, cp_map: Dict[str, float]) -> List[Dict[str, Any]]:
    by_instance_algo = {(r["instance"], r["algorithm"]): r for r in per_instance_stats}
    instances = sorted({str(r["instance"]) for r in per_instance_stats}, key=_instance_sort_key)
    comparisons: Dict[str, Dict[str, int]] = {
        "mean_A_vs_B": {"win": 0, "tie": 0, "loss": 0, "missing": 0},
        "best_A_vs_B": {"win": 0, "tie": 0, "loss": 0, "missing": 0},
        "mean_A_vs_CP": {"win": 0, "tie": 0, "loss": 0, "missing": 0},
        "mean_B_vs_CP": {"win": 0, "tie": 0, "loss": 0, "missing": 0},
    }

    for instance in instances:
        ra = by_instance_algo.get((instance, algorithm_a))
        rb = by_instance_algo.get((instance, algorithm_b))
        cp = cp_map.get(instance, math.nan)

        mean_a = float(ra["mean_objective"]) if ra is not None else math.nan
        mean_b = float(rb["mean_objective"]) if rb is not None else math.nan
        best_a = float(ra["best_objective"]) if ra is not None else math.nan
        best_b = float(rb["best_objective"]) if rb is not None else math.nan

        comparisons["mean_A_vs_B"][_win_loss(mean_a, mean_b)] += 1
        comparisons["best_A_vs_B"][_win_loss(best_a, best_b)] += 1
        comparisons["mean_A_vs_CP"][_win_loss(mean_a, cp)] += 1
        comparisons["mean_B_vs_CP"][_win_loss(mean_b, cp)] += 1

    rows: List[Dict[str, Any]] = []
    for metric, counts in comparisons.items():
        rows.append(
            {
                "comparison": metric,
                "wins": counts["win"],
                "ties": counts["tie"],
                "losses": counts["loss"],
                "missing": counts["missing"],
            }
        )
    return rows


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and abs(indexed[j + 1][1] - indexed[i][1]) <= EPS:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _wilcoxon_signed_rank_exact(diffs: Sequence[float]) -> Dict[str, float]:
    nz = [float(d) for d in diffs if math.isfinite(d) and abs(d) > EPS]
    if not nz:
        return {
            "n_nonzero": 0,
            "w_plus": math.nan,
            "w_minus": math.nan,
            "w_stat": math.nan,
            "p_value_two_sided": math.nan,
        }

    abs_vals = [abs(d) for d in nz]
    ranks = _average_ranks(abs_vals)
    w_plus = sum(rank for rank, diff in zip(ranks, nz) if diff > 0)
    rank_total = sum(ranks)
    w_minus = rank_total - w_plus
    w_stat = min(w_plus, w_minus)

    n = len(nz)
    count = 0
    total = 1 << n
    for mask in range(total):
        wp = 0.0
        for bit in range(n):
            if (mask >> bit) & 1:
                wp += ranks[bit]
        ws = min(wp, rank_total - wp)
        if ws <= w_stat + 1e-12:
            count += 1
    p_two_sided = count / total

    return {
        "n_nonzero": float(n),
        "w_plus": float(w_plus),
        "w_minus": float(w_minus),
        "w_stat": float(w_stat),
        "p_value_two_sided": float(min(1.0, p_two_sided)),
    }


def _paired_statistics(per_instance_stats: Sequence[Dict[str, Any]], algorithm_a: str, algorithm_b: str) -> List[Dict[str, Any]]:
    by_instance_algo = {(r["instance"], r["algorithm"]): r for r in per_instance_stats}
    instances = sorted({str(r["instance"]) for r in per_instance_stats}, key=_instance_sort_key)
    diffs: List[float] = []
    pct_improve_b_over_a: List[float] = []
    for instance in instances:
        ra = by_instance_algo.get((instance, algorithm_a))
        rb = by_instance_algo.get((instance, algorithm_b))
        if ra is None or rb is None:
            continue
        mean_a = float(ra["mean_objective"])
        mean_b = float(rb["mean_objective"])
        if not (math.isfinite(mean_a) and math.isfinite(mean_b)):
            continue
        diff = mean_a - mean_b
        diffs.append(diff)
        pct_improve_b_over_a.append(diff / (abs(mean_a) if abs(mean_a) > EPS else 1.0))

    wilcoxon = _wilcoxon_signed_rank_exact(diffs)
    rows = [
        {
            "metric": "mean_difference_A_minus_B",
            "value": _safe_mean(diffs),
        },
        {
            "metric": "median_difference_A_minus_B",
            "value": _safe_median(diffs),
        },
        {
            "metric": "wilcoxon_n_nonzero",
            "value": wilcoxon["n_nonzero"],
        },
        {
            "metric": "wilcoxon_w_stat",
            "value": wilcoxon["w_stat"],
        },
        {
            "metric": "wilcoxon_p_value_two_sided",
            "value": wilcoxon["p_value_two_sided"],
        },
        {
            "metric": "mean_pct_improvement_B_over_A",
            "value": _safe_mean(pct_improve_b_over_a),
        },
        {
            "metric": "median_pct_improvement_B_over_A",
            "value": _safe_median(pct_improve_b_over_a),
        },
    ]
    return rows


def _stability_across_instances(per_instance_stats: Sequence[Dict[str, Any]], algorithm_a: str, algorithm_b: str) -> List[Dict[str, Any]]:
    by_instance_algo = {(r["instance"], r["algorithm"]): r for r in per_instance_stats}
    instances = sorted({str(r["instance"]) for r in per_instance_stats}, key=_instance_sort_key)
    std_a: List[float] = []
    std_b: List[float] = []
    cv_a: List[float] = []
    cv_b: List[float] = []
    std_diff: List[float] = []
    for instance in instances:
        ra = by_instance_algo.get((instance, algorithm_a))
        rb = by_instance_algo.get((instance, algorithm_b))
        if ra is None or rb is None:
            continue
        sa = float(ra["std_objective"])
        sb = float(rb["std_objective"])
        ca = float(ra["cv_objective"])
        cb = float(rb["cv_objective"])
        if math.isfinite(sa):
            std_a.append(sa)
        if math.isfinite(sb):
            std_b.append(sb)
        if math.isfinite(ca):
            cv_a.append(ca)
        if math.isfinite(cb):
            cv_b.append(cb)
        if math.isfinite(sa) and math.isfinite(sb):
            std_diff.append(sa - sb)

    wilcoxon = _wilcoxon_signed_rank_exact(std_diff)
    rows = [
        {"metric": "mean_std_A", "value": _safe_mean(std_a)},
        {"metric": "mean_std_B", "value": _safe_mean(std_b)},
        {"metric": "median_std_A", "value": _safe_median(std_a)},
        {"metric": "median_std_B", "value": _safe_median(std_b)},
        {"metric": "mean_cv_A", "value": _safe_mean(cv_a)},
        {"metric": "mean_cv_B", "value": _safe_mean(cv_b)},
        {"metric": "wilcoxon_std_p_value_two_sided", "value": wilcoxon["p_value_two_sided"]},
    ]
    return rows


ITER_LINE_PATTERN = re.compile(r"^(?:\s*)iter=\d+\s")
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s]+)")


def _parse_iter_line(line: str) -> Dict[str, Any] | None:
    if not ITER_LINE_PATTERN.match(line):
        return None
    kv = {k: v for k, v in KEY_VALUE_PATTERN.findall(line)}
    if "iter" not in kv or "elapsed" not in kv:
        return None
    return {
        "iteration": _to_int(kv.get("iter")),
        "elapsed": _to_float(kv.get("elapsed", "").removesuffix("s")),
        "step_runtime": _to_float(kv.get("step_runtime", "").removesuffix("s")),
        "destroy": kv.get("destroy", ""),
        "repair": kv.get("repair", ""),
        "incumbent_objective": _to_float(kv.get("incumbent_objective")),
        "contender_objective": _to_float(kv.get("contender_objective")),
        "accepted": _to_bool(kv.get("accepted")),
        "repair_failed": _to_bool(kv.get("repair_failed")),
        "used_exploration": _to_bool(kv.get("used_exploration")),
        "stagnation_rounds": _to_int(kv.get("stagnation_rounds")),
        "weights_updated": _to_bool(kv.get("weights_updated")),
    }


def _parse_log_events(
    log_path: Path,
    algorithm: str,
    instance: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    curve_points: List[Dict[str, Any]] = []
    operator_events: List[Dict[str, Any]] = []
    best_so_far = math.inf
    improvements: List[Dict[str, Any]] = []

    for line in lines:
        parsed = _parse_iter_line(line)
        if parsed is None:
            continue

        before = float(parsed["incumbent_objective"])
        after = float(parsed["contender_objective"])
        elapsed = float(parsed["elapsed"])
        step_runtime = float(parsed["step_runtime"])
        iteration = int(parsed["iteration"]) if parsed["iteration"] is not None else -1

        seen_candidates = [x for x in (before, after) if math.isfinite(x)]
        seen_best = min(seen_candidates) if seen_candidates else math.nan
        if math.isfinite(seen_best):
            prev = best_so_far
            best_so_far = min(best_so_far, seen_best)
            curve_points.append(
                {
                    "algorithm": algorithm,
                    "instance": instance,
                    "seed": seed,
                    "iteration": iteration,
                    "elapsed": elapsed,
                    "best_so_far_objective": best_so_far,
                }
            )
            if best_so_far < prev - EPS:
                improvements.append(
                    {
                        "iteration": iteration,
                        "elapsed": elapsed,
                        "best_so_far_objective": best_so_far,
                        "destroy": parsed["destroy"],
                        "repair": parsed["repair"],
                    }
                )

        delta = after - before if (math.isfinite(before) and math.isfinite(after)) else math.nan
        for operator_type, operator_name in (("destroy", parsed["destroy"]), ("repair", parsed["repair"])):
            operator_events.append(
                {
                    "algorithm": algorithm,
                    "instance": instance,
                    "instance_index": _instance_index(instance),
                    "seed": seed,
                    "iteration": iteration,
                    "elapsed": elapsed,
                    "step_runtime": step_runtime,
                    "operator_type": operator_type,
                    "operator_name": operator_name,
                    "objective_before": before,
                    "objective_after": after,
                    "delta_objective": delta,
                    "accepted": parsed["accepted"],
                    "repair_failed": parsed["repair_failed"],
                    "used_exploration": parsed["used_exploration"],
                }
            )

    final_best = min((float(p["best_so_far_objective"]) for p in curve_points), default=math.nan)
    final_contrib_destroy = ""
    final_contrib_repair = ""
    if improvements and math.isfinite(final_best):
        for event in improvements:
            if float(event["best_so_far_objective"]) <= final_best + EPS:
                final_contrib_destroy = str(event["destroy"])
                final_contrib_repair = str(event["repair"])
                break

    run_meta = {
        "algorithm": algorithm,
        "instance": instance,
        "seed": seed,
        "log_path": str(log_path),
        "final_best_from_log": final_best,
        "final_best_contributor_destroy": final_contrib_destroy,
        "final_best_contributor_repair": final_contrib_repair,
        "n_iterations_in_log": len(curve_points),
    }
    return curve_points, operator_events, run_meta


def _step_curve_on_grid(points: Sequence[Tuple[float, float]], grid: np.ndarray) -> np.ndarray:
    if not points:
        return np.full_like(grid, np.nan, dtype=float)
    points_sorted = sorted(points, key=lambda t: t[0])
    out = np.full_like(grid, np.nan, dtype=float)
    best = float(points_sorted[0][1])
    j = 0
    for idx, t in enumerate(grid):
        while j < len(points_sorted) and points_sorted[j][0] <= t + EPS:
            best = min(best, float(points_sorted[j][1]))
            j += 1
        out[idx] = best
    return out


def _convergence_tables(
    curve_rows: Sequence[Dict[str, Any]],
    cp_map: Dict[str, float],
    time_step: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[RunKey, float]]:
    run_points: Dict[RunKey, List[Tuple[float, float]]] = {}
    max_time = 0.0
    for row in curve_rows:
        key = RunKey(algorithm=str(row["algorithm"]), instance=str(row["instance"]), seed=int(row["seed"]))
        elapsed = float(row["elapsed"])
        value = float(row["best_so_far_objective"])
        if math.isfinite(elapsed) and math.isfinite(value):
            run_points.setdefault(key, []).append((elapsed, value))
            max_time = max(max_time, elapsed)

    max_time = max(max_time, time_step)
    grid = np.arange(0.0, max_time + time_step, time_step)
    run_curves: Dict[RunKey, np.ndarray] = {}
    for key, points in run_points.items():
        run_curves[key] = _step_curve_on_grid(points, grid)

    run_curve_rows: List[Dict[str, Any]] = []
    for key, arr in run_curves.items():
        cp = cp_map.get(key.instance, math.nan)
        for t, v in zip(grid, arr):
            run_curve_rows.append(
                {
                    "algorithm": key.algorithm,
                    "instance": key.instance,
                    "instance_index": _instance_index(key.instance),
                    "seed": key.seed,
                    "time_seconds": float(t),
                    "best_so_far_objective": float(v),
                    "best_so_far_gap_to_cp": _gap_to_cp(float(v), cp),
                }
            )

    by_instance_algo: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for key, arr in run_curves.items():
        by_instance_algo.setdefault((key.instance, key.algorithm), []).append(arr)

    instance_curve_rows: List[Dict[str, Any]] = []
    instance_curves: Dict[Tuple[str, str], np.ndarray] = {}
    for (instance, algorithm), curves in by_instance_algo.items():
        stacked = np.vstack(curves)
        mean_curve = np.nanmean(stacked, axis=0)
        instance_curves[(instance, algorithm)] = mean_curve
        cp = cp_map.get(instance, math.nan)
        for t, v in zip(grid, mean_curve):
            instance_curve_rows.append(
                {
                    "algorithm": algorithm,
                    "instance": instance,
                    "instance_index": _instance_index(instance),
                    "time_seconds": float(t),
                    "mean_best_so_far_objective": float(v),
                    "mean_best_so_far_gap_to_cp": _gap_to_cp(float(v), cp),
                }
            )

    algorithms = sorted({key.algorithm for key in run_curves.keys()})
    global_rows: List[Dict[str, Any]] = []
    for algorithm in algorithms:
        curves = [curve for (instance, algo), curve in instance_curves.items() if algo == algorithm]
        if not curves:
            continue
        stacked = np.vstack(curves)
        mean_obj = np.nanmean(stacked, axis=0)

        gap_curves = []
        for (instance, algo), curve in instance_curves.items():
            if algo != algorithm:
                continue
            cp = cp_map.get(instance, math.nan)
            if not math.isfinite(cp):
                continue
            denom = _denom_for_gap(cp)
            gap_curves.append((curve - cp) / denom)
        mean_gap = np.nanmean(np.vstack(gap_curves), axis=0) if gap_curves else np.full_like(mean_obj, np.nan)

        for t, v_obj, v_gap in zip(grid, mean_obj, mean_gap):
            global_rows.append(
                {
                    "algorithm": algorithm,
                    "time_seconds": float(t),
                    "global_mean_objective": float(v_obj),
                    "global_mean_gap_to_cp": float(v_gap),
                }
            )

    return run_curve_rows, instance_curve_rows, global_rows, {k: float(grid[-1]) for k in run_curves.keys()}


def _time_to_target(
    curve_rows: Sequence[Dict[str, Any]],
    cp_map: Dict[str, float],
    target_gap: float,
    algorithm_a: str,
    algorithm_b: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_run: Dict[RunKey, List[Tuple[float, float]]] = {}
    for row in curve_rows:
        key = RunKey(algorithm=str(row["algorithm"]), instance=str(row["instance"]), seed=int(row["seed"]))
        by_run.setdefault(key, []).append((float(row["elapsed"]), float(row["best_so_far_objective"])))

    run_rows: List[Dict[str, Any]] = []
    for key, points in by_run.items():
        cp = cp_map.get(key.instance, math.nan)
        if not math.isfinite(cp):
            t_hit = math.nan
            target_value = math.nan
        else:
            target_value = cp + _denom_for_gap(cp) * target_gap
            points_sorted = sorted(points, key=lambda x: x[0])
            t_hit = math.nan
            for t, v in points_sorted:
                if math.isfinite(v) and v <= target_value + EPS:
                    t_hit = t
                    break
        run_rows.append(
            {
                "algorithm": key.algorithm,
                "instance": key.instance,
                "instance_index": _instance_index(key.instance),
                "seed": key.seed,
                "cp_objective": cp,
                "target_gap": target_gap,
                "target_objective": target_value,
                "time_to_target_seconds": t_hit,
            }
        )

    grouped = _group_rows(run_rows, ["instance", "algorithm"])
    instance_rows: List[Dict[str, Any]] = []
    for (instance, algorithm), rows in grouped.items():
        times = [float(r["time_to_target_seconds"]) for r in rows]
        instance_rows.append(
            {
                "instance": instance,
                "instance_index": _instance_index(str(instance)),
                "algorithm": algorithm,
                "mean_time_to_target_seconds": _safe_mean(times),
                "median_time_to_target_seconds": _safe_median(times),
                "n_runs": len(rows),
            }
        )
    instance_rows.sort(key=lambda r: (_instance_sort_key(str(r["instance"])), str(r["algorithm"])))

    by_instance_algo = {(r["instance"], r["algorithm"]): r for r in instance_rows}
    diffs: List[float] = []
    for instance in sorted({str(r["instance"]) for r in instance_rows}, key=_instance_sort_key):
        ra = by_instance_algo.get((instance, algorithm_a))
        rb = by_instance_algo.get((instance, algorithm_b))
        if ra is None or rb is None:
            continue
        ta = float(ra["mean_time_to_target_seconds"])
        tb = float(rb["mean_time_to_target_seconds"])
        if math.isfinite(ta) and math.isfinite(tb):
            diffs.append(ta - tb)
    wilcoxon = _wilcoxon_signed_rank_exact(diffs)
    summary = [
        {"metric": "mean_time_diff_A_minus_B", "value": _safe_mean(diffs)},
        {"metric": "median_time_diff_A_minus_B", "value": _safe_median(diffs)},
        {"metric": "wilcoxon_time_to_target_p_value_two_sided", "value": wilcoxon["p_value_two_sided"]},
    ]
    return run_rows, instance_rows, summary


def _operator_metrics(
    operator_events: Sequence[Dict[str, Any]],
    run_meta_rows: Sequence[Dict[str, Any]],
    alns_summary_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    events = list(operator_events)
    run_totals: Dict[Tuple[str, str, int, str], int] = {}
    for event in events:
        key = (str(event["algorithm"]), str(event["instance"]), int(event["seed"]), str(event["operator_type"]))
        run_totals[key] = run_totals.get(key, 0) + 1

    grouped_run_op = _group_rows(events, ["algorithm", "instance", "seed", "operator_type", "operator_name"])
    per_run_rows: List[Dict[str, Any]] = []
    for (algorithm, instance, seed, operator_type, operator_name), rows in grouped_run_op.items():
        deltas = [float(r["delta_objective"]) for r in rows]
        improving = [d for d in deltas if math.isfinite(d) and d < -EPS]
        accepted_values = [1.0 if _to_bool(r["accepted"]) else 0.0 for r in rows]
        total_in_type = run_totals.get((str(algorithm), str(instance), int(seed), str(operator_type)), len(rows))
        per_run_rows.append(
            {
                "algorithm": algorithm,
                "instance": instance,
                "instance_index": _instance_index(str(instance)),
                "seed": seed,
                "operator_type": operator_type,
                "operator_name": operator_name,
                "selection_count": len(rows),
                "selection_share": len(rows) / total_in_type if total_in_type > 0 else math.nan,
                "improvement_count": len(improving),
                "improvement_probability": len(improving) / len(rows) if rows else math.nan,
                "acceptance_rate": _safe_mean(accepted_values),
                "mean_delta": _safe_mean(deltas),
                "mean_improving_delta": _safe_mean(improving),
            }
        )

    per_instance_grouped = _group_rows(per_run_rows, ["algorithm", "instance", "operator_type", "operator_name"])
    per_instance_rows: List[Dict[str, Any]] = []
    for (algorithm, instance, operator_type, operator_name), rows in per_instance_grouped.items():
        per_instance_rows.append(
            {
                "algorithm": algorithm,
                "instance": instance,
                "instance_index": _instance_index(str(instance)),
                "operator_type": operator_type,
                "operator_name": operator_name,
                "mean_selection_share": _safe_mean([float(r["selection_share"]) for r in rows]),
                "mean_improvement_probability": _safe_mean([float(r["improvement_probability"]) for r in rows]),
                "mean_acceptance_rate": _safe_mean([float(r["acceptance_rate"]) for r in rows]),
                "mean_improving_delta": _safe_mean([float(r["mean_improving_delta"]) for r in rows]),
            }
        )

    global_grouped = _group_rows(per_instance_rows, ["algorithm", "operator_type", "operator_name"])
    global_rows: List[Dict[str, Any]] = []
    for (algorithm, operator_type, operator_name), rows in global_grouped.items():
        shares = [float(r["mean_selection_share"]) for r in rows]
        probs = [float(r["mean_improvement_probability"]) for r in rows]
        imp = [float(r["mean_improving_delta"]) for r in rows]
        global_rows.append(
            {
                "algorithm": algorithm,
                "operator_type": operator_type,
                "operator_name": operator_name,
                "mean_selection_share": _safe_mean(shares),
                "mean_improvement_probability": _safe_mean(probs),
                "mean_improving_delta": _safe_mean(imp),
                "std_selection_share_across_instances": _safe_std(shares),
                "n_instances": len(rows),
            }
        )

    # Entropy / acceptance dynamics / selection distribution comparison
    entropy_rows: List[Dict[str, Any]] = []
    run_grouped = _group_rows(events, ["algorithm", "instance", "seed", "operator_type"])
    for (algorithm, instance, seed, operator_type), rows in run_grouped.items():
        counts: Dict[str, int] = {}
        for row in rows:
            name = str(row["operator_name"])
            counts[name] = counts.get(name, 0) + 1
        total = sum(counts.values())
        entropy = 0.0
        if total > 0:
            for c in counts.values():
                p = c / total
                if p > 0:
                    entropy -= p * math.log(p)
        entropy_rows.append(
            {
                "algorithm": algorithm,
                "instance": instance,
                "instance_index": _instance_index(str(instance)),
                "seed": seed,
                "operator_type": operator_type,
                "selection_entropy": entropy,
            }
        )

    # Final best contributors
    run_row_lookup = {(r["algorithm"], r["instance"], int(r["seed"])): r for r in run_meta_rows}
    contribution_rows: List[Dict[str, Any]] = []
    for key, run in run_row_lookup.items():
        for operator_type, field in (("destroy", "final_best_contributor_destroy"), ("repair", "final_best_contributor_repair")):
            name = str(run.get(field, ""))
            if name == "":
                continue
            contribution_rows.append(
                {
                    "algorithm": run["algorithm"],
                    "instance": run["instance"],
                    "instance_index": _instance_index(str(run["instance"])),
                    "seed": run["seed"],
                    "operator_type": operator_type,
                    "operator_name": name,
                }
            )

    contribution_freq: List[Dict[str, Any]] = []
    contribution_grouped = _group_rows(contribution_rows, ["algorithm", "operator_type", "operator_name"])
    for (algorithm, operator_type, operator_name), rows in contribution_grouped.items():
        contribution_freq.append(
            {
                "algorithm": algorithm,
                "operator_type": operator_type,
                "operator_name": operator_name,
                "count_final_best_contributions": len(rows),
            }
        )

    # Task 19/20
    run_runtime_map = {
        (str(r["algorithm"]), str(r["instance"]), int(r["seed"])): float(r["runtime_seconds"])
        for r in alns_summary_rows
        if math.isfinite(float(r.get("runtime_seconds", math.nan)))
    }
    phase_events: List[Dict[str, Any]] = []
    for event in events:
        run_key = (str(event["algorithm"]), str(event["instance"]), int(event["seed"]))
        runtime = run_runtime_map.get(run_key, math.nan)
        elapsed = float(event["elapsed"])
        if math.isfinite(runtime) and runtime > EPS and math.isfinite(elapsed):
            ratio = max(0.0, min(1.0, elapsed / runtime))
            if ratio < 0.3:
                phase = "early_30pct"
            elif ratio < 0.7:
                phase = "middle_40pct"
            else:
                phase = "late_30pct"
        else:
            phase = "unknown"
        phase_events.append({**event, "phase": phase})

    phase_grouped = _group_rows(phase_events, ["algorithm", "phase", "operator_type", "operator_name"])
    phase_rows: List[Dict[str, Any]] = []
    for (algorithm, phase, operator_type, operator_name), rows in phase_grouped.items():
        deltas = [float(r["delta_objective"]) for r in rows]
        improving = [d for d in deltas if math.isfinite(d) and d < -EPS]
        phase_rows.append(
            {
                "algorithm": algorithm,
                "phase": phase,
                "operator_type": operator_type,
                "operator_name": operator_name,
                "selection_count": len(rows),
                "improvement_probability": len(improving) / len(rows) if rows else math.nan,
                "mean_delta": _safe_mean(deltas),
                "mean_improving_delta": _safe_mean(improving),
            }
        )

    efficiency_grouped = _group_rows(events, ["algorithm", "operator_type", "operator_name"])
    efficiency_rows: List[Dict[str, Any]] = []
    for (algorithm, operator_type, operator_name), rows in efficiency_grouped.items():
        total_improvement = 0.0
        total_time = 0.0
        for row in rows:
            d = float(row["delta_objective"])
            rt = float(row["step_runtime"])
            if math.isfinite(d) and d < -EPS:
                total_improvement += -d
            if math.isfinite(rt) and rt > 0:
                total_time += rt
        efficiency_rows.append(
            {
                "record_type": "efficiency",
                "algorithm": algorithm,
                "operator_type": operator_type,
                "operator_name": operator_name,
                "total_improvement": total_improvement,
                "total_operator_runtime_proxy_seconds": total_time,
                "efficiency_improvement_per_second": (total_improvement / total_time) if total_time > EPS else math.nan,
            }
        )
    acceptance_by_bin: Dict[Tuple[str, int], List[float]] = {}
    events_by_run = _group_rows(events, ["algorithm", "instance", "seed"])
    for (algorithm, instance, seed), rows in events_by_run.items():
        runtime = run_runtime_map.get((str(algorithm), str(instance), int(seed)), math.nan)
        rows_sorted = sorted(rows, key=lambda r: (float(r["elapsed"]), int(r["iteration"])))
        max_iter = max((int(r["iteration"]) for r in rows_sorted), default=1)
        for row in rows_sorted:
            elapsed = float(row["elapsed"])
            if math.isfinite(runtime) and runtime > EPS and math.isfinite(elapsed):
                ratio = max(0.0, min(0.999999, elapsed / runtime))
            else:
                ratio = max(0.0, min(0.999999, int(row["iteration"]) / max(1, max_iter)))
            bin_id = int(ratio * 10)
            acceptance_by_bin.setdefault((str(algorithm), bin_id), []).append(1.0 if _to_bool(row["accepted"]) else 0.0)

    acceptance_rows: List[Dict[str, Any]] = []
    for (algorithm, bin_id), values in sorted(acceptance_by_bin.items()):
        acceptance_rows.append(
            {
                "algorithm": algorithm,
                "progress_bin": f"{bin_id * 10:02d}-{(bin_id + 1) * 10:02d}pct",
                "progress_bin_index": bin_id,
                "acceptance_rate": _safe_mean(values),
                "n_events": len(values),
            }
        )

    for row in phase_rows:
        row["record_type"] = "phase_metrics"

    return (
        per_run_rows,
        per_instance_rows,
        global_rows,
        entropy_rows,
        contribution_freq,
        phase_rows + efficiency_rows,
        acceptance_rows,
    )


def _plot_boxplot_gaps(master_rows: Sequence[Dict[str, Any]], algorithm_a: str, algorithm_b: str, out_path: Path) -> None:
    vals_a = [float(r["gap_to_cp"]) for r in master_rows if r["algorithm"] == algorithm_a and math.isfinite(float(r["gap_to_cp"]))]
    vals_b = [float(r["gap_to_cp"]) for r in master_rows if r["algorithm"] == algorithm_b and math.isfinite(float(r["gap_to_cp"]))]
    if not vals_a and not vals_b:
        return
    plt.figure(figsize=(8, 5))
    plt.boxplot([vals_a, vals_b], tick_labels=[algorithm_a, algorithm_b], showmeans=True)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Final Relative Gap to CP")
    plt.title("Final Relative Gap Distribution")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_scatter_mean_ab(per_instance_rows: Sequence[Dict[str, Any]], algorithm_a: str, algorithm_b: str, out_path: Path) -> None:
    by_instance_algo = {(r["instance"], r["algorithm"]): r for r in per_instance_rows}
    points: List[Tuple[float, float, str]] = []
    for instance in sorted({str(r["instance"]) for r in per_instance_rows}, key=_instance_sort_key):
        ra = by_instance_algo.get((instance, algorithm_a))
        rb = by_instance_algo.get((instance, algorithm_b))
        if ra is None or rb is None:
            continue
        xa = float(ra["mean_objective"])
        yb = float(rb["mean_objective"])
        if math.isfinite(xa) and math.isfinite(yb):
            points.append((xa, yb, instance))
    if not points:
        return
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    lo = min(x + y)
    hi = max(x + y)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.85)
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    plt.xlabel(f"Mean Objective {algorithm_a}")
    plt.ylabel(f"Mean Objective {algorithm_b}")
    plt.title("Per-instance Mean Objective Comparison")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_global_convergence(global_curve_rows: Sequence[Dict[str, Any]], out_obj: Path, out_gap: Path) -> None:
    if not global_curve_rows:
        return
    grouped = _group_rows(global_curve_rows, ["algorithm"])
    plt.figure(figsize=(8, 5))
    for (algorithm,), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: float(r["time_seconds"]))
        x = [float(r["time_seconds"]) for r in rows_sorted]
        y = [float(r["global_mean_objective"]) for r in rows_sorted]
        plt.plot(x, y, label=str(algorithm))
    plt.xlabel("Time (s)")
    plt.ylabel("Global Mean Objective")
    plt.title("Global Convergence Curves")
    plt.legend()
    plt.tight_layout()
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_obj, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    for (algorithm,), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: float(r["time_seconds"]))
        x = [float(r["time_seconds"]) for r in rows_sorted]
        y = [float(r["global_mean_gap_to_cp"]) for r in rows_sorted]
        plt.plot(x, y, label=str(algorithm))
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Global Mean Gap to CP")
    plt.title("Global Relative-Gap Convergence")
    plt.legend()
    plt.tight_layout()
    out_gap.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_gap, dpi=140)
    plt.close()


def _plot_stability(per_instance_rows: Sequence[Dict[str, Any]], algorithm_a: str, algorithm_b: str, out_path: Path) -> None:
    by_instance_algo = {(r["instance"], r["algorithm"]): r for r in per_instance_rows}
    instances = sorted({str(r["instance"]) for r in per_instance_rows}, key=_instance_sort_key)
    x = np.arange(len(instances))
    y_a: List[float] = []
    y_b: List[float] = []
    labels: List[str] = []
    for instance in instances:
        ra = by_instance_algo.get((instance, algorithm_a))
        rb = by_instance_algo.get((instance, algorithm_b))
        if ra is None or rb is None:
            continue
        y_a.append(float(ra["std_objective"]))
        y_b.append(float(rb["std_objective"]))
        labels.append(instance)
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.42
    plt.figure(figsize=(max(8, len(labels) * 0.55), 5))
    plt.bar(x - width / 2, y_a, width=width, label=algorithm_a)
    plt.bar(x + width / 2, y_b, width=width, label=algorithm_b)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Std Across Seeds")
    plt.title("Within-instance Stability Comparison")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_selection_heatmap(per_instance_operator_rows: Sequence[Dict[str, Any]], algorithm_a: str, algorithm_b: str, out_path: Path) -> None:
    rows = list(per_instance_operator_rows)
    if not rows:
        return
    for row in rows:
        row["operator_id"] = f"{row['operator_type']}:{row['operator_name']}"

    op_shares: Dict[str, List[float]] = {}
    for row in rows:
        op = str(row["operator_id"])
        op_shares.setdefault(op, []).append(float(row["mean_selection_share"]))
    top_ops = sorted(op_shares.keys(), key=lambda o: _safe_mean(op_shares[o]), reverse=True)[:16]
    instances = sorted({str(r["instance"]) for r in rows}, key=_instance_sort_key)

    def matrix_for_algorithm(algorithm: str) -> np.ndarray:
        matrix = np.zeros((len(instances), len(top_ops)), dtype=float)
        lookup = {(str(r["instance"]), str(r["operator_id"])): float(r["mean_selection_share"]) for r in rows if r["algorithm"] == algorithm}
        for i, instance in enumerate(instances):
            for j, op in enumerate(top_ops):
                matrix[i, j] = lookup.get((instance, op), 0.0)
        return matrix

    mat_a = matrix_for_algorithm(algorithm_a)
    mat_b = matrix_for_algorithm(algorithm_b)

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(top_ops) * 0.6), max(6, len(instances) * 0.35)), constrained_layout=True)
    im0 = axes[0].imshow(mat_a, aspect="auto", cmap="YlGnBu")
    im1 = axes[1].imshow(mat_b, aspect="auto", cmap="YlGnBu")
    axes[0].set_title(algorithm_a)
    axes[1].set_title(algorithm_b)
    for ax in axes:
        ax.set_yticks(np.arange(len(instances)))
        ax.set_yticklabels(instances)
        ax.set_xticks(np.arange(len(top_ops)))
        ax.set_xticklabels(top_ops, rotation=70, ha="right", fontsize=8)
        ax.set_xlabel("Operator")
        ax.set_ylabel("Instance")
    fig.colorbar(im0, ax=axes, fraction=0.02, pad=0.02, label="Mean Selection Share")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_operator_improvement(global_operator_rows: Sequence[Dict[str, Any]], algorithm_a: str, algorithm_b: str, out_path: Path) -> None:
    rows = list(global_operator_rows)
    if not rows:
        return
    for row in rows:
        row["operator_id"] = f"{row['operator_type']}:{row['operator_name']}"
    a_lookup = {str(r["operator_id"]): float(r["mean_improvement_probability"]) for r in rows if r["algorithm"] == algorithm_a}
    b_lookup = {str(r["operator_id"]): float(r["mean_improvement_probability"]) for r in rows if r["algorithm"] == algorithm_b}
    ops = sorted(set(a_lookup.keys()) | set(b_lookup.keys()))
    ops = sorted(ops, key=lambda op: max(a_lookup.get(op, 0.0), b_lookup.get(op, 0.0)), reverse=True)[:18]
    if not ops:
        return
    x = np.arange(len(ops))
    width = 0.42
    plt.figure(figsize=(max(10, len(ops) * 0.6), 5))
    plt.bar(x - width / 2, [a_lookup.get(op, 0.0) for op in ops], width=width, label=algorithm_a)
    plt.bar(x + width / 2, [b_lookup.get(op, 0.0) for op in ops], width=width, label=algorithm_b)
    plt.xticks(x, ops, rotation=70, ha="right", fontsize=8)
    plt.ylabel("Mean Improvement Probability")
    plt.title("Operator Improvement Probability Comparison")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_cactus(time_to_target_run_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    grouped = _group_rows(time_to_target_run_rows, ["algorithm"])
    if not grouped:
        return
    plt.figure(figsize=(8, 5))
    plotted = 0
    for (algorithm,), rows in grouped.items():
        times = sorted([float(r["time_to_target_seconds"]) for r in rows if math.isfinite(float(r["time_to_target_seconds"]))])
        if not times:
            continue
        x = np.arange(1, len(times) + 1)
        plt.step(x, times, where="post", label=str(algorithm))
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Solved Runs (target reached count)")
    plt.ylabel("Time to Target (s)")
    plt.title("Cactus Plot: Time to Target")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    alns_root = (base_dir / args.alns_root).resolve() if not Path(args.alns_root).is_absolute() else Path(args.alns_root)
    baseline_dir = (base_dir / args.baseline_dir).resolve() if not Path(args.baseline_dir).is_absolute() else Path(args.baseline_dir)

    summary_a = _resolve_summary_path(alns_root=alns_root, variant=args.variant_a, run_id=args.run_id)
    summary_b = _resolve_summary_path(alns_root=alns_root, variant=args.variant_b, run_id=args.run_id)
    baseline_csv = (
        _latest_baseline_csv(baseline_dir) if args.baseline_csv == "" else
        ((base_dir / args.baseline_csv).resolve() if not Path(args.baseline_csv).is_absolute() else Path(args.baseline_csv))
    )
    if not baseline_csv.exists():
        raise FileNotFoundError(f"baseline csv not found: {baseline_csv}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        (base_dir / args.output_dir / timestamp).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir) / timestamp
    )
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    algo_a_label = "ALNS-A"
    algo_b_label = "ALNS-B"

    a_rows, a_logs = _read_alns_summary(summary_a, algorithm_label=algo_a_label)
    b_rows, b_logs = _read_alns_summary(summary_b, algorithm_label=algo_b_label)
    alns_rows = a_rows + b_rows
    log_map = {**a_logs, **b_logs}

    cp_rows, cp_map = _build_cp_baseline_rows(baseline_csv)
    master_rows = _build_master_table(alns_rows=alns_rows, cp_rows=cp_rows, cp_map=cp_map, log_map=log_map)
    _write_csv(
        tables_dir / "master_data.csv",
        master_rows,
        fieldnames=[
            "instance",
            "instance_index",
            "algorithm",
            "seed",
            "objective",
            "runtime_seconds",
            "gap_to_cp",
            "status",
            "timeout_hit",
            "solver_name",
            "log_path",
            "source_summary",
        ],
    )

    per_instance_rows = _per_instance_aggregation(alns_rows=alns_rows, cp_map=cp_map)
    _write_csv(
        tables_dir / "per_instance_aggregation.csv",
        per_instance_rows,
        fieldnames=[
            "instance",
            "instance_index",
            "algorithm",
            "n_seeds",
            "mean_objective",
            "median_objective",
            "best_objective",
            "std_objective",
            "min_objective",
            "max_objective",
            "mean_gap_to_cp",
            "best_gap_to_cp",
            "cv_objective",
        ],
    )

    win_loss_rows = _win_loss_summary(per_instance_rows, algorithm_a=algo_a_label, algorithm_b=algo_b_label, cp_map=cp_map)
    _write_csv(
        tables_dir / "win_loss_summary.csv",
        win_loss_rows,
        fieldnames=["comparison", "wins", "ties", "losses", "missing"],
    )

    stats_rows = _paired_statistics(per_instance_rows, algorithm_a=algo_a_label, algorithm_b=algo_b_label)
    _write_csv(tables_dir / "paired_statistics.csv", stats_rows, fieldnames=["metric", "value"])

    stability_rows = _stability_across_instances(per_instance_rows, algorithm_a=algo_a_label, algorithm_b=algo_b_label)
    _write_csv(tables_dir / "stability_across_instances.csv", stability_rows, fieldnames=["metric", "value"])

    # Parse logs for tasks 9-20.
    curve_rows: List[Dict[str, Any]] = []
    operator_events: List[Dict[str, Any]] = []
    run_meta_rows: List[Dict[str, Any]] = []
    for row in alns_rows:
        key = RunKey(algorithm=str(row["algorithm"]), instance=str(row["instance"]), seed=int(row["seed"]))
        log_path = log_map.get(key)
        if log_path is None or not log_path.exists():
            continue
        crows, op_rows, run_meta = _parse_log_events(
            log_path=log_path,
            algorithm=key.algorithm,
            instance=key.instance,
            seed=key.seed,
        )
        curve_rows.extend(crows)
        operator_events.extend(op_rows)
        run_meta_rows.append(run_meta)

    _write_csv(
        tables_dir / "run_log_meta.csv",
        run_meta_rows,
        fieldnames=[
            "algorithm",
            "instance",
            "seed",
            "log_path",
            "final_best_from_log",
            "final_best_contributor_destroy",
            "final_best_contributor_repair",
            "n_iterations_in_log",
        ],
    )

    _write_csv(
        tables_dir / "convergence_run_points.csv",
        curve_rows,
        fieldnames=[
            "algorithm",
            "instance",
            "seed",
            "iteration",
            "elapsed",
            "best_so_far_objective",
        ],
    )

    _write_csv(
        tables_dir / "operator_events.csv",
        operator_events,
        fieldnames=[
            "algorithm",
            "instance",
            "instance_index",
            "seed",
            "iteration",
            "elapsed",
            "step_runtime",
            "operator_type",
            "operator_name",
            "objective_before",
            "objective_after",
            "delta_objective",
            "accepted",
            "repair_failed",
            "used_exploration",
        ],
    )

    run_curve_rows, instance_curve_rows, global_curve_rows, _ = _convergence_tables(
        curve_rows=curve_rows,
        cp_map=cp_map,
        time_step=args.time_step,
    )
    _write_csv(
        tables_dir / "convergence_run_curves_on_grid.csv",
        run_curve_rows,
        fieldnames=[
            "algorithm",
            "instance",
            "instance_index",
            "seed",
            "time_seconds",
            "best_so_far_objective",
            "best_so_far_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "convergence_instance_mean_curves.csv",
        instance_curve_rows,
        fieldnames=[
            "algorithm",
            "instance",
            "instance_index",
            "time_seconds",
            "mean_best_so_far_objective",
            "mean_best_so_far_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "convergence_global_curves.csv",
        global_curve_rows,
        fieldnames=[
            "algorithm",
            "time_seconds",
            "global_mean_objective",
            "global_mean_gap_to_cp",
        ],
    )

    ttt_run_rows, ttt_instance_rows, ttt_summary_rows = _time_to_target(
        curve_rows=curve_rows,
        cp_map=cp_map,
        target_gap=args.target_gap,
        algorithm_a=algo_a_label,
        algorithm_b=algo_b_label,
    )
    _write_csv(
        tables_dir / "time_to_target_runs.csv",
        ttt_run_rows,
        fieldnames=[
            "algorithm",
            "instance",
            "instance_index",
            "seed",
            "cp_objective",
            "target_gap",
            "target_objective",
            "time_to_target_seconds",
        ],
    )
    _write_csv(
        tables_dir / "time_to_target_instance.csv",
        ttt_instance_rows,
        fieldnames=[
            "instance",
            "instance_index",
            "algorithm",
            "mean_time_to_target_seconds",
            "median_time_to_target_seconds",
            "n_runs",
        ],
    )
    _write_csv(
        tables_dir / "time_to_target_summary.csv",
        ttt_summary_rows,
        fieldnames=["metric", "value"],
    )

    (
        per_run_op_rows,
        per_instance_op_rows,
        global_op_rows,
        entropy_rows,
        contribution_rows,
        phase_and_eff_rows,
        acceptance_dynamics_rows,
    ) = _operator_metrics(
        operator_events=operator_events,
        run_meta_rows=run_meta_rows,
        alns_summary_rows=alns_rows,
    )
    _write_csv(
        tables_dir / "operator_metrics_per_run.csv",
        per_run_op_rows,
        fieldnames=[
            "algorithm",
            "instance",
            "instance_index",
            "seed",
            "operator_type",
            "operator_name",
            "selection_count",
            "selection_share",
            "improvement_count",
            "improvement_probability",
            "acceptance_rate",
            "mean_delta",
            "mean_improving_delta",
        ],
    )
    _write_csv(
        tables_dir / "operator_metrics_per_instance.csv",
        per_instance_op_rows,
        fieldnames=[
            "algorithm",
            "instance",
            "instance_index",
            "operator_type",
            "operator_name",
            "mean_selection_share",
            "mean_improvement_probability",
            "mean_acceptance_rate",
            "mean_improving_delta",
        ],
    )
    _write_csv(
        tables_dir / "operator_effectiveness_global.csv",
        global_op_rows,
        fieldnames=[
            "algorithm",
            "operator_type",
            "operator_name",
            "mean_selection_share",
            "mean_improvement_probability",
            "mean_improving_delta",
            "std_selection_share_across_instances",
            "n_instances",
        ],
    )
    _write_csv(
        tables_dir / "operator_entropy_per_run.csv",
        entropy_rows,
        fieldnames=["algorithm", "instance", "instance_index", "seed", "operator_type", "selection_entropy"],
    )
    _write_csv(
        tables_dir / "operator_final_best_contributions.csv",
        contribution_rows,
        fieldnames=["algorithm", "operator_type", "operator_name", "count_final_best_contributions"],
    )

    phase_rows = [r for r in phase_and_eff_rows if "phase" in r]
    efficiency_rows = [r for r in phase_and_eff_rows if "efficiency_improvement_per_second" in r]
    _write_csv(
        tables_dir / "operator_phase_metrics.csv",
        phase_rows,
        fieldnames=[
            "record_type",
            "algorithm",
            "phase",
            "operator_type",
            "operator_name",
            "selection_count",
            "improvement_probability",
            "mean_delta",
            "mean_improving_delta",
        ],
    )
    _write_csv(
        tables_dir / "operator_efficiency.csv",
        efficiency_rows,
        fieldnames=[
            "record_type",
            "algorithm",
            "operator_type",
            "operator_name",
            "total_improvement",
            "total_operator_runtime_proxy_seconds",
            "efficiency_improvement_per_second",
        ],
    )
    _write_csv(
        tables_dir / "operator_acceptance_dynamics.csv",
        acceptance_dynamics_rows,
        fieldnames=[
            "algorithm",
            "progress_bin",
            "progress_bin_index",
            "acceptance_rate",
            "n_events",
        ],
    )

    # Plots required by the plan.
    _plot_boxplot_gaps(master_rows, algorithm_a=algo_a_label, algorithm_b=algo_b_label, out_path=plots_dir / "boxplot_final_relative_gaps.png")
    _plot_scatter_mean_ab(per_instance_rows, algorithm_a=algo_a_label, algorithm_b=algo_b_label, out_path=plots_dir / "scatter_mean_A_vs_B.png")
    _plot_global_convergence(global_curve_rows, out_obj=plots_dir / "global_convergence_objective.png", out_gap=plots_dir / "global_convergence_gap.png")
    _plot_stability(per_instance_rows, algorithm_a=algo_a_label, algorithm_b=algo_b_label, out_path=plots_dir / "stability_std_comparison.png")
    _plot_selection_heatmap(per_instance_op_rows, algorithm_a=algo_a_label, algorithm_b=algo_b_label, out_path=plots_dir / "operator_selection_heatmap.png")
    _plot_operator_improvement(global_op_rows, algorithm_a=algo_a_label, algorithm_b=algo_b_label, out_path=plots_dir / "operator_improvement_probability.png")
    _plot_cactus(ttt_run_rows, out_path=plots_dir / "cactus_time_to_target.png")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "summary_a": str(summary_a),
        "summary_b": str(summary_b),
        "baseline_csv": str(baseline_csv),
        "algorithm_a_label": algo_a_label,
        "algorithm_b_label": algo_b_label,
        "target_gap": args.target_gap,
        "time_step": args.time_step,
        "n_master_rows": len(master_rows),
        "n_alns_rows": len(alns_rows),
        "n_cp_rows": len(cp_rows),
        "n_run_logs_parsed": len(run_meta_rows),
        "n_operator_events": len(operator_events),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full ALNS analysis pipeline implementing tasks from ALNS_Complete_Detailed_Analysis_Plan."
    )
    parser.add_argument("--alns-root", default="benchmark_alns", help="Path to benchmark_alns root.")
    parser.add_argument("--baseline-dir", default="benchmark_baseline", help="Directory with baseline_runs_*.csv.")
    parser.add_argument("--baseline-csv", default="", help="Explicit baseline csv path. If empty, latest is used.")
    parser.add_argument("--variant-a", default="alns_plain", help="Variant directory for ALNS-A.")
    parser.add_argument("--variant-b", default="alns_late_phase", help="Variant directory for ALNS-B.")
    parser.add_argument("--run-id", default=None, help="Run ID folder (e.g., 20260227-153053). If omitted, latest per variant.")
    parser.add_argument("--target-gap", type=float, default=0.02, help="Target gap above CP for time-to-target analysis.")
    parser.add_argument("--time-step", type=float, default=1.0, help="Time grid step in seconds for convergence aggregation.")
    parser.add_argument("--output-dir", default="analysis_alns", help="Output directory root.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    metadata = run_analysis(args)
    print("ALNS analysis finished.")
    print(f"Output directory: {metadata['output_dir']}")
    print(f"Rows: master={metadata['n_master_rows']} alns={metadata['n_alns_rows']} cp={metadata['n_cp_rows']}")
    print(f"Parsed logs={metadata['n_run_logs_parsed']} operator_events={metadata['n_operator_events']}")


if __name__ == "__main__":
    main()
