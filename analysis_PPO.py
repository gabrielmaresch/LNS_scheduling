from __future__ import annotations

import argparse
import csv
from datetime import datetime
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import subprocess
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
CHECKPOINT_LOG_PATTERN = re.compile(r"^===== instance=(\w+) run=(\d+)/(\d+) checkpoint=([^\s]+) =====$")
ITER_PATTERN = re.compile(r"^iter=(\d+)\s")
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s]+)")
REWARD_PATTERN = re.compile(r"reward=([+-]?\d+(?:\.\d+)?)")
STATE_DIM_PATTERN = re.compile(r"Detected state_dim=(\d+)")
SOLVED_PATTERN = re.compile(r"^solved=(True|False)\s+stop_time=([0-9.]+)s\s+reason=([A-Za-z_]+)")
COMPLETED_PATTERN = re.compile(
    r"^run_completed=True runtime=([0-9.]+)s best_objective=([^\s]+) current_objective=([^\s]+)"
)
ERROR_PATTERN = re.compile(r"^run_error=(.+)$")


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
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_list_csv(raw: str) -> List[str]:
    text = str(raw).strip()
    if text == "":
        return []
    return [t.strip() for t in text.split(",") if t.strip() != ""]


def _instance_index(name: str) -> int | None:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))
    return None


def _instance_sort_key(name: str) -> Tuple[int, str]:
    idx = _instance_index(name)
    if idx is None:
        return (10**9, name)
    return (idx, name)


def _file_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or "unknown"


def _action_plot_id(action_type: Any, action_name: Any) -> str:
    op_type = str(action_type).strip().lower()
    name = str(action_name).strip()
    if name == "":
        return op_type if op_type else "unknown"

    name_lower = name.lower()
    # Collapse duplicated explicit prefixes, e.g. "repair.repair_x" -> "repair_x".
    for sep in (".", ":"):
        token = f"{op_type}{sep}"
        while op_type and name_lower.startswith(token):
            name = name[len(token):].strip()
            name_lower = name.lower()
            if name == "":
                return op_type

    # If name already encodes type (common: repair_x / destroy_x), do not duplicate it.
    if op_type and (name_lower == op_type or name_lower.startswith(f"{op_type}_") or name_lower.startswith(f"{op_type}-")):
        return name
    if op_type:
        return f"{op_type}:{name}"
    return name


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


def _quantile(values: Sequence[float], q: float) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return math.nan
    arr = np.array(finite, dtype=float)
    return float(np.quantile(arr, q))


def _step_curve_on_grid(points: Sequence[Tuple[float, float]], grid: np.ndarray) -> np.ndarray:
    if not points:
        return np.full_like(grid, np.nan, dtype=float)
    pts = sorted([(float(t), float(v)) for t, v in points if math.isfinite(t) and math.isfinite(v)], key=lambda x: x[0])
    if not pts:
        return np.full_like(grid, np.nan, dtype=float)
    out = np.full_like(grid, np.nan, dtype=float)
    best = math.inf
    idx = 0
    for i, t in enumerate(grid):
        while idx < len(pts) and pts[idx][0] <= t + EPS:
            best = min(best, pts[idx][1])
            idx += 1
        out[i] = best
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_config(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


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


def _relative_diff(value: float, reference: float) -> float:
    if not (math.isfinite(value) and math.isfinite(reference)):
        return math.nan
    if abs(reference) > EPS:
        return (value - reference) / abs(reference)
    return value - reference


def _latest_baseline_csv(baseline_dir: Path) -> Path:
    candidates = sorted(baseline_dir.glob("baseline_runs_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no baseline_runs_*.csv found in {baseline_dir}")
    return candidates[-1]


def _build_cp_map(baseline_csv: Path) -> Dict[str, float]:
    rows = _read_csv(baseline_csv)
    by_instance: Dict[str, float] = {}
    for row in rows:
        instance = str(row.get("instance", "")).strip()
        if instance == "":
            continue
        objective = _to_float(row.get("objective"))
        if not math.isfinite(objective):
            continue
        by_instance[instance] = min(by_instance.get(instance, objective), objective)
    return by_instance


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and abs(indexed[j][1] - indexed[i][1]) <= EPS:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _wilcoxon_signed_rank_exact(diffs: Sequence[float]) -> Dict[str, float]:
    nz = [float(d) for d in diffs if math.isfinite(d) and abs(d) > EPS]
    n = len(nz)
    if n == 0:
        return {
            "n": 0.0,
            "w_plus": math.nan,
            "w_minus": math.nan,
            "w_stat": math.nan,
            "p_value_two_sided": math.nan,
        }

    abs_vals = [abs(v) for v in nz]
    ranks = _average_ranks(abs_vals)
    signed_ranks = [(ranks[i], 1 if nz[i] > 0 else -1) for i in range(n)]

    w_plus_obs = float(sum(rank for rank, sign in signed_ranks if sign > 0))
    w_minus_obs = float(sum(rank for rank, sign in signed_ranks if sign < 0))
    w_obs = min(w_plus_obs, w_minus_obs)

    if n <= 20:
        dist: Dict[float, int] = {}
        rank_vals = [rank for rank, _ in signed_ranks]
        total_states = 1 << n
        for mask in range(total_states):
            wp = 0.0
            for i, rank in enumerate(rank_vals):
                if (mask >> i) & 1:
                    wp += rank
            dist[wp] = dist.get(wp, 0) + 1
        center = sum(rank_vals) / 2.0
        w_obs_plus = w_plus_obs
        extreme = 0
        for wp, count in dist.items():
            if abs(wp - center) >= abs(w_obs_plus - center) - 1e-12:
                extreme += count
        p_two_sided = extreme / total_states
    else:
        p_two_sided = math.nan

    return {
        "n": float(n),
        "w_plus": w_plus_obs,
        "w_minus": w_minus_obs,
        "w_stat": w_obs,
        "p_value_two_sided": float(min(1.0, p_two_sided)) if math.isfinite(p_two_sided) else math.nan,
    }


def _adjust_pvalues_holm(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    indexed = [(i, p) for i, p in enumerate(p_values) if math.isfinite(p)]
    indexed.sort(key=lambda t: t[1])
    adjusted = [math.nan] * n
    running = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = (len(indexed) - rank + 1) * p
        running = max(running, adj)
        adjusted[idx] = min(1.0, running)
    return adjusted


def _adjust_pvalues_bh(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    indexed = [(i, p) for i, p in enumerate(p_values) if math.isfinite(p)]
    indexed.sort(key=lambda t: t[1])
    adjusted = [math.nan] * n
    running = 1.0
    m = len(indexed)
    for rank_rev, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_rev + 1
        adj = p * m / rank
        running = min(running, adj)
        adjusted[idx] = min(1.0, running)
    return adjusted


def _safe_git_commit(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _latest_run_id(variant_dir: Path) -> str:
    candidates = sorted(
        [
            p
            for p in variant_dir.iterdir()
            if p.is_dir() and ((p / "results" / "summary.csv").exists() or (p / "summary.csv").exists())
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"no run directories with summary.csv found in {variant_dir}")
    return candidates[-1].name


def _resolve_run_dir(variant_dir: Path, run_id: str | None) -> Tuple[Path, Path]:
    selected_run_id = run_id if run_id is not None else _latest_run_id(variant_dir)
    run_dir = variant_dir / selected_run_id
    summary_path = run_dir / "results" / "summary.csv"
    if not summary_path.exists():
        summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found in {run_dir}")
    return run_dir, summary_path


def _latest_checkpoint_benchmark_run(checkpoint_root: Path) -> str | None:
    if not checkpoint_root.exists():
        return None
    run_dirs = sorted([p for p in checkpoint_root.iterdir() if p.is_dir()])
    if not run_dirs:
        return None
    return run_dirs[-1].name


def _parse_checkpoint_log(log_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: Dict[str, Any] = {"source_log": str(log_path)}

    rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    iter_count = 0
    reward_sum = 0.0
    state_dim_value: int | None = None
    in_run_block = False

    def finalize_current() -> None:
        nonlocal current, iter_count, reward_sum, state_dim_value
        if current is None:
            return
        current["episode_length_steps"] = iter_count
        current["return_sum"] = reward_sum if iter_count > 0 else math.nan
        if current.get("state_dim") is None and state_dim_value is not None:
            current["state_dim"] = state_dim_value
        if "status" not in current:
            current["status"] = "incomplete"
        rows.append(current)
        current = None
        iter_count = 0
        reward_sum = 0.0
        state_dim_value = None

    for line in lines:
        line = line.strip()
        if line == "":
            continue

        if not in_run_block and line.startswith("====="):
            in_run_block = True

        if not in_run_block and "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
            continue

        start_match = CHECKPOINT_LOG_PATTERN.match(line)
        if start_match:
            finalize_current()
            instance, run_idx, runs_each, checkpoint = start_match.groups()
            current = {
                "checkpoint": checkpoint.strip(),
                "model_id": Path(checkpoint.strip()).stem,
                "test_instance_id": instance.strip(),
                "instance_index": _instance_index(instance.strip()),
                "eval_seed": _to_int(run_idx),
                "runs_each": _to_int(runs_each),
                "final_objective": math.nan,
                "best_objective_over_episode": math.nan,
                "episode_length_steps": 0,
                "runtime_seconds": math.nan,
                "termination_reason": "",
                "solved": False,
                "constraint_violation_metric": math.nan,
                "return_sum": math.nan,
                "discounted_return": math.nan,
                "state_dim": None,
                "status": "started",
                "error": "",
                "source_log": str(log_path),
            }
            continue

        if current is None:
            continue

        state_match = STATE_DIM_PATTERN.search(line)
        if state_match:
            state_dim_value = _to_int(state_match.group(1))
            if state_dim_value is not None:
                current["state_dim"] = state_dim_value

        if ITER_PATTERN.match(line):
            iter_count += 1
            reward_match = REWARD_PATTERN.search(line)
            if reward_match:
                reward_sum += _to_float(reward_match.group(1))
            kv = {k: v for k, v in KEY_VALUE_PATTERN.findall(line)}
            objective_raw = str(kv.get("objective", ""))
            before_raw = ""
            after_raw = ""
            if "->" in objective_raw:
                before_raw, after_raw = objective_raw.split("->", 1)
            objective_before = _to_float(before_raw)
            objective_after = _to_float(after_raw)
            delta = objective_after - objective_before if (math.isfinite(objective_before) and math.isfinite(objective_after)) else math.nan
            action_rows.append(
                {
                    "model_id": str(current.get("model_id", "")),
                    "checkpoint": str(current.get("checkpoint", "")),
                    "test_instance_id": str(current.get("test_instance_id", "")),
                    "instance_index": current.get("instance_index"),
                    "eval_seed": current.get("eval_seed"),
                    "iteration": _to_int(kv.get("iter")),
                    "elapsed_seconds": _to_float(str(kv.get("time", "")).removesuffix("s")),
                    "destroy_action": str(kv.get("destroy", "")),
                    "repair_action": str(kv.get("repair", "")),
                    "objective_before": objective_before,
                    "objective_after": objective_after,
                    "delta_objective": delta,
                    "accepted": _to_bool(kv.get("accepted")),
                    "reward": _to_float(kv.get("reward")),
                    "invalid_action": not math.isfinite(objective_after),
                    "source_log": str(log_path),
                }
            )

        solved_match = SOLVED_PATTERN.match(line)
        if solved_match:
            solved_text, _stop_time, reason = solved_match.groups()
            current["solved"] = _to_bool(solved_text)
            current["termination_reason"] = reason
            continue

        completed_match = COMPLETED_PATTERN.match(line)
        if completed_match:
            runtime, best_obj, current_obj = completed_match.groups()
            current["runtime_seconds"] = _to_float(runtime)
            current["best_objective_over_episode"] = _to_float(best_obj)
            current["final_objective"] = _to_float(current_obj)
            current["status"] = "ok"
            continue

        error_match = ERROR_PATTERN.match(line)
        if error_match:
            current["status"] = "error"
            current["error"] = error_match.group(1)
            if current.get("termination_reason", "") == "":
                current["termination_reason"] = "error"
            continue

    finalize_current()
    return rows, header, action_rows


def _build_model_card_rows(
    summary_rows: Sequence[Dict[str, str]],
    variant: str,
    run_id: str,
    config: Dict[str, str],
    commit_hash: str,
    eval_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    state_dim_by_model: Dict[str, int] = {}
    for row in eval_rows:
        model_id = str(row.get("model_id", ""))
        state_dim = _to_int(row.get("state_dim"))
        if model_id != "" and state_dim is not None:
            state_dim_by_model[model_id] = state_dim

    out: List[Dict[str, Any]] = []
    for row in summary_rows:
        checkpoint = str(row.get("checkpoint", "")).strip()
        if checkpoint == "":
            continue
        model_id = Path(checkpoint).stem
        checkpoint_seed = _to_int(row.get("seed"))
        train_instances = _parse_list_csv(str(row.get("train_instances", "")))
        test_instances = _parse_list_csv(str(row.get("test_instances", "")))
        out.append(
            {
                "model_id": model_id,
                "variant": variant,
                "run_id": run_id,
                "checkpoint": checkpoint,
                "checkpoint_seed": checkpoint_seed if checkpoint_seed is not None else "",
                "train_instances": ",".join(train_instances),
                "test_instances": ",".join(test_instances),
                "n_train_instances": len(train_instances),
                "n_test_instances": len(test_instances),
                "training_budget_timesteps": _to_int(row.get("max_iterations")),
                "training_runtime_seconds": _to_float(row.get("runtime_seconds")),
                "training_status": str(row.get("status", "")),
                "executed_steps_training": _to_int(row.get("executed_steps")),
                "timeout_seconds_training": _to_float(row.get("timeout_seconds")),
                "reward_definition": "See DRL_PPO_with_late_phase.py reward logging (reward=...)",
                "observation_definition": f"state_dim={state_dim_by_model.get(model_id, '')}",
                "action_definition": "Policy selects destroy/repair actions",
                "domain_randomization": "",
                "environment_version_commit": commit_hash,
                "run_config_timeout_seconds": config.get("timeout_seconds", ""),
                "run_config_max_iterations": config.get("max_iterations", ""),
                "run_config_n_instances": config.get("n_instances", ""),
                "run_config_train_count": config.get("train_count", ""),
                "run_config_test_count": config.get("test_count", ""),
                "run_config_n_seeds": config.get("n_seeds", ""),
            }
        )

    out.sort(key=lambda r: (_to_int(r.get("checkpoint_seed")) or 10**9, str(r.get("model_id", ""))))
    return out


def _augment_eval_rows(
    raw_eval_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, str]],
    variant: str,
    run_id: str,
    benchmark_run_id: str | None,
    cp_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    summary_by_checkpoint = {str(r.get("checkpoint", "")).strip(): r for r in summary_rows}
    out: List[Dict[str, Any]] = []

    for row in raw_eval_rows:
        checkpoint = str(row.get("checkpoint", "")).strip()
        srow = summary_by_checkpoint.get(checkpoint, {})
        model_seed = _to_int(srow.get("seed"))
        train_instances = str(srow.get("train_instances", ""))
        test_instances = str(srow.get("test_instances", ""))
        instance_name = str(row.get("test_instance_id", ""))
        final_obj = _to_float(row.get("final_objective"))
        cp_obj = _to_float(cp_map.get(instance_name))
        out.append(
            {
                "model_id": str(row.get("model_id", "")),
                "variant": variant,
                "run_id": run_id,
                "checkpoint_benchmark_run_id": benchmark_run_id or "",
                "checkpoint": checkpoint,
                "checkpoint_seed": model_seed if model_seed is not None else "",
                "test_instance_id": instance_name,
                "instance_index": _to_int(row.get("instance_index")),
                "eval_seed": _to_int(row.get("eval_seed")),
                "final_objective": final_obj,
                "best_objective_over_episode": _to_float(row.get("best_objective_over_episode")),
                "episode_length_steps": _to_int(row.get("episode_length_steps")),
                "runtime_seconds": _to_float(row.get("runtime_seconds")),
                "termination_reason": str(row.get("termination_reason", "")),
                "solved": _to_bool(row.get("solved")),
                "constraint_violation_metric": _to_float(row.get("constraint_violation_metric")),
                "return_sum": _to_float(row.get("return_sum")),
                "discounted_return": _to_float(row.get("discounted_return")),
                "state_dim": _to_int(row.get("state_dim")),
                "status": str(row.get("status", "")),
                "error": str(row.get("error", "")),
                "runs_each": _to_int(row.get("runs_each")),
                "train_instances": train_instances,
                "summary_test_instances": test_instances,
                "source_log": str(row.get("source_log", "")),
                "cp_objective": cp_obj,
                "gap_to_cp": _gap_to_cp(final_obj, cp_obj),
            }
        )

    out.sort(
        key=lambda r: (
            _to_int(r.get("checkpoint_seed")) or 10**9,
            _instance_sort_key(str(r.get("test_instance_id", ""))),
            _to_int(r.get("eval_seed")) or 10**9,
        )
    )
    return out


def _per_model_instance_summary(eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok_rows = [
        row
        for row in eval_rows
        if str(row.get("status", "")) == "ok"
        and math.isfinite(_to_float(row.get("final_objective")))
    ]
    grouped = _group_rows(ok_rows, ["model_id", "checkpoint_seed", "test_instance_id", "instance_index"])
    out: List[Dict[str, Any]] = []

    for (model_id, checkpoint_seed, instance, instance_index), rows in grouped.items():
        values = [_to_float(r.get("final_objective")) for r in rows]
        gaps = [_to_float(r.get("gap_to_cp")) for r in rows]
        out.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "test_instance_id": instance,
                "instance_index": instance_index,
                "n_runs": len(rows),
                "mean_objective": _safe_mean(values),
                "median_objective": _safe_median(values),
                "best_objective": _safe_min(values),
                "worst_objective": _safe_max(values),
                "std_objective": _safe_std(values),
                "range_objective": _safe_max(values) - _safe_min(values) if values else math.nan,
                "q10_objective": _quantile(values, 0.10),
                "q25_objective": _quantile(values, 0.25),
                "q50_objective": _quantile(values, 0.50),
                "q75_objective": _quantile(values, 0.75),
                "q90_objective": _quantile(values, 0.90),
                "mean_gap_to_cp": _safe_mean(gaps),
                "median_gap_to_cp": _safe_median(gaps),
                "best_gap_to_cp": _safe_min(gaps),
                "std_gap_to_cp": _safe_std(gaps),
            }
        )

    out.sort(key=lambda r: (_to_int(r.get("checkpoint_seed")) or 10**9, _instance_sort_key(str(r.get("test_instance_id", "")))))
    return out


def _ppo_all_instance_summary(eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok_rows = [
        row
        for row in eval_rows
        if str(row.get("status", "")) == "ok"
        and math.isfinite(_to_float(row.get("final_objective")))
    ]
    grouped = _group_rows(ok_rows, ["test_instance_id", "instance_index"])
    out: List[Dict[str, Any]] = []
    for (instance, instance_index), rows in grouped.items():
        values = [_to_float(r.get("final_objective")) for r in rows]
        gaps = [_to_float(r.get("gap_to_cp")) for r in rows]
        out.append(
            {
                "model_id": "PPO_ALL",
                "checkpoint_seed": "",
                "test_instance_id": instance,
                "instance_index": instance_index,
                "n_runs": len(rows),
                "mean_objective": _safe_mean(values),
                "median_objective": _safe_median(values),
                "best_objective": _safe_min(values),
                "worst_objective": _safe_max(values),
                "std_objective": _safe_std(values),
                "range_objective": _safe_max(values) - _safe_min(values) if values else math.nan,
                "q10_objective": _quantile(values, 0.10),
                "q25_objective": _quantile(values, 0.25),
                "q50_objective": _quantile(values, 0.50),
                "q75_objective": _quantile(values, 0.75),
                "q90_objective": _quantile(values, 0.90),
                "mean_gap_to_cp": _safe_mean(gaps),
                "median_gap_to_cp": _safe_median(gaps),
                "best_gap_to_cp": _safe_min(gaps),
                "std_gap_to_cp": _safe_std(gaps),
            }
        )
    out.sort(key=lambda r: _instance_sort_key(str(r.get("test_instance_id", ""))))
    return out


def _load_alns_eval_rows(
    alns_root: Path,
    variant: str,
    run_id: str | None,
    cp_map: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], str, str] | None:
    variant_dir = alns_root / variant
    if not variant_dir.exists():
        return None
    run_dir, summary_path = _resolve_run_dir(variant_dir=variant_dir, run_id=run_id)
    summary_rows = _read_csv(summary_path)
    out: List[Dict[str, Any]] = []
    for row in summary_rows:
        instance = str(row.get("instance", "")).strip()
        objective = _to_float(row.get("objective"))
        cp_obj = _to_float(cp_map.get(instance))
        out.append(
            {
                "algorithm": variant,
                "variant": variant,
                "run_id": run_dir.name,
                "source_summary": str(summary_path),
                "test_instance_id": instance,
                "instance_index": _to_int(row.get("instance_index")),
                "seed": _to_int(row.get("seed")),
                "status": str(row.get("status", "")),
                "objective": objective,
                "runtime_seconds": _to_float(row.get("runtime_seconds")),
                "iterations": _to_int(row.get("iterations")),
                "solved": _to_bool(row.get("solved")),
                "cp_objective": cp_obj,
                "gap_to_cp": _gap_to_cp(objective, cp_obj),
            }
        )
    out.sort(key=lambda r: (_instance_sort_key(str(r.get("test_instance_id", ""))), _to_int(r.get("seed")) or 10**9))
    return out, run_dir.name, str(summary_path)


def _alns_per_instance_summary(alns_eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok_rows = [r for r in alns_eval_rows if math.isfinite(_to_float(r.get("objective")))]
    grouped = _group_rows(ok_rows, ["algorithm", "test_instance_id", "instance_index"])
    out: List[Dict[str, Any]] = []
    for (algorithm, instance, instance_index), rows in grouped.items():
        vals = [_to_float(r.get("objective")) for r in rows]
        gaps = [_to_float(r.get("gap_to_cp")) for r in rows]
        out.append(
            {
                "algorithm": algorithm,
                "test_instance_id": instance,
                "instance_index": instance_index,
                "n_runs": len(rows),
                "mean_objective": _safe_mean(vals),
                "median_objective": _safe_median(vals),
                "best_objective": _safe_min(vals),
                "worst_objective": _safe_max(vals),
                "std_objective": _safe_std(vals),
                "mean_gap_to_cp": _safe_mean(gaps),
                "median_gap_to_cp": _safe_median(gaps),
                "best_gap_to_cp": _safe_min(gaps),
                "std_gap_to_cp": _safe_std(gaps),
            }
        )
    out.sort(key=lambda r: (str(r.get("algorithm", "")), _instance_sort_key(str(r.get("test_instance_id", "")))))
    return out


def _ppo_vs_alns_comparison(
    ppo_instance_rows: Sequence[Dict[str, Any]],
    alns_instance_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    alns_map = {
        (str(r.get("algorithm", "")), str(r.get("test_instance_id", ""))): r
        for r in alns_instance_rows
    }
    alns_algorithms = sorted({str(r.get("algorithm", "")) for r in alns_instance_rows})
    cmp_rows: List[Dict[str, Any]] = []
    for prow in ppo_instance_rows:
        model_id = str(prow.get("model_id", ""))
        checkpoint_seed = prow.get("checkpoint_seed")
        instance = str(prow.get("test_instance_id", ""))
        ppo_mean = _to_float(prow.get("mean_objective"))
        ppo_best = _to_float(prow.get("best_objective"))
        ppo_gap = _to_float(prow.get("mean_gap_to_cp"))
        if not math.isfinite(ppo_mean):
            continue
        for alg in alns_algorithms:
            arow = alns_map.get((alg, instance))
            if arow is None:
                continue
            alns_mean = _to_float(arow.get("mean_objective"))
            alns_best = _to_float(arow.get("best_objective"))
            alns_gap = _to_float(arow.get("mean_gap_to_cp"))
            diff_mean = ppo_mean - alns_mean if math.isfinite(alns_mean) else math.nan
            diff_best = ppo_best - alns_best if (math.isfinite(ppo_best) and math.isfinite(alns_best)) else math.nan
            cmp_rows.append(
                {
                    "model_id": model_id,
                    "checkpoint_seed": checkpoint_seed,
                    "test_instance_id": instance,
                    "instance_index": prow.get("instance_index"),
                    "alns_algorithm": alg,
                    "ppo_mean_objective": ppo_mean,
                    "alns_mean_objective": alns_mean,
                    "diff_mean_ppo_minus_alns": diff_mean,
                    "rel_diff_mean_vs_alns": _relative_diff(ppo_mean, alns_mean),
                    "ppo_best_objective": ppo_best,
                    "alns_best_objective": alns_best,
                    "diff_best_ppo_minus_alns": diff_best,
                    "ppo_mean_gap_to_cp": ppo_gap,
                    "alns_mean_gap_to_cp": alns_gap,
                    "diff_gap_to_cp_ppo_minus_alns": ppo_gap - alns_gap if (math.isfinite(ppo_gap) and math.isfinite(alns_gap)) else math.nan,
                    "ppo_better_mean": bool(math.isfinite(diff_mean) and diff_mean < -EPS),
                }
            )

    cmp_rows.sort(
        key=lambda r: (
            str(r.get("model_id", "")) != "PPO_ALL",
            _to_int(str(r.get("model_id", "")).replace("checkpoint_seed", "")) or 10**9,
            str(r.get("model_id", "")),
            str(r.get("alns_algorithm", "")),
            _instance_sort_key(str(r.get("test_instance_id", ""))),
        )
    )

    summary_grouped = _group_rows(cmp_rows, ["model_id", "checkpoint_seed", "alns_algorithm"])
    summary_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, alg), rows in summary_grouped.items():
        diffs = [_to_float(r.get("diff_mean_ppo_minus_alns")) for r in rows]
        rel_diffs = [_to_float(r.get("rel_diff_mean_vs_alns")) for r in rows]
        gap_diffs = [_to_float(r.get("diff_gap_to_cp_ppo_minus_alns")) for r in rows]
        summary_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "alns_algorithm": alg,
                "n_instances_compared": len(rows),
                "mean_diff_mean_ppo_minus_alns": _safe_mean(diffs),
                "median_diff_mean_ppo_minus_alns": _safe_median(diffs),
                "mean_rel_diff_mean_vs_alns": _safe_mean(rel_diffs),
                "mean_diff_gap_to_cp_ppo_minus_alns": _safe_mean(gap_diffs),
                "ppo_better_count": sum(1 for v in diffs if math.isfinite(v) and v < -EPS),
                "alns_better_count": sum(1 for v in diffs if math.isfinite(v) and v > EPS),
                "tie_count": sum(1 for v in diffs if math.isfinite(v) and abs(v) <= EPS),
            }
        )
    summary_rows.sort(
        key=lambda r: (
            str(r.get("model_id", "")) != "PPO_ALL",
            _to_int(str(r.get("model_id", "")).replace("checkpoint_seed", "")) or 10**9,
            str(r.get("model_id", "")),
            str(r.get("alns_algorithm", "")),
        )
    )

    matrix_rows: List[Dict[str, Any]] = []
    for row in ppo_instance_rows:
        matrix_rows.append(
            {
                "algorithm_id": str(row.get("model_id", "")),
                "source_family": "PPO",
                "test_instance_id": str(row.get("test_instance_id", "")),
                "instance_index": row.get("instance_index"),
                "mean_objective": _to_float(row.get("mean_objective")),
                "mean_gap_to_cp": _to_float(row.get("mean_gap_to_cp")),
            }
        )
    for row in alns_instance_rows:
        matrix_rows.append(
            {
                "algorithm_id": str(row.get("algorithm", "")),
                "source_family": "ALNS",
                "test_instance_id": str(row.get("test_instance_id", "")),
                "instance_index": row.get("instance_index"),
                "mean_objective": _to_float(row.get("mean_objective")),
                "mean_gap_to_cp": _to_float(row.get("mean_gap_to_cp")),
            }
        )
    matrix_rows.sort(key=lambda r: (str(r.get("algorithm_id", "")), _instance_sort_key(str(r.get("test_instance_id", "")))))
    return cmp_rows, summary_rows, matrix_rows


def _heldout_scores_6a(per_instance_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_rows(per_instance_rows, ["model_id", "checkpoint_seed"])
    out: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed), rows in grouped.items():
        means = [_to_float(r.get("mean_objective")) for r in rows]
        finite_means = [x for x in means if math.isfinite(x)]
        out.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "n_test_instances": len(rows),
                "score_mean": _safe_mean(finite_means),
                "score_median": _safe_median(finite_means),
                "score_worstcase": _safe_max(finite_means),
                "score_std_across_test_instances": _safe_std(finite_means),
            }
        )
    out.sort(key=lambda r: (_to_int(r.get("checkpoint_seed")) or 10**9, str(r.get("model_id", ""))))
    return out


def _cross_model_matrix(per_instance_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    models = sorted(
        {str(r.get("model_id", "")) for r in per_instance_rows},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    instances = sorted({str(r.get("test_instance_id", "")) for r in per_instance_rows}, key=_instance_sort_key)
    lookup = {(str(r.get("model_id", "")), str(r.get("test_instance_id", ""))): _to_float(r.get("mean_objective")) for r in per_instance_rows}
    rows: List[Dict[str, Any]] = []
    for instance in instances:
        for model in models:
            value = lookup.get((model, instance), math.nan)
            rows.append(
                {
                    "test_instance_id": instance,
                    "instance_index": _instance_index(instance),
                    "model_id": model,
                    "mean_objective": value,
                    "evaluated": bool(math.isfinite(value)),
                }
            )
    return rows, models, instances


def _ranking_and_dominance(
    matrix_rows: Sequence[Dict[str, Any]],
    models: Sequence[str],
    instances: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = {(str(r["model_id"]), str(r["test_instance_id"])): _to_float(r["mean_objective"]) for r in matrix_rows}

    rank_rows: List[Dict[str, Any]] = []
    for instance in instances:
        present = [(model, lookup.get((model, instance), math.nan)) for model in models]
        present = [(m, v) for m, v in present if math.isfinite(v)]
        if not present:
            continue
        ranks = _average_ranks([v for _, v in present])
        for idx, (model, value) in enumerate(present):
            rank_rows.append(
                {
                    "test_instance_id": instance,
                    "instance_index": _instance_index(instance),
                    "model_id": model,
                    "mean_objective": value,
                    "rank": ranks[idx],
                }
            )

    by_model = _group_rows(rank_rows, ["model_id"])
    rank_summary_rows: List[Dict[str, Any]] = []
    for (model,), rows in by_model.items():
        ranks = [_to_float(r["rank"]) for r in rows]
        rank_summary_rows.append(
            {
                "model_id": model,
                "n_instances_ranked": len(rows),
                "wins_rank1": sum(1 for r in rows if abs(_to_float(r["rank"]) - 1.0) <= EPS),
                "top2_count": sum(1 for r in rows if _to_float(r["rank"]) <= 2.0 + EPS),
                "average_rank": _safe_mean(ranks),
            }
        )
    rank_summary_rows.sort(key=lambda r: (_to_float(r["average_rank"]), str(r["model_id"])))

    dominance_rows: List[Dict[str, Any]] = []
    for model_a, model_b in combinations(models, 2):
        better_a = 0
        better_b = 0
        ties = 0
        shared = 0
        for instance in instances:
            va = lookup.get((model_a, instance), math.nan)
            vb = lookup.get((model_b, instance), math.nan)
            if not (math.isfinite(va) and math.isfinite(vb)):
                continue
            shared += 1
            if va < vb - EPS:
                better_a += 1
            elif vb < va - EPS:
                better_b += 1
            else:
                ties += 1
        dominates = shared > 0 and better_a >= better_b and better_a > 0
        dominance_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "shared_instances": shared,
                "better_a_count": better_a,
                "better_b_count": better_b,
                "tie_count": ties,
                "a_dominates_b": dominates,
            }
        )
    return rank_rows, rank_summary_rows, dominance_rows


def _pairwise_model_tests(
    matrix_rows: Sequence[Dict[str, Any]],
    models: Sequence[str],
    instances: Sequence[str],
) -> List[Dict[str, Any]]:
    lookup = {(str(r["model_id"]), str(r["test_instance_id"])): _to_float(r["mean_objective"]) for r in matrix_rows}
    rows: List[Dict[str, Any]] = []
    for model_a, model_b in combinations(models, 2):
        diffs: List[float] = []
        better_a = 0
        better_b = 0
        ties = 0
        for instance in instances:
            va = lookup.get((model_a, instance), math.nan)
            vb = lookup.get((model_b, instance), math.nan)
            if not (math.isfinite(va) and math.isfinite(vb)):
                continue
            d = va - vb
            diffs.append(d)
            if d < -EPS:
                better_a += 1
            elif d > EPS:
                better_b += 1
            else:
                ties += 1
        w = _wilcoxon_signed_rank_exact(diffs)
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_common_instances": len(diffs),
                "mean_diff_A_minus_B": _safe_mean(diffs),
                "median_diff_A_minus_B": _safe_median(diffs),
                "better_a_count": better_a,
                "better_b_count": better_b,
                "tie_count": ties,
                "wilcoxon_p_value_two_sided": w["p_value_two_sided"],
                "wilcoxon_n_nonzero": w["n"],
            }
        )

    p_values = [_to_float(r["wilcoxon_p_value_two_sided"]) for r in rows]
    holm = _adjust_pvalues_holm(p_values)
    bh = _adjust_pvalues_bh(p_values)
    for i, row in enumerate(rows):
        row["p_holm"] = holm[i]
        row["p_bh_fdr"] = bh[i]
    return rows


def _risk_profile(eval_rows: Sequence[Dict[str, Any]], bad_run_k: float = 1.5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ok_rows = [
        row
        for row in eval_rows
        if str(row.get("status", "")) == "ok" and math.isfinite(_to_float(row.get("final_objective")))
    ]
    grouped = _group_rows(ok_rows, ["model_id", "checkpoint_seed", "test_instance_id", "instance_index"])
    per_instance_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, instance, instance_index), rows in grouped.items():
        values = [_to_float(r.get("final_objective")) for r in rows]
        values = [x for x in values if math.isfinite(x)]
        if not values:
            continue
        q25 = _quantile(values, 0.25)
        q50 = _quantile(values, 0.50)
        q75 = _quantile(values, 0.75)
        q90 = _quantile(values, 0.90)
        worst = _safe_max(values)
        iqr = q75 - q25 if (math.isfinite(q75) and math.isfinite(q25)) else math.nan
        bad_thr = q75 + bad_run_k * iqr if math.isfinite(iqr) else math.nan
        bad_frac = _safe_mean([1.0 if (math.isfinite(bad_thr) and v > bad_thr + EPS) else 0.0 for v in values])
        mean_obj = _safe_mean(values)
        std_obj = _safe_std(values)
        cv = std_obj / abs(mean_obj) if (math.isfinite(std_obj) and math.isfinite(mean_obj) and abs(mean_obj) > EPS) else math.nan
        per_instance_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "test_instance_id": instance,
                "instance_index": instance_index,
                "n_runs": len(values),
                "std_objective": std_obj,
                "cv_objective": cv,
                "tail_q90_minus_q50": q90 - q50 if math.isfinite(q90) and math.isfinite(q50) else math.nan,
                "tail_worst_minus_q50": worst - q50 if math.isfinite(worst) and math.isfinite(q50) else math.nan,
                "bad_run_threshold": bad_thr,
                "bad_run_fraction": bad_frac,
            }
        )

    grouped_model = _group_rows(per_instance_rows, ["model_id", "checkpoint_seed"])
    per_model_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed), rows in grouped_model.items():
        per_model_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "n_test_instances": len(rows),
                "mean_std_objective": _safe_mean([_to_float(r["std_objective"]) for r in rows]),
                "mean_cv_objective": _safe_mean([_to_float(r["cv_objective"]) for r in rows]),
                "mean_tail_q90_minus_q50": _safe_mean([_to_float(r["tail_q90_minus_q50"]) for r in rows]),
                "mean_tail_worst_minus_q50": _safe_mean([_to_float(r["tail_worst_minus_q50"]) for r in rows]),
                "mean_bad_run_fraction": _safe_mean([_to_float(r["bad_run_fraction"]) for r in rows]),
            }
        )
    return per_instance_rows, per_model_rows


def _within_episode_progress(action_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped = _group_rows(action_rows, ["model_id", "checkpoint_seed", "test_instance_id", "eval_seed"])
    run_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, instance, eval_seed), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: (_to_int(r.get("iteration")) or 10**9, _to_float(r.get("elapsed_seconds"))))
        best = math.inf
        for row in rows_sorted:
            v = _to_float(row.get("objective_after"))
            if math.isfinite(v):
                best = min(best, v)
                run_rows.append(
                    {
                        "model_id": model_id,
                        "checkpoint_seed": checkpoint_seed,
                        "test_instance_id": instance,
                        "instance_index": _instance_index(str(instance)),
                        "eval_seed": eval_seed,
                        "iteration": _to_int(row.get("iteration")),
                        "elapsed_seconds": _to_float(row.get("elapsed_seconds")),
                        "best_so_far_objective": best,
                    }
                )
    grouped_mean = _group_rows(run_rows, ["model_id", "checkpoint_seed", "test_instance_id", "instance_index", "iteration"])
    mean_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, instance, instance_index, iteration), rows in grouped_mean.items():
        mean_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "test_instance_id": instance,
                "instance_index": instance_index,
                "iteration": iteration,
                "mean_best_so_far_objective": _safe_mean([_to_float(r["best_so_far_objective"]) for r in rows]),
            }
        )
    return run_rows, mean_rows


def _convergence_tables(
    progress_run_rows: Sequence[Dict[str, Any]],
    cp_map: Dict[str, float],
    time_step: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_points: Dict[Tuple[str, Any, str, Any], List[Tuple[float, float]]] = {}
    max_time = 0.0
    for row in progress_run_rows:
        model_id = str(row.get("model_id", ""))
        checkpoint_seed = row.get("checkpoint_seed")
        instance = str(row.get("test_instance_id", ""))
        eval_seed = row.get("eval_seed")
        value = _to_float(row.get("best_so_far_objective"))
        elapsed = _to_float(row.get("elapsed_seconds"))
        if not math.isfinite(elapsed):
            elapsed = float(_to_int(row.get("iteration")) or math.nan)
        if not math.isfinite(value) or not math.isfinite(elapsed):
            continue
        key = (model_id, checkpoint_seed, instance, eval_seed)
        run_points.setdefault(key, []).append((elapsed, value))
        max_time = max(max_time, elapsed)

    if not run_points:
        return [], [], []

    step = max(float(time_step), 1e-6)
    max_time = max(max_time, step)
    grid = np.arange(0.0, max_time + step, step)

    run_curves: Dict[Tuple[str, Any, str, Any], np.ndarray] = {}
    for key, points in run_points.items():
        run_curves[key] = _step_curve_on_grid(points, grid)

    run_curve_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, instance, eval_seed), curve in run_curves.items():
        cp_obj = _to_float(cp_map.get(instance))
        for t, v in zip(grid, curve):
            run_curve_rows.append(
                {
                    "model_id": model_id,
                    "checkpoint_seed": checkpoint_seed,
                    "test_instance_id": instance,
                    "instance_index": _instance_index(instance),
                    "eval_seed": eval_seed,
                    "time_seconds": float(t),
                    "best_so_far_objective": float(v),
                    "best_so_far_gap_to_cp": _gap_to_cp(float(v), cp_obj),
                }
            )

    by_model_instance: Dict[Tuple[str, Any, str], List[np.ndarray]] = {}
    for (model_id, checkpoint_seed, instance, _eval_seed), curve in run_curves.items():
        by_model_instance.setdefault((model_id, checkpoint_seed, instance), []).append(curve)

    instance_curve_rows: List[Dict[str, Any]] = []
    instance_curves: Dict[Tuple[str, Any, str], np.ndarray] = {}
    for (model_id, checkpoint_seed, instance), curves in by_model_instance.items():
        mean_curve = np.nanmean(np.vstack(curves), axis=0)
        instance_curves[(model_id, checkpoint_seed, instance)] = mean_curve
        cp_obj = _to_float(cp_map.get(instance))
        for t, v in zip(grid, mean_curve):
            instance_curve_rows.append(
                {
                    "model_id": model_id,
                    "checkpoint_seed": checkpoint_seed,
                    "test_instance_id": instance,
                    "instance_index": _instance_index(instance),
                    "time_seconds": float(t),
                    "mean_best_so_far_objective": float(v),
                    "mean_best_so_far_gap_to_cp": _gap_to_cp(float(v), cp_obj),
                }
            )

    model_ids = sorted(
        {str(k[0]) for k in instance_curves.keys()},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    global_curve_rows: List[Dict[str, Any]] = []
    for model_id in model_ids:
        model_curves = [curve for (m, _seed, _inst), curve in instance_curves.items() if m == model_id]
        if not model_curves:
            continue
        mean_obj = np.nanmean(np.vstack(model_curves), axis=0)
        gap_curves: List[np.ndarray] = []
        for (m, _seed, instance), curve in instance_curves.items():
            if m != model_id:
                continue
            cp_obj = _to_float(cp_map.get(instance))
            if not math.isfinite(cp_obj):
                continue
            denom = _denom_for_gap(cp_obj)
            gap_curves.append((curve - cp_obj) / denom)
        mean_gap = np.nanmean(np.vstack(gap_curves), axis=0) if gap_curves else np.full_like(mean_obj, np.nan)
        for t, v_obj, v_gap in zip(grid, mean_obj, mean_gap):
            global_curve_rows.append(
                {
                    "model_id": model_id,
                    "time_seconds": float(t),
                    "global_mean_objective": float(v_obj),
                    "global_mean_gap_to_cp": float(v_gap),
                }
            )

    all_curves = list(instance_curves.values())
    if all_curves:
        all_obj = np.nanmean(np.vstack(all_curves), axis=0)
        all_gap_curves: List[np.ndarray] = []
        for (_m, _seed, instance), curve in instance_curves.items():
            cp_obj = _to_float(cp_map.get(instance))
            if not math.isfinite(cp_obj):
                continue
            all_gap_curves.append((curve - cp_obj) / _denom_for_gap(cp_obj))
        all_gap = np.nanmean(np.vstack(all_gap_curves), axis=0) if all_gap_curves else np.full_like(all_obj, np.nan)
        for t, v_obj, v_gap in zip(grid, all_obj, all_gap):
            global_curve_rows.append(
                {
                    "model_id": "ALL_MODELS",
                    "time_seconds": float(t),
                    "global_mean_objective": float(v_obj),
                    "global_mean_gap_to_cp": float(v_gap),
                }
            )

    return run_curve_rows, instance_curve_rows, global_curve_rows


def _time_to_target(
    progress_run_rows: Sequence[Dict[str, Any]],
    cp_map: Dict[str, float],
    target_gap: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_run: Dict[Tuple[str, Any, str, Any], List[Tuple[float, float]]] = {}
    for row in progress_run_rows:
        model_id = str(row.get("model_id", ""))
        checkpoint_seed = row.get("checkpoint_seed")
        instance = str(row.get("test_instance_id", ""))
        eval_seed = row.get("eval_seed")
        value = _to_float(row.get("best_so_far_objective"))
        elapsed = _to_float(row.get("elapsed_seconds"))
        if not math.isfinite(elapsed):
            elapsed = float(_to_int(row.get("iteration")) or math.nan)
        if not math.isfinite(value) or not math.isfinite(elapsed):
            continue
        by_run.setdefault((model_id, checkpoint_seed, instance, eval_seed), []).append((elapsed, value))

    run_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, instance, eval_seed), points in by_run.items():
        cp_obj = _to_float(cp_map.get(instance))
        if math.isfinite(cp_obj):
            target_value = cp_obj + _denom_for_gap(cp_obj) * float(target_gap)
            t_hit = math.nan
            for t, v in sorted(points, key=lambda p: p[0]):
                if math.isfinite(v) and v <= target_value + EPS:
                    t_hit = float(t)
                    break
        else:
            target_value = math.nan
            t_hit = math.nan
        run_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "test_instance_id": instance,
                "instance_index": _instance_index(instance),
                "eval_seed": eval_seed,
                "cp_objective": cp_obj,
                "target_gap": float(target_gap),
                "target_objective": target_value,
                "time_to_target_seconds": t_hit,
                "reached_target": bool(math.isfinite(t_hit)),
            }
        )

    by_instance_model = _group_rows(run_rows, ["model_id", "checkpoint_seed", "test_instance_id", "instance_index"])
    instance_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, instance, instance_index), rows in by_instance_model.items():
        times = [_to_float(r.get("time_to_target_seconds")) for r in rows]
        finite_times = [t for t in times if math.isfinite(t)]
        instance_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "test_instance_id": instance,
                "instance_index": instance_index,
                "n_runs": len(rows),
                "solved_runs": len(finite_times),
                "solved_fraction": len(finite_times) / len(rows) if rows else math.nan,
                "mean_time_to_target_seconds": _safe_mean(times),
                "median_time_to_target_seconds": _safe_median(times),
            }
        )

    by_model = _group_rows(instance_rows, ["model_id", "checkpoint_seed"])
    summary_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed), rows in by_model.items():
        summary_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "n_test_instances": len(rows),
                "mean_solved_fraction": _safe_mean([_to_float(r.get("solved_fraction")) for r in rows]),
                "mean_time_to_target_seconds": _safe_mean([_to_float(r.get("mean_time_to_target_seconds")) for r in rows]),
                "median_time_to_target_seconds": _safe_median([_to_float(r.get("median_time_to_target_seconds")) for r in rows]),
            }
        )

    summary_rows.sort(key=lambda r: (_to_int(r.get("checkpoint_seed")) or 10**9, str(r.get("model_id", ""))))
    return run_rows, instance_rows, summary_rows


def _action_usage_tables(action_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_run_type = _group_rows(action_rows, ["model_id", "checkpoint_seed", "test_instance_id", "eval_seed"])
    per_run_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, instance, eval_seed), rows in by_run_type.items():
        for action_type, key in (("destroy", "destroy_action"), ("repair", "repair_action")):
            counts: Dict[str, int] = {}
            invalid = 0
            for row in rows:
                action = str(row.get(key, ""))
                if action == "":
                    continue
                counts[action] = counts.get(action, 0) + 1
                if _to_bool(row.get("invalid_action")):
                    invalid += 1
            total = sum(counts.values())
            entropy = 0.0
            if total > 0:
                for c in counts.values():
                    p = c / total
                    if p > 0:
                        entropy -= p * math.log(p)
            for action, count in counts.items():
                per_run_rows.append(
                    {
                        "model_id": model_id,
                        "checkpoint_seed": checkpoint_seed,
                        "test_instance_id": instance,
                        "instance_index": _instance_index(str(instance)),
                        "eval_seed": eval_seed,
                        "action_type": action_type,
                        "action_name": action,
                        "selection_count": count,
                        "selection_share": count / total if total > 0 else math.nan,
                        "action_entropy": entropy,
                        "invalid_action_fraction": invalid / total if total > 0 else math.nan,
                    }
                )

    per_instance_grouped = _group_rows(per_run_rows, ["model_id", "checkpoint_seed", "test_instance_id", "instance_index", "action_type", "action_name"])
    per_instance_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, instance, instance_index, action_type, action_name), rows in per_instance_grouped.items():
        per_instance_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "test_instance_id": instance,
                "instance_index": instance_index,
                "action_type": action_type,
                "action_name": action_name,
                "mean_selection_share": _safe_mean([_to_float(r["selection_share"]) for r in rows]),
                "mean_action_entropy": _safe_mean([_to_float(r["action_entropy"]) for r in rows]),
                "mean_invalid_action_fraction": _safe_mean([_to_float(r["invalid_action_fraction"]) for r in rows]),
                "n_runs": len(rows),
            }
        )

    global_grouped = _group_rows(per_instance_rows, ["model_id", "checkpoint_seed", "action_type", "action_name"])
    global_rows: List[Dict[str, Any]] = []
    for (model_id, checkpoint_seed, action_type, action_name), rows in global_grouped.items():
        global_rows.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "action_type": action_type,
                "action_name": action_name,
                "mean_selection_share": _safe_mean([_to_float(r["mean_selection_share"]) for r in rows]),
                "std_selection_share_across_instances": _safe_std([_to_float(r["mean_selection_share"]) for r in rows]),
                "mean_invalid_action_fraction": _safe_mean([_to_float(r["mean_invalid_action_fraction"]) for r in rows]),
                "n_instances": len(rows),
            }
        )
    return per_run_rows, per_instance_rows, global_rows


def _action_effectiveness(action_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []
    for row in action_rows:
        for action_type, key in (("destroy", "destroy_action"), ("repair", "repair_action")):
            action_name = str(row.get(key, ""))
            if action_name == "":
                continue
            enriched.append(
                {
                    "model_id": row.get("model_id"),
                    "checkpoint_seed": row.get("checkpoint_seed"),
                    "action_type": action_type,
                    "action_name": action_name,
                    "delta_objective": _to_float(row.get("delta_objective")),
                    "invalid_action": _to_bool(row.get("invalid_action")),
                }
            )
    grouped = _group_rows(enriched, ["model_id", "checkpoint_seed", "action_type", "action_name"])
    for (model_id, checkpoint_seed, action_type, action_name), rows in grouped.items():
        deltas = [_to_float(r["delta_objective"]) for r in rows]
        improving = [d for d in deltas if math.isfinite(d) and d < -EPS]
        worsening = [d for d in deltas if math.isfinite(d) and d > EPS]
        finite = [d for d in deltas if math.isfinite(d)]
        invalid_frac = _safe_mean([1.0 if _to_bool(r["invalid_action"]) else 0.0 for r in rows])
        out.append(
            {
                "model_id": model_id,
                "checkpoint_seed": checkpoint_seed,
                "action_type": action_type,
                "action_name": action_name,
                "selection_count": len(rows),
                "improve_probability": len(improving) / len(finite) if finite else math.nan,
                "mean_delta": _safe_mean(finite),
                "mean_improving_delta": _safe_mean(improving),
                "mean_worsening_delta": _safe_mean(worsening),
                "invalid_action_fraction": invalid_frac,
            }
        )
    return out


def _training_diagnostics(run_dir: Path, summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    logs_dir = run_dir / "logs"
    out: List[Dict[str, Any]] = []
    if not logs_dir.exists():
        return out
    by_seed_checkpoint = {str(r.get("seed", "")).strip(): str(r.get("checkpoint", "")).strip() for r in summary_rows}
    for log_path in sorted(logs_dir.glob("seed*_training.json")):
        seed_match = re.search(r"seed(\d+)_training\.json$", log_path.name)
        seed = seed_match.group(1) if seed_match else ""
        checkpoint = by_seed_checkpoint.get(seed, "")
        model_id = Path(checkpoint).stem if checkpoint else f"seed{seed}"
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ret = [_to_float(x) for x in payload.get("episode_return", [])]
        pol = [_to_float(x) for x in payload.get("policy_loss", [])]
        val = [_to_float(x) for x in payload.get("value_loss", [])]
        ent = [_to_float(x) for x in payload.get("entropy", [])]
        reward = [_to_float(x) for x in payload.get("reward", [])]
        trend = payload.get("reward_trend", {}) if isinstance(payload.get("reward_trend", {}), dict) else {}
        out.append(
            {
                "model_id": model_id,
                "checkpoint_seed": _to_int(seed),
                "log_path": str(log_path),
                "n_episode_returns": len(ret),
                "mean_episode_return": _safe_mean(ret),
                "final_episode_return": ret[-1] if ret else math.nan,
                "mean_policy_loss": _safe_mean(pol),
                "mean_value_loss": _safe_mean(val),
                "mean_entropy": _safe_mean(ent),
                "mean_reward_step": _safe_mean(reward),
                "reward_trend_label": str(trend.get("label", "")),
                "reward_trend_early_slope": _to_float(trend.get("early_slope")),
                "reward_trend_late_slope": _to_float(trend.get("late_slope")),
                "reward_trend_slope_ratio": _to_float(trend.get("slope_ratio")),
            }
        )
    out.sort(key=lambda r: (_to_int(r.get("checkpoint_seed")) or 10**9, str(r.get("model_id", ""))))
    return out

def _build_data_quality_checks(
    summary_rows: Sequence[Dict[str, str]],
    eval_rows: Sequence[Dict[str, Any]],
    expected_eval_seeds: int,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add_check(check_id: str, passed: bool, scope: str, details: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else "FAIL",
                "scope": scope,
                "details": details,
            }
        )

    add_check(
        check_id="task1_model_count",
        passed=len(summary_rows) > 0,
        scope="summary.csv",
        details=f"models_detected={len(summary_rows)}",
    )

    all_have_splits = all(
        str(row.get("checkpoint", "")).strip() != ""
        and len(_parse_list_csv(str(row.get("train_instances", "")))) > 0
        and len(_parse_list_csv(str(row.get("test_instances", "")))) > 0
        for row in summary_rows
    )
    add_check(
        check_id="task1_train_test_splits_present",
        passed=all_have_splits,
        scope="summary.csv",
        details="checkpoint/train_instances/test_instances present for all models",
    )

    add_check(
        check_id="task2_eval_rows_present",
        passed=len(eval_rows) > 0,
        scope="checkpoint_benchmark logs",
        details=f"eval_rows={len(eval_rows)}",
    )

    ok_rows = [r for r in eval_rows if str(r.get("status", "")) == "ok"]
    add_check(
        check_id="task2_successful_eval_rows_present",
        passed=len(ok_rows) > 0,
        scope="checkpoint_benchmark logs",
        details=f"successful_eval_rows={len(ok_rows)}",
    )

    eval_group = _group_rows(ok_rows, ["checkpoint", "test_instance_id"])
    expected_pairs = 0
    missing_pairs = 0
    for row in summary_rows:
        checkpoint = str(row.get("checkpoint", "")).strip()
        for test_instance in _parse_list_csv(str(row.get("test_instances", ""))):
            expected_pairs += 1
            actual = len(eval_group.get((checkpoint, test_instance), []))
            if actual != expected_eval_seeds:
                missing_pairs += 1
    add_check(
        check_id="task3_expected_seed_count_per_model_instance",
        passed=missing_pairs == 0 and expected_pairs > 0,
        scope="(checkpoint, test_instance)",
        details=f"expected_per_pair={expected_eval_seeds} pairs={expected_pairs} mismatched_pairs={missing_pairs}",
    )

    timeout_values = {_to_float(r.get("timeout_seconds")) for r in summary_rows if math.isfinite(_to_float(r.get("timeout_seconds")))}
    max_iter_values = {_to_int(r.get("max_iterations")) for r in summary_rows if _to_int(r.get("max_iterations")) is not None}
    add_check(
        check_id="task3_identical_env_settings_across_models",
        passed=len(timeout_values) <= 1 and len(max_iter_values) <= 1,
        scope="summary.csv",
        details=f"timeout_values={sorted(timeout_values)} max_iterations_values={sorted(max_iter_values)}",
    )

    missing_key_count = 0
    for row in ok_rows:
        if not math.isfinite(_to_float(row.get("final_objective"))):
            missing_key_count += 1
        if not math.isfinite(_to_float(row.get("runtime_seconds"))):
            missing_key_count += 1
        if not math.isfinite(_to_float(row.get("best_objective_over_episode"))):
            missing_key_count += 1
        if str(row.get("termination_reason", "")).strip() == "":
            missing_key_count += 1
    add_check(
        check_id="task3_no_missing_key_values",
        passed=missing_key_count == 0,
        scope="successful eval rows",
        details=f"missing_key_value_count={missing_key_count}",
    )

    valid_reasons = {"timeout", "solved", "max_steps", "max_iterations", "invalid_action", "error", "stopped", "unknown"}
    invalid_reasons = sorted(
        {
            str(r.get("termination_reason", "")).strip()
            for r in eval_rows
            if str(r.get("termination_reason", "")).strip() != ""
            and str(r.get("termination_reason", "")).strip() not in valid_reasons
        }
    )
    add_check(
        check_id="task3_termination_reason_consistency",
        passed=len(invalid_reasons) == 0,
        scope="evaluation runs",
        details=f"invalid_reasons={invalid_reasons}",
    )

    direction_violations = 0
    for row in ok_rows:
        best_obj = _to_float(row.get("best_objective_over_episode"))
        final_obj = _to_float(row.get("final_objective"))
        if math.isfinite(best_obj) and math.isfinite(final_obj) and best_obj > final_obj + EPS:
            direction_violations += 1
    add_check(
        check_id="task3_minimization_consistency",
        passed=direction_violations == 0,
        scope="successful eval rows",
        details=f"best_objective>final_objective violations={direction_violations}",
    )

    error_rows = [r for r in eval_rows if str(r.get("status", "")) == "error"]
    add_check(
        check_id="task3_run_errors",
        passed=len(error_rows) == 0,
        scope="evaluation runs",
        details=f"error_rows={len(error_rows)}",
    )

    return checks


def _write_data_quality_md(path: Path, checks: Sequence[Dict[str, Any]]) -> None:
    lines = ["# Data Quality", ""]
    for check in checks:
        status = str(check.get("status", ""))
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        lines.append(f"- {icon} {check.get('check_id')}: {check.get('details')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_boxplot_objective_by_model(eval_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    ok_rows = [r for r in eval_rows if str(r.get("status", "")) == "ok" and math.isfinite(_to_float(r.get("final_objective")))]
    if not ok_rows:
        return
    grouped = _group_rows(ok_rows, ["model_id"])
    model_ids = sorted(grouped.keys(), key=lambda k: (_to_int(str(k[0]).replace("checkpoint_seed", "")) or 10**9, str(k[0])))
    labels: List[str] = []
    series: List[List[float]] = []
    for (model_id,) in model_ids:
        values = [_to_float(r.get("final_objective")) for r in grouped[(model_id,)]]
        finite = [x for x in values if math.isfinite(x)]
        if not finite:
            continue
        labels.append(str(model_id))
        series.append(finite)
    if not series:
        return
    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.boxplot(series, tick_labels=labels, showmeans=True)
    plt.ylabel("Final Objective")
    plt.title("PPO Test Objectives by Model")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_boxplot_final_relative_gaps_by_model(eval_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    ok_rows = [
        r
        for r in eval_rows
        if str(r.get("status", "")) == "ok" and math.isfinite(_to_float(r.get("gap_to_cp")))
    ]
    if not ok_rows:
        return
    grouped = _group_rows(ok_rows, ["model_id"])
    model_ids = sorted(grouped.keys(), key=lambda k: (_to_int(str(k[0]).replace("checkpoint_seed", "")) or 10**9, str(k[0])))
    labels: List[str] = []
    series: List[List[float]] = []
    for (model_id,) in model_ids:
        values = [_to_float(r.get("gap_to_cp")) for r in grouped[(model_id,)]]
        finite = [x for x in values if math.isfinite(x)]
        if not finite:
            continue
        labels.append(str(model_id))
        series.append(finite)
    if not series:
        return
    all_vals = np.array([v for arr in series for v in arr], dtype=float)
    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    if all_vals.size >= 10:
        y_lo, y_hi = [float(v) for v in np.nanpercentile(all_vals, [2.0, 98.0])]
    else:
        y_lo, y_hi = y_min, y_max
    if not (math.isfinite(y_lo) and math.isfinite(y_hi)) or y_hi <= y_lo:
        y_lo, y_hi = y_min, y_max
    if y_hi <= y_lo:
        pad = max(abs(y_lo) * 0.05, 1e-3)
        y_lo -= pad
        y_hi += pad
    else:
        pad = max((y_hi - y_lo) * 0.1, 1e-4)
        y_lo -= pad
        y_hi += pad
    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.boxplot(series, tick_labels=labels, showmeans=True)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.ylim(y_lo, y_hi)
    plt.ylabel("Final Relative Gap to CP")
    plt.title("Final Relative Gap Distribution by Model")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_boxplots_objective_by_instance_per_model(eval_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    ok_rows = [r for r in eval_rows if str(r.get("status", "")) == "ok" and math.isfinite(_to_float(r.get("final_objective")))]
    if not ok_rows:
        return
    model_ids = sorted(
        {str(r.get("model_id", "")) for r in ok_rows},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        model_rows = [r for r in ok_rows if str(r.get("model_id", "")) == model_id]
        grouped = _group_rows(model_rows, ["test_instance_id"])
        instances = sorted((str(k[0]) for k in grouped.keys()), key=_instance_sort_key)
        labels: List[str] = []
        series: List[List[float]] = []
        for instance in instances:
            values = [_to_float(r.get("final_objective")) for r in grouped[(instance,)]]
            finite = [x for x in values if math.isfinite(x)]
            if not finite:
                continue
            labels.append(instance)
            series.append(finite)
        if not series:
            continue
        plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
        plt.boxplot(series, tick_labels=labels, showmeans=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Final Objective")
        plt.title(f"PPO Test Objectives by Instance - {model_id}")
        plt.tight_layout()
        plt.savefig(out_dir / f"boxplot_objective_by_instance_{_file_token(model_id)}.png", dpi=140)
        plt.close()


def _plot_heatmap_model_instance_means(per_instance_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not per_instance_rows:
        return
    models = sorted(
        {str(r.get("model_id", "")) for r in per_instance_rows},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    instances = sorted({str(r.get("test_instance_id", "")) for r in per_instance_rows}, key=_instance_sort_key)
    if not models or not instances:
        return

    lookup = {
        (str(r.get("model_id", "")), str(r.get("test_instance_id", ""))): _to_float(r.get("mean_objective"))
        for r in per_instance_rows
    }
    mat = np.full((len(models), len(instances)), np.nan, dtype=float)
    for i, model in enumerate(models):
        for j, instance in enumerate(instances):
            value = lookup.get((model, instance), math.nan)
            mat[i, j] = value

    masked = np.ma.masked_invalid(mat)
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="lightgray")

    plt.figure(figsize=(max(8, len(instances) * 0.6), max(4.5, len(models) * 0.6)))
    im = plt.imshow(masked, aspect="auto", cmap=cmap)
    plt.xticks(np.arange(len(instances)), instances, rotation=60, ha="right")
    plt.yticks(np.arange(len(models)), models)
    plt.xlabel("Test Instance")
    plt.ylabel("Model")
    plt.title("Mean Objective Heatmap (Model x Test Instance)")
    plt.colorbar(im, fraction=0.03, pad=0.03, label="Mean Objective")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_heatmap_model_instance_gap_to_cp(per_instance_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not per_instance_rows:
        return
    models = sorted(
        {str(r.get("model_id", "")) for r in per_instance_rows},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    instances = sorted({str(r.get("test_instance_id", "")) for r in per_instance_rows}, key=_instance_sort_key)
    if not models or not instances:
        return
    lookup = {
        (str(r.get("model_id", "")), str(r.get("test_instance_id", ""))): _to_float(r.get("mean_gap_to_cp"))
        for r in per_instance_rows
    }
    mat = np.full((len(models), len(instances)), np.nan, dtype=float)
    for i, model in enumerate(models):
        for j, instance in enumerate(instances):
            mat[i, j] = lookup.get((model, instance), math.nan)
    if not np.isfinite(mat).any():
        return
    masked = np.ma.masked_invalid(mat)
    cmap = matplotlib.colormaps["RdYlGn_r"].copy()
    cmap.set_bad(color="lightgray")
    plt.figure(figsize=(max(8, len(instances) * 0.6), max(4.5, len(models) * 0.6)))
    im = plt.imshow(masked, aspect="auto", cmap=cmap)
    plt.xticks(np.arange(len(instances)), instances, rotation=60, ha="right")
    plt.yticks(np.arange(len(models)), models)
    plt.xlabel("Test Instance")
    plt.ylabel("Model")
    plt.title("Mean Gap to CP Heatmap (Model x Test Instance)")
    plt.colorbar(im, fraction=0.03, pad=0.03, label="Mean Gap to CP")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_best_ppo_vs_benchmark_logy(eval_rows: Sequence[Dict[str, Any]], cp_map: Dict[str, float], out_path: Path) -> None:
    ok_rows = [
        r
        for r in eval_rows
        if str(r.get("status", "")) == "ok"
        and math.isfinite(_to_float(r.get("final_objective")))
    ]
    if not ok_rows:
        return
    by_instance = _group_rows(ok_rows, ["test_instance_id"])
    labels: List[str] = []
    vals_ppo: List[float] = []
    vals_cp: List[float] = []
    for (instance,), rows in sorted(by_instance.items(), key=lambda kv: _instance_sort_key(str(kv[0][0]))):
        cp_obj = _to_float(cp_map.get(str(instance)))
        best_ppo = _safe_min([_to_float(r.get("final_objective")) for r in rows])
        if not (math.isfinite(cp_obj) and math.isfinite(best_ppo)):
            continue
        labels.append(str(instance))
        vals_ppo.append(float(best_ppo))
        vals_cp.append(float(cp_obj))
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.38
    all_vals = vals_ppo + vals_cp
    positive_vals = [v for v in all_vals if v > 0]
    floor = (min(positive_vals) * 0.1) if positive_vals else 1e-6
    vals_ppo_plot = [v if v > 0 else floor for v in vals_ppo]
    vals_cp_plot = [v if v > 0 else floor for v in vals_cp]
    plt.figure(figsize=(max(9, len(labels) * 0.6), 5))
    plt.bar(x - width / 2, vals_ppo_plot, width=width, label="Best PPO (all seeds)")
    plt.bar(x + width / 2, vals_cp_plot, width=width, label="CP benchmark")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Objective (log scale)")
    plt.title("Per-instance Best PPO Objective vs CP Benchmark")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_risk_profile(per_model_risk_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not per_model_risk_rows:
        return
    rows = sorted(
        per_model_risk_rows,
        key=lambda r: (_to_int(str(r.get("model_id", "")).replace("checkpoint_seed", "")) or 10**9, str(r.get("model_id", ""))),
    )
    labels = [str(r.get("model_id", "")) for r in rows]
    cv_vals = [_to_float(r.get("mean_cv_objective")) for r in rows]
    bad_vals = [_to_float(r.get("mean_bad_run_fraction")) for r in rows]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.bar(x - width / 2, cv_vals, width=width, label="Mean CV")
    plt.bar(x + width / 2, bad_vals, width=width, label="Mean Bad-run Fraction")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Risk Metric Value")
    plt.title("Seed Risk Profile by Model")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_cv_distribution_by_model(risk_instance_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not risk_instance_rows:
        return
    grouped = _group_rows(risk_instance_rows, ["model_id"])
    model_ids = sorted(grouped.keys(), key=lambda k: (_to_int(str(k[0]).replace("checkpoint_seed", "")) or 10**9, str(k[0])))
    labels: List[str] = []
    series: List[List[float]] = []
    for (model_id,) in model_ids:
        vals = [_to_float(r.get("cv_objective")) for r in grouped[(model_id,)]]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        labels.append(str(model_id))
        series.append(vals)
    if not series:
        return
    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.boxplot(series, tick_labels=labels, showmeans=True)
    plt.ylabel("CV Across Seeds (Std / |Mean|)")
    plt.title("CV Distribution by Model")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_risk_profile_per_instance(risk_instance_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not risk_instance_rows:
        return
    models = sorted(
        {str(r.get("model_id", "")) for r in risk_instance_rows},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    instances = sorted({str(r.get("test_instance_id", "")) for r in risk_instance_rows}, key=_instance_sort_key)
    if not models or not instances:
        return

    metrics = [
        ("std_objective", "Std Objective"),
        ("tail_q90_minus_q50", "Tail Spread q90-q50"),
        ("tail_worst_minus_q50", "Tail Spread worst-q50"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(12, len(instances) * 1.3), max(5, len(models) * 0.55)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        lookup = {
            (str(r.get("model_id", "")), str(r.get("test_instance_id", ""))): _to_float(r.get(metric_key))
            for r in risk_instance_rows
        }
        mat = np.full((len(models), len(instances)), np.nan, dtype=float)
        for i, model in enumerate(models):
            for j, instance in enumerate(instances):
                mat[i, j] = lookup.get((model, instance), math.nan)
        masked = np.ma.masked_invalid(mat)
        cmap = matplotlib.colormaps["YlOrRd"].copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(masked, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(instances)))
        ax.set_xticklabels(instances, rotation=60, ha="right")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel("Test Instance")
        ax.set_ylabel("Model")
        ax.set_title(metric_label)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.suptitle("Risk Metrics per (Model, Instance)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_global_convergence(global_curve_rows: Sequence[Dict[str, Any]], out_obj: Path, out_gap: Path) -> None:
    if not global_curve_rows:
        return
    grouped = _group_rows(global_curve_rows, ["model_id"])
    ordered_models = sorted(
        [str(k[0]) for k in grouped.keys()],
        key=lambda m: (m != "ALL_MODELS", _to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    plt.figure(figsize=(8, 5))
    plotted = 0
    for model_id in ordered_models:
        rows = grouped.get((model_id,), [])
        rows_sorted = sorted(rows, key=lambda r: _to_float(r.get("time_seconds")))
        x = [_to_float(r.get("time_seconds")) for r in rows_sorted]
        y = [_to_float(r.get("global_mean_objective")) for r in rows_sorted]
        if not x or not any(math.isfinite(v) for v in y):
            continue
        plt.plot(x, y, label=model_id)
        plotted += 1
    if plotted > 0:
        plt.xlabel("Time (s)")
        plt.ylabel("Global Mean Objective")
        plt.title("Global Convergence Curves (All Seeds)")
        plt.legend()
        plt.tight_layout()
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_obj, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plotted = 0
    for model_id in ordered_models:
        rows = grouped.get((model_id,), [])
        rows_sorted = sorted(rows, key=lambda r: _to_float(r.get("time_seconds")))
        x = [_to_float(r.get("time_seconds")) for r in rows_sorted]
        y = [_to_float(r.get("global_mean_gap_to_cp")) for r in rows_sorted]
        if not x or not any(math.isfinite(v) for v in y):
            continue
        plt.plot(x, y, label=model_id)
        plotted += 1
    if plotted > 0:
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Global Mean Gap to CP")
        plt.title("Global Relative-Gap Convergence (All Seeds)")
        plt.legend()
        plt.tight_layout()
        out_gap.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_gap, dpi=140)
    plt.close()


def _plot_within_episode_progress(progress_mean_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not progress_mean_rows:
        return
    grouped = _group_rows(progress_mean_rows, ["model_id", "iteration"])
    by_model_iter: Dict[Tuple[str, int], float] = {}
    for (model_id, iteration), rows in grouped.items():
        by_model_iter[(str(model_id), int(iteration))] = _safe_mean([_to_float(r.get("mean_best_so_far_objective")) for r in rows])
    models = sorted({m for (m, _i) in by_model_iter.keys()}, key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m))
    plt.figure(figsize=(9, 5))
    for model in models:
        points = sorted([(i, v) for (m, i), v in by_model_iter.items() if m == model and math.isfinite(v)], key=lambda t: t[0])
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y, label=model)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Best-so-far Objective")
    plt.title("Within-episode Progress by Model")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_cactus(time_to_target_run_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not time_to_target_run_rows:
        return
    grouped = _group_rows(time_to_target_run_rows, ["model_id"])
    ordered_models = sorted(
        [str(k[0]) for k in grouped.keys()],
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    plt.figure(figsize=(8, 5))
    plotted = 0
    for model_id in ordered_models:
        rows = grouped.get((model_id,), [])
        times = sorted([_to_float(r.get("time_to_target_seconds")) for r in rows if math.isfinite(_to_float(r.get("time_to_target_seconds")))])
        if not times:
            continue
        x = np.arange(1, len(times) + 1)
        plt.step(x, times, where="post", label=model_id)
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Solved Runs (target reached count)")
    plt.ylabel("Time to Target (s)")
    plt.title("Cactus Plot: Time to Target by Model")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_pairwise_scatter_model_means(
    matrix_rows: Sequence[Dict[str, Any]],
    models: Sequence[str],
    instances: Sequence[str],
    out_dir: Path,
) -> None:
    if not matrix_rows:
        return
    lookup = {(str(r.get("model_id", "")), str(r.get("test_instance_id", ""))): _to_float(r.get("mean_objective")) for r in matrix_rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_a, model_b in combinations(models, 2):
        points: List[Tuple[str, float, float]] = []
        for instance in instances:
            va = lookup.get((model_a, instance), math.nan)
            vb = lookup.get((model_b, instance), math.nan)
            if math.isfinite(va) and math.isfinite(vb):
                points.append((instance, va, vb))
        if not points:
            continue
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        min_xy = min(xs + ys)
        max_xy = max(xs + ys)
        pad = max((max_xy - min_xy) * 0.05, 1e-9)
        lo = min_xy - pad
        hi = max_xy + pad

        plt.figure(figsize=(6.5, 6))
        plt.scatter(xs, ys, s=34, alpha=0.85)
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="gray")
        if len(points) <= 25:
            for instance, x, y in points:
                plt.annotate(instance, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel(f"{model_a} mean objective")
        plt.ylabel(f"{model_b} mean objective")
        plt.title(f"Pairwise Mean Objective Scatter: {model_a} vs {model_b} (n={len(points)})")
        plt.tight_layout()
        file_name = f"pairwise_scatter_{_file_token(model_a)}_vs_{_file_token(model_b)}.png"
        plt.savefig(out_dir / file_name, dpi=140)
        plt.close()


def _plot_action_usage_heatmap(action_global_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not action_global_rows:
        return
    models = sorted(
        {str(r.get("model_id", "")) for r in action_global_rows},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    op_scores: Dict[str, List[float]] = {}
    for row in action_global_rows:
        action_type = str(row.get("action_type", ""))
        action_name = str(row.get("action_name", ""))
        key = _action_plot_id(action_type, action_name)
        op_scores.setdefault(key, []).append(_to_float(row.get("mean_selection_share")))
    top_actions = sorted(op_scores.keys(), key=lambda k: _safe_mean(op_scores[k]), reverse=True)[:20]
    if not models or not top_actions:
        return
    lookup = {
        (
            str(r.get("model_id", "")),
            _action_plot_id(r.get("action_type"), r.get("action_name")),
        ): _to_float(r.get("mean_selection_share"))
        for r in action_global_rows
    }
    mat = np.zeros((len(models), len(top_actions)), dtype=float)
    for i, model in enumerate(models):
        for j, action in enumerate(top_actions):
            mat[i, j] = lookup.get((model, action), 0.0)
    plt.figure(figsize=(max(10, len(top_actions) * 0.55), max(4.5, len(models) * 0.6)))
    im = plt.imshow(mat, aspect="auto", cmap="YlGnBu")
    plt.xticks(np.arange(len(top_actions)), top_actions, rotation=70, ha="right", fontsize=8)
    plt.yticks(np.arange(len(models)), models)
    plt.xlabel("Action")
    plt.ylabel("Model")
    plt.title("Action Usage Heatmap (Mean Selection Share)")
    plt.colorbar(im, fraction=0.03, pad=0.03, label="Mean Selection Share")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_action_usage_heatmap_per_model(action_instance_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not action_instance_rows:
        return
    model_ids = sorted(
        {str(r.get("model_id", "")) for r in action_instance_rows},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        rows = [r for r in action_instance_rows if str(r.get("model_id", "")) == model_id]
        if not rows:
            continue
        instances = sorted({str(r.get("test_instance_id", "")) for r in rows}, key=_instance_sort_key)
        action_scores: Dict[str, List[float]] = {}
        for row in rows:
            action_id = _action_plot_id(row.get("action_type"), row.get("action_name"))
            action_scores.setdefault(action_id, []).append(_to_float(row.get("mean_selection_share")))
        top_actions = sorted(action_scores.keys(), key=lambda a: _safe_mean(action_scores[a]), reverse=True)[:20]
        if not instances or not top_actions:
            continue
        lookup = {
            (str(r.get("test_instance_id", "")), _action_plot_id(r.get("action_type"), r.get("action_name"))): _to_float(r.get("mean_selection_share"))
            for r in rows
        }
        mat = np.zeros((len(instances), len(top_actions)), dtype=float)
        for i, instance in enumerate(instances):
            for j, action_id in enumerate(top_actions):
                mat[i, j] = lookup.get((instance, action_id), 0.0)
        plt.figure(figsize=(max(10, len(top_actions) * 0.55), max(4.5, len(instances) * 0.6)))
        im = plt.imshow(mat, aspect="auto", cmap="YlGnBu")
        plt.xticks(np.arange(len(top_actions)), top_actions, rotation=70, ha="right", fontsize=8)
        plt.yticks(np.arange(len(instances)), instances)
        plt.xlabel("Action")
        plt.ylabel("Test Instance")
        plt.title(f"Action Usage Heatmap by Instance - {model_id}")
        plt.colorbar(im, fraction=0.03, pad=0.03, label="Mean Selection Share")
        plt.tight_layout()
        plt.savefig(out_dir / f"action_usage_heatmap_instances_{_file_token(model_id)}.png", dpi=140)
        plt.close()


def _plot_action_effectiveness_bars_per_model(action_effectiveness_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not action_effectiveness_rows:
        return
    model_ids = sorted(
        {str(r.get("model_id", "")) for r in action_effectiveness_rows},
        key=lambda m: (_to_int(m.replace("checkpoint_seed", "")) or 10**9, m),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        rows = [r for r in action_effectiveness_rows if str(r.get("model_id", "")) == model_id]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: _to_float(r.get("selection_count")), reverse=True)
        top_rows = rows_sorted[:16]
        if not top_rows:
            continue
        labels = [_action_plot_id(r.get("action_type"), r.get("action_name")) for r in top_rows]
        improve_probs = [_to_float(r.get("improve_probability")) for r in top_rows]
        mean_deltas = [_to_float(r.get("mean_delta")) for r in top_rows]
        x = np.arange(len(labels))

        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.7), 8), constrained_layout=True)
        axes[0].bar(x, improve_probs, color="#1f77b4")
        axes[0].set_ylabel("P(improve)")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("Action Improve Probability")

        delta_colors = ["#2ca02c" if (math.isfinite(v) and v < 0) else "#d62728" for v in mean_deltas]
        axes[1].bar(x, mean_deltas, color=delta_colors)
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].set_ylabel("Mean Delta Objective")
        axes[1].set_title("Action Mean Delta (negative is better)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        fig.suptitle(f"Action Effectiveness Bars - {model_id}")
        fig.savefig(out_dir / f"action_effectiveness_bars_{_file_token(model_id)}.png", dpi=140)
        plt.close(fig)


def _plot_ppo_vs_alns_heatmap(matrix_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not matrix_rows:
        return
    algorithms = sorted(
        {str(r.get("algorithm_id", "")) for r in matrix_rows},
        key=lambda a: (
            a != "PPO_ALL",
            not a.startswith("checkpoint_seed"),
            _to_int(a.replace("checkpoint_seed", "")) or 10**9,
            a,
        ),
    )
    instances = sorted({str(r.get("test_instance_id", "")) for r in matrix_rows}, key=_instance_sort_key)
    if not algorithms or not instances:
        return
    lookup = {
        (str(r.get("algorithm_id", "")), str(r.get("test_instance_id", ""))): _to_float(r.get("mean_objective"))
        for r in matrix_rows
    }
    mat = np.full((len(algorithms), len(instances)), np.nan, dtype=float)
    for i, algo in enumerate(algorithms):
        for j, instance in enumerate(instances):
            mat[i, j] = lookup.get((algo, instance), math.nan)
    if not np.isfinite(mat).any():
        return
    masked = np.ma.masked_invalid(mat)
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="lightgray")
    plt.figure(figsize=(max(10, len(instances) * 0.55), max(5.5, len(algorithms) * 0.5)))
    im = plt.imshow(masked, aspect="auto", cmap=cmap)
    plt.xticks(np.arange(len(instances)), instances, rotation=60, ha="right")
    plt.yticks(np.arange(len(algorithms)), algorithms)
    plt.xlabel("Instance")
    plt.ylabel("Algorithm / Model")
    plt.title("PPO vs ALNS Mean Objective Heatmap")
    plt.colorbar(im, fraction=0.03, pad=0.03, label="Mean Objective")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_ppo_all_vs_alns_scatter(cmp_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = [r for r in cmp_rows if str(r.get("model_id", "")) == "PPO_ALL"]
    if not rows:
        return
    algorithms = sorted({str(r.get("alns_algorithm", "")) for r in rows})
    if not algorithms:
        return
    fig, axes = plt.subplots(1, len(algorithms), figsize=(max(6, 5.5 * len(algorithms)), 5.5), constrained_layout=True)
    if len(algorithms) == 1:
        axes = [axes]
    plotted_any = False
    for ax, alg in zip(axes, algorithms):
        subset = [r for r in rows if str(r.get("alns_algorithm", "")) == alg]
        xs = [_to_float(r.get("alns_mean_objective")) for r in subset]
        ys = [_to_float(r.get("ppo_mean_objective")) for r in subset]
        pts = [
            (str(r.get("test_instance_id", "")), x, y)
            for r, x, y in zip(subset, xs, ys)
            if math.isfinite(x) and math.isfinite(y)
        ]
        if not pts:
            ax.set_title(f"PPO_ALL vs {alg}\n(no overlap)")
            continue
        plotted_any = True
        px = [p[1] for p in pts]
        py = [p[2] for p in pts]
        lo = min(px + py)
        hi = max(px + py)
        pad = max((hi - lo) * 0.05, 1e-9)
        lo -= pad
        hi += pad
        ax.scatter(px, py, s=34, alpha=0.85)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="gray")
        if len(pts) <= 25:
            for instance, x, y in pts:
                ax.annotate(instance, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{alg} mean objective")
        ax.set_ylabel("PPO_ALL mean objective")
        ax.set_title(f"PPO_ALL vs {alg} (n={len(pts)})")
    if not plotted_any:
        plt.close(fig)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_ppo_all_vs_alns_diff_boxplot(cmp_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = [r for r in cmp_rows if str(r.get("model_id", "")) == "PPO_ALL"]
    if not rows:
        return
    algorithms = sorted({str(r.get("alns_algorithm", "")) for r in rows})
    labels: List[str] = []
    series: List[List[float]] = []
    for alg in algorithms:
        vals = [
            _to_float(r.get("diff_mean_ppo_minus_alns"))
            for r in rows
            if str(r.get("alns_algorithm", "")) == alg
        ]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        labels.append(f"PPO_ALL-{alg}")
        series.append(vals)
    if not series:
        return
    plt.figure(figsize=(max(8, len(labels) * 1.8), 5))
    plt.boxplot(series, tick_labels=labels, showmeans=True)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Mean Objective Difference (PPO - ALNS)")
    plt.title("PPO_ALL vs ALNS Per-instance Mean Objective Difference")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    ppo_root = (BASE_DIR / args.ppo_root).resolve() if not Path(args.ppo_root).is_absolute() else Path(args.ppo_root)
    alns_root = (BASE_DIR / args.alns_root).resolve() if not Path(args.alns_root).is_absolute() else Path(args.alns_root)
    variant_dir = ppo_root / args.variant
    if not variant_dir.exists():
        raise FileNotFoundError(f"variant directory not found: {variant_dir}")

    run_dir, summary_path = _resolve_run_dir(variant_dir=variant_dir, run_id=args.run_id)
    run_id = run_dir.name
    summary_rows = _read_csv(summary_path)
    config = _read_config(run_dir / "config.txt")
    commit_hash = _safe_git_commit(BASE_DIR)

    baseline_csv_path = Path(args.baseline_csv) if args.baseline_csv != "" else None
    if baseline_csv_path is not None and not baseline_csv_path.is_absolute():
        baseline_csv_path = (BASE_DIR / baseline_csv_path).resolve()
    if baseline_csv_path is None:
        baseline_dir = (BASE_DIR / args.baseline_dir).resolve() if not Path(args.baseline_dir).is_absolute() else Path(args.baseline_dir)
        baseline_csv_path = _latest_baseline_csv(baseline_dir)
    cp_map = _build_cp_map(baseline_csv_path)

    alns_eval_rows: List[Dict[str, Any]] = []
    alns_summary_per_instance_rows: List[Dict[str, Any]] = []
    alns_sources: List[Dict[str, Any]] = []
    alns_variants = [v for v in [args.alns_variant_a, args.alns_variant_b] if str(v).strip() != ""]
    seen_variants: set[str] = set()
    for variant in alns_variants:
        if variant in seen_variants:
            continue
        seen_variants.add(variant)
        loaded = _load_alns_eval_rows(
            alns_root=alns_root,
            variant=variant,
            run_id=args.alns_run_id,
            cp_map=cp_map,
        )
        if loaded is None:
            continue
        rows, alns_run_id, alns_summary_path = loaded
        alns_eval_rows.extend(rows)
        alns_sources.append(
            {
                "variant": variant,
                "run_id": alns_run_id,
                "summary_path": alns_summary_path,
                "n_rows": len(rows),
            }
        )
    if alns_eval_rows:
        alns_summary_per_instance_rows = _alns_per_instance_summary(alns_eval_rows)

    checkpoint_root = run_dir / "checkpoint_benchmark"
    selected_checkpoint_run = args.checkpoint_run_id
    if selected_checkpoint_run is None:
        selected_checkpoint_run = _latest_checkpoint_benchmark_run(checkpoint_root)
    checkpoint_dir = checkpoint_root / selected_checkpoint_run if selected_checkpoint_run is not None else None

    raw_eval_rows: List[Dict[str, Any]] = []
    header_rows: List[Dict[str, Any]] = []
    raw_action_rows: List[Dict[str, Any]] = []
    if checkpoint_dir is not None and checkpoint_dir.exists():
        for log_path in sorted(checkpoint_dir.glob("checkpoint_seed*.log")):
            rows, header, action_rows = _parse_checkpoint_log(log_path)
            raw_eval_rows.extend(rows)
            header_rows.append(header)
            raw_action_rows.extend(action_rows)

    eval_rows = _augment_eval_rows(
        raw_eval_rows=raw_eval_rows,
        summary_rows=summary_rows,
        variant=args.variant,
        run_id=run_id,
        benchmark_run_id=selected_checkpoint_run,
        cp_map=cp_map,
    )
    summary_by_checkpoint = {str(r.get("checkpoint", "")).strip(): r for r in summary_rows}
    action_rows: List[Dict[str, Any]] = []
    for row in raw_action_rows:
        checkpoint = str(row.get("checkpoint", "")).strip()
        srow = summary_by_checkpoint.get(checkpoint, {})
        instance = str(row.get("test_instance_id", ""))
        cp_obj = _to_float(cp_map.get(instance))
        obj_after = _to_float(row.get("objective_after"))
        action_rows.append(
            {
                "model_id": str(row.get("model_id", "")),
                "variant": args.variant,
                "run_id": run_id,
                "checkpoint_benchmark_run_id": selected_checkpoint_run or "",
                "checkpoint": checkpoint,
                "checkpoint_seed": _to_int(srow.get("seed")),
                "test_instance_id": instance,
                "instance_index": _to_int(row.get("instance_index")),
                "eval_seed": _to_int(row.get("eval_seed")),
                "iteration": _to_int(row.get("iteration")),
                "elapsed_seconds": _to_float(row.get("elapsed_seconds")),
                "destroy_action": str(row.get("destroy_action", "")),
                "repair_action": str(row.get("repair_action", "")),
                "objective_before": _to_float(row.get("objective_before")),
                "objective_after": obj_after,
                "delta_objective": _to_float(row.get("delta_objective")),
                "accepted": _to_bool(row.get("accepted")),
                "reward": _to_float(row.get("reward")),
                "invalid_action": _to_bool(row.get("invalid_action")),
                "cp_objective": cp_obj,
                "gap_to_cp_after": _gap_to_cp(obj_after, cp_obj),
                "source_log": str(row.get("source_log", "")),
            }
        )
    model_card_rows = _build_model_card_rows(
        summary_rows=summary_rows,
        variant=args.variant,
        run_id=run_id,
        config=config,
        commit_hash=commit_hash,
        eval_rows=eval_rows,
    )
    per_instance_rows = _per_model_instance_summary(eval_rows)
    ppo_all_rows = _ppo_all_instance_summary(eval_rows)
    ppo_cmp_rows = list(per_instance_rows) + list(ppo_all_rows)
    ppo_vs_alns_rows: List[Dict[str, Any]] = []
    ppo_vs_alns_summary_rows: List[Dict[str, Any]] = []
    ppo_vs_alns_matrix_rows: List[Dict[str, Any]] = []
    if alns_summary_per_instance_rows:
        (
            ppo_vs_alns_rows,
            ppo_vs_alns_summary_rows,
            ppo_vs_alns_matrix_rows,
        ) = _ppo_vs_alns_comparison(
            ppo_instance_rows=ppo_cmp_rows,
            alns_instance_rows=alns_summary_per_instance_rows,
        )
    heldout_rows = _heldout_scores_6a(per_instance_rows)
    matrix_rows, models, instances = _cross_model_matrix(per_instance_rows)
    rank_rows, rank_summary_rows, dominance_rows = _ranking_and_dominance(matrix_rows, models=models, instances=instances)
    pairwise_rows = _pairwise_model_tests(matrix_rows, models=models, instances=instances)
    risk_instance_rows, risk_model_rows = _risk_profile(eval_rows)
    progress_run_rows, progress_mean_rows = _within_episode_progress(action_rows)
    conv_run_rows, conv_instance_rows, conv_global_rows = _convergence_tables(
        progress_run_rows=progress_run_rows,
        cp_map=cp_map,
        time_step=args.time_step,
    )
    ttt_run_rows, ttt_instance_rows, ttt_summary_rows = _time_to_target(
        progress_run_rows=progress_run_rows,
        cp_map=cp_map,
        target_gap=args.target_gap,
    )
    action_usage_run_rows, action_usage_instance_rows, action_usage_global_rows = _action_usage_tables(action_rows)
    action_effectiveness_rows = _action_effectiveness(action_rows)
    training_rows = _training_diagnostics(run_dir=run_dir, summary_rows=summary_rows)
    quality_checks = _build_data_quality_checks(
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        expected_eval_seeds=args.expected_eval_seeds,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (BASE_DIR / args.output_dir / timestamp).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir) / timestamp
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        tables_dir / "model_card.csv",
        model_card_rows,
        fieldnames=[
            "model_id",
            "variant",
            "run_id",
            "checkpoint",
            "checkpoint_seed",
            "train_instances",
            "test_instances",
            "n_train_instances",
            "n_test_instances",
            "training_budget_timesteps",
            "training_runtime_seconds",
            "training_status",
            "executed_steps_training",
            "timeout_seconds_training",
            "reward_definition",
            "observation_definition",
            "action_definition",
            "domain_randomization",
            "environment_version_commit",
            "run_config_timeout_seconds",
            "run_config_max_iterations",
            "run_config_n_instances",
            "run_config_train_count",
            "run_config_test_count",
            "run_config_n_seeds",
        ],
    )
    _write_csv(
        tables_dir / "evaluation_runs.csv",
        eval_rows,
        fieldnames=[
            "model_id",
            "variant",
            "run_id",
            "checkpoint_benchmark_run_id",
            "checkpoint",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "eval_seed",
            "final_objective",
            "best_objective_over_episode",
            "episode_length_steps",
            "runtime_seconds",
            "termination_reason",
            "solved",
            "constraint_violation_metric",
            "return_sum",
            "discounted_return",
            "state_dim",
            "status",
            "error",
            "runs_each",
            "train_instances",
            "summary_test_instances",
            "source_log",
            "cp_objective",
            "gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "evaluation_summary_per_model_instance.csv",
        per_instance_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "n_runs",
            "mean_objective",
            "median_objective",
            "best_objective",
            "worst_objective",
            "std_objective",
            "range_objective",
            "q10_objective",
            "q25_objective",
            "q50_objective",
            "q75_objective",
            "q90_objective",
            "mean_gap_to_cp",
            "median_gap_to_cp",
            "best_gap_to_cp",
            "std_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "evaluation_summary_ppo_all_instances.csv",
        ppo_all_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "n_runs",
            "mean_objective",
            "median_objective",
            "best_objective",
            "worst_objective",
            "std_objective",
            "range_objective",
            "q10_objective",
            "q25_objective",
            "q50_objective",
            "q75_objective",
            "q90_objective",
            "mean_gap_to_cp",
            "median_gap_to_cp",
            "best_gap_to_cp",
            "std_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "alns_eval_runs.csv",
        alns_eval_rows,
        fieldnames=[
            "algorithm",
            "variant",
            "run_id",
            "source_summary",
            "test_instance_id",
            "instance_index",
            "seed",
            "status",
            "objective",
            "runtime_seconds",
            "iterations",
            "solved",
            "cp_objective",
            "gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "alns_summary_per_instance.csv",
        alns_summary_per_instance_rows,
        fieldnames=[
            "algorithm",
            "test_instance_id",
            "instance_index",
            "n_runs",
            "mean_objective",
            "median_objective",
            "best_objective",
            "worst_objective",
            "std_objective",
            "mean_gap_to_cp",
            "median_gap_to_cp",
            "best_gap_to_cp",
            "std_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "ppo_vs_alns_per_instance.csv",
        ppo_vs_alns_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "alns_algorithm",
            "ppo_mean_objective",
            "alns_mean_objective",
            "diff_mean_ppo_minus_alns",
            "rel_diff_mean_vs_alns",
            "ppo_best_objective",
            "alns_best_objective",
            "diff_best_ppo_minus_alns",
            "ppo_mean_gap_to_cp",
            "alns_mean_gap_to_cp",
            "diff_gap_to_cp_ppo_minus_alns",
            "ppo_better_mean",
        ],
    )
    _write_csv(
        tables_dir / "ppo_vs_alns_summary.csv",
        ppo_vs_alns_summary_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "alns_algorithm",
            "n_instances_compared",
            "mean_diff_mean_ppo_minus_alns",
            "median_diff_mean_ppo_minus_alns",
            "mean_rel_diff_mean_vs_alns",
            "mean_diff_gap_to_cp_ppo_minus_alns",
            "ppo_better_count",
            "alns_better_count",
            "tie_count",
        ],
    )
    _write_csv(
        tables_dir / "ppo_vs_alns_matrix.csv",
        ppo_vs_alns_matrix_rows,
        fieldnames=[
            "algorithm_id",
            "source_family",
            "test_instance_id",
            "instance_index",
            "mean_objective",
            "mean_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "heldout_scores_6A.csv",
        heldout_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "n_test_instances",
            "score_mean",
            "score_median",
            "score_worstcase",
            "score_std_across_test_instances",
        ],
    )
    _write_csv(
        tables_dir / "cross_model_matrix.csv",
        matrix_rows,
        fieldnames=["test_instance_id", "instance_index", "model_id", "mean_objective", "evaluated"],
    )
    _write_csv(
        tables_dir / "rank_per_instance.csv",
        rank_rows,
        fieldnames=["test_instance_id", "instance_index", "model_id", "mean_objective", "rank"],
    )
    _write_csv(
        tables_dir / "rank_summary.csv",
        rank_summary_rows,
        fieldnames=["model_id", "n_instances_ranked", "wins_rank1", "top2_count", "average_rank"],
    )
    _write_csv(
        tables_dir / "dominance_summary.csv",
        dominance_rows,
        fieldnames=["model_a", "model_b", "shared_instances", "better_a_count", "better_b_count", "tie_count", "a_dominates_b"],
    )
    _write_csv(
        tables_dir / "pairwise_model_tests.csv",
        pairwise_rows,
        fieldnames=[
            "model_a",
            "model_b",
            "n_common_instances",
            "mean_diff_A_minus_B",
            "median_diff_A_minus_B",
            "better_a_count",
            "better_b_count",
            "tie_count",
            "wilcoxon_p_value_two_sided",
            "wilcoxon_n_nonzero",
            "p_holm",
            "p_bh_fdr",
        ],
    )
    _write_csv(
        tables_dir / "risk_profile_per_instance.csv",
        risk_instance_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "n_runs",
            "std_objective",
            "cv_objective",
            "tail_q90_minus_q50",
            "tail_worst_minus_q50",
            "bad_run_threshold",
            "bad_run_fraction",
        ],
    )
    _write_csv(
        tables_dir / "risk_profile_per_model.csv",
        risk_model_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "n_test_instances",
            "mean_std_objective",
            "mean_cv_objective",
            "mean_tail_q90_minus_q50",
            "mean_tail_worst_minus_q50",
            "mean_bad_run_fraction",
        ],
    )
    _write_csv(
        tables_dir / "action_events.csv",
        action_rows,
        fieldnames=[
            "model_id",
            "variant",
            "run_id",
            "checkpoint_benchmark_run_id",
            "checkpoint",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "eval_seed",
            "iteration",
            "elapsed_seconds",
            "destroy_action",
            "repair_action",
            "objective_before",
            "objective_after",
            "delta_objective",
            "accepted",
            "reward",
            "invalid_action",
            "cp_objective",
            "gap_to_cp_after",
            "source_log",
        ],
    )
    _write_csv(
        tables_dir / "action_usage_per_run.csv",
        action_usage_run_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "eval_seed",
            "action_type",
            "action_name",
            "selection_count",
            "selection_share",
            "action_entropy",
            "invalid_action_fraction",
        ],
    )
    _write_csv(
        tables_dir / "action_usage_per_instance.csv",
        action_usage_instance_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "action_type",
            "action_name",
            "mean_selection_share",
            "mean_action_entropy",
            "mean_invalid_action_fraction",
            "n_runs",
        ],
    )
    _write_csv(
        tables_dir / "action_usage_global.csv",
        action_usage_global_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "action_type",
            "action_name",
            "mean_selection_share",
            "std_selection_share_across_instances",
            "mean_invalid_action_fraction",
            "n_instances",
        ],
    )
    _write_csv(
        tables_dir / "action_effectiveness.csv",
        action_effectiveness_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "action_type",
            "action_name",
            "selection_count",
            "improve_probability",
            "mean_delta",
            "mean_improving_delta",
            "mean_worsening_delta",
            "invalid_action_fraction",
        ],
    )
    _write_csv(
        tables_dir / "within_episode_progress_runs.csv",
        progress_run_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "eval_seed",
            "iteration",
            "elapsed_seconds",
            "best_so_far_objective",
        ],
    )
    _write_csv(
        tables_dir / "within_episode_progress_means.csv",
        progress_mean_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "iteration",
            "mean_best_so_far_objective",
        ],
    )
    _write_csv(
        tables_dir / "convergence_run_curves_on_grid.csv",
        conv_run_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "eval_seed",
            "time_seconds",
            "best_so_far_objective",
            "best_so_far_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "convergence_instance_mean_curves.csv",
        conv_instance_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "time_seconds",
            "mean_best_so_far_objective",
            "mean_best_so_far_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "convergence_global_curves.csv",
        conv_global_rows,
        fieldnames=[
            "model_id",
            "time_seconds",
            "global_mean_objective",
            "global_mean_gap_to_cp",
        ],
    )
    _write_csv(
        tables_dir / "time_to_target_runs.csv",
        ttt_run_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "eval_seed",
            "cp_objective",
            "target_gap",
            "target_objective",
            "time_to_target_seconds",
            "reached_target",
        ],
    )
    _write_csv(
        tables_dir / "time_to_target_instance.csv",
        ttt_instance_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "test_instance_id",
            "instance_index",
            "n_runs",
            "solved_runs",
            "solved_fraction",
            "mean_time_to_target_seconds",
            "median_time_to_target_seconds",
        ],
    )
    _write_csv(
        tables_dir / "time_to_target_summary.csv",
        ttt_summary_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "n_test_instances",
            "mean_solved_fraction",
            "mean_time_to_target_seconds",
            "median_time_to_target_seconds",
        ],
    )
    _write_csv(
        tables_dir / "training_diagnostics.csv",
        training_rows,
        fieldnames=[
            "model_id",
            "checkpoint_seed",
            "log_path",
            "n_episode_returns",
            "mean_episode_return",
            "final_episode_return",
            "mean_policy_loss",
            "mean_value_loss",
            "mean_entropy",
            "mean_reward_step",
            "reward_trend_label",
            "reward_trend_early_slope",
            "reward_trend_late_slope",
            "reward_trend_slope_ratio",
        ],
    )
    _write_csv(
        tables_dir / "data_quality_checks.csv",
        quality_checks,
        fieldnames=["check_id", "status", "scope", "details"],
    )
    _write_csv(
        tables_dir / "checkpoint_log_headers.csv",
        header_rows,
        fieldnames=sorted({k for row in header_rows for k in row.keys()}),
    )
    _write_data_quality_md(output_dir / "data_quality.md", quality_checks)

    _plot_boxplot_objective_by_model(eval_rows, out_path=plots_dir / "boxplot_objective_by_model.png")
    _plot_boxplot_final_relative_gaps_by_model(eval_rows, out_path=plots_dir / "boxplot_final_relative_gaps_by_model.png")
    _plot_boxplots_objective_by_instance_per_model(eval_rows, out_dir=plots_dir / "boxplots_by_model_instance")
    _plot_heatmap_model_instance_means(per_instance_rows, out_path=plots_dir / "heatmap_model_instance_mean_objective.png")
    _plot_heatmap_model_instance_gap_to_cp(per_instance_rows, out_path=plots_dir / "heatmap_model_instance_mean_gap_to_cp.png")
    _plot_best_ppo_vs_benchmark_logy(eval_rows, cp_map=cp_map, out_path=plots_dir / "per_instance_best_ppo_vs_benchmark_logy.png")
    _plot_risk_profile(risk_model_rows, out_path=plots_dir / "risk_profile_by_model.png")
    _plot_cv_distribution_by_model(risk_instance_rows, out_path=plots_dir / "cv_distribution_by_model.png")
    _plot_risk_profile_per_instance(risk_instance_rows, out_path=plots_dir / "risk_profile_per_instance_heatmaps.png")
    _plot_global_convergence(
        conv_global_rows,
        out_obj=plots_dir / "global_convergence_objective_all_seeds.png",
        out_gap=plots_dir / "global_convergence_gap_all_seeds.png",
    )
    _plot_cactus(ttt_run_rows, out_path=plots_dir / "cactus_time_to_target_by_model.png")
    _plot_within_episode_progress(progress_mean_rows, out_path=plots_dir / "within_episode_progress_by_model.png")
    _plot_action_usage_heatmap(action_usage_global_rows, out_path=plots_dir / "action_usage_heatmap.png")
    _plot_action_usage_heatmap_per_model(action_usage_instance_rows, out_dir=plots_dir / "action_usage_by_model")
    _plot_action_effectiveness_bars_per_model(action_effectiveness_rows, out_dir=plots_dir / "action_effectiveness_by_model")
    _plot_pairwise_scatter_model_means(matrix_rows, models=models, instances=instances, out_dir=plots_dir / "pairwise_scatter")
    if ppo_vs_alns_matrix_rows:
        ppo_alns_plots_dir = plots_dir / "ppo_vs_alns"
        _plot_ppo_vs_alns_heatmap(ppo_vs_alns_matrix_rows, out_path=ppo_alns_plots_dir / "heatmap_mean_objective.png")
        _plot_ppo_all_vs_alns_scatter(ppo_vs_alns_rows, out_path=ppo_alns_plots_dir / "ppo_all_vs_alns_scatter.png")
        _plot_ppo_all_vs_alns_diff_boxplot(ppo_vs_alns_rows, out_path=ppo_alns_plots_dir / "ppo_all_vs_alns_diff_boxplot.png")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "ppo_root": str(ppo_root),
        "variant": args.variant,
        "run_id": run_id,
        "summary_path": str(summary_path),
        "alns_root": str(alns_root),
        "alns_variant_a": args.alns_variant_a,
        "alns_variant_b": args.alns_variant_b,
        "alns_run_id": args.alns_run_id or "",
        "alns_sources": alns_sources,
        "baseline_csv": str(baseline_csv_path),
        "checkpoint_benchmark_run_id": selected_checkpoint_run or "",
        "checkpoint_benchmark_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
        "expected_eval_seeds": args.expected_eval_seeds,
        "target_gap": args.target_gap,
        "time_step": args.time_step,
        "n_models": len(model_card_rows),
        "n_eval_rows_total": len(eval_rows),
        "n_eval_rows_ok": sum(1 for r in eval_rows if str(r.get("status", "")) == "ok"),
        "n_eval_rows_error": sum(1 for r in eval_rows if str(r.get("status", "")) == "error"),
        "n_ppo_all_rows": len(ppo_all_rows),
        "n_alns_eval_rows": len(alns_eval_rows),
        "n_alns_summary_per_instance_rows": len(alns_summary_per_instance_rows),
        "n_ppo_vs_alns_rows": len(ppo_vs_alns_rows),
        "n_ppo_vs_alns_summary_rows": len(ppo_vs_alns_summary_rows),
        "n_action_rows": len(action_rows),
        "n_convergence_run_rows": len(conv_run_rows),
        "n_time_to_target_rows": len(ttt_run_rows),
        "n_training_diagnostics_rows": len(training_rows),
        "n_quality_pass": sum(1 for r in quality_checks if str(r.get("status")) == "PASS"),
        "n_quality_fail": sum(1 for r in quality_checks if str(r.get("status")) == "FAIL"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO model evaluation analysis with ALNS-style output structure.")
    parser.add_argument("--ppo-root", default="benchmark_PPO", help="Path to benchmark_PPO root.")
    parser.add_argument("--variant", default="ppo_late_phase", help="Variant directory under benchmark_PPO.")
    parser.add_argument("--run-id", default=None, help="PPO run folder id (e.g. 20260227-210927). If omitted, latest is used.")
    parser.add_argument("--alns-root", default="benchmark_alns", help="Path to benchmark_alns root for PPO-vs-ALNS comparison.")
    parser.add_argument("--alns-variant-a", default="alns_plain", help="First ALNS variant under benchmark_alns.")
    parser.add_argument("--alns-variant-b", default="alns_late_phase", help="Second ALNS variant under benchmark_alns.")
    parser.add_argument("--alns-run-id", default=None, help="ALNS run folder id (if omitted, latest per ALNS variant is used).")
    parser.add_argument("--baseline-dir", default="benchmark_baseline", help="Directory with baseline_runs_*.csv for gap metrics.")
    parser.add_argument("--baseline-csv", default="", help="Explicit baseline csv path. If empty, latest baseline_runs_*.csv is used.")
    parser.add_argument(
        "--checkpoint-run-id",
        default=None,
        help="Checkpoint benchmark run id under run/checkpoint_benchmark. If omitted, latest is used.",
    )
    parser.add_argument("--expected-eval-seeds", type=int, default=20, help="Expected number of evaluation seeds per (model,test_instance).")
    parser.add_argument("--target-gap", type=float, default=0.02, help="Target relative gap above CP for time-to-target analysis.")
    parser.add_argument("--time-step", type=float, default=1.0, help="Time grid step in seconds for convergence aggregation.")
    parser.add_argument("--output-dir", default="analysis_ppo", help="Output directory root.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    metadata = run_analysis(args)
    print("PPO analysis finished.")
    print(f"Output directory: {metadata['output_dir']}")
    print(
        "Rows:"
        f" models={metadata['n_models']}"
        f" eval_total={metadata['n_eval_rows_total']}"
        f" eval_ok={metadata['n_eval_rows_ok']}"
        f" eval_error={metadata['n_eval_rows_error']}"
    )
    print(f"Quality checks: pass={metadata['n_quality_pass']} fail={metadata['n_quality_fail']}")


if __name__ == "__main__":
    main()
