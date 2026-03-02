from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Dict, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = Path(__file__).resolve().parent


SUMMARY_FIELDS = [
    "variant",
    "instance",
    "instance_index",
    "seed",
    "timeout_seconds",
    "status",
    "objective",
    "runtime_seconds",
    "iterations",
    "solved",
    "timeout_hit",
    "error",
    "solver_name",
]


def _to_int(value: Any) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _instance_index(name: str) -> int | None:
    digits = ""
    for ch in reversed(name):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else None


def _instance_sort_key(name: str) -> tuple[int, str]:
    idx = _instance_index(name)
    if idx is None:
        return (10**9, name)
    return (idx, name)


def _read_config(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _read_summary(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _ask_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return default if raw == "" else raw


def _ask_bool(prompt: str, default_yes: bool = True) -> bool:
    default = "y" if default_yes else "n"
    raw = input(f"{prompt} [y/n] [{default}]: ").strip().lower()
    value = default if raw == "" else raw
    return value in {"y", "yes", "1", "true"}


def _ask_variant_mode() -> List[str]:
    raw = input("Variants to merge both/plain/late_phase [both]: ").strip().lower()
    mode = "both" if raw == "" else raw
    if mode in {"both", "all"}:
        return ["alns_plain", "alns_late_phase"]
    if mode in {"plain", "alns_plain"}:
        return ["alns_plain"]
    if mode in {"late", "late_phase", "alns_late_phase"}:
        return ["alns_late_phase"]
    raise ValueError("variant mode must be both/plain/late_phase")


def _list_run_names_lex(variant: str) -> List[str]:
    variant_dir = BENCHMARK_ROOT / variant
    if not variant_dir.exists():
        return []
    return sorted([p.name for p in variant_dir.iterdir() if p.is_dir()])


def _latest_variant_lex() -> str:
    candidates: List[Tuple[str, str]] = []
    for variant in ("alns_plain", "alns_late_phase"):
        names = _list_run_names_lex(variant)
        if names:
            candidates.append((names[-1], variant))
    if not candidates:
        return "alns_plain"
    candidates.sort()
    return candidates[-1][1]


def _default_pair(variant: str, run_names: Sequence[str]) -> Tuple[str, str]:
    valid = []
    for run_name in run_names:
        summary_path = BENCHMARK_ROOT / variant / run_name / "summary.csv"
        if summary_path.exists():
            valid.append(run_name)
    if len(valid) >= 2:
        return valid[-2], valid[-1]
    if len(run_names) < 2:
        raise ValueError("need at least two run folders to merge")
    return run_names[-2], run_names[-1]


def _extract_timeout(config: Dict[str, str], rows: Sequence[Dict[str, str]]) -> float | None:
    cfg_timeout = _to_float(config.get("timeout_seconds"))
    if cfg_timeout is not None:
        return cfg_timeout
    vals = sorted({_to_float(r.get("timeout_seconds")) for r in rows if _to_float(r.get("timeout_seconds")) is not None})
    if len(vals) == 1:
        return vals[0]
    return None


def _run_specs(variant: str, run_name: str) -> Dict[str, Any]:
    run_dir = BENCHMARK_ROOT / variant / run_name
    summary_path = run_dir / "summary.csv"
    logs_dir = run_dir / "logs"
    config_path = run_dir / "config.txt"
    summary_rows = _read_summary(summary_path)
    config = _read_config(config_path)
    instances = sorted(
        {str(r.get("instance", "")).strip() for r in summary_rows if str(r.get("instance", "")).strip() != ""},
        key=_instance_sort_key,
    )
    seeds = sorted({_to_int(r.get("seed")) for r in summary_rows if _to_int(r.get("seed")) is not None})
    timeout_values = sorted({_to_float(r.get("timeout_seconds")) for r in summary_rows if _to_float(r.get("timeout_seconds")) is not None})
    status_counts: Dict[str, int] = {}
    for row in summary_rows:
        status = str(row.get("status", "")).strip() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "variant": variant,
        "run_name": run_name,
        "run_dir": run_dir,
        "summary_path": summary_path,
        "logs_dir": logs_dir,
        "config_path": config_path,
        "summary_rows": summary_rows,
        "config": config,
        "instances": instances,
        "seeds": [int(s) for s in seeds if s is not None],
        "timeout_values": [float(v) for v in timeout_values if v is not None],
        "timeout_hint": _extract_timeout(config, summary_rows),
        "status_counts": status_counts,
        "logs_count": len(list(logs_dir.glob("*.log"))) if logs_dir.exists() else 0,
    }


def _print_specs_readback(variant: str, left: Dict[str, Any], right: Dict[str, Any]) -> None:
    print(f"\nVariant={variant}")
    print(f"  merge_from_a={left['run_name']}")
    print(f"    summary_exists={left['summary_path'].exists()} rows={len(left['summary_rows'])} logs={left['logs_count']}")
    print(f"    instances={len(left['instances'])} seeds={left['seeds']} timeout_hint={left['timeout_hint']}")
    print(f"    status_counts={left['status_counts']}")
    print(f"  merge_from_b={right['run_name']}")
    print(f"    summary_exists={right['summary_path'].exists()} rows={len(right['summary_rows'])} logs={right['logs_count']}")
    print(f"    instances={len(right['instances'])} seeds={right['seeds']} timeout_hint={right['timeout_hint']}")
    print(f"    status_counts={right['status_counts']}")


def _seed_mapping(rows_a: Sequence[Dict[str, str]], rows_b: Sequence[Dict[str, str]]) -> Dict[Tuple[int, int], int]:
    source_seed_pairs = set()
    for row in rows_a:
        seed = _to_int(row.get("seed"))
        if seed is not None:
            source_seed_pairs.add((0, seed))
    for row in rows_b:
        seed = _to_int(row.get("seed"))
        if seed is not None:
            source_seed_pairs.add((1, seed))
    ordered = sorted(source_seed_pairs, key=lambda t: (t[0], t[1]))
    return {pair: new_seed for new_seed, pair in enumerate(ordered)}


def _row_sort_key(row: Dict[str, Any]) -> Tuple[int, int]:
    idx = _to_int(row.get("instance_index"))
    if idx is None:
        idx = _instance_index(str(row.get("instance", ""))) or 10**9
    seed = _to_int(row.get("seed"))
    return (idx, seed if seed is not None else 10**9)


def _find_source_log(logs_dir: Path, instance: str, old_seed: int) -> Path | None:
    candidates = sorted(logs_dir.glob(f"*_{instance}_seed{old_seed}.log"))
    return candidates[0] if candidates else None


def _merge_variant(
    variant: str,
    run_a: Dict[str, Any],
    run_b: Dict[str, Any],
    out_run_name: str,
) -> None:
    rows_a = run_a["summary_rows"]
    rows_b = run_b["summary_rows"]
    if not rows_a and not rows_b:
        raise ValueError(f"{variant}: both summaries empty")

    timeout_a = run_a["timeout_hint"]
    timeout_b = run_b["timeout_hint"]
    if timeout_a is None or timeout_b is None:
        print(
            f"\nWARNING {variant}: timeout could not be verified "
            f"(a={timeout_a}, b={timeout_b})"
        )
        if not _ask_bool("Continue merge without a reliable timeout check?", default_yes=False):
            raise RuntimeError(f"{variant}: merge canceled due to missing timeout information")
    elif abs(timeout_a - timeout_b) > 1e-9:
        print(
            f"\nWARNING {variant}: timeout mismatch "
            f"a={timeout_a} vs b={timeout_b}"
        )
        if not _ask_bool("Continue merge with mixed timeout values?", default_yes=False):
            raise RuntimeError(f"{variant}: merge canceled due to timeout mismatch")

    out_dir = BENCHMARK_ROOT / variant / out_run_name
    out_logs = out_dir / "logs"
    if out_dir.exists():
        raise FileExistsError(f"output run already exists: {out_dir}")
    out_logs.mkdir(parents=True, exist_ok=False)

    mapping = _seed_mapping(rows_a, rows_b)
    merged_rows: List[Dict[str, Any]] = []
    missing_logs: List[str] = []
    sources = [(0, run_a, rows_a), (1, run_b, rows_b)]

    max_idx = -1
    for _source_id, _run, rows in sources:
        for row in rows:
            idx = _to_int(row.get("instance_index"))
            if idx is None:
                idx = _instance_index(str(row.get("instance", "")))
            if idx is not None:
                max_idx = max(max_idx, idx)
    index_width = max(2, len(str(max_idx))) if max_idx >= 0 else 2

    for source_id, run, rows in sources:
        for row in rows:
            instance = str(row.get("instance", "")).strip()
            if instance == "":
                continue
            old_seed = _to_int(row.get("seed"))
            if old_seed is None:
                continue
            new_seed = mapping[(source_id, old_seed)]
            new_row = {key: row.get(key, "") for key in SUMMARY_FIELDS}
            new_row["variant"] = variant
            idx = _to_int(new_row.get("instance_index"))
            if idx is None:
                idx = _instance_index(instance)
            new_row["instance_index"] = idx if idx is not None else ""
            new_row["seed"] = new_seed
            merged_rows.append(new_row)

            src_log = _find_source_log(run["logs_dir"], instance=instance, old_seed=old_seed)
            idx_tag = "na" if idx is None else f"{idx:0{index_width}d}"
            dst_log = out_logs / f"{idx_tag}_{instance}_seed{new_seed}.log"
            if src_log is None:
                missing_logs.append(f"{run['run_name']}:{instance}:seed{old_seed}")
            else:
                shutil.copy2(src_log, dst_log)

    merged_rows.sort(key=_row_sort_key)
    _write_summary(out_dir / "summary.csv", merged_rows)

    config_lines = [
        f"run_id={out_run_name}",
        f"variant={variant}",
        f"merged_from_a={run_a['run_name']}",
        f"merged_from_b={run_b['run_name']}",
        f"merged_at={datetime.now().isoformat(timespec='seconds')}",
        f"rows_total={len(merged_rows)}",
        f"instances={len({str(r.get('instance', '')).strip() for r in merged_rows})}",
        f"seeds_consecutive=0..{len(mapping)-1}",
        f"source_timeout_a={timeout_a}",
        f"source_timeout_b={timeout_b}",
        f"solver_setup_a={run_a['config'].get('solver_setup', '')}",
        f"solver_setup_b={run_b['config'].get('solver_setup', '')}",
    ]
    (out_dir / "config.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    map_rows = [
        {
            "source_run": run_a["run_name"] if source_id == 0 else run_b["run_name"],
            "source_id": source_id,
            "old_seed": old_seed,
            "new_seed": new_seed,
        }
        for (source_id, old_seed), new_seed in sorted(mapping.items(), key=lambda t: t[1])
    ]
    with (out_dir / "seed_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_run", "source_id", "old_seed", "new_seed"])
        writer.writeheader()
        writer.writerows(map_rows)

    if missing_logs:
        (out_dir / "missing_logs.txt").write_text(
            "\n".join(missing_logs) + "\n",
            encoding="utf-8",
        )

    print(f"\nMerged {variant} -> {out_dir}")
    print(f"  merged_rows={len(merged_rows)}")
    print(f"  new_seed_count={len(mapping)}")
    print(f"  missing_logs={len(missing_logs)}")


def main() -> None:
    default_variant_mode = "both"
    suggested_variant = _latest_variant_lex()
    print(f"Latest variant by lexicographic folder name: {suggested_variant}")
    variants = _ask_variant_mode() if default_variant_mode == "both" else [suggested_variant]

    selections: Dict[str, Tuple[str, str]] = {}
    for variant in variants:
        run_names = _list_run_names_lex(variant)
        if len(run_names) < 2:
            raise ValueError(f"{variant}: need at least two folders to merge")
        default_a, default_b = _default_pair(variant, run_names)
        print(f"\n{variant} available runs (lex): {', '.join(run_names[-6:])}")
        run_a = _ask_str(f"{variant} source run A", default_a)
        run_b = _ask_str(f"{variant} source run B", default_b)
        if run_a == run_b:
            raise ValueError(f"{variant}: run A and run B must differ")
        selections[variant] = (run_a, run_b)

    out_run_name = _ask_str("Merged output run folder name", datetime.now().strftime("%Y%m%d-%H%M%S"))

    specs: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for variant in variants:
        run_a_name, run_b_name = selections[variant]
        left = _run_specs(variant, run_a_name)
        right = _run_specs(variant, run_b_name)
        _print_specs_readback(variant, left, right)
        if not left["summary_path"].exists() or not right["summary_path"].exists():
            raise FileNotFoundError(
                f"{variant}: both source folders must contain summary.csv "
                f"(got a={left['summary_path'].exists()} b={right['summary_path'].exists()})"
            )
        specs[variant] = (left, right)

    if not _ask_bool("\nProceed with merge?", default_yes=False):
        print("Merge canceled.")
        return

    for variant in variants:
        left, right = specs[variant]
        _merge_variant(
            variant=variant,
            run_a=left,
            run_b=right,
            out_run_name=out_run_name,
        )

    print("\nMerge complete.")


if __name__ == "__main__":
    main()
