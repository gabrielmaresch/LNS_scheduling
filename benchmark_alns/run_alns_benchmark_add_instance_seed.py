from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from run_alns_benchmark import _run_single


FIELDNAMES = [
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


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _instance_index_from_stem(stem: str) -> int | None:
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    if digits == "":
        return None
    return int(digits)


def _instance_sort_key(name: str) -> tuple[int, str]:
    idx = _instance_index_from_stem(name)
    if idx is None:
        return (10**9, name)
    return (idx, name)


def _ask_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return default if raw == "" else raw


def _ask_int(prompt: str, default: int, minimum: int = 0) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else int(raw)
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _ask_float(prompt: str, default: float, minimum: float = 0.1) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else float(raw)
    if value < minimum:
        raise ValueError(f"value must be >= {minimum}")
    return value


def _ask_variant(default: str) -> str:
    raw = input(f"Variant alns_plain/alns_late_phase [{default}]: ").strip().lower()
    value = default if raw == "" else raw
    if value not in {"alns_plain", "alns_late_phase"}:
        raise ValueError("variant must be alns_plain or alns_late_phase")
    return value


def _ask_mode(default: str = "repair_single") -> str:
    raw = input(
        f"Operation mode repair_single/add_seed/add_instance [{default}]: "
    ).strip().lower()
    value = default if raw == "" else raw
    if value not in {"repair_single", "add_seed", "add_instance"}:
        raise ValueError("mode must be repair_single, add_seed, or add_instance")
    return value


def _ask_merge_mode(default: str) -> str:
    raw = input(f"Merge mode add/replace [{default}]: ").strip().lower()
    value = default if raw == "" else raw
    if value not in {"add", "replace"}:
        raise ValueError("merge mode must be add or replace")
    return value


def _read_config(path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def _read_summary(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _benchmark_root() -> Path:
    return Path(__file__).resolve().parent


def _variant_dir(variant: str) -> Path:
    return _benchmark_root() / variant


def _list_run_names_lex(variant: str) -> List[str]:
    root = _variant_dir(variant)
    if not root.exists():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(names)


def _latest_run_name_lex(variant: str) -> str:
    names = _list_run_names_lex(variant)
    return names[-1] if names else ""


def _latest_variant_lex(default: str = "alns_plain") -> str:
    candidates: List[Tuple[str, str]] = []
    for variant in ("alns_plain", "alns_late_phase"):
        latest = _latest_run_name_lex(variant)
        if latest != "":
            candidates.append((latest, variant))
    if not candidates:
        return default
    candidates.sort()
    return candidates[-1][1]


def _specs_readback(
    variant: str,
    run_dir: Path,
    summary_rows: Sequence[Dict[str, Any]],
    config: Dict[str, str],
) -> None:
    summary_path = run_dir / "summary.csv"
    logs_dir = run_dir / "logs"
    run_names = _list_run_names_lex(variant)
    print("\nSelected benchmark specs:")
    print(f"  variant={variant}")
    print(f"  run_dir={run_dir}")
    print(
        f"  lex_latest_run_for_variant={run_names[-1] if run_names else 'n/a'} "
        f"(default rule: lexicographic latest)"
    )
    print(f"  summary_exists={summary_path.exists()}")
    print(f"  config_exists={(run_dir / 'config.txt').exists()}")
    if config:
        for key in (
            "run_id",
            "variant",
            "n_instances",
            "n_seeds",
            "timeout_seconds",
            "solver_setup",
            "repair_weight_init",
        ):
            if key in config:
                print(f"  config.{key}={config[key]}")

    if summary_rows:
        instances = sorted(
            {str(r.get("instance", "")).strip() for r in summary_rows if str(r.get("instance", "")).strip() != ""},
            key=_instance_sort_key,
        )
        seeds = sorted(
            {_to_int(r.get("seed")) for r in summary_rows if _to_int(r.get("seed")) is not None}
        )
        timeout_vals = sorted(
            {_to_float(r.get("timeout_seconds")) for r in summary_rows if _to_float(r.get("timeout_seconds")) is not None}
        )
        status_counts: Dict[str, int] = {}
        for row in summary_rows:
            status = str(row.get("status", "")).strip() or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
        print(f"  summary_rows={len(summary_rows)}")
        print(f"  summary_unique_instances={len(instances)}")
        print(f"  summary_unique_seeds={len(seeds)}")
        print(f"  summary_timeouts={timeout_vals if timeout_vals else 'n/a'}")
        print(f"  summary_status_counts={status_counts}")
    else:
        print("  summary_rows=0")

    if logs_dir.exists():
        log_files = sorted([p.name for p in logs_dir.glob("*.log")])
        print(f"  logs_count={len(log_files)}")
    else:
        print("  logs_count=0")


def _derive_instances_from_summary_or_config(
    summary_rows: Sequence[Dict[str, Any]],
    config: Dict[str, str],
) -> List[str]:
    from_summary = sorted(
        {str(r.get("instance", "")).strip() for r in summary_rows if str(r.get("instance", "")).strip() != ""},
        key=_instance_sort_key,
    )
    if from_summary:
        return from_summary

    n_cfg = _to_int(config.get("n_instances"))
    pool = sorted(
        [p.stem for p in (BASE_DIR / "Instances2000").glob("Example*.txt")],
        key=_instance_sort_key,
    )
    if n_cfg is not None and n_cfg > 0:
        return pool[: min(n_cfg, len(pool))]
    return pool


def _derive_seeds_from_summary_or_config(
    summary_rows: Sequence[Dict[str, Any]],
    config: Dict[str, str],
) -> List[int]:
    from_summary = sorted(
        {_to_int(r.get("seed")) for r in summary_rows if _to_int(r.get("seed")) is not None}
    )
    if from_summary:
        return [int(s) for s in from_summary if s is not None]

    n_cfg = _to_int(config.get("n_seeds"))
    if n_cfg is not None and n_cfg > 0:
        return list(range(n_cfg))
    return [0]


def _target_pairs(
    mode: str,
    summary_rows: Sequence[Dict[str, Any]],
    config: Dict[str, str],
) -> List[Tuple[str, int]]:
    if mode == "repair_single":
        ex_num = _ask_int("Instance number (ExampleN)", default=0, minimum=0)
        seed = _ask_int("Seed", default=0, minimum=0)
        return [(f"Example{ex_num}", seed)]

    if mode == "add_seed":
        seed = _ask_int("Seed to add", default=0, minimum=0)
        instances = _derive_instances_from_summary_or_config(summary_rows, config)
        if not instances:
            raise ValueError("could not infer instance list from summary/config")
        print(f"Adding seed={seed} for {len(instances)} instances.")
        return [(instance, seed) for instance in instances]

    if mode == "add_instance":
        ex_num = _ask_int("Instance number to add (ExampleN)", default=0, minimum=0)
        instance = f"Example{ex_num}"
        seeds = _derive_seeds_from_summary_or_config(summary_rows, config)
        if not seeds:
            raise ValueError("could not infer seed list from summary/config")
        print(f"Adding instance={instance} for seeds={seeds}.")
        return [(instance, seed) for seed in seeds]

    raise ValueError(f"unsupported mode: {mode}")


def _rows_sort_key(row: Dict[str, Any]) -> tuple[int, int]:
    idx = _to_int(row.get("instance_index"))
    seed = _to_int(row.get("seed"))
    return (idx if idx is not None else 10**9, seed if seed is not None else 10**9)


def _instance_tag(index: int | None, width: int) -> str:
    if index is None:
        return "na"
    return f"{index:0{width}d}"


def main() -> None:
    default_variant = _latest_variant_lex(default="alns_plain")
    variant = _ask_variant(default=default_variant)
    run_names = _list_run_names_lex(variant)
    default_run = run_names[-1] if run_names else datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = _ask_str("Run folder name", default_run)

    run_dir = _variant_dir(variant) / run_name
    summary_path = run_dir / "summary.csv"
    config_path = run_dir / "config.txt"
    logs_dir = run_dir / "logs"

    summary_rows = _read_summary(summary_path)
    config = _read_config(config_path)
    _specs_readback(variant=variant, run_dir=run_dir, summary_rows=summary_rows, config=config)

    default_timeout = (
        _to_float(config.get("timeout_seconds"))
        or (_to_float(summary_rows[0].get("timeout_seconds")) if summary_rows else None)
        or 60.0
    )
    timeout_seconds = _ask_float("Timeout per solve (seconds)", default=float(default_timeout), minimum=0.1)
    mode = _ask_mode(default="repair_single")
    default_merge = "replace" if mode == "repair_single" else "add"
    merge_mode = _ask_merge_mode(default=default_merge)

    pairs = _target_pairs(mode=mode, summary_rows=summary_rows, config=config)
    if not pairs:
        raise ValueError("no targets selected")
    target_keys = {(instance, seed) for instance, seed in pairs}

    if merge_mode == "replace":
        before = len(summary_rows)
        summary_rows = [
            row
            for row in summary_rows
            if (str(row.get("instance", "")).strip(), _to_int(row.get("seed"))) not in target_keys
        ]
        print(f"Removed {before - len(summary_rows)} existing row(s) due to replace mode.")

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_indexes = [
        _instance_index_from_stem(instance)
        for instance, _seed in pairs
    ]
    all_indexes.extend([_to_int(r.get("instance_index")) for r in summary_rows])
    known_indexes = [idx for idx in all_indexes if idx is not None]
    index_width = max(2, len(str(max(known_indexes)))) if known_indexes else 2

    total = len(pairs)
    for i, (instance, seed) in enumerate(pairs, start=1):
        instance_path = BASE_DIR / "Instances2000" / f"{instance}.txt"
        if not instance_path.exists():
            raise FileNotFoundError(f"instance not found: {instance_path}")
        print(f"[{i}/{total}] variant={variant} instance={instance} seed={seed}")
        row, log_text = _run_single(
            instance_path=instance_path,
            seed=seed,
            timeout_seconds=timeout_seconds,
            variant=variant,
        )
        summary_rows.append(row)
        tag = _instance_tag(_instance_index_from_stem(instance), index_width)
        log_path = logs_dir / f"{tag}_{instance}_seed{seed}.log"
        log_path.write_text(log_text, encoding="utf-8")
        print(
            f"[{i}/{total}] status={row['status']} runtime={row['runtime_seconds']:.3f}s "
            f"objective={row['objective']} log={log_path.name}"
        )

    summary_rows = sorted(summary_rows, key=_rows_sort_key)

    if summary_path.exists():
        backup = summary_path.with_suffix(summary_path.suffix + f".bak_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(summary_path, backup)
        print(f"Created backup: {backup}")

    _write_summary(summary_path, summary_rows)
    print(f"Updated summary: {summary_path}")
    print(f"Total rows now: {len(summary_rows)}")


if __name__ == "__main__":
    main()
