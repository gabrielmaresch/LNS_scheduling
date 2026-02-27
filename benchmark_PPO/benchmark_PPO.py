from __future__ import annotations

import csv
from datetime import datetime
import io
from pathlib import Path
import re
import sys
from contextlib import redirect_stdout


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from DRL_PPO_run_checkpoint_auto import run_checkpoint


def _instance_index_from_stem(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None


def _instance_sort_key(stem: str) -> tuple[int, str]:
    idx = _instance_index_from_stem(stem)
    if idx is not None:
        return (idx, stem)
    return (10**9, stem)


def _parse_list_csv(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _read_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def _variant_root(variant: str) -> Path:
    base = Path(__file__).resolve().parent
    if variant == "latephase":
        return base / "ppo_late_phase"
    if variant == "plain":
        return base / "ppo_plain"
    raise ValueError("variant must be 'plain' or 'latephase'")


def _latest_run_dir(variant: str) -> Path | None:
    root = _variant_root(variant)
    candidates: list[Path] = []
    if not root.exists():
        return None
    for path in root.iterdir():
        if path.is_dir() and (
            (path / "results" / "summary.csv").exists() or (path / "summary.csv").exists()
        ):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _resolve_run_dir_by_name(variant: str, run_name: str) -> tuple[Path, Path, Path]:
    root = _variant_root(variant)
    run_dir = root / run_name
    if (run_dir / "results" / "summary.csv").exists():
        return run_dir, run_dir / "results", run_dir / "config.txt"
    if (run_dir / "summary.csv").exists():
        return run_dir, run_dir, run_dir / "config.txt"
    raise FileNotFoundError(
        f"Run folder '{run_name}' not found under {root} with summary.csv"
    )


def _recent_run_names(variant: str, limit: int = 5) -> list[str]:
    root = _variant_root(variant)
    if not root.exists():
        return []
    runs: list[Path] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        if (run_dir / "results" / "summary.csv").exists() or (run_dir / "summary.csv").exists():
            runs.append(run_dir)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [run.name for run in runs[:limit]]


def _latest_variant() -> str:
    roots = {
        "latephase": _variant_root("latephase"),
        "plain": _variant_root("plain"),
    }
    newest_variant = "latephase"
    newest_time = -1.0
    for variant, root in roots.items():
        if not root.exists():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            if not ((run_dir / "results" / "summary.csv").exists() or (run_dir / "summary.csv").exists()):
                continue
            mtime = run_dir.stat().st_mtime
            if mtime > newest_time:
                newest_time = mtime
                newest_variant = variant
    return newest_variant


def _ask_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return default if raw == "" else raw


def _ask_variant(default: str) -> str:
    raw = input(f"PPO variant plain/latephase [{default}]: ").strip().lower()
    value = default if raw == "" else raw
    if value not in {"plain", "latephase"}:
        raise ValueError("variant must be 'plain' or 'latephase'")
    return value


def _ask_int(prompt: str, default: int, minimum: int = 1) -> int:
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


def _extract_test_instances(row: dict[str, str], pool: list[str]) -> list[str]:
    explicit_test = _parse_list_csv(row.get("test_instances", ""))
    if explicit_test:
        return sorted(explicit_test, key=_instance_sort_key)
    trained = set(_parse_list_csv(row.get("train_instances", "")))
    inferred = [stem for stem in pool if stem not in trained]
    return sorted(inferred, key=_instance_sort_key)


def main() -> None:
    default_variant = _latest_variant()
    variant = _ask_variant(default=default_variant)
    latest_run = _latest_run_dir(variant)
    default_name = latest_run.name if latest_run is not None else ""
    suggestions = _recent_run_names(variant=variant, limit=5)
    if suggestions:
        print(f"Recent {variant} run folder suggestions: " + ", ".join(suggestions))
    selected_name = _ask_str(
        "PPO run folder name",
        default=default_name,
    )
    if selected_name == "":
        raise ValueError("run folder name is required")
    run_dir, results_dir, config_path = _resolve_run_dir_by_name(variant, selected_name)

    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found in {results_dir}")

    config = _read_config(config_path)
    pool_instances = _parse_list_csv(config.get("pool_instances", ""))

    runs_each = _ask_int("Number of runs per test instance", default=10, minimum=1)
    timeout_seconds = _ask_float("Timeout per checkpoint run (seconds)", default=120.0, minimum=0.1)

    out_dir = run_dir / "checkpoint_benchmark" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: (value or "") for key, value in row.items()})

    if not rows:
        raise ValueError(f"No rows in {summary_path}")

    for row in rows:
        checkpoint_name = row.get("checkpoint", "").strip()
        if not checkpoint_name:
            continue
        checkpoint_path = results_dir / checkpoint_name
        if not checkpoint_path.exists():
            print(f"skip checkpoint missing: {checkpoint_path}")
            continue

        seed = row.get("seed", "na")
        test_instances = _extract_test_instances(row=row, pool=pool_instances)
        if not test_instances:
            print(f"skip checkpoint no_test_instances: {checkpoint_name}")
            continue

        log_path = out_dir / f"{Path(checkpoint_name).stem}.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"run_dir={run_dir}\n")
            log_handle.write(f"results_dir={results_dir}\n")
            log_handle.write(f"checkpoint={checkpoint_name}\n")
            log_handle.write(f"seed={seed}\n")
            log_handle.write(f"runs_each={runs_each}\n")
            log_handle.write(f"timeout_seconds={timeout_seconds}\n")
            log_handle.write("test_instances=" + ",".join(test_instances) + "\n\n")

            total = len(test_instances) * runs_each
            counter = 0
            for test_stem in test_instances:
                instance_index = _instance_index_from_stem(test_stem)
                if instance_index is None:
                    log_handle.write(f"skip instance invalid_name={test_stem}\n")
                    continue

                for rep in range(1, runs_each + 1):
                    counter += 1
                    print(
                        f"checkpoint={checkpoint_name} seed={seed} "
                        f"instance={test_stem} run={rep}/{runs_each} [{counter}/{total}]"
                    )
                    log_handle.write(
                        f"\n===== instance={test_stem} run={rep}/{runs_each} checkpoint={checkpoint_name} =====\n"
                    )
                    capture = io.StringIO()
                    try:
                        with redirect_stdout(capture):
                            run_checkpoint(
                                example_number=instance_index,
                                checkpoint_path=checkpoint_path,
                                timeout_seconds=timeout_seconds,
                                show_final_schedule=False,
                            )
                    except Exception as exc:
                        capture.write(f"run_error={type(exc).__name__}: {exc}\n")
                    log_handle.write(capture.getvalue())

        print(f"wrote logfile: {log_path}")

    print(f"Benchmark complete. Logs directory: {out_dir}")


if __name__ == "__main__":
    main()
