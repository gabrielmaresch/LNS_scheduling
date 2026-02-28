from __future__ import annotations

import csv
from datetime import datetime
import io
from pathlib import Path
import random
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


def _ask_mode(default: str = "run") -> str:
    raw = input(f"Benchmark mode run/direct [{default}]: ").strip().lower()
    value = default if raw == "" else raw
    if value not in {"run", "direct"}:
        raise ValueError("benchmark mode must be 'run' or 'direct'")
    return value


def _default_checkpoint_path() -> Path:
    checkpoint_dir = BASE_DIR / "PPO_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return checkpoint_dir / "drl_ppo_checkpoint.pt"


def _resolve_checkpoint_path(raw: str) -> Path:
    if raw == "":
        return _default_checkpoint_path()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    direct = (Path.cwd() / path).resolve()
    if direct.exists():
        return direct
    return (BASE_DIR / "PPO_checkpoints" / path).resolve()


def _discover_instance_stems() -> list[str]:
    instances_dir = BASE_DIR / "Instances1-50"
    if not instances_dir.exists():
        return []
    stems = [path.stem for path in instances_dir.glob("Example*.txt")]
    return sorted(stems, key=_instance_sort_key)


def _ask_test_instances(default_pool: list[str]) -> list[str]:
    preview = ", ".join(default_pool[:8]) if default_pool else "none"
    raw = input(
        "Test instances CSV (blank=all discovered "
        f"{len(default_pool)}; e.g. Example2,Example9) [{preview}...]: "
    ).strip()
    if raw == "":
        return list(default_pool)
    selected = _parse_list_csv(raw)
    return sorted(selected, key=_instance_sort_key)


def _sample_test_instances(pool: list[str], count: int, sample_seed: int) -> list[str]:
    if count > len(pool):
        raise ValueError(f"requested {count} instances but only {len(pool)} discovered")
    rng = random.Random(sample_seed)
    return rng.sample(pool, k=count)


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _run_checkpoint_benchmark(
    *,
    run_dir: Path,
    results_dir: Path,
    checkpoint_name: str,
    checkpoint_path: Path,
    seed: str,
    test_instances: list[str],
    runs_each: int,
    timeout_seconds: float,
    out_dir: Path,
    eval_seeds: list[int] | None = None,
) -> None:
    if not test_instances:
        print(f"skip checkpoint no_test_instances: {checkpoint_name}")
        return

    # Recreate benchmark output dirs in case they were removed externally
    # (e.g., cloud sync/cleanup) during long checkpoint runs.
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{Path(checkpoint_name).stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"run_dir={run_dir}\n")
        log_handle.write(f"results_dir={results_dir}\n")
        log_handle.write(f"checkpoint={checkpoint_name}\n")
        log_handle.write(f"seed={seed}\n")
        log_handle.write(f"runs_each={runs_each}\n")
        log_handle.write(f"timeout_seconds={timeout_seconds}\n")
        if eval_seeds is not None:
            log_handle.write("eval_seeds=" + ",".join(str(s) for s in eval_seeds) + "\n")
        log_handle.write("test_instances=" + ",".join(test_instances) + "\n\n")

        total = len(test_instances) * (len(eval_seeds) if eval_seeds is not None else runs_each)
        counter = 0
        for test_stem in test_instances:
            instance_index = _instance_index_from_stem(test_stem)
            if instance_index is None:
                log_handle.write(f"skip instance invalid_name={test_stem}\n")
                continue

            run_tokens = (
                [f"seed={eval_seed}" for eval_seed in eval_seeds]
                if eval_seeds is not None
                else [f"run={rep}/{runs_each}" for rep in range(1, runs_each + 1)]
            )
            for run_idx, run_token in enumerate(run_tokens, start=1):
                counter += 1
                if eval_seeds is not None:
                    eval_seed = eval_seeds[run_idx - 1]
                    _set_eval_seed(eval_seed)
                print(
                    f"checkpoint={checkpoint_name} seed={seed} "
                    f"instance={test_stem} {run_token} [{counter}/{total}]"
                )
                log_handle.write(
                    f"\n===== instance={test_stem} {run_token} checkpoint={checkpoint_name} =====\n"
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


def _main_run_mode(runs_each: int, timeout_seconds: float) -> None:
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
        _run_checkpoint_benchmark(
            run_dir=run_dir,
            results_dir=results_dir,
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            seed=seed,
            test_instances=test_instances,
            runs_each=runs_each,
            timeout_seconds=timeout_seconds,
            out_dir=out_dir,
        )

    print(f"Benchmark complete. Logs directory: {out_dir}")


def _main_direct_mode(timeout_seconds: float) -> None:
    default_path = _default_checkpoint_path()
    checkpoint_input = input(f"Checkpoint path [{default_path}]: ").strip()
    checkpoint_path = _resolve_checkpoint_path(checkpoint_input)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    default_pool = _discover_instance_stems()
    if not default_pool:
        raise ValueError("no instances discovered in Instances1-50")
    n_random_instances = _ask_int("Number of random test instances", default=10, minimum=1)
    sample_seed = _ask_int("Random-instance sampler seed", default=0, minimum=0)
    test_instances = _sample_test_instances(default_pool, n_random_instances, sample_seed)

    n_eval_seeds = _ask_int("Number of evaluation seeds", default=3, minimum=1)
    eval_seed_start = _ask_int("Evaluation seed start", default=0, minimum=0)
    eval_seeds = [eval_seed_start + i for i in range(n_eval_seeds)]
    print("Selected instances: " + ",".join(test_instances))
    print("Evaluation seeds: " + ",".join(str(s) for s in eval_seeds))

    if not test_instances:
        raise ValueError("no test instances selected")

    out_dir = (
        Path(__file__).resolve().parent
        / "manual_checkpoint_benchmark"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = checkpoint_path.name
    _run_checkpoint_benchmark(
        run_dir=checkpoint_path.parent,
        results_dir=checkpoint_path.parent,
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
        seed="direct",
        test_instances=test_instances,
        runs_each=len(eval_seeds),
        timeout_seconds=timeout_seconds,
        out_dir=out_dir,
        eval_seeds=eval_seeds,
    )

    print(f"Benchmark complete. Logs directory: {out_dir}")


def main() -> None:
    mode = _ask_mode(default="run")
    timeout_seconds = _ask_float("Timeout per checkpoint run (seconds)", default=120.0, minimum=0.1)
    if mode == "direct":
        _main_direct_mode(timeout_seconds=timeout_seconds)
    else:
        runs_each = _ask_int("Number of runs per test instance", default=10, minimum=1)
        _main_run_mode(runs_each=runs_each, timeout_seconds=timeout_seconds)


if __name__ == "__main__":
    main()
