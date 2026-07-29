"""Matched AtlasWM comparisons with nested regularizer selection and small-sample statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml

METHODS = {
    "prediction_only": {"regularizer": {"name": "none"}, "lambdas": [0.0]},
    "covariance": {"regularizer": {"name": "covariance"}},
    "full_gaussian_mmd": {"regularizer": {"name": "full_gaussian_mmd", "beta": 1.0}},
    "iid_haar_ecf": {
        "regularizer": {
            "name": "atlas",
            "design": "haar",
            "n_haar_projections": 1024,
            "rotation_mode": "none",
            "target": "gaussian",
            "kernel": "single",
            "lambda_": 1.0,
            "n_knots": 33,
        }
    },
    "orthogonal_ecf": {
        "regularizer": {
            "name": "atlas",
            "design": "cross_polytope",
            "deduplicate_antipodes": True,
            "rotation_mode": "haar",
            "target": "gaussian",
            "kernel": "single",
            "lambda_": 1.0,
            "n_knots": 33,
        }
    },
    "atlas": {"regularizer": {"name": "atlas"}},
}

_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    24: 2.064,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        return 0.0
    for threshold in sorted(_T_CRITICAL_975):
        if degrees_of_freedom <= threshold:
            return _T_CRITICAL_975[threshold]
    return 1.960


def confidence_interval_95(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return t_critical_975(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))


def hedges_g_paired(differences: list[float]) -> float | None:
    if len(differences) < 2:
        return 0.0
    standard_deviation = statistics.stdev(differences)
    if standard_deviation == 0.0:
        return None if any(differences) else 0.0
    raw = statistics.fmean(differences) / standard_deviation
    correction = 1.0 - 3.0 / max(4.0 * len(differences) - 5.0, 1.0)
    return correction * raw


def build_method_config(
    base: dict,
    method: str,
    seed: int,
    output_dir: Path,
    weight: float,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    config = deepcopy(base)
    config["regularizer"] = deepcopy(METHODS[method]["regularizer"])
    config["seed"] = int(seed)
    config["trainer"]["lambda_reg"] = float(weight)
    config["output"]["dir"] = str(output_dir)
    return config


def aggregate(records: list[dict]) -> list[dict]:
    rows = []
    methods = sorted({record["method"] for record in records})
    for method in methods:
        method_records = [record for record in records if record["method"] == method]
        numeric_names = sorted(
            set.intersection(
                *[
                    {key for key, value in record.items() if isinstance(value, (int, float))}
                    for record in method_records
                ]
            )
            - {"seed", "lambda_reg"}
        )
        row = {"method": method, "seeds": len(method_records)}
        for metric in numeric_names:
            values = [float(record[metric]) for record in method_records]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = confidence_interval_95(values)
        rows.append(row)
    return rows


def paired_comparisons(records: list[dict], baseline: str = "prediction_only") -> list[dict]:
    baseline_by_seed = {
        record["seed"]: record for record in records if record["method"] == baseline
    }
    rows = []
    for method in sorted({record["method"] for record in records} - {baseline}):
        method_by_seed = {
            record["seed"]: record for record in records if record["method"] == method
        }
        shared_seeds = sorted(set(baseline_by_seed) & set(method_by_seed))
        if not shared_seeds:
            continue
        metrics = sorted(
            set.intersection(
                *[
                    {
                        key
                        for key, value in baseline_by_seed[seed].items()
                        if isinstance(value, (int, float))
                        and key not in {"seed", "lambda_reg"}
                        and isinstance(method_by_seed[seed].get(key), (int, float))
                    }
                    for seed in shared_seeds
                ]
            )
        )
        for metric in metrics:
            differences = [
                float(method_by_seed[seed][metric])
                - float(baseline_by_seed[seed][metric])
                for seed in shared_seeds
            ]
            rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "seeds": len(differences),
                    "mean_difference": statistics.fmean(differences),
                    "std_difference": (
                        statistics.stdev(differences) if len(differences) > 1 else 0.0
                    ),
                    "ci95_difference": confidence_interval_95(differences),
                    "hedges_g_paired": hedges_g_paired(differences),
                    "positive_difference_fraction": (
                        sum(value > 0 for value in differences) / len(differences)
                    ),
                }
            )
    return rows


def select_weight(records: list[dict], metric: str, direction: str) -> tuple[float, list[dict]]:
    candidates = sorted({float(record["lambda_reg"]) for record in records})
    summaries = []
    for candidate in candidates:
        values = [
            float(record[metric])
            for record in records
            if float(record["lambda_reg"]) == candidate
        ]
        summaries.append(
            {
                "lambda_reg": candidate,
                "metric": metric,
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci95": confidence_interval_95(values),
                "seeds": len(values),
            }
        )
    reverse = direction == "max"
    chosen = sorted(summaries, key=lambda item: item["mean"], reverse=reverse)[0]
    return float(chosen["lambda_reg"]), summaries


def run_training(config: dict, config_path: Path, run_dir: Path) -> dict:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        check=True,
    )
    elapsed = time.perf_counter() - started
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    return {**evaluation, "wall_clock_seconds": elapsed}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/benchmark")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=[101, 103, 107])
    parser.add_argument(
        "--evaluation-seeds",
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 23, 37, 53, 71],
    )
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.1, 0.3],
    )
    parser.add_argument("--selection-metric", default="one_step_mse")
    parser.add_argument("--selection-direction", choices=["min", "max"], default="min")
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    selection_report = {}
    chosen_weights = {}

    for method in args.methods:
        weights = METHODS[method].get("lambdas", args.lambda_grid)
        selection_records = []
        for weight in weights:
            for seed in args.selection_seeds:
                run_dir = (
                    root
                    / "selection_runs"
                    / method
                    / f"lambda_{weight:g}"
                    / f"seed_{seed}"
                )
                config = build_method_config(base, method, seed, run_dir, weight)
                config_path = (
                    root
                    / "configs"
                    / f"selection_{method}_lambda{weight:g}_seed{seed}.yaml"
                )
                result = run_training(config, config_path, run_dir)
                selection_records.append(
                    {
                        "method": method,
                        "seed": seed,
                        "lambda_reg": weight,
                        "run_dir": str(run_dir),
                        **result,
                    }
                )
        chosen, summaries = select_weight(
            selection_records,
            args.selection_metric,
            args.selection_direction,
        )
        chosen_weights[method] = chosen
        selection_report[method] = {
            "chosen_lambda_reg": chosen,
            "candidates": summaries,
        }

    records = []
    raw_path = root / "runs.jsonl"
    raw_path.unlink(missing_ok=True)
    for method in args.methods:
        weight = chosen_weights[method]
        for seed in args.evaluation_seeds:
            run_dir = root / "runs" / method / f"seed_{seed}"
            config = build_method_config(base, method, seed, run_dir, weight)
            config_path = root / "configs" / f"final_{method}_seed{seed}.yaml"
            result = run_training(config, config_path, run_dir)
            record = {
                "method": method,
                "seed": seed,
                "lambda_reg": weight,
                "run_dir": str(run_dir),
                **result,
            }
            records.append(record)
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    rows = aggregate(records)
    comparisons = paired_comparisons(records)
    (root / "selection.json").write_text(
        json.dumps(selection_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "aggregate.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(root / "aggregate.csv", rows)
    write_csv(root / "paired_comparisons.csv", comparisons)
    print(
        json.dumps(
            {"selection": selection_report, "aggregate": rows, "paired": comparisons},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
