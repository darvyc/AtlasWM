"""Run matched multi-seed AtlasWM comparisons and aggregate raw outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from atlaswm.config import load_config, validate_config

METHODS = {
    "prediction_only": ({"name": "none"}, 0.0),
    "covariance": ({"name": "covariance"}, 0.1),
    "full_gaussian_mmd": ({"name": "full_gaussian_mmd", "beta": 1.0}, 0.1),
    "iid_haar_ecf": (
        {
            "name": "atlas",
            "design": "haar",
            "n_haar_projections": 1024,
            "rotation_mode": "none",
            "target": "gaussian",
            "kernel": "single",
            "lambda_": 1.0,
            "n_knots": 33,
        },
        0.1,
    ),
    "atlas": ({"name": "atlas"}, 0.1),
}


def build_method_config(base: dict, method: str, seed: int, output_dir: Path) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    config = deepcopy(base)
    regularizer, default_weight = METHODS[method]
    if method == "atlas":
        config["regularizer"]["name"] = "atlas"
    else:
        config["regularizer"] = deepcopy(regularizer)
    config["seed"] = seed
    config["trainer"]["lambda_reg"] = default_weight
    config["output"]["dir"] = str(output_dir)
    validate_config(config)
    return config


def aggregate(records: list[dict]) -> list[dict]:
    methods = sorted({record["method"] for record in records})
    metric_names = sorted(
        key
        for key in records[0]
        if key not in {"method", "seed", "run_dir"}
        and isinstance(records[0][key], (int, float))
    )
    rows = []
    for method in methods:
        method_records = [record for record in records if record["method"] == method]
        row = {"method": method, "seeds": len(method_records)}
        for metric in metric_names:
            values = [float(record[metric]) for record in method_records if metric in record]
            if not values:
                continue
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = (
                1.96 * row[f"{metric}_std"] / len(values) ** 0.5 if len(values) > 1 else 0.0
            )
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
        common_metrics = sorted(
            key
            for key in baseline_by_seed[shared_seeds[0]]
            if key not in {"method", "seed", "run_dir"}
            and isinstance(baseline_by_seed[shared_seeds[0]][key], (int, float))
            and key in method_by_seed[shared_seeds[0]]
        )
        for metric in common_metrics:
            differences = [
                float(method_by_seed[seed][metric])
                - float(baseline_by_seed[seed][metric])
                for seed in shared_seeds
            ]
            standard_deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
            rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "seeds": len(differences),
                    "mean_difference": statistics.fmean(differences),
                    "std_difference": standard_deviation,
                    "ci95_difference": (
                        1.96 * standard_deviation / len(differences) ** 0.5
                        if len(differences) > 1
                        else 0.0
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/benchmark")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 53, 71])
    args = parser.parse_args()
    base = load_config(args.config)
    root = Path(args.output)
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    records = []
    raw_path = root / "runs.jsonl"
    raw_path.unlink(missing_ok=True)
    for method in args.methods:
        for seed in args.seeds:
            run_dir = root / "runs" / method / f"seed_{seed}"
            config = build_method_config(base, method, seed, run_dir)
            config_path = config_dir / f"{method}_seed{seed}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            subprocess.run(
                [sys.executable, "scripts/train.py", "--config", str(config_path)],
                check=True,
            )
            evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
            record = {"method": method, "seed": seed, "run_dir": str(run_dir), **evaluation}
            records.append(record)
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    rows = aggregate(records)
    comparisons = paired_comparisons(records)
    (root / "aggregate.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if comparisons:
        comparison_fields = sorted({key for row in comparisons for key in row})
        with (root / "paired_comparisons.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison_fields)
            writer.writeheader()
            writer.writerows(comparisons)
    print(json.dumps({"aggregate": rows, "paired": comparisons}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
