import json
import sys
from pathlib import Path

import yaml

from atlaswm.cli import evaluate_main, train_main


ROOT = Path(__file__).resolve().parents[1]


def test_training_and_evaluation_entry_points(tmp_path: Path, monkeypatch, capsys):
    config = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text(encoding="utf-8"))
    config["output"]["dir"] = str(tmp_path / "run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["atlaswm-train", "--config", str(config_path)])
    train_main()
    checkpoint = tmp_path / "run" / "checkpoint_last.pt"
    assert checkpoint.is_file()
    assert (tmp_path / "run" / "evaluation.json").is_file()
    assert (tmp_path / "run" / "resolved_config.yaml").is_file()
    assert (tmp_path / "run" / "system.json").is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "atlaswm-evaluate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--max-batches",
            "1",
        ],
    )
    evaluate_main()
    output = capsys.readouterr().out
    assert "one_step_mse" in output
