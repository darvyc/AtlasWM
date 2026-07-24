"""Convert a compact NPZ archive into memory-mappable NPY trajectory arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--obs-key", default="obs")
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--state-key")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as archive:
        observations = np.asarray(archive[args.obs_key])
        actions = np.asarray(archive[args.action_key])
        states = None if args.state_key is None else np.asarray(archive[args.state_key])
    np.save(output / "observations.npy", observations, allow_pickle=False)
    np.save(output / "actions.npy", actions, allow_pickle=False)
    if states is not None:
        np.save(output / "states.npy", states, allow_pickle=False)
    metadata = {
        "observations_shape": list(observations.shape),
        "observations_dtype": str(observations.dtype),
        "actions_shape": list(actions.shape),
        "actions_dtype": str(actions.dtype),
        "states_shape": None if states is None else list(states.shape),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
