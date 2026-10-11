"""Independent bookkeeping audit for the train/validation calibration run."""
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np

BASE = Path(__file__).resolve().parent
EXP = BASE/"experiments"


def rows(name):
    with (EXP/name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((EXP/"calibration_manifest.json").read_text(encoding="utf-8"))
    prespec = json.loads((EXP/"calibration_prespecification.json").read_text(encoding="utf-8"))
    fit = rows("calibration_fit_results.csv")
    result = rows("calibration_results.csv")
    train_predictions = rows("calibration_train_predictions.csv")
    sample = np.load(EXP/"calibration_validation_samples.npz")
    assert sha(EXP/manifest["numerical_source_snapshot"]) == manifest["numerical_source_sha256"]
    assert set(int(row["T"]) for row in fit) == {32, 128, 512, 2048}
    assert all(sum(int(row["T"]) == T for row in fit) == 5 for T in (32, 128, 512, 2048))
    assert len(train_predictions) == 40
    assert len(result) == 43
    assert sum(row["policy"].startswith("oracle_") for row in result) == 3
    assert sum(row["policy"] in ("source", "energy") for row in result) == 40
    ids = sample["candidate_ids"].tolist()
    energy = sample["energy"]
    assert energy.shape[0] == 30000 and energy.shape[1] == len(ids) == 6
    assert ids[2] == "oracle_energy"
    oracle_mean = energy[:, 2].mean()
    by_id = {row["candidate_id"]: row for row in result}
    # Every candidate row must reproduce the held-out regret from the saved
    # energy mean, including duplicate learned maps.
    for row in result:
        levels = tuple(int(value) for value in row["map_levels"].split(";"))
        map_index = next(index for index, candidate_levels in enumerate(sample["candidate_levels"])
                         if tuple(candidate_levels.tolist()) == levels)
        expected = math.sqrt(float(energy[:, map_index].mean())/float(oracle_mean))-1
        assert abs(expected-float(row["validation_regret_to_oracle_energy"])) < 2e-12
    assert manifest["candidate_count"] == 43
    assert manifest["unique_map_count"] == 6
    assert prespec["validation_trials"] == 30000
    assert prespec["training_replicates"] == 5
    print("PASS: calibration fit counts, candidate retention, held-out dimensions, and all regret rows")


if __name__ == "__main__":
    main()
