"""Predeclared extra independent vectors after the first paired-group audit.

The first 10,000-vector streams are immutable. Each 32x32 profile receives
50,000 additional independently labeled vectors. Primary estimates pool all
60,000 vectors; first-only, additional-only and combined results are retained.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import time

import numpy as np
from scipy.fft import dstn

import elliptic2d_application as experiment
import poisson_application as codec
import run_revision as core

OUT = Path(__file__).resolve().parent
EXTRA_TRIALS = 50000
BOOTSTRAP = 300


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
        for key in ("mesh", "n", "block_size", "blocks", "trials"):
            row[key] = int(row[key])
    return rows


def bootstrap_rms(values, reference, labels):
    generator = core.rng_for("elliptic2d_replication", "bootstrap", *labels)
    estimates = np.empty(BOOTSTRAP)
    for j in range(BOOTSTRAP):
        draws = generator.integers(len(values), size=len(values))
        estimates[j] = math.sqrt(float(values[draws].mean())/reference)
    return np.percentile(estimates, [2.5, 97.5])


def estimate(row, values, label, profile):
    result = dict(row)
    rms = math.sqrt(float(values.mean())/row["reference_expected_energy"])
    low, high = bootstrap_rms(values, row["reference_expected_energy"], (profile, row["id"], label))
    result.update(dataset=label, trials=len(values), mc_energy_mse=float(values.mean()),
        mc_relative_rms=rms, mc_relative_rms_low=float(low), mc_relative_rms_high=float(high),
        exact_in_mc_interval=bool(low <= row["exact_relative_rms"] <= high),
        relative_prediction_error=rms/row["exact_relative_rms"]-1)
    return result


def paired_policies(primary_rows, saved):
    comparisons = []
    for (mesh, profile), (energy, ids) in saved.items():
        pairs = []
        for size in experiment.BLOCK_SIZES:
            baseline = f"b{size}_contiguous_uniform"
            pairs.extend((baseline, candidate) for candidate in ids
                         if candidate != baseline and candidate.startswith(f"b{size}_"))
            for layout in ("contiguous", "interleaved"):
                source, exact = f"b{size}_{layout}_source", f"b{size}_{layout}_energy"
                if source in ids and exact in ids:
                    pairs.append((source, exact))
        pairs.append(("b8_contiguous_uniform", "b32_contiguous_energy"))
        pairs.append(("b8_interleaved_uniform", "b32_interleaved_energy"))
        for baseline, candidate in pairs:
            if baseline not in ids or candidate not in ids:
                continue
            first, second = energy[:, ids.index(baseline)], energy[:, ids.index(candidate)]
            generator = core.rng_for("elliptic2d_replication", "paired", mesh, profile, baseline, candidate)
            values = []
            for _ in range(BOOTSTRAP):
                draws = generator.integers(len(energy), size=len(energy))
                values.append(1-math.sqrt(float(second[draws].mean()/first[draws].mean())))
            low, high = np.percentile(values, [2.5, 97.5])
            popfirst = next(row for row in primary_rows if row["mesh"] == mesh and row["profile"] == profile and row["id"] == baseline)
            popsecond = next(row for row in primary_rows if row["mesh"] == mesh and row["profile"] == profile and row["id"] == candidate)
            comparisons.append(dict(mesh=mesh, profile=profile, baseline=baseline, candidate=candidate,
                exact_fractional_rms_reduction=1-popsecond["exact_relative_rms"]/popfirst["exact_relative_rms"],
                mc_fractional_rms_reduction=1-math.sqrt(float(second.mean()/first.mean())),
                mc_fractional_rms_reduction_low=float(low), mc_fractional_rms_reduction_high=float(high),
                bootstrap_replicates=BOOTSTRAP, trials=len(energy)))
    return comparisons


def run():
    began = time.time()
    initial_manifest = json.loads((OUT/"elliptic2d_manifest.json").read_text(encoding="utf-8"))
    configuration_record = json.loads((OUT/"elliptic2d_prespecification.json").read_text(encoding="utf-8"))
    initial_rows = read_rows(OUT/"elliptic2d_results.csv")
    prespecification = dict(date="2026-10-10 Asia/Shanghai", extra_trials=EXTRA_TRIALS,
        profile_mesh_pairs=[[32, profile] for profile in experiment.PROFILES],
        first_trials=10000, pooled_trials=60000, bootstrap_replicates=BOOTSTRAP,
        rationale="The first32x32hotspot paired group had predominantly positive2-3standard-error RMS deviations and26/40pointwise intervals covered. Increase precision in BOTH32x32 profiles, retaining the initial streams and every declared configuration.",
        selection="One extra independent stream per profile; no seed search, no replacement or sample-dependent deletion; primary uses all60000vectors.",
        stream_labels={profile: ["elliptic2d", "replication", "input", 32, profile, 1]
                       for profile in experiment.PROFILES},
        master_seed=core.MASTER_SEED, rng="PCG64 from SHA256-labeled SeedSequence",
        initial_source_sha256=initial_manifest["numerical_source_sha256"],
        experiment_dependency_sha256=experiment.sha(OUT/"elliptic2d_application.py"),
        replication_source_sha256=experiment.sha(Path(__file__)),
        initial_results_sha256=experiment.sha(OUT/"elliptic2d_results.csv"),
        initial_sample_hashes={profile: experiment.sha(OUT/f"elliptic2d_samples_m32_{profile}.npz")
                               for profile in experiment.PROFILES},
        configurations={profile: configuration_record["configurations"][f"m32_{profile}"]
                        for profile in experiment.PROFILES})
    (OUT/"elliptic2d_replication_prespecification.json").write_text(
        json.dumps(prespecification, indent=2)+"\n", encoding="utf-8")
    primary_rows = [dict(row, dataset="first") for row in initial_rows if row["mesh"] == 16]
    history_rows = [dict(row, dataset="first") for row in initial_rows]
    saved = {}
    for profile in experiment.PROFILES:
        mesh, n = 32, 1024
        scales = experiment.source_scales(mesh, profile)
        h, green, eigenvalues, _ = experiment.geometry(mesh)
        configs = prespecification["configurations"][profile]
        ids = [cfg["id"] for cfg in configs]
        generator = core.rng_for(*prespecification["stream_labels"][profile])
        energy = np.empty((EXTRA_TRIALS, len(configs)))
        minexps, maxexps = np.full(len(configs), 10000), np.full(len(configs), -10000)
        input_hash = hashlib.sha256()
        batch = 500000//n
        for begin in range(0, EXTRA_TRIALS, batch):
            inputs = generator.lognormal(size=(min(batch, EXTRA_TRIALS-begin), n))*scales
            input_hash.update(inputs.tobytes(order="C"))
            for j, cfg in enumerate(configs):
                errors, emin, emax = codec.bounded_quantize(inputs, cfg)
                transformed = dstn(errors.reshape(-1, mesh, mesh), type=1, norm="ortho", axes=(1, 2))
                energy[begin:begin+len(inputs), j] = h*h*np.sum(transformed**2/eigenvalues, axis=(1, 2))
                minexps[j], maxexps[j] = min(minexps[j], emin), max(maxexps[j], emax)
        np.savez_compressed(OUT/f"elliptic2d_replication_samples_m32_{profile}.npz",
            energy=energy, codec_ids=np.array(ids), exponent_min=minexps, exponent_max=maxexps,
            input_sha256=np.array(input_hash.hexdigest()))
        first = np.load(OUT/f"elliptic2d_samples_m32_{profile}.npz")
        if first["codec_ids"].tolist() != ids:
            raise RuntimeError("first/extra codec order mismatch")
        combined = np.concatenate((first["energy"], energy))
        saved[mesh, profile] = combined, ids
        np.savez_compressed(OUT/f"elliptic2d_combined_samples_m32_{profile}.npz", energy=combined,
            codec_ids=np.array(ids), first_vectors=len(first["energy"]), extra_vectors=EXTRA_TRIALS)
        for j, name in enumerate(ids):
            row = next(row for row in initial_rows if row["mesh"] == mesh and row["profile"] == profile and row["id"] == name)
            replica = estimate(row, energy[:, j], "replication", profile)
            combined_row = estimate(row, combined[:, j], "combined", profile)
            history_rows.extend((replica, combined_row))
            primary_rows.append(combined_row)
        codec.write_csv(OUT/"elliptic2d_all_stream_results.csv", history_rows)
        print(f"Replicated {profile}: first10000 + independent50000 = pooled60000; all14codecs retained.", flush=True)
    for profile in experiment.PROFILES:
        if experiment.sha(OUT/f"elliptic2d_samples_m32_{profile}.npz") != prespecification["initial_sample_hashes"][profile]:
            raise RuntimeError("initial paired stream was modified")
    for profile in experiment.PROFILES:
        first = np.load(OUT/f"elliptic2d_samples_m16_{profile}.npz")
        saved[16, profile] = first["energy"], first["codec_ids"].tolist()
    codec.write_csv(OUT/"elliptic2d_primary_results.csv", primary_rows)
    codec.write_csv(OUT/"elliptic2d_primary_paired_comparisons.csv", paired_policies(primary_rows, saved))
    codec.write_csv(OUT/"elliptic2d_primary_rankings.csv", experiment.ranking(primary_rows))
    manifest = dict(prespecification, duration_seconds=time.time()-began,
        primary_configuration_count=len(primary_rows),
        primary_exact_rms_ci_coverage=sum(row["exact_in_mc_interval"] for row in primary_rows),
        primary_max_relative_rms_prediction_error=max(abs(row["relative_prediction_error"]) for row in primary_rows),
        first_exact_rms_ci_coverage=initial_manifest["exact_rms_ci_coverage"],
        first_max_relative_rms_prediction_error=initial_manifest["maximum_relative_rms_prediction_error"],
        group_summary=[dict(mesh=mesh, profile=profile,
            configurations=len(group := [row for row in primary_rows if row["mesh"] == mesh and row["profile"] == profile]),
            coverage=sum(row["exact_in_mc_interval"] for row in group),
            max_relative_prediction_error=max(abs(row["relative_prediction_error"]) for row in group))
                       for mesh in experiment.MESHES for profile in experiment.PROFILES])
    names = ["elliptic2d_replication.py", "elliptic2d_replication_prespecification.json",
        "elliptic2d_all_stream_results.csv", "elliptic2d_primary_results.csv",
        "elliptic2d_primary_paired_comparisons.csv", "elliptic2d_primary_rankings.csv"]
    names.extend(path.name for path in sorted(OUT.glob("elliptic2d_replication_samples_*.npz")))
    names.extend(path.name for path in sorted(OUT.glob("elliptic2d_combined_samples_*.npz")))
    manifest["files"] = {name: experiment.sha(OUT/name) for name in names}
    (OUT/"elliptic2d_replication_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Primary40configs coverage {manifest['primary_exact_rms_ci_coverage']}/40; maxRMSdiscrepancy"
          f" {100*manifest['primary_max_relative_rms_prediction_error']:.3f}%; initialstream preserved.", flush=True)


if __name__ == "__main__":
    run()
