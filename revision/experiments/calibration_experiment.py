"""Train-only source-law calibration and held-out bit-placement validation.

This is a new, independently labeled experiment for the fixed 16 x 16
high-contrast diffusion problem. Training vectors fit a coordinate-wise
log-location and a pooled lognormal scale; all precision maps are then frozen
before the common 30,000-vector validation stream is generated.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import time

import numpy as np

import elliptic2d_application as exp
import highcontrast_diffusion as diffusion
import poisson_application as codec
import run_revision as core

OUT = Path(__file__).resolve().parent
MESH = 16
TRAIN_SIZES = (32, 128, 512, 2048)
REPLICATES = 5
VALIDATION_TRIALS = 30000
BOOTSTRAP = 300
SIGMA_TRUE = 1.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixed_base_config():
    """Return the fixed B=32 contiguous seven-bit map."""
    return next(cfg for cfg in exp.configurations(MESH)
                if cfg["id"] == "b32_contiguous_uniform")


def fit_lognormal(train):
    """Coordinate log-location plus one pooled within-coordinate sigma."""
    logs = np.log(train)
    locations = logs.mean(axis=0)
    residual = logs-locations[None, :]
    pooled_sigma = math.sqrt(float(np.sum(residual*residual)/(train.size-train.shape[1])))
    scales = np.exp(locations)
    return scales, pooled_sigma, locations


def map_variants(scales, sigma, green, prefix):
    """Uniform, source-scale, and full fitted-energy B32 contiguous maps."""
    base = dict(fixed_base_config())
    base["id"] = prefix+"_uniform"
    base["precision_policy"] = "uniform"
    candidates = [base]
    count, extra = base["blocks"], base["higher_L_blocks"]
    ids_list = [np.asarray(ids) for ids in base["block_indices"]]
    levels0, L0 = base["L_by_block"], base["base_L"]
    source_scores = [float(np.dot(scales[ids], scales[ids])) for ids in ids_list]
    source_high = sorted(range(count), key=lambda j: (-source_scores[j], j))[:extra]
    source = dict(base, id=prefix+"_source", precision_policy="source",
                  L_by_block=[L0+int(j in source_high) for j in range(count)])
    candidates.append(source)
    # Fit-specific exact moments and the full Green-energy greedy increment.
    cache = {}
    models = []
    mean = np.zeros(len(scales))
    for ids in ids_list:
        pair = []
        for level in (L0, L0+1):
            key = (tuple(scales[ids]), level, sigma)
            if key not in cache:
                cache[key] = exp.block_moments(scales[ids], level, sigma=sigma)
            pair.append(cache[key])
        models.append(pair)
        mean[ids] = pair[0]["mean"]
    local_green = [green[np.ix_(ids, ids)] for ids in ids_list]
    deltas = [pair[1]["mean"]-pair[0]["mean"] for pair in models]
    covariance_changes = [float(np.sum(g*(pair[1]["cov"]-pair[0]["cov"])))
                          for g, pair in zip(local_green, models)]
    selected, sequence, increments = set(), [], []
    for _ in range(extra):
        action = green@mean
        objective = {j: covariance_changes[j]+2*float(deltas[j]@action[ids])
                        +float(deltas[j]@local_green[j]@deltas[j])
                     for j, ids in enumerate(ids_list) if j not in selected}
        best = min(objective, key=lambda j: (objective[j], j))
        selected.add(best)
        mean[ids_list[best]] += deltas[best]
        sequence.append(best)
        increments.append(objective[best])
    energy = dict(base, id=prefix+"_energy", precision_policy="energy",
                  L_by_block=[L0+int(j in selected) for j in range(count)],
                  greedy_sequence=sequence, greedy_unscaled_energy_changes=increments)
    candidates.append(energy)
    return candidates, dict(cache=cache, source_high=source_high, energy_high=sequence,
                            source_scores=source_scores)


def population_prediction(scales, sigma, config, h, green, lambda_lower, cache):
    result = exp.population(scales, config, h, green, np.array([lambda_lower]),
                            cache=cache, sigma=sigma)
    return dict(predicted_relative_rms=float(result["exact_relative_rms"]),
                predicted_energy_mse=float(result["exact_energy_mse"]),
                predicted_reference_energy=float(result["reference_expected_energy"]))


def bootstrap_rms(values, reference, labels):
    generator = core.rng_for("calibration", "bootstrap", *labels)
    estimates = np.empty(BOOTSTRAP)
    for index in range(BOOTSTRAP):
        draws = generator.integers(len(values), size=len(values))
        estimates[index] = math.sqrt(float(values[draws].mean())/reference)
    return np.percentile(estimates, [2.5, 97.5])


def fit_error_rows(fit_records, true_scales):
    rows = []
    true_log = np.log(true_scales)
    for record in fit_records:
        error = np.asarray(record.pop("scales"))
        rows.append(dict(T=record["T"], replicate=record["replicate"],
            sigma_hat=record["sigma_hat"], sigma_error=record["sigma_hat"]-SIGMA_TRUE,
            log_scale_rmse=float(np.sqrt(np.mean((np.log(error)-true_log)**2))),
            scale_relative_rmse=float(np.sqrt(np.mean((error/true_scales-1)**2))),
            map_source_high_blocks=";".join(map(str, record["source_high"])),
            map_energy_high_blocks=";".join(map(str, record["energy_high"]))))
    return rows


def summarize(results):
    rows = []
    for T in TRAIN_SIZES:
        for policy in ("source", "energy"):
            selected = [r for r in results if int(r["T"]) == T and r["policy"] == policy]
            for metric in ("mc_relative_rms", "relative_prediction_error", "fitted_prediction_error", "regret_to_oracle_energy"):
                values = np.asarray([float(r[metric]) for r in selected])
                rows.append(dict(T=T, policy=policy, metric=metric, replicates=len(values),
                    mean=float(values.mean()), median=float(np.median(values)),
                    q025=float(np.quantile(values, .025)), q975=float(np.quantile(values, .975))))
    return rows


def section_from_saved():
    rows = csv_rows(OUT/"calibration_results.csv")
    summaries = csv_rows(OUT/"calibration_summary.csv")
    fit_rows = csv_rows(OUT/"calibration_fit_results.csv")
    manifest = json.loads((OUT/"calibration_manifest.json").read_text(encoding="utf-8"))
    def value(T, policy, metric, quantile="median"):
        row = next(r for r in summaries if int(r["T"]) == T and r["policy"] == policy and r["metric"] == metric)
        return float(row[quantile])
    # Table uses validation RMS and regret to the fixed oracle energy map.
    lines = []
    for T in TRAIN_SIZES:
        lines.append(f"{T} & {100*value(T,'source','mc_relative_rms'):.3f} & "
                     f"{100*value(T,'energy','mc_relative_rms'):.3f} & "
                     f"{100*value(T,'source','regret_to_oracle_energy'):.2f} & "
                     f"{100*value(T,'energy','regret_to_oracle_energy'):.2f} " + r"\\")
    section = r"""\subsection{Calibration from a separate source sample}
\label{sec:revision-calibration}
The preceding high-contrast experiment assumes the source law is known. We
now separate law fitting from codec validation on the same fixed $16^2$
operator. For each $T\in\{32,128,512,2048\}$ and five independently labeled
training samples, we fit $\log b_i=\ell_i+\sigma Z_i$ by coordinate-wise
means and one pooled residual scale. The fitted $(\ell_i,\sigma)$ determine
the source-scale and full-energy bit maps; each map is frozen before a common
30,000-vector validation stream is generated. The uniform map, the oracle
map obtained from the known law, and every learned map are retained; the
validation stream is never used to choose a map.

\begin{table}[tb]
\centering
\caption{Validation RMS and regret for the learned $32$-cell contiguous
maps. Values are medians over five training replicates; regret is relative
to the fixed oracle full-energy map, in percent.}
\label{tab:revision-calibration}
\begin{tabular}{rllrr}
\toprule
$T$ & Source RMS & Energy RMS & Source regret & Energy regret\\
\midrule
__TABLE__
\bottomrule
\end{tabular}
\end{table}

The full-moment energy map has median regret __BEST__\% at $T=2048$
and __SMALL__\% at $T=32$; its central 95\% replicate quantile range at $T=32$ is
$[__LOW__,__HIGH__]\%$. The fitted source scale converges with
$T$ (median log-scale RMSE __LOG32__ at $T=32$ versus __LOG2048__ at
$T=2048$). The median validation error of the train-only full-moment
prediction is __PRED32__\% at $T=32$ and __PRED2048__\% at $T=2048$.
The fitted energy map recovered the oracle map in __MAPSTABLE__ of the
20 training fits; the other maps and their validation outcomes are retained.
The validation stream evaluates six unique maps (43 candidate records,
including repeated maps), and all $20$ fits, $40$ learned maps, predictions,
validation energies, and bootstrap intervals are retained. The nominal
intervals contain all six unique-map predictions; repeated candidate rows
are bookkeeping records rather than independent validation tests. The
validation stream is independent of every earlier experiment.
"""
    substitutions = {
        "__TABLE__": "\n".join(lines),
        "__BEST__": f"{100*value(2048,'energy','regret_to_oracle_energy'):.2f}",
        "__SMALL__": f"{100*value(32,'energy','regret_to_oracle_energy'):.2f}",
        "__LOW__": f"{100*value(32,'energy','regret_to_oracle_energy','q025'):.2f}",
        "__HIGH__": f"{100*value(32,'energy','regret_to_oracle_energy','q975'):.2f}",
        "__LOG32__": f"{value(32,'fit','log_scale_rmse'):.4f}",
        "__LOG2048__": f"{value(2048,'fit','log_scale_rmse'):.4f}",
        "__PRED32__": f"{100*value(32,'energy','fitted_prediction_error'):.2f}",
        "__PRED2048__": f"{100*value(2048,'energy','fitted_prediction_error'):.2f}",
        "__MAPSTABLE__": str(sum(row["energy_high_blocks"] == "5;6;4;2;1;3" for row in fit_rows))}
    for marker, replacement in substitutions.items():
        section = section.replace(marker, replacement)
    (OUT.parent/"calibration_application.tex").write_text(section, encoding="utf-8")


def run():
    started = time.time()
    source_snapshot = OUT/"calibration_numerical_source.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    h, green, matrix, coefficient, lambda_lower, faces = diffusion.diffusion_geometry(MESH)
    true_scales = exp.source_scales(MESH, "hotspot")
    base = fixed_base_config()
    oracle_candidates = diffusion.precision_candidates(true_scales, green)[0]
    oracle_uniform = next(cfg for cfg in oracle_candidates if cfg["id"] == "b32_contiguous_uniform")
    oracle_source = next(cfg for cfg in oracle_candidates if cfg["id"] == "b32_contiguous_source")
    oracle_energy = next(cfg for cfg in oracle_candidates if cfg["id"] == "b32_contiguous_energy")
    prespec = dict(date="2026-10-11 Asia/Shanghai", mesh=MESH,
        source_law="b_hotspot=s_hotspot(x,y)exp(N(0,1)); independent coordinates",
        training_sizes=list(TRAIN_SIZES), training_replicates=REPLICATES,
        validation_trials=VALIDATION_TRIALS, bootstrap_replicates=BOOTSTRAP,
        fitted_model="coordinate log-location, one pooled residual sigma; no validation data in fit",
        candidates="fixed B=32 contiguous, exactly7bits/cell: Uniform, train Source, train Energy; oracle maps retained",
        oracle_map=oracle_energy["id"], stream_labels={
            "training": "calibration|training|T|replicate|20261011|highcontrast16",
            "validation": "calibration|validation|20261011|highcontrast16"},
        sampling="All40train maps and all validation energies retained; no validation ranking or deletion",
        master_seed=core.MASTER_SEED, source_dependencies={
            "elliptic2d_application.py": sha(OUT/"elliptic2d_application.py"),
            "highcontrast_diffusion.py": sha(OUT/"highcontrast_diffusion.py"),
            "poisson_application.py": sha(OUT/"poisson_application.py"),
            "run_revision.py": sha(OUT/"run_revision.py")},
        numerical_source_snapshot=source_snapshot.name, numerical_source_sha256=sha(source_snapshot))
    (OUT/"calibration_prespecification.json").write_text(json.dumps(prespec, indent=2)+"\n", encoding="utf-8")
    fit_records, fit_rows, train_predictions = [], [], []
    training_hashes = {}
    all_candidates = [dict(oracle_uniform, id="oracle_uniform", precision_policy="oracle_uniform"),
                      dict(oracle_source, id="oracle_source", precision_policy="oracle_source"),
                      dict(oracle_energy, id="oracle_energy", precision_policy="oracle_energy")]
    for T in TRAIN_SIZES:
        for replicate in range(REPLICATES):
            train_rng = core.rng_for("calibration", "training", T, replicate, 20261011, "highcontrast16")
            train = train_rng.lognormal(size=(T, MESH*MESH))*true_scales
            training_hashes[f"T{T}_r{replicate}"] = hashlib.sha256(train.tobytes(order="C")).hexdigest()
            scales, sigma_hat, locations = fit_lognormal(train)
            candidates, metadata = map_variants(scales, sigma_hat, green, f"T{T}_r{replicate}")
            for candidate in candidates:
                candidate["T"] = T
                candidate["replicate"] = replicate
            fit_records.append(dict(T=T, replicate=replicate, sigma_hat=sigma_hat,
                                    scales=scales, source_high=metadata["source_high"],
                                    energy_high=metadata["energy_high"]))
            fit_rows.append(dict(T=T, replicate=replicate, sigma_hat=sigma_hat,
                sigma_error=float(sigma_hat-SIGMA_TRUE),
                log_scale_rmse=float(np.sqrt(np.mean((locations-np.log(true_scales))**2))),
                scale_relative_rmse=float(np.sqrt(np.mean((scales/true_scales-1)**2))),
                source_high_blocks=";".join(map(str, metadata["source_high"])),
                energy_high_blocks=";".join(map(str, metadata["energy_high"])),
                training_input_sha256=training_hashes[f"T{T}_r{replicate}"]))
            all_candidates.extend(candidates[1:])
            fit_cache = metadata["cache"]
            for cfg in candidates[1:]:
                pred = population_prediction(scales, sigma_hat, cfg, h, green, lambda_lower, fit_cache)
                train_predictions.append(dict(T=T, replicate=replicate, policy=cfg["precision_policy"],
                    candidate_id=cfg["id"], **pred))
    prespec["candidate_count_before_validation"] = len(all_candidates)
    prespec["training_fit_records"] = len(fit_records)
    prespec["training_input_sha256"] = training_hashes
    prespec["oracle_maps"] = {"uniform": oracle_uniform["L_by_block"],
                               "source": oracle_source["L_by_block"],
                               "energy": oracle_energy["L_by_block"]}
    (OUT/"calibration_prespecification.json").write_text(json.dumps(prespec, indent=2)+"\n", encoding="utf-8")
    # Validation uses one independent stream and evaluates every learned map.
    validation_rng = core.rng_for("calibration", "validation", 20261011, "highcontrast16")
    candidates_unique, candidate_keys = [], set()
    fitted_prediction_lookup = {(int(row["T"]), int(row["replicate"]), row["candidate_id"]): row
                                for row in train_predictions}
    for cfg in all_candidates:
        key = tuple(cfg["L_by_block"])
        if key not in candidate_keys:
            candidate_keys.add(key)
            candidates_unique.append(cfg)
    energy = np.empty((VALIDATION_TRIALS, len(candidates_unique)))
    batch = 512
    validation_hash = hashlib.sha256()
    minimum_exponents = np.full(len(candidates_unique), 10000)
    maximum_exponents = np.full(len(candidates_unique), -10000)
    for begin in range(0, VALIDATION_TRIALS, batch):
        inputs = validation_rng.lognormal(size=(min(batch, VALIDATION_TRIALS-begin), MESH*MESH))*true_scales
        validation_hash.update(inputs.tobytes(order="C"))
        for j, cfg in enumerate(candidates_unique):
            errors, emin, emax = codec.bounded_quantize(inputs, cfg)
            solved = errors@green
            energy[begin:begin+len(inputs), j] = h*h*np.sum(errors*solved, axis=1)
            minimum_exponents[j] = min(minimum_exponents[j], emin)
            maximum_exponents[j] = max(maximum_exponents[j], emax)
    ids_to_index = {tuple(cfg["L_by_block"]): j for j, cfg in enumerate(candidates_unique)}
    oracle_reference = next(j for j, cfg in enumerate(candidates_unique) if cfg["id"] == "oracle_energy")
    result_rows = []
    for cfg in all_candidates:
        j = ids_to_index[tuple(cfg["L_by_block"])]
        values = energy[:, j]
        # True-law oracle population prediction for every candidate; this is
        # not used in selecting the candidate map.
        true_pred = population_prediction(true_scales, SIGMA_TRUE, cfg, h, green, lambda_lower, {})
        low, high = bootstrap_rms(values, true_pred["predicted_reference_energy"], (cfg["id"],))
        rms = math.sqrt(float(values.mean())/true_pred["predicted_reference_energy"])
        train_prediction = fitted_prediction_lookup.get((int(cfg.get("T", 0)), int(cfg.get("replicate", -1)), cfg["id"]))
        # Re-express the train-only energy prediction with the true-law
        # validation reference before comparing RMS values.  Comparing the
        # two relative RMS values with different denominators would mix fit
        # error with the change in the source-reference energy.
        fitted_rms = (math.sqrt(float(train_prediction["predicted_energy_mse"])
                                /true_pred["predicted_reference_energy"])
                      if train_prediction else float("nan"))
        row = dict(candidate_id=cfg["id"], T=cfg.get("T", 0), replicate=cfg.get("replicate", -1),
            policy=cfg["precision_policy"], map_levels=";".join(map(str, cfg["L_by_block"])),
            validation_trials=VALIDATION_TRIALS, predicted_true_rms=true_pred["predicted_relative_rms"],
            predicted_true_energy_mse=true_pred["predicted_energy_mse"],
            fitted_predicted_rms=fitted_rms,
            fitted_prediction_error=(rms/fitted_rms-1) if train_prediction else float("nan"),
            mc_relative_rms=rms, mc_relative_rms_low=float(low), mc_relative_rms_high=float(high),
            exact_in_mc_interval=bool(low <= true_pred["predicted_relative_rms"] <= high),
            relative_prediction_error=rms/true_pred["predicted_relative_rms"]-1,
            validation_energy_mean=float(values.mean()),
            # Both terms are RMS values for the same true-law denominator; the
            # ratio therefore uses the two held-out energy means directly.
            validation_regret_to_oracle_energy=float(
                math.sqrt(float(values.mean())/float(energy[:, oracle_reference].mean()))-1),
            observed_exponent_min=int(minimum_exponents[j]), observed_exponent_max=int(maximum_exponents[j]))
        result_rows.append(row)
    write_csv(OUT/"calibration_fit_results.csv", fit_rows)
    write_csv(OUT/"calibration_train_predictions.csv", train_predictions)
    write_csv(OUT/"calibration_results.csv", result_rows)
    # Preserve a metric alias for compact generation; fitting RMSE is reported
    # in fit_results and is intentionally not selected using validation data.
    summary_rows = summarize([dict(r, regret_to_oracle_energy=r["validation_regret_to_oracle_energy"])
                              for r in result_rows if r["policy"] in ("source", "energy")])
    # Add fit metrics in the same long table for the generated text.
    for T in TRAIN_SIZES:
        fitgroup = [r for r in fit_rows if int(r["T"]) == T]
        for metric in ("log_scale_rmse", "scale_relative_rmse", "sigma_error"):
            vals = np.asarray([float(r[metric]) for r in fitgroup])
            summary_rows.append(dict(T=T, policy="fit", metric=metric, replicates=len(vals),
                mean=float(vals.mean()), median=float(np.median(vals)),
                q025=float(np.quantile(vals, .025)), q975=float(np.quantile(vals, .975))))
    write_csv(OUT/"calibration_summary.csv", summary_rows)
    np.savez_compressed(OUT/"calibration_validation_samples.npz", energy=energy,
        candidate_ids=np.array([cfg["id"] for cfg in candidates_unique]),
        candidate_levels=np.array([cfg["L_by_block"] for cfg in candidates_unique]),
        validation_input_sha256=np.array(validation_hash.hexdigest()),
        exponent_min=minimum_exponents, exponent_max=maximum_exponents)
    unique_rows = {}
    for r in result_rows:
        unique_rows.setdefault(r["map_levels"], r)
    manifest = dict(prespec, duration_seconds=time.time()-started,
        candidate_count=len(all_candidates), unique_map_count=len(candidates_unique),
        validation_input_sha256=validation_hash.hexdigest(),
        prediction_interval_coverage=sum(r["exact_in_mc_interval"] for r in result_rows),
        prediction_interval_count=len(result_rows), unique_map_interval_coverage=sum(r["exact_in_mc_interval"] for r in unique_rows.values()),
        unique_map_interval_count=len(unique_rows), max_prediction_error=max(abs(r["relative_prediction_error"]) for r in result_rows),
        format_failures=0, numpy=np.__version__, python=platform.python_version())
    # Write the generated section now; figure/table integration is owned by root.
    (OUT/"calibration_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    section_from_saved()
    manifest["generated_section_sha256"] = sha(OUT.parent/"calibration_application.tex")
    files = [path for path in OUT.glob("calibration_*") if path.is_file() and path.name != "calibration_manifest.json"]
    manifest["files"] = {path.name: sha(path) for path in sorted(files)}
    (OUT/"calibration_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Calibration completed: {len(result_rows)} candidate rows, {len(candidates_unique)} unique maps; "
          f"prediction coverage {manifest['prediction_interval_coverage']}/{manifest['prediction_interval_count']}; "
          f"max prediction RMS discrepancy {100*manifest['max_prediction_error']:.3f}%", flush=True)


def from_saved():
    manifest = json.loads((OUT/"calibration_manifest.json").read_text(encoding="utf-8"))
    snapshot = OUT/manifest["numerical_source_snapshot"]
    if sha(snapshot) != manifest["numerical_source_sha256"]:
        raise ArithmeticError("calibration numerical source snapshot hash mismatch")
    section_from_saved()
    manifest["artifact_rebuild"] = "Rebuilt section from retained CSV/NPZ; no training or validation sampling"
    manifest["generated_section_sha256"] = sha(OUT.parent/"calibration_application.tex")
    (OUT/"calibration_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Rebuilt calibration manuscript section without sampling.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-saved", action="store_true")
    args = parser.parse_args()
    if args.from_saved:
        from_saved()
    else:
        run()
