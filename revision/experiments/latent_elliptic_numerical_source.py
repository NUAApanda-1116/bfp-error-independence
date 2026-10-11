"""Prescribed latent-factor correlated sources in a 2D elliptic solve.

Run this file for quadrature, held-out sampling, actual codec packing, CG
checks, and artifacts. --from-saved rebuilds artifacts without recomputation.
All six equal-payload codecs and all quadrature orders remain in the package.
"""
from pathlib import Path
import argparse
import json
import math
import platform
import time

import numpy as np
import scipy
from scipy import special
from scipy.fft import dstn

import elliptic2d_application as elliptic
import poisson_application as codec
import run_revision as core

OUT = Path(__file__).resolve().parent
MESH, TAU = 16, .6
ORDERS = (21, 41, 61, 81, 101)
RELATIVE_QUADRATURE_TARGET = 1e-7


def loading(mesh):
    points = np.arange(1, mesh+1)/(mesh+1)
    x, y = np.meshgrid(points, points, indexing="ij")
    return (np.sin(np.pi*x)*np.sin(np.pi*y)).ravel()


def source_reference(scales, phi, tau, h, green):
    """Analytical whole-source energy for the correlated lognormal law."""
    log_cov = tau*tau*np.outer(phi, phi)
    log_variance = 1+tau*tau*phi*phi
    mean = scales*np.exp(log_variance/2)
    raw_second = np.outer(mean, mean)*np.exp(log_cov)
    np.fill_diagonal(raw_second, scales*scales*np.exp(2*log_variance))
    reference = h*h*float(np.sum(green*raw_second))
    source_second = scales*scales*np.exp(2*log_variance)
    norm_fourth_pairs = np.outer(source_second, source_second)*np.exp(4*log_cov)
    np.fill_diagonal(norm_fourth_pairs, scales**4*np.exp(8*log_variance))
    return reference, float(norm_fourth_pairs.sum())


def failure_audit(scales, phi, tau, config, h, eigenvalues, fourth_norm,
                  exponent_min=-128, exponent_max=127, skipped_states=0):
    """Correlated-input union bound; no product of marginal probabilities.

    P(all X_i<T) <= min_i P(X_i<T) in each block. The upper event is bounded
    by the sum of marginal tails. The same construction bounds outer strata
    of the population evaluator; skipped probabilities add at most their
    declared per-state threshold, independently of the latent factor.
    """
    standard_deviation = np.sqrt(1+tau*tau*phi*phi)
    lower_log = exponent_min*math.log(2)
    upper_log = (exponent_max+1)*math.log(2)
    lower_logs = special.log_ndtr((lower_log-np.log(scales))/standard_deviation)
    lower_block_logs = [float(np.min(lower_logs[np.asarray(ids)]))
                        for ids in config["block_indices"]]
    upper_logs = special.log_ndtr(-(upper_log-np.log(scales))/standard_deviation)
    event_logs = lower_block_logs+upper_logs.tolist()
    if skipped_states:
        event_logs.append(math.log(config["blocks"]*skipped_states*1e-25))
    event_log = min(0., float(special.logsumexp(event_logs)))
    factor = h*h/eigenvalues.min()*config["block_size"]*2.**(-2*(config["base_L"]+1))
    energy_log = math.log(factor)+.5*(math.log(fourth_norm)+event_log)
    return dict(probability_log10_bound=event_log/math.log(10),
                residual_energy_log10_bound=energy_log/math.log(10))


def quadrature(configs, mesh=MESH, tau=TAU, order=61):
    """Integrate full conditional responses; all codecs share node caches."""
    h, green, eigenvalues, _ = elliptic.geometry(mesh)
    scales, phi = elliptic.source_scales(mesh, "hotspot"), loading(mesh)
    reference, fourth_norm = source_reference(scales, phi, tau, h, green)
    points, weights = np.polynomial.hermite.hermgauss(order)
    points, weights = np.sqrt(2)*points, weights/math.sqrt(math.pi)
    accumulators = [dict(exact=0., independent=0., uniform=0., centered=0.,
                        conditional_mean_energy=0., mean=np.zeros(mesh*mesh),
                        second=np.zeros(mesh*mesh)) for _ in configs]
    for node, weight in zip(points, weights):
        local_scales, cache = scales*np.exp(tau*phi*node), {}
        for j, config in enumerate(configs):
            prediction = elliptic.population(local_scales, config, h, green, eigenvalues,
                                            cache, sigma=1., return_moments=True)
            accumulator = accumulators[j]
            accumulator["exact"] += weight*prediction["exact_energy_mse"]
            accumulator["independent"] += weight*prediction["independent_energy_mse"]
            accumulator["uniform"] += weight*prediction["uniform_energy_mse"]
            accumulator["centered"] += weight*prediction["exact_energy_variance"]
            accumulator["conditional_mean_energy"] += weight*prediction["energy_squared_bias"]
            error_mean = prediction["error_mean"]
            accumulator["mean"] += weight*error_mean
            accumulator["second"] += weight*(np.diag(prediction["error_covariance"])+error_mean**2)
    results = []
    for config, accumulator in zip(configs, accumulators):
        mean = accumulator["mean"]
        coherent_bias = h*h*float(mean@green@mean)
        unconditional_diagonal = h*h*float(np.dot(np.diag(green), accumulator["second"]-mean**2))
        failure = failure_audit(scales, phi, tau, config, h, eigenvalues, fourth_norm)
        # The evaluator loops over 76 exponents, each omitted below 1e-25.
        omitted = failure_audit(scales, phi, tau, config, h, eigenvalues, fourth_norm,
                                exponent_min=-40, exponent_max=35, skipped_states=76)
        omitted_residual = 10.**omitted["residual_energy_log10_bound"]
        omission_difference = 2*math.sqrt(accumulator["exact"]*omitted_residual)+omitted_residual
        row = {key: value for key, value in config.items()
               if key not in ("block_indices", "L_by_block")}
        row.update(tau=tau, quadrature_order=order, reference_expected_energy=reference,
            exact_energy_mse=accumulator["exact"], independent_energy_mse=accumulator["independent"],
            uniform_energy_mse=accumulator["uniform"],
            unconditional_diagonal_energy_mse=unconditional_diagonal+coherent_bias,
            conditional_centered_energy=accumulator["centered"],
            conditional_mean_energy=accumulator["conditional_mean_energy"],
            coherent_squared_bias_energy=coherent_bias,
            latent_mean_fluctuation_energy=accumulator["conditional_mean_energy"]-coherent_bias,
            exact_relative_rms=math.sqrt(accumulator["exact"]/reference),
            independent_relative_rms=math.sqrt(accumulator["independent"]/reference),
            uniform_relative_rms=math.sqrt(accumulator["uniform"]/reference),
            unconditional_diagonal_relative_rms=math.sqrt((unconditional_diagonal+coherent_bias)/reference),
            failure_probability_log10_bound=failure["probability_log10_bound"],
            failure_energy_mse_log10_bound=failure["residual_energy_log10_bound"],
            omitted_probability_log10_bound=omitted["probability_log10_bound"],
            omitted_energy_mse_bound=omission_difference)
        results.append(row)
    return results


def latent_population(config, mesh=MESH, tau=TAU, order=61):
    return quadrature([config], mesh, tau, order)[0]


def quadrature_convergence(configs, mesh=MESH, tau=TAU):
    all_rows, prior, final = [], None, None
    for order in ORDERS:
        start = time.time()
        results = quadrature(configs, mesh, tau, order)
        maximum_change = math.inf
        if prior is not None:
            maximum_change = max(abs(row[metric]/old[metric]-1)
                                 for row, old in zip(results, prior)
                                 for metric in ("exact_energy_mse", "independent_energy_mse", "conditional_mean_energy"))
        for row in results:
            row["maximum_relative_change_from_previous_order"] = maximum_change
            all_rows.append(row.copy())
        codec.write_csv(OUT/"latent_elliptic_quadrature.csv", all_rows)
        print(f"Latent GH order {order}: largest relative change {maximum_change:.3e}; "
              f"{time.time()-start:.1f}s", flush=True)
        prior, final = results, results
        if order >= 61 and maximum_change < RELATIVE_QUADRATURE_TARGET:
            return final, all_rows, maximum_change
    return final, all_rows, maximum_change


def write_artifacts(rows, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    positions = np.arange(len(rows))
    exact = np.array([row["exact_relative_rms"] for row in rows])*100
    independent = np.array([row["independent_relative_rms"] for row in rows])*100
    mc = np.array([row["mc_relative_rms"] for row in rows])*100
    low = np.array([row["mc_relative_rms_low"] for row in rows])*100
    high = np.array([row["mc_relative_rms_high"] for row in rows])*100
    with plt.rc_context({"font.family": "serif", "font.size": 9}):
        figure, axis = plt.subplots(figsize=(5.8, 2.65))
        axis.plot(positions, exact, "o-", label="Full conditional covariance", color="#174c72")
        axis.plot(positions, independent, "s--", label="Conditional diagonal, latent response retained", color="#b65c12")
        axis.errorbar(positions+.07, mc, yerr=np.array([mc-low, high-mc]), fmt=".", color="black", capsize=3,
                      label="30,000 paired vectors, 95% bootstrap")
        axis.set_xticks(positions, [f"{int(row['block_size'])}\n"+('Cont.' if row['layout']=='contiguous' else 'Inter.') for row in rows])
        axis.set_xlabel("Block length and spatial layout (7 bits/cell)")
        axis.set_ylabel("Relative energy RMS (%)")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=.2)
        figure.tight_layout()
        figure.savefig(OUT/"latent_elliptic_application.pdf", bbox_inches="tight")
        figure.savefig(OUT/"latent_elliptic_application.png", dpi=200, bbox_inches="tight")
        plt.close(figure)
    table = []
    for row in rows:
        layout = "Cont." if row["layout"] == "contiguous" else "Inter."
        table.append(f"{int(row['block_size'])} & {layout} & {100*row['exact_relative_rms']:.3f} & "
                     f"{100*row['independent_relative_rms']:.3f} & "
                     f"[{100*row['mc_relative_rms_low']:.3f},{100*row['mc_relative_rms_high']:.3f}] " + r"\\")
    maximum_factor = max(row["exact_energy_mse"]/row["independent_energy_mse"] for row in rows)
    section = r"""\subsection{A correlated source field}\label{sec:revision-latent}
The conditional extension in Corollary~\ref{cor:revision-latent} gives a second
test with correlated inputs on the $16\times16$ grid. With the same hotspot
envelope $a_i$, let
\[
 b_i=a_i\exp(Z_i+0.6\phi_iY),\qquad
 \phi_i=\sin(\pi x_i)\sin(\pi y_i),
\]
where $Y,Z_1,\ldots,Z_N$ are independent standard normals. The $b_i$ are
independent conditional on $Y$, so each block is evaluated using
Proposition~\ref{prop:revision-nonidentical}. Writing $\mu(y)$ and $\Sigma(y)$
for the whole-vector conditional error moments gives
\begin{equation}\label{eq:revision-latent-energy}
 \mathbb E\|\widehat u-u\|_{A,h}^2
 =h^2\mathbb E_Y\!\left[\operatorname{tr}(G\Sigma(Y))
                         +\mu(Y)^TG\mu(Y)\right].
\end{equation}
The diagonal comparison replaces $\Sigma(Y)$ by its diagonal while retaining
the entire second term, including the correlations induced by the latent
factor across distinct blocks. The reference source energy is analytical:
for $M_i=a_i\exp((1+0.36\phi_i^2)/2)$,
$\mathbb E[b_ib_j]=M_iM_j\exp(0.36\phi_i\phi_j)$ for $i\ne j$, and
$\mathbb E[b_i^2]=a_i^2\exp(2(1+0.36\phi_i^2))$.

All six prescribed block/layout codecs use $7N$ bits. Gauss--Hermite orders
__ORDERS__ are retained; the final relative change in energy MSE and the
conditional-mean contribution is __CHANGE__. This is a numerical
convergence check. A new stream of __TRIALS__ complete source vectors is paired
across codecs, with __BOOT__ vector-bootstrap replicates. The measured
intervals cover __COVERAGE__ of six population predictions. Actual CG solves
on __CG__ vectors per codec have maximum relative energy difference
__CGERR__ from the direct source-error energies.

\begin{table}[tb]
\centering
\caption{Correlated hotspot sources: relative energy RMS (percent).
The diagonal prediction retains the exact conditional mean response to $Y$.}
\label{tab:revision-latent}
\begingroup\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{rlrrl}
\toprule
$|B_j|$ & Layout & Full & Diagonal & MC $95\%$ interval\\
\midrule
__TABLE__
\bottomrule
\end{tabular}\endgroup
\end{table}
The diagonal model underestimates energy MSE by as much as a factor
__FACTOR__. This comparison isolates the shared-exponent contribution after
the latent source correlation is included.
"""
    changes = {"__TABLE__": "\n".join(table), "__ORDERS__": ", ".join(map(str, manifest["quadrature_orders"])),
               "__CHANGE__": f"${manifest['maximum_final_quadrature_relative_change']:.1e}$",
               "__TRIALS__": f"{manifest['trials']:,}".replace(",", r"\,"),
               "__BOOT__": str(manifest["bootstrap_replicates"]),
               "__COVERAGE__": str(manifest["exact_rms_ci_coverage"]),
               "__CG__": str(manifest["cg_vectors_per_codec"]),
               "__CGERR__": f"${manifest['maximum_cg_relative_energy_error']:.1e}$",
               "__FACTOR__": f"{maximum_factor:.2f}"}
    for key, value in changes.items():
        section = section.replace(key, value)
    # Convert scientific notation to native LaTeX once, without changing values.
    import re
    section = re.sub(r"\$([0-9.]+)e([+-][0-9]+)\$",
                     lambda match: "$"+match.group(1)+r"\times10^{"+str(int(match.group(2)))+"}$", section)
    (OUT.parent/"latent_application.tex").write_text(section, encoding="utf-8")
    files = [path for path in OUT.glob("latent_elliptic_*")
             if path.is_file() and path.name != "latent_elliptic_manifest.json"]
    manifest["files"] = {path.name: elliptic.sha(path) for path in sorted(files)}
    manifest["generated_section_sha256"] = elliptic.sha(OUT.parent/"latent_application.tex")
    (OUT/"latent_elliptic_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


def run(trials=30000, bootstrap=300, cg_vectors=100):
    started = time.time()
    configs = elliptic.configurations(MESH)
    numerical_source = OUT/"latent_elliptic_numerical_source.py"
    numerical_source.write_bytes(Path(__file__).read_bytes())
    prespecification = dict(date="2026-10-10 Asia/Shanghai", mesh=MESH, n=MESH*MESH, tau=TAU,
        source="a_i exp(Z_i+tau*phi_i*Y); Y,Z_i independent N(0,1)",
        envelope="1+7exp(-((x-.3)^2+(y-.65)^2)/.02)",
        loading="sin(pi*x)sin(pi*y)", configurations=configs,
        trials=trials, bootstrap_replicates=bootstrap, cg_vectors_per_codec=cg_vectors,
        quadrature_orders_planned=list(ORDERS), relative_quadrature_target=RELATIVE_QUADRATURE_TARGET,
        comparison="Drop conditional within-block error covariances; retain E[mu(Y)^T G mu(Y)], including latent correlations and bias",
        reference="Analytical correlated-lognormal raw source moments propagated through fixed2D Green matrix",
        codec="Actual signed int8 exponents and L+3 coefficient bits, fixed7bits/cell; no escape or saturation",
        sampling="All6 codecs share a new held-out source stream; whole-vector percentile bootstrap; no configuration/interval deletion",
        rng="core.rng_for('latent_elliptic','heldout','input',16,.6), PCG64 SHA256labels",
        solver="scipy CG rtol1e-11, atol0, maxiter4N; first100 complete vectors/codecs",
        design="tau=.6 and all six codecs prescribed before quadrature or sampling; no tau/seed scan",
        stage="Before quadrature and sampling", numerical_source_sha256=elliptic.sha(numerical_source),
        dependencies={"elliptic2d_application.py": elliptic.sha(OUT/"elliptic2d_application.py"),
                      "poisson_application.py": elliptic.sha(OUT/"poisson_application.py"),
                      "run_revision.py": elliptic.sha(OUT/"run_revision.py")})
    (OUT/"latent_elliptic_prespecification.json").write_text(json.dumps(prespecification, indent=2)+"\n", encoding="utf-8")
    predictions, quadrature_rows, maximum_change = quadrature_convergence(configs)
    prespecification.update(stage="Before Monte Carlo, after recorded quadrature convergence",
                            quadrature_orders=sorted({row["quadrature_order"] for row in quadrature_rows}),
                            maximum_final_quadrature_relative_change=maximum_change,
                            final_quadrature_order=predictions[0]["quadrature_order"],
                            numerical_quadrature_target_met=bool(maximum_change < RELATIVE_QUADRATURE_TARGET))
    (OUT/"latent_elliptic_prespecification.json").write_text(json.dumps(prespecification, indent=2)+"\n", encoding="utf-8")
    h, green, eigenvalues, matrix = elliptic.geometry(MESH)
    scales, phi = elliptic.source_scales(MESH, "hotspot"), loading(MESH)
    generator = core.rng_for("latent_elliptic", "heldout", "input", MESH, TAU)
    energy = np.empty((trials, len(configs)))
    minima, maxima = np.full(len(configs), 10000), np.full(len(configs), -10000)
    cg_inputs, cg_errors = [], [[] for _ in configs]
    batch = max(1, 500000//(MESH*MESH))
    for begin in range(0, trials, batch):
        count = min(batch, trials-begin)
        factors = generator.normal(size=count)
        inputs = scales*np.exp(generator.normal(size=(count, MESH*MESH))+TAU*factors[:, None]*phi)
        to_check = max(0, min(cg_vectors-begin, count))
        if to_check:
            cg_inputs.extend(inputs[:to_check])
        for j, config in enumerate(configs):
            errors, emin, emax = codec.bounded_quantize(inputs, config)
            transformed = dstn(errors.reshape(-1, MESH, MESH), type=1, norm="ortho", axes=(1, 2))
            energy[begin:begin+count, j] = h*h*np.sum(transformed**2/eigenvalues, axis=(1, 2))
            minima[j], maxima[j] = min(minima[j], emin), max(maxima[j], emax)
            if to_check:
                cg_errors[j].extend(errors[:to_check])
    np.savez_compressed(OUT/"latent_elliptic_samples.npz", energy=energy,
                        codec_ids=np.array([config["id"] for config in configs]),
                        exponent_min=minima, exponent_max=maxima,
                        cg_inputs=np.array(cg_inputs), cg_errors=np.array(cg_errors))
    rows, solver_rows = [], []
    for j, row in enumerate(predictions):
        values = energy[:, j]
        rms = math.sqrt(float(values.mean())/row["reference_expected_energy"])
        low, high = codec.bootstrap_rms(values, row["reference_expected_energy"],
                                       ("latent_elliptic", MESH, TAU, row["id"]), bootstrap)
        audits = elliptic.cg_audit(np.array(cg_inputs), np.array(cg_errors[j]), matrix, eigenvalues, h, MESH)
        solver_rows.extend(dict(codec=row["id"], **audit) for audit in audits)
        row.update(trials=trials, mc_energy_mse=float(values.mean()), mc_relative_rms=rms,
                   mc_relative_rms_low=float(low), mc_relative_rms_high=float(high),
                   exact_in_mc_interval=bool(low <= row["exact_relative_rms"] <= high),
                   relative_prediction_error=rms/row["exact_relative_rms"]-1,
                   observed_exponent_min=int(minima[j]), observed_exponent_max=int(maxima[j]),
                   cg_vectors=len(audits),
                   cg_max_relative_energy_error=max(a["relative_energy_error"] for a in audits),
                   cg_max_relative_energy_norm_certificate=max(a["relative_energy_norm_certificate"] for a in audits),
                   format_failures=0)
        row.pop("maximum_relative_change_from_previous_order", None)
        rows.append(row)
    codec.write_csv(OUT/"latent_elliptic_results.csv", rows)
    codec.write_csv(OUT/"latent_elliptic_cg_audit.csv", solver_rows)
    manifest = dict(prespecification, numerical_source_snapshot=numerical_source.name,
        duration_seconds=time.time()-started, configuration_count=len(rows),
        exact_rms_ci_coverage=sum(row["exact_in_mc_interval"] for row in rows),
        maximum_relative_rms_prediction_error=max(abs(row["relative_prediction_error"]) for row in rows),
        maximum_cg_relative_energy_error=max(row["cg_max_relative_energy_error"] for row in rows),
        maximum_cg_relative_energy_norm_certificate=max(row["cg_max_relative_energy_norm_certificate"] for row in rows),
        maximum_omitted_energy_mse_bound=max(row["omitted_energy_mse_bound"] for row in rows),
        failure_probability_log10_bound=max(row["failure_probability_log10_bound"] for row in rows),
        failure_energy_mse_log10_bound=max(row["failure_energy_mse_log10_bound"] for row in rows),
        format_failures=0, numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version())
    write_artifacts(rows, manifest)
    print(f"Latent experiment retained all6 codecs, coverage {manifest['exact_rms_ci_coverage']}/6; "
          f"max RMS discrepancy {100*manifest['maximum_relative_rms_prediction_error']:.2f}%; "
          f"CG max energy error {manifest['maximum_cg_relative_energy_error']:.2e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-saved", action="store_true")
    parser.add_argument("--trials", type=int, default=30000)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--cg-vectors", type=int, default=100)
    args = parser.parse_args()
    if args.from_saved:
        write_artifacts(codec.read_rows("latent_elliptic_results.csv"),
                        json.loads((OUT/"latent_elliptic_manifest.json").read_text(encoding="utf-8")))
    else:
        run(args.trials, args.bootstrap, args.cg_vectors)
