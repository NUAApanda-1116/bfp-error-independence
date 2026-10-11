"""Fixed high-contrast diffusion case, separate from all earlier experiments.

Independent, nonidentical lognormal sources; actual fixed-budget codec; all
three declared bit-placement rules; held-out paired vectors and Jacobi PCG.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu

import elliptic2d_application as exp
import poisson_application as codec
import run_revision as core

OUT = Path(__file__).resolve().parent
MESH = 16
CONTRAST = .01
TRIALS = 30000
BOOTSTRAP = 300
PCG_VECTORS = 100


def conductivity(x, y):
    return np.where((x-.7)**2+(y-.3)**2 <= .05, CONTRAST, 1.)


def harmonic(left, right):
    return 2*left*right/(left+right)


def diffusion_geometry(mesh=MESH):
    """Nodal diffusion graph with harmonic edges and zero-Dirichlet faces.

    Conductivities are sampled at the interior grid points; internal face
    weights use the harmonic mean, while boundary faces use the harmonic
    mean of the adjacent interior point and the prescribed boundary value.
    Every edge weight is >= .01, giving A >= .01*A_Poisson.
    """
    h = 1/(mesh+1)
    points = np.arange(1, mesh+1)*h
    x, y = np.meshgrid(points, points, indexing="ij")
    coefficient = conductivity(x, y)
    wx = np.empty((mesh+1, mesh))
    wy = np.empty((mesh, mesh+1))
    wx[1:-1] = harmonic(coefficient[:-1], coefficient[1:])
    wy[:, 1:-1] = harmonic(coefficient[:, :-1], coefficient[:, 1:])
    wx[0] = harmonic(coefficient[0], conductivity(0., points))
    wx[-1] = harmonic(coefficient[-1], conductivity(1., points))
    wy[:, 0] = harmonic(coefficient[:, 0], conductivity(points, 0.))
    wy[:, -1] = harmonic(coefficient[:, -1], conductivity(points, 1.))
    diagonal = wx[:-1]+wx[1:]+wy[:, :-1]+wy[:, 1:]
    rows, columns, values = [], [], []
    for i in range(mesh):
        for j in range(mesh):
            index = i*mesh+j
            rows.append(index)
            columns.append(index)
            values.append(diagonal[i, j])
            if i+1 < mesh:
                weight = wx[i+1, j]
                rows.extend((index, index+mesh))
                columns.extend((index+mesh, index))
                values.extend((-weight, -weight))
            if j+1 < mesh:
                weight = wy[i, j+1]
                rows.extend((index, index+1))
                columns.extend((index+1, index))
                values.extend((-weight, -weight))
    matrix = sparse.coo_matrix((np.asarray(values)/h**2, (rows, columns)),
                              shape=(mesh*mesh, mesh*mesh)).tocsr()
    factor = splu(matrix.tocsc())
    green = factor.solve(np.eye(mesh*mesh))
    poisson_smallest = 8/h**2*math.sin(math.pi/(2*(mesh+1)))**2
    lambda_lower = CONTRAST*poisson_smallest
    if np.max(np.abs(matrix@green-np.eye(mesh*mesh))) > 2e-12:
        raise ArithmeticError("diffusion inverse defect")
    return h, green, matrix, coefficient, lambda_lower, dict(x=wx, y=wy)


def precision_candidates(scales, green, mesh=MESH):
    """Generic version of the declared three fixed placement families."""
    cache, candidates = {}, []
    for baseline in exp.configurations(mesh):
        candidates.append(baseline)
        if not baseline["higher_L_blocks"]:
            continue
        base, extra, count = baseline["base_L"], baseline["higher_L_blocks"], baseline["blocks"]
        ids_list = [np.asarray(indices) for indices in baseline["block_indices"]]
        scores = [float(np.dot(scales[ids], scales[ids])) for ids in ids_list]
        source_high = sorted(range(count), key=lambda j: (-scores[j], j))[:extra]
        candidates.append(dict(baseline, id=baseline["id"].replace("_uniform", "_source"),
            precision_policy="source", L_by_block=[base+int(j in source_high) for j in range(count)]))
        models, mean = [], np.zeros(len(scales))
        for ids in ids_list:
            pair = []
            for level in (base, base+1):
                key = (tuple(scales[ids]), level, 1.)
                if key not in cache:
                    cache[key] = exp.block_moments(scales[ids], level)
                pair.append(cache[key])
            models.append(pair)
            mean[ids] = pair[0]["mean"]
        local_green = [green[np.ix_(ids, ids)] for ids in ids_list]
        deltas = [pair[1]["mean"]-pair[0]["mean"] for pair in models]
        changes = [float(np.sum(g*(pair[1]["cov"]-pair[0]["cov"])))
                   for g, pair in zip(local_green, models)]
        high, sequence, increments = set(), [], []
        for _ in range(extra):
            action = green@mean
            objectives = {j: changes[j]+2*float(deltas[j]@action[ids])
                            +float(deltas[j]@local_green[j]@deltas[j])
                          for j, ids in enumerate(ids_list) if j not in high}
            best = min(objectives, key=lambda j: (objectives[j], j))
            high.add(best)
            mean[ids_list[best]] += deltas[best]
            sequence.append(best)
            increments.append(objectives[best])
        candidates.append(dict(baseline, id=baseline["id"].replace("_uniform", "_energy"),
            precision_policy="energy", L_by_block=[base+int(j in high) for j in range(count)],
            greedy_sequence=sequence, greedy_unscaled_energy_changes=increments))
    return candidates, cache


def predictions(scales, candidates, h, green, lambda_lower, cache):
    # Population uses the smallest eigenvalue only for a rigorous tail bound;
    # the coefficient-weighted Green matrix supplies the full energy itself.
    return [dict(profile="hotspot", **exp.population(scales, cfg, h, green,
                     np.array([lambda_lower]), cache),
                     **exp.format_audit(scales, cfg, h, np.array([lambda_lower])))
            for cfg in candidates]


def pilot():
    h, green, matrix, coefficient, lambda_lower, faces = diffusion_geometry()
    scales = exp.source_scales(MESH, "hotspot")
    candidates, cache = precision_candidates(scales, green)
    rows = predictions(scales, candidates, h, green, lambda_lower, cache)
    codec.write_csv(OUT/"highcontrast_diffusion_analytic_pilot.csv", rows)
    for row in rows:
        print(f"{row['id']}: exact={100*row['exact_relative_rms']:.4f}% "
              f"ind={100*row['independent_relative_rms']:.4f}% "
              f"uniform={100*row['uniform_relative_rms']:.4f}%", flush=True)
    for size in (32, 128):
        for layout in ("contiguous", "interleaved"):
            source = next(cfg for cfg in candidates if cfg["id"] == f"b{size}_{layout}_source")
            energy = next(cfg for cfg in candidates if cfg["id"] == f"b{size}_{layout}_energy")
            src = next(row for row in rows if row["id"] == source["id"])
            eng = next(row for row in rows if row["id"] == energy["id"])
            print(f"B{size},{layout}: source→energy fractionalRMSgain={1-eng['exact_relative_rms']/src['exact_relative_rms']:.4%}; "
                  f"changedblocks={sum(a != b for a,b in zip(source['L_by_block'],energy['L_by_block']))}", flush=True)


def pcg_audit(inputs, errors, matrix, green, h, lambda_lower):
    """Actual independent original/quantized Jacobi-PCG solves."""
    preconditioner = sparse.diags(1/matrix.diagonal(), format="csr")
    records = []
    for index, (b, e) in enumerate(zip(inputs, errors)):
        counts = [0, 0]
        def first_callback(_):
            counts[0] += 1
        def second_callback(_):
            counts[1] += 1
        u, status = cg(matrix, b, M=preconditioner, rtol=1e-11, atol=0.,
                       maxiter=4*len(b), callback=first_callback)
        uq, statusq = cg(matrix, b+e, M=preconditioner, rtol=1e-11, atol=0.,
                         maxiter=4*len(b), callback=second_callback)
        if status or statusq:
            raise ArithmeticError(f"Jacobi PCG failure {status},{statusq}")
        delta = uq-u
        residual = matrix@delta-e
        energy = h*h*float(delta@(matrix@delta))
        reference_energy = h*h*float(e@green@e)
        certificate = h*float(np.linalg.norm(residual))/math.sqrt(lambda_lower)
        records.append(dict(vector=index, energy=energy, reference_energy=reference_energy,
            relative_energy_error=abs(energy/reference_energy-1),
            relative_energy_norm_certificate=certificate/math.sqrt(reference_energy),
            original_iterations=counts[0], quantized_iterations=counts[1],
            original_relative_residual=float(np.linalg.norm(matrix@u-b)/np.linalg.norm(b)),
            quantized_relative_residual=float(np.linalg.norm(matrix@uq-(b+e))/np.linalg.norm(b+e))))
    return records


def bootstrap_rms(values, reference, label):
    generator = core.rng_for("highcontrast_diffusion", "bootstrap", label)
    samples = []
    for _ in range(BOOTSTRAP):
        draws = generator.integers(len(values), size=len(values))
        samples.append(math.sqrt(float(values[draws].mean())/reference))
    return np.percentile(samples, [2.5, 97.5])


def paired_comparisons(energies, ids, rows):
    pairs = []
    for size in (32, 128):
        for layout in ("contiguous", "interleaved"):
            pairs.append((f"b{size}_{layout}_source", f"b{size}_{layout}_energy"))
            pairs.append((f"b{size}_{layout}_uniform", f"b{size}_{layout}_energy"))
    pairs.append(("b8_contiguous_uniform", "b32_contiguous_energy"))
    results = []
    for baseline, candidate in pairs:
        first, second = energies[:, ids.index(baseline)], energies[:, ids.index(candidate)]
        generator = core.rng_for("highcontrast_diffusion", "paired", baseline, candidate)
        estimates = []
        for _ in range(BOOTSTRAP):
            draws = generator.integers(len(energies), size=len(energies))
            estimates.append(1-math.sqrt(float(second[draws].mean()/first[draws].mean())))
        low, high = np.percentile(estimates, [2.5, 97.5])
        popfirst = next(row for row in rows if row["id"] == baseline)
        popsecond = next(row for row in rows if row["id"] == candidate)
        results.append(dict(baseline=baseline, candidate=candidate,
            exact_fractional_rms_reduction=1-popsecond["exact_relative_rms"]/popfirst["exact_relative_rms"],
            mc_fractional_rms_reduction=1-math.sqrt(float(second.mean()/first.mean())),
            mc_fractional_rms_reduction_low=float(low), mc_fractional_rms_reduction_high=float(high),
            bootstrap_replicates=BOOTSTRAP, vectors=len(energies)))
    return results


def rank_rows(rows):
    result = []
    for method in ("exact", "independent", "uniform", "mc"):
        ordered = sorted(rows, key=lambda row: (row[method+"_relative_rms"], row["id"]))
        winner = ordered[0]
        result.append(dict(method=method, chosen_codec=winner["id"],
            model_relative_rms=winner[method+"_relative_rms"], observed_relative_rms=winner["mc_relative_rms"],
            ordered_codecs=";".join(row["id"] for row in ordered)))
    return result


def write_section(rows, comparisons, manifest):
    lines = []
    for size, policy in ((8, "uniform"), (32, "uniform"), (32, "source"), (32, "energy"),
                         (128, "source"), (128, "energy")):
        row = next(row for row in rows if row["id"] == f"b{size}_contiguous_{policy}")
        label = dict(uniform="Uniform", source="Source", energy="Energy")[policy]
        lines.append(f"{size} & {label} & {100*row['exact_relative_rms']:.3f} & "
                     f"{100*row['independent_relative_rms']:.3f} & [{100*row['mc_relative_rms_low']:.3f},{100*row['mc_relative_rms_high']:.3f}] " + r"\\")
    primary = next(row for row in comparisons if row["baseline"] == "b32_contiguous_source"
                   and row["candidate"] == "b32_contiguous_energy")
    secondary = next(row for row in comparisons if row["baseline"] == "b128_contiguous_source"
                     and row["candidate"] == "b128_contiguous_energy")
    section = r"""\subsection{High-contrast diffusion and spatial bit placement}
\label{sec:revision-highcontrast}
We next use $-\nabla\cdot(\alpha\nabla u)=f$ on the same $16^2$ grid,
where $\alpha=0.01$ inside $(x-0.7)^2+(y-0.3)^2\le0.05$ and
$\alpha=1$ elsewhere. Harmonic face conductivities, including the
Dirichlet boundary faces, define the symmetric positive definite operator.
The low-conductivity region is separated from the source hotspot, so the
compression objective accounts for spatial response as well as source scale.
Using the same hotspot source law and $7$ bits per cell, we retain all
$14$ prescribed codecs and $30\,000$ paired source vectors. At length
$32$, the full-energy rule changes two precision blocks relative to Source
placement and reduces measured RMS by __GAIN__\%
(paired $95\%$ interval $[__GAINLO__,__GAINHI__]\%$); at length $128$,
the reduction is __LARGEGAIN__\% ($[__LARGELO__,__LARGEHI__]\%$).
Table~\ref{tab:revision-highcontrast} gives the contiguous results;
both layouts and coincident maps remain in the package. The intervals
cover __COVERAGE__ of $14$ exact RMS predictions, with maximum
discrepancy __MAXERR__\%. We also solve $100$ paired sources per codec
by Jacobi-preconditioned conjugate gradients at tolerance $10^{-11}$.
Since $A\succeq0.01A_{\rm Poisson}$, the preceding residual certificate
uses $0.01\lambda_{\min}(A_{\rm Poisson})$; its largest relative bound is
$__CERT__$, while PCG and direct-solve energy values agree within
$__PCGERR__$ relatively.

\begin{table}[tb]
\centering
\caption{High-contrast diffusion, $16^2$ sources and exactly $1792$ payload
bits. All entries use contiguous blocks; relative energy RMS is in percent.}
\label{tab:revision-highcontrast}
\begingroup\footnotesize\setlength{\tabcolsep}{5pt}
\begin{tabular}{rlrrr}
\toprule
$|B|$ & Placement & Exact & Ind. & MC $95\%$ interval\\
\midrule
__TABLE__
\bottomrule
\end{tabular}\endgroup
\end{table}
"""
    scientific = lambda value: f"{value:.1e}".replace("e-", r"\times10^{-")+"}"
    values = {"__GAIN__": f"{100*primary['mc_fractional_rms_reduction']:.2f}",
        "__GAINLO__": f"{100*primary['mc_fractional_rms_reduction_low']:.2f}",
        "__GAINHI__": f"{100*primary['mc_fractional_rms_reduction_high']:.2f}",
        "__LARGEGAIN__": f"{100*secondary['mc_fractional_rms_reduction']:.2f}",
        "__LARGELO__": f"{100*secondary['mc_fractional_rms_reduction_low']:.2f}",
        "__LARGEHI__": f"{100*secondary['mc_fractional_rms_reduction_high']:.2f}",
        "__COVERAGE__": str(manifest["exact_rms_ci_coverage"]),
        "__MAXERR__": f"{100*manifest['maximum_relative_rms_prediction_error']:.2f}",
        "__CERT__": scientific(manifest["maximum_pcg_relative_energy_norm_certificate"]),
        "__PCGERR__": scientific(manifest["maximum_pcg_relative_energy_error"]),
        "__TABLE__": "\n".join(lines)}
    for key, value in values.items():
        section = section.replace(key, value)
    (OUT.parent/"highcontrast_diffusion.tex").write_text(section, encoding="utf-8")


def run():
    began = time.time()
    source_snapshot = OUT/"highcontrast_diffusion_numerical_source.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    stream_labels = ["highcontrast_diffusion", "heldout", "input", MESH, "hotspot", 1]
    prespecification = dict(date="2026-10-10 Asia/Shanghai", mesh=MESH, trials=TRIALS,
        law="independent b_i=s_hot(x_i,y_i)exp(N(0,1)); samefixedhotspotlaw",
        conductivity=".01 if(x-.7)^2+(y-.3)^2<=.05; 1elsewhere",
        operator="h^-2 weightedfive-pointgraph; harmonicinternalandboundaryfaceconductivities; Dirichletzero",
        contrast=CONTRAST, block_sizes=[8, 32, 128], layouts=["contiguous", "interleaved"],
        policies=["uniform", "source", "energy"], bits_per_entry=7,
        sampling="all14declaredconfigurationresults retained includingidenticalmaps, pairedsourcevectors",
        analytic_pilot="Onefixedcoefficient/sourcecase, nosweeporseedsearch; pilotdataretained",
        bootstrap_replicates=BOOTSTRAP, bootstrap_unit="completeindependentsourcevectors, pairedacrosscodecs",
        pcg_vectors_per_codec=PCG_VECTORS, solver="actual JacobiPCG,rtol=1e-11,atol=0,maxiter=4N",
        eigenvalue_certificate="A>=.01A_Poisson, lambda_lower=.01*lambda_min(A_Poisson)",
        master_seed=core.MASTER_SEED, stream_labels=stream_labels,
        numerical_source_sha256=exp.sha(source_snapshot), numerical_source_snapshot=source_snapshot.name,
        source_dependencies={"elliptic2d_application.py": exp.sha(OUT/"elliptic2d_application.py"),
            "poisson_application.py": exp.sha(OUT/"poisson_application.py"),
            "run_revision.py": exp.sha(OUT/"run_revision.py")})
    (OUT/"highcontrast_diffusion_prespecification.json").write_text(json.dumps(prespecification, indent=2)+"\n", encoding="utf-8")
    h, green, matrix, coefficient, lambda_lower, faces = diffusion_geometry()
    scales = exp.source_scales(MESH, "hotspot")
    candidates, cache = precision_candidates(scales, green)
    prespecification["configurations"] = candidates
    prespecification["lambda_lower"] = lambda_lower
    (OUT/"highcontrast_diffusion_prespecification.json").write_text(json.dumps(prespecification, indent=2)+"\n", encoding="utf-8")
    rows = predictions(scales, candidates, h, green, lambda_lower, cache)
    energies = np.empty((TRIALS, len(candidates)))
    minima, maxima = np.full(len(candidates), 10000), np.full(len(candidates), -10000)
    generator = core.rng_for(*stream_labels)
    input_hash = hashlib.sha256()
    first_inputs, first_errors = [], [[] for _ in candidates]
    batch = 512
    for begin in range(0, TRIALS, batch):
        inputs = generator.lognormal(size=(min(batch, TRIALS-begin), MESH*MESH))*scales
        input_hash.update(inputs.tobytes(order="C"))
        checks = max(0, min(PCG_VECTORS-begin, len(inputs)))
        if checks:
            first_inputs.extend(inputs[:checks])
        for j, cfg in enumerate(candidates):
            errors, emin, emax = codec.bounded_quantize(inputs, cfg)
            solved = errors@green
            energies[begin:begin+len(inputs), j] = h*h*np.sum(errors*solved, axis=1)
            minima[j], maxima[j] = min(minima[j], emin), max(maxima[j], emax)
            if checks:
                first_errors[j].extend(errors[:checks])
    ids = [cfg["id"] for cfg in candidates]
    np.savez_compressed(OUT/"highcontrast_diffusion_samples.npz", energy=energies,
        codec_ids=np.array(ids), exponent_min=minima, exponent_max=maxima,
        input_sha256=np.array(input_hash.hexdigest()), pcg_inputs=np.asarray(first_inputs),
        pcg_errors=np.asarray(first_errors), conductivity=coefficient,
        face_weights_x=faces["x"], face_weights_y=faces["y"], scales=scales)
    pcg_rows = []
    for j, row in enumerate(rows):
        values = energies[:, j]
        low, high = bootstrap_rms(values, row["reference_expected_energy"], row["id"])
        rms = math.sqrt(float(values.mean())/row["reference_expected_energy"])
        checks = pcg_audit(np.asarray(first_inputs), np.asarray(first_errors[j]), matrix, green, h, lambda_lower)
        pcg_rows.extend(dict(codec=row["id"], **record) for record in checks)
        row.update(trials=TRIALS, mc_energy_mse=float(values.mean()), mc_relative_rms=rms,
            mc_relative_rms_low=float(low), mc_relative_rms_high=float(high),
            exact_in_mc_interval=bool(low <= row["exact_relative_rms"] <= high),
            relative_prediction_error=rms/row["exact_relative_rms"]-1,
            observed_exponent_min=int(minima[j]), observed_exponent_max=int(maxima[j]),
            pcg_vectors=len(checks), pcg_max_relative_energy_error=max(a["relative_energy_error"] for a in checks),
            pcg_max_relative_energy_norm_certificate=max(a["relative_energy_norm_certificate"] for a in checks),
            pcg_max_iterations=max(max(a["original_iterations"], a["quantized_iterations"]) for a in checks),
            format_failures=0)
    comparisons = paired_comparisons(energies, ids, rows)
    codec.write_csv(OUT/"highcontrast_diffusion_results.csv", rows)
    codec.write_csv(OUT/"highcontrast_diffusion_pcg_audit.csv", pcg_rows)
    codec.write_csv(OUT/"highcontrast_diffusion_paired_comparisons.csv", comparisons)
    codec.write_csv(OUT/"highcontrast_diffusion_rankings.csv", rank_rows(rows))
    manifest = dict(prespecification, duration_seconds=time.time()-began, configurations_count=len(rows),
        exact_rms_ci_coverage=sum(row["exact_in_mc_interval"] for row in rows),
        maximum_relative_rms_prediction_error=max(abs(row["relative_prediction_error"]) for row in rows),
        maximum_pcg_relative_energy_error=max(row["pcg_max_relative_energy_error"] for row in rows),
        maximum_pcg_relative_energy_norm_certificate=max(row["pcg_max_relative_energy_norm_certificate"] for row in rows),
        maximum_pcg_iterations=max(row["pcg_max_iterations"] for row in rows),
        eigenvalue_lower_bound=lambda_lower, minimum_face_weight=float(min(faces["x"].min(), faces["y"].min())),
        source_input_sha256=input_hash.hexdigest(), format_failures=0)
    write_section(rows, comparisons, manifest)
    manifest["files"] = {path.name: exp.sha(path) for path in sorted(OUT.glob("highcontrast_diffusion_*"))
                         if path.is_file() and path.name != "highcontrast_diffusion_manifest.json"}
    manifest["generated_section_sha256"] = exp.sha(OUT.parent/"highcontrast_diffusion.tex")
    (OUT/"highcontrast_diffusion_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Completed14codecs,{TRIALS}pairedvectors:coverage{manifest['exact_rms_ci_coverage']}/14; "
          f"maxRMSdifference{100*manifest['maximum_relative_rms_prediction_error']:.3f}%; "
          f"PCGmaxrelativeenergyerror{manifest['maximum_pcg_relative_energy_error']:.2e}, "
          f"certificate{manifest['maximum_pcg_relative_energy_norm_certificate']:.2e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        pilot()
    else:
        run()
