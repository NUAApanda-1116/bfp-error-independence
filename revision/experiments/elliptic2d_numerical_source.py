"""Independent, nonidentically distributed sources in a 2D elliptic solve.

New experiment, separate from the earlier 1D and IID experiments.  The public
block_moments and geometry functions are exposed for independent audits.
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
import scipy
from scipy import special, sparse
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import cg

import poisson_application as codec
import run_revision as core

OUT = Path(__file__).resolve().parent
MESHES = (16, 32)
PROFILES = ("smooth", "hotspot")
BLOCK_SIZES = (8, 32, 128)
SIGMA = 1.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cdf(scales, bound, sigma=SIGMA):
    scales = np.asarray(scales, dtype=float)
    if bound <= 0:
        return np.zeros_like(scales)
    return special.ndtr(np.log(bound/scales)/sigma)


def _normal_interval(lower, upper):
    """Stable Phi(upper)-Phi(lower), including either far tail."""
    left = special.ndtr(upper)-special.ndtr(lower)
    right = special.ndtr(-lower)-special.ndtr(-upper)
    return np.where(upper <= 0, left, right)


def residual_integrals(scales, step, upper, sigma=SIGMA):
    """Unnormalized integrals of e and e^2 over 0<X<upper.

    Each input has X=scale*exp(N(0,sigma^2)).  Every intersecting nearest-
    rounding cell is integrated with analytical truncated raw moments.
    """
    scales = np.asarray(scales, dtype=float)
    centers = np.arange(int(math.ceil(upper/step+.5)))*step
    lower, higher = np.maximum(centers-step/2, 0), np.minimum(centers+step/2, upper)
    valid = higher > lower
    centers, lower, higher = centers[valid], lower[valid], higher[valid]
    zl = np.log(np.maximum(lower[None, :], np.finfo(float).tiny)/scales[:, None])/sigma
    zh = np.log(higher[None, :]/scales[:, None])/sigma
    zl[:, lower == 0] = -np.inf
    moments = [_normal_interval(zl-j*sigma, zh-j*sigma)
               *scales[:, None]**j*math.exp(j*j*sigma*sigma/2) for j in range(3)]
    p, first, second = moments
    e1 = np.sum(centers[None, :]*p-first, axis=1)
    e2 = np.sum(centers[None, :]**2*p-2*centers[None, :]*first+second, axis=1)
    if np.any(e2 < -2e-11*scales**2):
        raise ArithmeticError("negative residual second moment")
    return e1, np.maximum(e2, 0.)


def block_moments(scales, level, sigma=SIGMA, kmin=-40, kmax=35):
    """Exact finite-block full mean/covariance via maximum strata.

    Tiny outer strata are omitted without normalization and their probability
    and a fourth-moment error bound are returned.  No Monte Carlo is used.
    """
    scales = np.asarray(scales, dtype=float)
    size = len(scales)
    layers = []
    mean = np.zeros(size)
    second = np.zeros(size)
    conditional_accumulator = np.zeros((size, size))
    skipped_probabilities = []
    mean_step2 = 0.
    for exponent in range(kmin, kmax+1):
        lower, upper, step = 2.**exponent, 2.**(exponent+1), 2.**(exponent-level)
        logp = special.log_ndtr(np.log(lower/scales)/sigma)
        logq = special.log_ndtr(np.log(upper/scales)/sigma)
        log_lower, log_upper = float(logp.sum()), float(logq.sum())
        d = -math.expm1(log_lower-log_upper)
        probability = math.exp(log_upper)*d
        if probability < 1e-25:
            skipped_probabilities.append(probability)
            continue
        p, q = np.exp(logp), np.exp(logq)
        ia, qa = residual_integrals(scales, step, lower, sigma)
        ib, qb = residual_integrals(scales, step, upper, sigma)
        u, z = ia/p, ib/q
        u2, z2 = qa/p, qb/q
        r = math.exp(log_lower-log_upper)
        conditional_mean = (z-r*u)/d
        conditional_second = (z2-r*u2)/d
        conditional_cov = -r*np.outer(z-u, z-u)/d**2
        np.fill_diagonal(conditional_cov, conditional_second-conditional_mean**2)
        mean += probability*conditional_mean
        second += probability*conditional_second
        conditional_accumulator += probability*conditional_cov
        mean_step2 += probability*step*step
        layers.append(dict(k=exponent, probability=probability, step=step,
            conditional_mean=conditional_mean.tolist(),
            conditional_cov_min=float(np.min(conditional_cov-np.diag(np.diag(conditional_cov)))),
            conditional_cov_max=float(np.max(conditional_cov-np.diag(np.diag(conditional_cov))))))
    retained = math.fsum(layer["probability"] for layer in layers)
    covariance = conditional_accumulator.copy()
    for layer in layers:
        difference = np.asarray(layer["conditional_mean"])-mean
        covariance += layer["probability"]*np.outer(difference, difference)
    # e'=0 on omitted strata; its contribution to centered second moments is
    # missed_probability*mean*mean^T. Obtain the missed probability from the
    # tails and skipped strata, independently of subtracting a nearly-one sum.
    log_lower_mass = float(special.log_ndtr(np.log(2.**kmin/scales)/sigma).sum())
    lower_missing = math.exp(log_lower_mass)
    upper_missing = -math.expm1(float(special.log_ndtr(np.log(2.**(kmax+1)/scales)/sigma).sum()))
    missed = math.fsum(skipped_probabilities+[lower_missing, upper_missing])
    covariance += missed*np.outer(mean, mean)
    np.fill_diagonal(covariance, second-mean**2)
    # Quantization obeys ||e||² <= size*2^-2(level+1)||X||².
    # Cauchy bounds a missed-strata contribution by this factor times the
    # fourth norm moment and the square root of missed probability.
    x2 = scales**2*math.exp(2*sigma*sigma)
    x4 = scales**4*math.exp(8*sigma*sigma)
    fourth_norm = float(x4.sum()+x2.sum()**2-np.dot(x2, x2))
    numerical_tail_bound = size*2.**(-2*(level+1))*math.sqrt(fourth_norm*missed)
    return dict(mean=mean, cov=covariance, second=second, mean_step2=mean_step2,
                retained_probability=retained, omitted_error_second_norm_bound=numerical_tail_bound,
                omitted_probability_bound=missed,
                layers=layers)


def geometry(mesh):
    h = 1/(mesh+1)
    index = np.arange(1, mesh+1)
    one_eig = 4/h**2*np.sin(np.pi*index/(2*(mesh+1)))**2
    eigenvalues = one_eig[:, None]+one_eig[None, :]
    n = mesh*mesh
    eye = np.eye(n).reshape(n, mesh, mesh)
    green = idstn(dstn(eye, type=1, norm="ortho", axes=(1, 2))/eigenvalues,
                  type=1, norm="ortho", axes=(1, 2)).reshape(n, n)
    t = sparse.diags((-np.ones(mesh-1), 2*np.ones(mesh), -np.ones(mesh-1)),
                      offsets=(-1, 0, 1), format="csr")
    matrix = (sparse.kron(sparse.eye(mesh), t)+sparse.kron(t, sparse.eye(mesh)))/h**2
    defect = float(np.max(np.abs(matrix@green-np.eye(n))))
    if defect > 1e-12 or not np.allclose(green, green.T, rtol=1e-13, atol=1e-16):
        raise ArithmeticError("2D Green matrix identity failed")
    return h, green, eigenvalues, matrix.tocsr()


def source_scales(mesh, profile):
    points = np.arange(1, mesh+1)/(mesh+1)
    x, y = np.meshgrid(points, points, indexing="ij")
    if profile == "smooth":
        scales = 1+3*np.sin(np.pi*x)**2*np.sin(np.pi*y)**2
    elif profile == "hotspot":
        scales = 1+7*np.exp(-((x-.3)**2+(y-.65)**2)/.02)
    else:
        raise ValueError(profile)
    return scales.ravel()


def configurations(mesh):
    n, budget = mesh*mesh, 7*mesh*mesh
    configs = []
    for size in BLOCK_SIZES:
        count = n//size
        base = (budget-8*count)//n-3
        extra = (budget-8*count-n*(base+3))//size
        high = set(np.floor((np.arange(extra)+.5)*count/extra).astype(int)) if extra else set()
        levels = [base+int(j in high) for j in range(count)]
        for layout in ("contiguous", "interleaved"):
            ids = (np.arange(n).reshape(count, size) if layout == "contiguous"
                   else np.arange(n).reshape(size, count).T)
            payload = size*sum(level+3 for level in levels)+8*count
            configs.append(dict(id=f"b{size}_{layout}_uniform", mesh=mesh, n=n, block_size=size,
                blocks=count, layout=layout, base_L=base, higher_L_blocks=extra,
                precision_policy="uniform",
                L_by_block=levels, block_indices=ids.tolist(),
                coefficient_bits=payload-8*count, exponent_bits=8*count,
                padding_bits=budget-payload, total_bits=budget, bits_per_entry=7))
    return configs


def full_error_moments(scales, config, cache=None, sigma=SIGMA):
    """Unconditional full error mean and block-diagonal covariance."""
    cache = {} if cache is None else cache
    n = len(scales)
    mean, covariance = np.zeros(n), np.zeros((n, n))
    blocks = []
    for indices, level in zip(config["block_indices"], config["L_by_block"]):
        ids = np.array(indices)
        key = (tuple(scales[ids]), level, sigma)
        if key not in cache:
            cache[key] = block_moments(scales[ids], level, sigma=sigma)
        model = cache[key]
        mean[ids] = model["mean"]
        covariance[np.ix_(ids, ids)] = model["cov"]
        blocks.append(model)
    return dict(mean=mean, cov=covariance, blocks=blocks)


def allocation_variants(scales, mesh, green, cache=None):
    """All prescribed placements: uniform, source scale, exact energy greedy.

    Extra policies are tested on mesh=32; when no extra bits can be allocated,
    the identical policies are represented by a single uniform configuration.
    These are fixed spatial choices made before drawing the held-out vectors.
    """
    cache = {} if cache is None else cache
    result = []
    for cfg in configurations(mesh):
        result.append(cfg)
        if mesh != 32 or cfg["higher_L_blocks"] == 0:
            continue
        base, extra, count = cfg["base_L"], cfg["higher_L_blocks"], cfg["blocks"]
        ids_list = [np.asarray(indices) for indices in cfg["block_indices"]]
        scores = [float(np.dot(scales[ids], scales[ids])) for ids in ids_list]
        high = sorted(range(count), key=lambda j: (-scores[j], j))[:extra]
        source = dict(cfg, id=cfg["id"].replace("_uniform", "_source"),
                      precision_policy="source", L_by_block=[base+int(j in high) for j in range(count)])
        result.append(source)
        # Start at common lower precision. Each greedy increment minimizes the
        # full centered energy plus coherent squared bias, with fixed tie rule.
        models = []
        mean = np.zeros(len(scales))
        for ids in ids_list:
            pair = []
            for level in (base, base+1):
                key = (tuple(scales[ids]), level, SIGMA)
                if key not in cache:
                    cache[key] = block_moments(scales[ids], level)
                pair.append(cache[key])
            models.append(pair)
            mean[ids] = pair[0]["mean"]
        deltas = [pair[1]["mean"]-pair[0]["mean"] for pair in models]
        local_green = [green[np.ix_(ids, ids)] for ids in ids_list]
        centered_changes = [float(np.sum(g*(pair[1]["cov"]-pair[0]["cov"])))
                            for pair, g in zip(models, local_green)]
        high = set()
        greedy_sequence, greedy_changes = [], []
        for _ in range(extra):
            action = green@mean
            improvements = {}
            for j, ids in enumerate(ids_list):
                if j in high:
                    continue
                delta = deltas[j]
                improvements[j] = centered_changes[j]+2*float(delta@action[ids])+float(delta@local_green[j]@delta)
            best = min(improvements, key=lambda j: (improvements[j], j))
            high.add(best)
            mean[ids_list[best]] += deltas[best]
            greedy_sequence.append(best)
            greedy_changes.append(improvements[best])
        energy = dict(cfg, id=cfg["id"].replace("_uniform", "_energy"),
                      precision_policy="energy", L_by_block=[base+int(j in high) for j in range(count)],
                      greedy_sequence=greedy_sequence, greedy_unscaled_energy_changes=greedy_changes)
        result.append(energy)
    return result


def population(scales, config, h, green, eigenvalues, cache=None, sigma=SIGMA, return_moments=False):
    cache = {} if cache is None else cache
    n = len(scales)
    error_mean = np.zeros(n)
    exact_centered, independent_centered, uniform = 0., 0., 0.
    retained, omitted_norm_bound = 1., 0.
    conditional_min, conditional_max = 0., 0.
    for indices, level in zip(config["block_indices"], config["L_by_block"]):
        ids = np.array(indices)
        local_scales = scales[ids]
        key = (tuple(local_scales), level, sigma)
        if key not in cache:
            cache[key] = block_moments(local_scales, level, sigma=sigma)
        model = cache[key]
        local_green = green[np.ix_(ids, ids)]
        error_mean[ids] = model["mean"]
        exact_centered += float(np.sum(local_green*model["cov"]))
        independent_centered += float(np.dot(np.diag(local_green), np.diag(model["cov"])))
        uniform += model["mean_step2"]/12*float(np.trace(local_green))
        retained = min(retained, model["retained_probability"])
        omitted_norm_bound += model["omitted_error_second_norm_bound"]
        conditional_min = min(conditional_min, min(t["conditional_cov_min"] for t in model["layers"]))
        conditional_max = max(conditional_max, max(t["conditional_cov_max"] for t in model["layers"]))
    bias = float(error_mean@green@error_mean)
    source_mean = scales*math.exp(.5*sigma**2)
    source_var = scales**2*(math.exp(2*sigma**2)-math.exp(sigma**2))
    reference = h*h*(float(np.dot(source_var, np.diag(green)))+float(source_mean@green@source_mean))
    exact, independent, uniform = h*h*(exact_centered+bias), h*h*(independent_centered+bias), h*h*uniform
    row = {k: v for k, v in config.items() if k not in ("block_indices", "L_by_block", "greedy_sequence", "greedy_unscaled_energy_changes")}
    row.update(exact_energy_mse=exact, exact_energy_variance=h*h*exact_centered,
        energy_squared_bias=h*h*bias, independent_energy_mse=independent,
        uniform_energy_mse=uniform, reference_expected_energy=reference,
        exact_relative_rms=math.sqrt(exact/reference),
        independent_relative_rms=math.sqrt(independent/reference),
        uniform_relative_rms=math.sqrt(uniform/reference),
        minimum_retained_layer_probability=retained,
        omitted_residual_energy_bound=h*h/eigenvalues.min()*omitted_norm_bound,
        omitted_energy_mse_bound=2*math.sqrt(exact*h*h/eigenvalues.min()*omitted_norm_bound)
                                  +h*h/eigenvalues.min()*omitted_norm_bound,
        conditional_covariance_min=conditional_min, conditional_covariance_max=conditional_max)
    if return_moments:
        row["error_mean"] = error_mean
        row["error_covariance"] = full_error_moments(scales, config, cache, sigma=sigma)["cov"]
    return row


def pilot():
    rows = []
    start = time.time()
    for mesh in MESHES:
        h, green, eigenvalues, _ = geometry(mesh)
        for profile in PROFILES:
            scales = source_scales(mesh, profile)
            cache = {}
            for config in allocation_variants(scales, mesh, green, cache):
                row = dict(profile=profile, **population(scales, config, h, green, eigenvalues, cache))
                rows.append(row)
                print(f"{mesh} {profile} {config['id']}: exact={100*row['exact_relative_rms']:.4f}% "
                      f"ind={100*row['independent_relative_rms']:.4f}% "
                      f"unif={100*row['uniform_relative_rms']:.4f}%", flush=True)
    codec.write_csv(OUT/"elliptic2d_analytic_pilot.csv", rows)
    print(f"Pilot completed in {time.time()-start:.1f}s", flush=True)


def format_audit(scales, config, h, eigenvalues):
    """Exponent failure union bound and unconditional energy on failure."""
    low_log, high_log = -128*math.log(2), 128*math.log(2)
    lower_block_logs = []
    for indices in config["block_indices"]:
        ids = np.asarray(indices)
        lower_block_logs.append(float(special.log_ndtr((low_log-np.log(scales[ids]))/SIGMA).sum()))
    upper_logs = special.log_ndtr(-(high_log-np.log(scales))/SIGMA)
    log_failure = min(0., float(special.logsumexp(np.concatenate((lower_block_logs, upper_logs)))))
    source2 = scales**2*math.exp(2*SIGMA**2)
    source4 = scales**4*math.exp(8*SIGMA**2)
    fourth_norm = float(source4.sum()+source2.sum()**2-np.dot(source2, source2))
    factor = h*h/eigenvalues.min()*config["block_size"]*2.**(-2*(config["base_L"]+1))
    energy_log = math.log(factor)+.5*(math.log(fourth_norm)+log_failure)
    return dict(failure_probability_log10_bound=log_failure/math.log(10),
                failure_energy_mse_log10_bound=energy_log/math.log(10))


def cg_audit(inputs, residuals, matrix, eigenvalues, h, mesh):
    """Actual two CG solves; residuals certify the extra solution energy."""
    rows = []
    count = len(inputs)
    for index in range(count):
        b, e = inputs[index], residuals[index]
        u, info = cg(matrix, b, rtol=1e-11, atol=0., maxiter=4*mesh*mesh)
        uq, infoq = cg(matrix, b+e, rtol=1e-11, atol=0., maxiter=4*mesh*mesh)
        if info or infoq:
            raise RuntimeError(f"CG did not converge: {info},{infoq}")
        delta = uq-u
        delta_residual = matrix@delta-e
        r, rq = matrix@u-b, matrix@uq-(b+e)
        actual_energy = h*h*float(delta@(matrix@delta))
        transformed = dstn(e.reshape(mesh, mesh), type=1, norm="ortho")
        target_energy = h*h*float(np.sum(transformed**2/eigenvalues))
        certified_energy_error = h*float(np.linalg.norm(delta_residual))/math.sqrt(float(eigenvalues.min()))
        rows.append(dict(vector=index, target_energy=target_energy, cg_energy=actual_energy,
            relative_energy_error=abs(actual_energy/target_energy-1),
            relative_energy_norm_certificate=certified_energy_error/math.sqrt(target_energy),
            original_relative_residual=float(np.linalg.norm(r)/np.linalg.norm(b)),
            quantized_relative_residual=float(np.linalg.norm(rq)/np.linalg.norm(b+e)),
            delta_residual_norm=float(np.linalg.norm(delta_residual))))
    return rows


def paired_comparisons(rows, saved, bootstrap):
    comparisons = []
    for (mesh, profile), (energy, ids) in saved.items():
        for size in BLOCK_SIZES:
            baseline = f"b{size}_contiguous_uniform"
            for candidate in ids:
                if candidate == baseline or not candidate.startswith(f"b{size}_"):
                    continue
                first, second = energy[:, ids.index(baseline)], energy[:, ids.index(candidate)]
                generator = core.rng_for("elliptic2d", "comparison", mesh, profile, baseline, candidate)
                estimates = []
                for _ in range(bootstrap):
                    draws = generator.integers(len(energy), size=len(energy))
                    estimates.append(1-math.sqrt(second[draws].mean()/first[draws].mean()))
                low, high = np.percentile(estimates, [2.5, 97.5])
                population_first = next(row for row in rows if row["mesh"] == mesh and row["profile"] == profile and row["id"] == baseline)
                population_second = next(row for row in rows if row["mesh"] == mesh and row["profile"] == profile and row["id"] == candidate)
                comparisons.append(dict(mesh=mesh, profile=profile, block_size=size,
                    baseline=baseline, candidate=candidate,
                    exact_fractional_rms_reduction=1-population_second["exact_relative_rms"]/population_first["exact_relative_rms"],
                    mc_fractional_rms_reduction=1-math.sqrt(second.mean()/first.mean()),
                    mc_fractional_rms_reduction_low=float(low), mc_fractional_rms_reduction_high=float(high),
                    bootstrap_replicates=bootstrap, trials=len(energy)))
    return comparisons


def ranking(rows):
    rankings = []
    for mesh in MESHES:
        for profile in PROFILES:
            group = [row for row in rows if row["mesh"] == mesh and row["profile"] == profile]
            for method in ("exact", "independent", "uniform", "mc"):
                ordered = sorted(group, key=lambda row: (row[method+"_relative_rms"], row["id"]))
                winner = ordered[0]
                rankings.append(dict(mesh=mesh, profile=profile, method=method, chosen_codec=winner["id"],
                    predicted_relative_rms=winner[method+"_relative_rms"],
                    observed_relative_rms=winner["mc_relative_rms"],
                    ordered_codecs=";".join(row["id"] for row in ordered)))
    return rankings


def run(trials16=20000, trials32=10000, bootstrap=300, cg_vectors=100):
    start_time = time.time()
    trials = {16: trials16, 32: trials32}
    source_snapshot = OUT/"elliptic2d_numerical_source.py"
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    # This declaration is written before evaluating placements or sampling.
    prespecification = dict(date="2026-10-10 Asia/Shanghai", meshes=list(MESHES), profiles=list(PROFILES),
        laws="b_i=s_i exp(Z_i), mutually independent Z_i~N(0,1)",
        source_envelopes={"smooth": "1+3 sin(pi*x)^2 sin(pi*y)^2",
            "hotspot": "1+7 exp(-((x-.3)^2+(y-.65)^2)/.02)"},
        trials_by_mesh=trials, bootstrap_replicates=bootstrap, cg_vectors_per_codec=cg_vectors,
        solver="scipy.sparse.linalg.cg, rtol=1e-11, atol=0, maxiter=4N; unquantized fixed2Doperator",
        reference="h^2 E[b^T A^-1 b]; deterministic population reference",
        codec="same actual fixed7bits/cell packed two's-complement codec as1D; no escaping/saturation",
        policies={"uniform": "evenly spaced extra-bit blocks, fixed prescribed locations",
            "source": "descending sum s_i^2, ties by block index; mesh32 only",
            "energy": "greedy exact centered-energy+squared-bias increments, ties by block index; mesh32 only"},
        block_sizes=list(BLOCK_SIZES), layouts=["contiguous", "interleaved"],
        sampling="all prescribed distinct layouts/policies retained, paired source vectors acrosscodecs",
        rng="core.rng_for('elliptic2d','heldout','input',mesh,profile), PCG64 SHA256labels",
        analytic_pilot="initial24config pilot before placement policies; all pilot results retained",
        numerical_source_sha256=sha(source_snapshot),
        source_dependencies={"poisson_application.py": sha(OUT/"poisson_application.py"),
            "run_revision.py": sha(OUT/"run_revision.py")})
    (OUT/"elliptic2d_prespecification.json").write_text(json.dumps(prespecification, indent=2)+"\n", encoding="utf-8")
    setups = {}
    for mesh in MESHES:
        h, green, eigenvalues, matrix = geometry(mesh)
        for profile in PROFILES:
            scales, cache = source_scales(mesh, profile), {}
            configs = allocation_variants(scales, mesh, green, cache)
            setups[mesh, profile] = (h, green, eigenvalues, matrix, scales, configs, cache)
    prespecification["configurations"] = {f"m{mesh}_{profile}": setups[mesh, profile][5]
                                          for mesh in MESHES for profile in PROFILES}
    # Exact spatial placements are now durably recorded before the first sample.
    (OUT/"elliptic2d_prespecification.json").write_text(json.dumps(prespecification, indent=2)+"\n", encoding="utf-8")
    all_rows, solver_rows, saved = [], [], {}
    for mesh in MESHES:
        n = mesh*mesh
        for profile in PROFILES:
            h, green, eigenvalues, matrix, scales, configs, cache = setups[mesh, profile]
            predictions = [dict(profile=profile, **population(scales, cfg, h, green, eigenvalues, cache),
                                **format_audit(scales, cfg, h, eigenvalues)) for cfg in configs]
            generator = core.rng_for("elliptic2d", "heldout", "input", mesh, profile)
            energy = np.empty((trials[mesh], len(configs)))
            minima, maxima = np.full(len(configs), 10000), np.full(len(configs), -10000)
            cg_inputs, cg_errors = [], [[] for _ in configs]
            batch = max(1, 500000//n)
            for begin in range(0, trials[mesh], batch):
                inputs = generator.lognormal(size=(min(batch, trials[mesh]-begin), n))*scales
                to_check = max(0, min(cg_vectors-begin, len(inputs)))
                if to_check:
                    cg_inputs.extend(inputs[:to_check])
                for j, cfg in enumerate(configs):
                    errors, emin, emax = codec.bounded_quantize(inputs, cfg)
                    transformed = dstn(errors.reshape(-1, mesh, mesh), type=1, norm="ortho", axes=(1, 2))
                    energy[begin:begin+len(inputs), j] = h*h*np.sum(transformed**2/eigenvalues, axis=(1, 2))
                    minima[j], maxima[j] = min(minima[j], emin), max(maxima[j], emax)
                    if to_check:
                        cg_errors[j].extend(errors[:to_check])
            ids = [cfg["id"] for cfg in configs]
            np.savez_compressed(OUT/f"elliptic2d_samples_m{mesh}_{profile}.npz", energy=energy,
                                codec_ids=np.array(ids), exponent_min=minima, exponent_max=maxima,
                                cg_inputs=np.array(cg_inputs), cg_errors=np.array(cg_errors))
            saved[mesh, profile] = energy, ids
            for j, row in enumerate(predictions):
                values = energy[:, j]
                rms = math.sqrt(float(values.mean())/row["reference_expected_energy"])
                low, high = codec.bootstrap_rms(values, row["reference_expected_energy"],
                    ("elliptic2d", mesh, profile, row["id"]), bootstrap)
                audits = cg_audit(np.asarray(cg_inputs), np.asarray(cg_errors[j]), matrix, eigenvalues, h, mesh)
                for audit in audits:
                    solver_rows.append(dict(mesh=mesh, profile=profile, codec=row["id"], **audit))
                row.update(trials=trials[mesh], mc_energy_mse=float(values.mean()), mc_relative_rms=rms,
                    mc_relative_rms_low=float(low), mc_relative_rms_high=float(high),
                    exact_in_mc_interval=bool(low <= row["exact_relative_rms"] <= high),
                    relative_prediction_error=rms/row["exact_relative_rms"]-1,
                    observed_exponent_min=int(minima[j]), observed_exponent_max=int(maxima[j]),
                    cg_vectors=len(audits), cg_max_relative_energy_error=max(a["relative_energy_error"] for a in audits),
                    cg_max_relative_energy_norm_certificate=max(a["relative_energy_norm_certificate"] for a in audits),
                    format_failures=0)
                all_rows.append(row)
            codec.write_csv(OUT/"elliptic2d_results.csv", all_rows)
            print(f"Completed {mesh}x{mesh} {profile}: {trials[mesh]} paired vectors, {len(configs)} codecs, "
                  f"{len(cg_inputs)} CG checks/codec.", flush=True)
    codec.write_csv(OUT/"elliptic2d_cg_audit.csv", solver_rows)
    codec.write_csv(OUT/"elliptic2d_paired_comparisons.csv", paired_comparisons(all_rows, saved, bootstrap))
    codec.write_csv(OUT/"elliptic2d_rankings.csv", ranking(all_rows))
    manifest = dict(prespecification, numerical_source_snapshot=source_snapshot.name,
        duration_seconds=time.time()-start_time, configuration_count=len(all_rows),
        exact_rms_ci_coverage=sum(row["exact_in_mc_interval"] for row in all_rows),
        maximum_relative_rms_prediction_error=max(abs(row["relative_prediction_error"]) for row in all_rows),
        maximum_cg_relative_energy_error=max(row["cg_max_relative_energy_error"] for row in all_rows),
        maximum_cg_relative_energy_norm_certificate=max(row["cg_max_relative_energy_norm_certificate"] for row in all_rows),
        maximum_omitted_energy_mse_bound=max(row["omitted_energy_mse_bound"] for row in all_rows),
        failure_probability_log10_bound=max(row["failure_probability_log10_bound"] for row in all_rows),
        failure_energy_mse_log10_bound=max(row["failure_energy_mse_log10_bound"] for row in all_rows),
        format_failures=0, numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
        platform=platform.platform())
    files = [path for path in OUT.glob("elliptic2d_*") if path.is_file() and path.name != "elliptic2d_manifest.json"]
    manifest["files"] = {path.name: sha(path) for path in sorted(files)}
    (OUT/"elliptic2d_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Retained {len(all_rows)} configs; exact coverage {manifest['exact_rms_ci_coverage']}/{len(all_rows)}; "
          f"max RMS discrepancy {100*manifest['maximum_relative_rms_prediction_error']:.2f}%; "
          f"max CG energy relative error {manifest['maximum_cg_relative_energy_error']:.2e}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--trials16", type=int, default=20000)
    parser.add_argument("--trials32", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--cg-vectors", type=int, default=100)
    args = parser.parse_args()
    if args.pilot:
        pilot()
    else:
        run(args.trials16, args.trials32, args.bootstrap, args.cg_vectors)
