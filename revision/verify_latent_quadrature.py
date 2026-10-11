"""Independent outer-quadrature and source-reference audit for latent sources.

Adaptive Gauss--Kronrod integration is compared with the producer's Hermite
mixture on two small configurations. Geometry uses an independently inverted
rational stencil. Inner moments were separately audited at 60/90 digits.
The full held-out source stream is also regenerated, without quantization, to
measure the analytical population reference; no prediction is fitted to it.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
from scipy import integrate, special

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"experiments"))
import elliptic2d_application as elliptic
import latent_elliptic_application as producer
import poisson_application as codec
import run_revision as core
import verify_poisson2d_nonidentical as rational


def configuration(name, blocks, levels):
    return dict(id=name, n=4, mesh=2, block_indices=blocks, L_by_block=levels,
                block_size=max(map(len, blocks)), blocks=len(blocks),
                base_L=min(levels), higher_L_blocks=0, layout=name,
                total_bits=sum(len(ids)*(level+3)+8 for ids, level in zip(blocks, levels)))


def outer_audit(config):
    mesh, tau, limit = 2, .6, 10.
    hx, hy, exact_green = rational.geometry(mesh, mesh)
    green = np.array([[float(value) for value in row] for row in exact_green])
    h = float(hx)
    scales, phi = elliptic.source_scales(mesh, "hotspot"), producer.loading(mesh)
    eigenvalue_min = float(np.linalg.eigvalsh(np.linalg.inv(green))[0])
    reference, _ = producer.source_reference(scales, phi, tau, h, green)
    def integrand(y):
        local_scales = scales*np.exp(tau*phi*y)
        moments = elliptic.full_error_moments(local_scales, config)
        mean, cov = moments["mean"], moments["cov"]
        bias = float(mean@green@mean)
        exact = h*h*(float(np.sum(green*cov))+bias)
        diagonal = h*h*(float(np.dot(np.diag(green), np.diag(cov)))+bias)
        source_mean = local_scales*math.exp(.5)
        source_variance = local_scales**2*(math.exp(2)-math.e)
        source = h*h*(float(np.dot(np.diag(green), source_variance))+float(source_mean@green@source_mean))
        return np.array([exact, diagonal, source])*math.exp(-y*y/2)/math.sqrt(2*math.pi)
    values, estimated_error = integrate.quad_vec(integrand, -limit, limit, epsabs=1e-13, epsrel=1e-11)
    raw_second_tail = scales**2*np.exp(2+2*tau*tau*phi*phi)*(
        special.ndtr(2*tau*phi-limit)+special.ndtr(-limit-2*tau*phi))
    source_tail_bound = h*h/eigenvalue_min*float(raw_second_tail.sum())
    error_tail_bound = source_tail_bound*config["block_size"]*2.**(-2*(config["base_L"]+1))
    rows = []
    for order in (21, 41, 61):
        prediction = producer.latent_population(config, mesh=mesh, tau=tau, order=order)
        differences = dict(exact_energy=abs(prediction["exact_energy_mse"]/values[0]-1),
                           conditional_diagonal_energy=abs(prediction["independent_energy_mse"]/values[1]-1),
                           analytical_source_energy=abs(reference/values[2]-1))
        assert max(differences.values()) < 1e-8
        rows.append(dict(order=order, relative_differences=differences,
                         hermite_exact_energy=prediction["exact_energy_mse"],
                         hermite_conditional_diagonal_energy=prediction["independent_energy_mse"]))
    return dict(config=config, status="PASS", adaptive_integral_range=[-limit, limit],
                adaptive_energy=float(values[0]), adaptive_conditional_diagonal_energy=float(values[1]),
                adaptive_source_energy=float(values[2]), analytical_source_energy=reference,
                adaptive_estimated_absolute_error=float(estimated_error),
                omitted_outer_source_energy_bound=source_tail_bound,
                omitted_outer_error_energy_bound=error_tail_bound, comparisons=rows)


def source_stream_audit():
    manifest = json.loads((ROOT/"experiments/latent_elliptic_manifest.json").read_text(encoding="utf-8"))
    mesh, tau, trials = manifest["mesh"], manifest["tau"], manifest["trials"]
    h, green, eigenvalues, _ = elliptic.geometry(mesh)
    scales, phi = elliptic.source_scales(mesh, "hotspot"), producer.loading(mesh)
    reference, _ = producer.source_reference(scales, phi, tau, h, green)
    stream = core.rng_for("latent_elliptic", "heldout", "input", mesh, tau)
    values = np.empty(trials)
    batch = max(1, 500000//(mesh*mesh))
    retained = np.load(ROOT/"experiments/latent_elliptic_samples.npz")
    for begin in range(0, trials, batch):
        count = min(batch, trials-begin)
        factor = stream.normal(size=count)
        inputs = scales*np.exp(stream.normal(size=(count, mesh*mesh))+tau*factor[:, None]*phi)
        if begin == 0:
            assert np.array_equal(inputs[:len(retained["cg_inputs"])], retained["cg_inputs"])
        # A fresh direct DST solve of the original sources; no codec errors enter.
        from scipy.fft import dstn
        transformed = dstn(inputs.reshape(-1, mesh, mesh), type=1, norm="ortho", axes=(1, 2))
        values[begin:begin+count] = h*h*np.sum(transformed**2/eigenvalues, axis=(1, 2))
    generator = core.rng_for("latent_elliptic", "source_reference_bootstrap")
    averages = np.array([values[generator.integers(trials, size=trials)].mean() for _ in range(300)])
    low, high = np.percentile(averages, [2.5, 97.5])
    np.savez_compressed(ROOT/"experiments/latent_elliptic_source_energy.npz", source_energy=values)
    return dict(trials=trials, analytical_expected_energy=reference,
                measured_expected_energy=float(values.mean()),
                relative_measurement_difference=float(values.mean()/reference-1),
                vector_bootstrap_interval=[float(low), float(high)],
                reference_in_interval=bool(low <= reference <= high),
                original_source_stream_regeneration="First100 saved full input vectors agree exactly",
                source_energy_file="experiments/latent_elliptic_source_energy.npz")


def main():
    cases = [configuration("single", [[0, 1, 2, 3]], [3]),
             configuration("cross_block_latent", [[0, 2], [1, 3]], [1, 3])]
    results = [outer_audit(config) for config in cases]
    measured = source_stream_audit()
    path = Path(__file__).resolve()
    report = dict(status="PASS", source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  producer_sha256=hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest(),
                  inner_source_sha256=hashlib.sha256(Path(elliptic.__file__).read_bytes()).hexdigest(),
                  method="Independent adaptive Gauss--Kronrod outer integral with rational2D inverse; analytical correlated source reference and regenerated held-out source measurement",
                  independence="Inner block evaluator has a separate high-precision audit; outer adaptive integration imports no Hermite nodes or weights",
                  quadrature_results=results, source_reference_measurement=measured)
    path.with_name("latent_quadrature_audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    maximum = max(error for row in results for order in row["comparisons"]
                  for error in order["relative_differences"].values())
    print(f"PASS: two latent-mixture configurations; largest GH/adaptive relative difference {maximum:.3e}")
    print(f"Analytical source reference relative MC difference {measured['relative_measurement_difference']:.3e}; "
          f"inside interval: {measured['reference_in_interval']}")


if __name__ == "__main__":
    main()
