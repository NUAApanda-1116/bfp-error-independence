"""Independent operator and solver audit of the high-contrast diffusion case.

Run from SISC_submission:
    python revision/verify_highcontrast_diffusion.py

Only the public diffusion_geometry function is imported from the producer.
The oracle constructs nodal conductivities and edge weights with exact
rational comparisons, assembles the operator from its edge list, evaluates
energies as sums of squared edge differences, and implements Jacobi-PCG
without scipy.sparse.linalg.cg or the producer's solve/energy code.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import sys

import numpy as np
import scipy
from scipy import linalg, sparse


HERE = Path(__file__).resolve().parent
LOW = F(1, 100)


def exact_conductivity(x, y):
    """The disk predicate uses rational coordinates, including its boundary."""
    return LOW if (x-F(7, 10))**2+(y-F(3, 10))**2 <= F(1, 20) else F(1)


def rational_geometry(mesh):
    """Enumerate undirected interior edges and individual Dirichlet edges."""
    h = F(1, mesh+1)
    coefficient = [[exact_conductivity((i+1)*h, (j+1)*h)
                    for j in range(mesh)] for i in range(mesh)]
    edges = []
    faces_x = np.zeros((mesh+1, mesh))
    faces_y = np.zeros((mesh, mesh+1))
    for i in range(mesh):
        for j in range(mesh):
            node = i*mesh+j
            current = coefficient[i][j]
            # Each interior edge appears once; each boundary side appears once.
            for di, dj in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                ni, nj = i+di, j+dj
                interior = 0 <= ni < mesh and 0 <= nj < mesh
                if interior and (di < 0 or dj < 0):
                    continue
                other = coefficient[ni][nj] if interior else exact_conductivity(
                    (ni+1)*h, (nj+1)*h)
                weight = 2*current*other/(current+other)
                neighbor = ni*mesh+nj if interior else -1
                edges.append((node, neighbor, weight))
                if di:
                    faces_x[i+int(di > 0), j] = float(weight)
                else:
                    faces_y[i, j+int(dj > 0)] = float(weight)
    n = mesh*mesh
    rows, columns, values = [], [], []
    # Sum rank-one edge contributions, independent of the producer's stencil.
    for left, right, weight in edges:
        scaled = float(weight/h**2)
        rows.append(left)
        columns.append(left)
        values.append(scaled)
        if right >= 0:
            rows.extend((right, left, right))
            columns.extend((right, right, left))
            values.extend((scaled, -scaled, -scaled))
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()
    return float(h), matrix, np.array(coefficient, dtype=float), edges, dict(
        x=faces_x, y=faces_y)


def edge_energy(vector, edges):
    """No matrix multiplication: include both endpoints and zero boundary."""
    terms = (float(weight)*(float(vector[left])-
             (float(vector[right]) if right >= 0 else 0.))**2
             for left, right, weight in edges)
    return math.fsum(terms)


def jacobi_pcg(matrix, rhs, relative_tolerance=2e-11):
    """Independent PCG, with a recomputed residual before accepting a solve."""
    rhs = np.asarray(rhs, dtype=float)
    rhs_norm = float(np.linalg.norm(rhs))
    solution = np.zeros_like(rhs)
    if rhs_norm == 0:
        return solution, 0, 0.
    diagonal = matrix.diagonal()
    residual = rhs.copy()
    preconditioned = residual/diagonal
    direction = preconditioned.copy()
    old_product = float(residual@preconditioned)
    for iteration in range(1, 8*len(rhs)+1):
        action = matrix@direction
        denominator = float(direction@action)
        assert denominator > 0, "PCG lost positive definiteness"
        step = old_product/denominator
        solution += step*direction
        residual -= step*action
        if np.linalg.norm(residual) <= relative_tolerance*rhs_norm:
            true_residual = rhs-matrix@solution
            true_relative = float(np.linalg.norm(true_residual)/rhs_norm)
            if true_relative <= relative_tolerance:
                return solution, iteration, true_relative
            # Restart only if recursive cancellation understates the residual.
            residual = true_residual
            preconditioned = residual/diagonal
            direction = preconditioned.copy()
            old_product = float(residual@preconditioned)
            continue
        preconditioned = residual/diagonal
        new_product = float(residual@preconditioned)
        assert new_product > 0, "PCG preconditioned residual became nonpositive"
        direction = preconditioned+(new_product/old_product)*direction
        old_product = new_product
    raise ArithmeticError("Independent Jacobi-PCG failed to reach tolerance")


def assert_small(value, tolerance, name):
    if value > tolerance:
        raise AssertionError(f"{name}: {value:.6e} exceeds {tolerance:.6e}")


def audit(mesh, vectors, producer_geometry):
    h, green, matrix, coefficient, lambda_lower, faces = producer_geometry(mesh)
    oh, oracle, oc, edges, ofaces = rational_geometry(mesh)
    n = mesh*mesh
    assert h == oh
    assert matrix.shape == oracle.shape == (n, n)
    assert green.shape == (n, n)
    assert np.all(np.isfinite(green))
    coefficient_defect = float(np.max(np.abs(np.asarray(coefficient)-oc)))
    face_defect = max(float(np.max(np.abs(faces[axis]-ofaces[axis])))
                      for axis in ("x", "y"))
    matrix_defect = float(np.max(np.abs((matrix-oracle).toarray())))
    matrix_scale = float(np.max(np.abs(oracle.toarray())))
    assert_small(coefficient_defect, 1e-15, "nodal conductivity reconstruction")
    assert_small(face_defect, 2e-15, "harmonic face reconstruction")
    assert_small(matrix_defect, 2e-13*matrix_scale, "edge operator reconstruction")

    dense = oracle.toarray()
    matrix_symmetry = float(np.max(np.abs(dense-dense.T)))
    green_symmetry = float(np.max(np.abs(green-green.T)))
    assert_small(matrix_symmetry, 1e-14*matrix_scale, "matrix symmetry")
    assert_small(green_symmetry, 2e-12*float(np.max(np.abs(green))), "Green symmetry")
    factor = linalg.cho_factor(dense, lower=True, check_finite=True)
    independent_green = linalg.cho_solve(factor, np.eye(n))
    green_defect = float(np.max(np.abs(green-independent_green)))
    inverse_defect = float(np.max(np.abs(matrix@green-np.eye(n))))
    assert_small(green_defect, 2e-11*float(np.max(np.abs(independent_green))),
                 "Cholesky inverse versus producer sparse LU inverse")
    assert_small(inverse_defect, 2e-11, "sparse A times Green identity")

    one_min = 4/h**2*math.sin(math.pi/(2*(mesh+1)))**2
    expected_lower = float(LOW)*2*one_min
    assert_small(abs(lambda_lower-expected_lower), 1e-14,
                 "declared Poisson comparison lower bound")
    eigenvalues = linalg.eigvalsh(dense)
    minimum_eigenvalue = float(eigenvalues[0])
    assert minimum_eigenvalue > 0
    assert minimum_eigenvalue >= lambda_lower*(1-1e-12)
    # The rational edge inequality proves the lower bound for every vector;
    # the eigenvalue calculation is an additional numerical check.
    assert all(weight >= LOW for _, _, weight in edges)
    plain = sparse.diags((-np.ones(mesh-1), 2*np.ones(mesh), -np.ones(mesh-1)),
                         offsets=(-1, 0, 1), format="csr")
    poisson = (sparse.kron(sparse.eye(mesh), plain)+
               sparse.kron(plain, sparse.eye(mesh)))/h**2
    comparison_min = float(linalg.eigvalsh(dense-float(LOW)*poisson.toarray())[0])
    assert comparison_min >= -1e-11*matrix_scale

    generator = np.random.default_rng(20261010+mesh)
    energy_defects = []
    pcg_rows = []
    for index in range(vectors):
        potential = generator.standard_normal(n)
        direct = edge_energy(potential, edges)
        quadratic = h*h*float(potential@(matrix@potential))
        energy_defects.append(abs(quadratic/direct-1))
        rhs = generator.standard_normal(n)*10.**generator.uniform(-2, 2, n)
        error = generator.uniform(-.25, .25, n)
        reference = linalg.cho_solve(factor, error)
        direct_rhs_energy = edge_energy(reference, edges)
        green_rhs_energy = h*h*float(error@green@error)
        energy_defects.append(abs(green_rhs_energy/direct_rhs_energy-1))
        original, steps, residual = jacobi_pcg(oracle, rhs)
        quantized, qsteps, qresidual = jacobi_pcg(oracle, rhs+error)
        delta = quantized-original
        delta_residual = error-oracle@delta
        true_norm_error = math.sqrt(edge_energy(delta-reference, edges))
        certificate = h*float(np.linalg.norm(delta_residual))/math.sqrt(lambda_lower)
        # Reference solve roundoff is negligible at the requested PCG tolerance.
        reference_floor = 1e-12*math.sqrt(direct_rhs_energy)
        assert true_norm_error <= certificate*(1+1e-6)+reference_floor
        cg_energy = edge_energy(delta, edges)
        pcg_rows.append(dict(vector=index, iterations=steps, quantized_iterations=qsteps,
            original_relative_residual=residual, quantized_relative_residual=qresidual,
            target_energy=direct_rhs_energy, pcg_energy=cg_energy,
            relative_energy_error=abs(cg_energy/direct_rhs_energy-1),
            relative_true_energy_norm_error=true_norm_error/math.sqrt(direct_rhs_energy),
            relative_energy_norm_certificate=certificate/math.sqrt(direct_rhs_energy),
            true_norm_error_to_certificate=true_norm_error/certificate if certificate else 0.))
    maximum_energy_defect = max(energy_defects)
    assert_small(maximum_energy_defect, 2e-11, "explicit edge versus quadratic energy")
    return dict(mesh=mesh, unknowns=n, low_conductivity_nodes=int(np.count_nonzero(oc < 1)),
        interior_edges=sum(right >= 0 for _, right, _ in edges),
        dirichlet_edges=sum(right < 0 for _, right, _ in edges),
        exact_weight_values=sorted({str(weight) for _, _, weight in edges}),
        coefficient_max_defect=coefficient_defect, face_max_defect=face_defect,
        matrix_max_defect=matrix_defect, matrix_symmetry_max_defect=matrix_symmetry,
        green_symmetry_max_defect=green_symmetry, inverse_max_defect=inverse_defect,
        independent_inverse_max_defect=green_defect, minimum_eigenvalue=minimum_eigenvalue,
        rigorous_lambda_lower=lambda_lower, comparison_minimum_eigenvalue=comparison_min,
        maximum_relative_energy_identity_defect=maximum_energy_defect,
        pcg_pairs=vectors,
        maximum_pcg_relative_residual=max(max(row["original_relative_residual"],
                                             row["quantized_relative_residual"])
                                          for row in pcg_rows),
        maximum_pcg_relative_energy_error=max(row["relative_energy_error"] for row in pcg_rows),
        maximum_relative_energy_norm_certificate=max(row["relative_energy_norm_certificate"]
                                                    for row in pcg_rows),
        maximum_true_error_to_certificate=max(row["true_norm_error_to_certificate"]
                                             for row in pcg_rows),
        pcg_results=pcg_rows, status="PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--vectors", type=int, default=12)
    args = parser.parse_args()
    assert args.vectors > 0 and all(mesh >= 2 for mesh in args.meshes)
    sys.path.insert(0, str(HERE/"experiments"))
    import highcontrast_diffusion as producer
    results = [audit(mesh, args.vectors, producer.diffusion_geometry) for mesh in args.meshes]
    report = dict(status="PASS", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        independence="Only diffusion_geometry imported; rational edge oracle, independent Cholesky and manual Jacobi-PCG",
        certificate="||delta-delta_exact||_{h^2 A} <= h ||e-A delta||_2 / sqrt(lambda_lower)",
        lower_bound_proof="Every exact rational edge weight >= 1/100; edge sums imply A >= A_Poisson/100",
        results=results)
    output = HERE/"highcontrast_diffusion_audit.json"
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    for row in results:
        print(f"PASS mesh={row['mesh']}: eigenvalue={row['minimum_eigenvalue']:.6g} "
              f">= {row['rigorous_lambda_lower']:.6g}; "
              f"edge-energy defect={row['maximum_relative_energy_identity_defect']:.3g}; "
              f"PCG pairs={row['pcg_pairs']}, "
              f"max relative energy error={row['maximum_pcg_relative_energy_error']:.3g}")
    print(f"Audit report: {output}")


if __name__ == "__main__":
    main()
