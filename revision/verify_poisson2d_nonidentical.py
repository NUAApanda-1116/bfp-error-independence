"""Independent rational audit of heterogeneous BFP sources in a 2D solve.

No producer or numerical experiment is imported. Nested rectangular events
give the full nonidentical block moments. A separate complete RHS enumeration
quantizes rational values, solves with an exact inverse obtained by Gaussian
elimination, and evaluates squared grid differences including the boundary.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json


def multiply(values):
    result = F(1)
    for value in values:
        result *= value
    return result


def exponent(values):
    maximum = max(map(abs, values))
    if maximum == 0:
        return None
    k, scale = 0, F(1)
    while maximum < scale:
        scale /= 2
        k -= 1
    while maximum >= 2 * scale:
        scale *= 2
        k += 1
    return k


def error(value, k, level):
    if k is None:
        return F(0)
    step = F(2) ** (k - level)
    scaled = value / step
    lower = scaled.numerator // scaled.denominator
    remainder = scaled - lower
    rounded = lower + int(remainder > F(1, 2)
                          or (remainder == F(1, 2) and lower % 2))
    return rounded * step - value


def event_moments(laws, level):
    """Direct product-event subtraction, without normalized covariance formulas."""
    n = len(laws)
    possible_states = sorted({exponent([x]) for law in laws for x, _ in law if x})
    means = [F(0)] * n
    raw = [[F(0)] * n for _ in range(n)]
    mass = multiply(sum(p for x, p in law if x == 0) for law in laws)
    states = []
    for k in possible_states:
        a, b = F(2) ** k, F(2) ** (k + 1)
        lower = [sum(p for x, p in law if abs(x) < a) for law in laws]
        upper = [sum(p for x, p in law if abs(x) < b) for law in laws]
        probability = multiply(upper) - multiply(lower)
        if probability == 0:
            continue
        tl = [[sum(p * error(x, k, level) ** ell
                   for x, p in law if abs(x) < a) for ell in (1, 2)]
              for law in laws]
        tu = [[sum(p * error(x, k, level) ** ell
                   for x, p in law if abs(x) < b) for ell in (1, 2)]
              for law in laws]
        conditional_mean = []
        conditional_raw = [[F(0)] * n for _ in range(n)]
        for i in range(n):
            keep = [j for j in range(n) if j != i]
            numerator = tu[i][0] * multiply(upper[j] for j in keep)
            numerator -= tl[i][0] * multiply(lower[j] for j in keep)
            means[i] += numerator
            conditional_mean.append(numerator / probability)
            diagonal = tu[i][1] * multiply(upper[j] for j in keep)
            diagonal -= tl[i][1] * multiply(lower[j] for j in keep)
            raw[i][i] += diagonal
            conditional_raw[i][i] = diagonal / probability
            for j in range(i):
                keep_pair = [r for r in range(n) if r not in (i, j)]
                pair = tu[i][0] * tu[j][0] * multiply(upper[r] for r in keep_pair)
                pair -= tl[i][0] * tl[j][0] * multiply(lower[r] for r in keep_pair)
                raw[i][j] += pair
                raw[j][i] += pair
                conditional_raw[i][j] = conditional_raw[j][i] = pair / probability
        mass += probability
        states.append(dict(k=k, p=probability, mean=conditional_mean, raw=conditional_raw))
    assert mass == 1
    covariance = [[raw[i][j] - means[i] * means[j] for j in range(n)]
                  for i in range(n)]
    return means, covariance, states


def inverse(matrix):
    n = len(matrix)
    rows = [list(matrix[i]) + [F(i == j) for j in range(n)] for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]
        rows[j] = [entry / divisor for entry in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [left - factor * right for left, right in zip(rows[i], rows[j])]
    assert all(rows[i][j] == int(i == j) for i in range(n) for j in range(n))
    return [row[n:] for row in rows]


def geometry(rows, columns):
    hx, hy = F(1, columns + 1), F(1, rows + 1)
    n = rows * columns
    matrix = [[F(0)] * n for _ in range(n)]
    for r in range(rows):
        for c in range(columns):
            i = r * columns + c
            matrix[i][i] = 2 / hx**2 + 2 / hy**2
            for rr, cc, spacing in ((r - 1, c, hy), (r + 1, c, hy),
                                    (r, c - 1, hx), (r, c + 1, hx)):
                if 0 <= rr < rows and 0 <= cc < columns:
                    matrix[i][rr * columns + cc] = -1 / spacing**2
    green = inverse(matrix)
    assert all(sum(matrix[i][r] * green[r][j] for r in range(n)) == int(i == j)
               for i in range(n) for j in range(n))
    return hx, hy, green


def bilinear(left, matrix, right):
    return sum(left[i] * matrix[i][j] * right[j]
               for i in range(len(left)) for j in range(len(right)))


def gradient_energy(rhs, rows, columns, hx, hy, green):
    solution = [sum(row[j] * rhs[j] for j in range(len(rhs))) for row in green]
    energy = F(0)
    for r in range(rows):
        values = [F(0)] + solution[r * columns:(r + 1) * columns] + [F(0)]
        energy += sum((b - a)**2 * hy / hx for a, b in zip(values, values[1:]))
    for c in range(columns):
        values = [F(0)] + [solution[r * columns + c] for r in range(rows)] + [F(0)]
        energy += sum((b - a)**2 * hx / hy for a, b in zip(values, values[1:]))
    return energy


def coordinate_laws(n, profile):
    scales = [F(1), F(2), F(1, 2), F(3, 2), F(3, 4), F(5, 4)][:n]
    if profile == "one_sided":
        return [[(scale * F(13, 10), F(i + 1, i + 3)),
                 (scale * F(41, 20), F(2, i + 3))]
                for i, scale in enumerate(scales)]
    return [[(F(0), F(1, 4)), (scale * F((-1)**i * 3, 4), F(1, 4)),
             (scale * F(9, 4), F(1, 2))] for i, scale in enumerate(scales)]


def verify(rows, columns, profile, layout, level):
    n = rows * columns
    laws = coordinate_laws(n, profile)
    if layout == "single":
        blocks = [list(range(n))]
    elif layout == "row":
        blocks = [list(range(r * columns, (r + 1) * columns)) for r in range(rows)]
    else:
        blocks = [list(range(c, n, columns)) for c in range(columns)]
    levels = [level + j % 2 for j in range(len(blocks))]
    hx, hy, green = geometry(rows, columns)
    means = [F(0)] * n
    covariance = [[F(0)] * n for _ in range(n)]
    conditional_signs = set()
    for ids, local_level in zip(blocks, levels):
        local_mean, local_cov, states = event_moments([laws[i] for i in ids], local_level)
        for i, coordinate in enumerate(ids):
            means[coordinate] = local_mean[i]
            for j, other in enumerate(ids):
                covariance[coordinate][other] = local_cov[i][j]
        for state in states:
            for i in range(len(ids)):
                for j in range(i):
                    cov = state["raw"][i][j] - state["mean"][i] * state["mean"][j]
                    conditional_signs.add(int(cov > 0) - int(cov < 0))
    predicted = hx * hy * (sum(covariance[i][j] * green[i][j]
                              for i in range(n) for j in range(n))
                          + bilinear(means, green, means))
    direct = mass = F(0)
    direct_means = [F(0)] * n
    direct_raw = [[F(0)] * n for _ in range(n)]
    states = 0
    for outcome in product(*laws):
        rhs = [x for x, _ in outcome]
        probability = multiply(p for _, p in outcome)
        errors = [F(0)] * n
        for ids, local_level in zip(blocks, levels):
            k = exponent([rhs[i] for i in ids])
            for i in ids:
                errors[i] = error(rhs[i], k, local_level)
        direct += probability * gradient_energy(errors, rows, columns, hx, hy, green)
        for i in range(n):
            direct_means[i] += probability * errors[i]
            for j in range(n):
                direct_raw[i][j] += probability * errors[i] * errors[j]
        mass += probability
        states += 1
    assert mass == 1
    assert direct == predicted
    assert means == direct_means
    assert all(covariance[i][j] == direct_raw[i][j] - direct_means[i] * direct_means[j]
               for i in range(n) for j in range(n))
    return dict(grid=[rows, columns], profile=profile, layout=layout,
                levels=levels, enumerated_rhs=states, status="PASS",
                conditional_covariance_signs=sorted(conditional_signs),
                energy_mse=str(direct), energy_difference="0", full_covariance_difference="0")


def main():
    results = [verify(rows, columns, profile, layout, level)
               for rows, columns in ((2, 2), (2, 3))
               for profile in ("one_sided", "zeros_ties")
               for layout in ("single", "row", "column")
               for level in (0, 2)]
    signs = {sign for row in results for sign in row["conditional_covariance_signs"]}
    assert signs == {-1, 0, 1}
    path = Path(__file__).resolve()
    report = dict(status="PASS", configurations=len(results),
                  enumerated_rhs=sum(row["enumerated_rhs"] for row in results),
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  exact_arithmetic="fractions.Fraction",
                  independence="No imports from producer or experiment implementations",
                  method="Nested product-event moments versus complete joint quantization, exact 2D inverse, and squared grid differences",
                  results=results)
    path.with_name("poisson2d_nonidentical_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {report['configurations']} heterogeneous 2D configurations; "
          f"{report['enumerated_rhs']} complete rational right-hand sides")


if __name__ == "__main__":
    main()
