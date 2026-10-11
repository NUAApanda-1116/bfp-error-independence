"""Independent high-precision Pareto cell-integral audit of the population model."""
from pathlib import Path
import csv
import json
import mpmath as mp

mp.mp.dps = 80
Z = mp.mpf(0)


def antiderivative(power, q, x):
    if power == 1:
        return -q * x ** -4 + mp.mpf(4) / 3 * x ** -3
    return -q * q * x ** -4 + mp.mpf(8) / 3 * q * x ** -3 - 2 * x ** -2


def truncated(k, L, t, power):
    s = mp.power(2, k - L)
    result = Z
    for j in range(2 ** (L + 1) + 1):
        lo = max(mp.mpf(1), (j - mp.mpf('.5')) * s)
        hi = min(t, (j + mp.mpf('.5')) * s)
        if hi > lo:
            result += antiderivative(power, j * s, hi) - antiderivative(power, j * s, lo)
    return result


def population(n, L):
    layers = []
    for k in range(40):
        a, b = mp.power(2, k), mp.power(2, k + 1)
        p, q = 1 - a ** -4, 1 - b ** -4
        mass = q ** n - p ** n
        if not mass:
            continue
        m = (q ** (n - 1) * truncated(k, L, b, 1)
             - p ** (n - 1) * truncated(k, L, a, 1)) / mass
        second = (q ** (n - 1) * truncated(k, L, b, 2)
                  - p ** (n - 1) * truncated(k, L, a, 2)) / mass
        if p:
            r = (p / q) ** n
            u, z = truncated(k, L, a, 1) / p, truncated(k, L, b, 1) / q
            c = -r * (z - u) ** 2 / (1 - r) ** 2
        else:
            c = Z
        layers.append((mass, m, second, c))
    mean = sum(P * m for P, m, _, _ in layers)
    variance = sum(P * t for P, _, t, _ in layers) - mean ** 2
    between = sum(P * (m - mean) ** 2 for P, m, _, _ in layers)
    conditional = sum(P * c for P, _, _, c in layers)
    return dict(mean=mean, variance=variance, inflation=1 + (n - 1) * (between + conditional) / variance)


if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    rows = list(csv.DictReader((base / 'experiments' / 'benchmark.csv').open(encoding='utf-8')))
    results = []
    for row in rows:
        if row['distribution'] != 'pareto4':
            continue
        ref = population(int(row['n']), int(row['Lm']))
        diff = abs(mp.mpf(row['exact_inflation']) / ref['inflation'] - 1)
        results.append(dict(n=int(row['n']), L=int(row['Lm']),
                            inflation_80_digits=mp.nstr(ref['inflation'], 50),
                            relative_binary64_difference=float(diff)))
    worst = max(r['relative_binary64_difference'] for r in results)
    (base / 'pareto_precision_audit.json').write_text(json.dumps(dict(
        decimal_digits=80, relative_difference_max=worst, records=results), indent=2), encoding='utf-8')
    print(f'Independent 80-digit Pareto audit: {len(results)} configurations, max relative difference {worst:.3g}')
    assert worst < 1e-5
