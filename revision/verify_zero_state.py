"""Independent small-block simulation checks the ReLU all-zero mixture.

Unlike the main driver, this quantizer is implemented directly below.  The
fixed seed and all configurations are retained; predictions are not fitted.
"""
from pathlib import Path
import json
import math
import sys
import numpy as np

base = Path(__file__).resolve().parent
sys.path.insert(0, str(base / 'experiments'))
from run_revision import exact_model

seed = 2026101001
blocks = 300000
rng = np.random.default_rng(seed)
rows = []
for n in (2, 4):
    x = np.maximum(rng.standard_normal((blocks, n)), 0)
    maxima = x.max(axis=1)
    step = np.zeros(blocks)
    nonzero = maxima > 0
    step[nonzero] = np.exp2(np.floor(np.log2(maxima[nonzero])) - 2)
    scaled = x / np.where(nonzero, step, 1)[:, None]
    for mode in ('rne', 'sr'):
        if mode == 'rne':
            rounded = np.rint(scaled)
        else:
            lower = np.floor(scaled)
            rounded = lower + (rng.random(x.shape) < scaled - lower)
        errors = (rounded - scaled) * step[:, None]
        model = exact_model('relu', n, 2, mode)
        assert math.isclose(model['zero_state_probability'], 2.0**(-n))
        assert abs(model['retained_probability'] - 1) < 2e-8
        statistics = {
            'mean': (errors.mean(axis=1), model['mean']),
            'second_moment': ((errors**2).mean(axis=1), model['var'] + model['mean']**2),
            'covariance': ((errors[:, 0]-model['mean'])*(errors[:, 1]-model['mean']), model['cov']),
        }
        measurements = {}
        for key, (sample, target) in statistics.items():
            estimate = float(sample.mean())
            standard_error = float(sample.std(ddof=1) / math.sqrt(blocks))
            z = abs(estimate-target)/standard_error
            assert z < 5, (n, mode, key, z)
            measurements[key] = dict(prediction=target, estimate=estimate,
                                     standard_error=standard_error, absolute_z=z)
        rows.append(dict(n=n, L=2, rounding=mode,
                         zero_probability=model['zero_state_probability'],
                         observed_zero_probability=float((~nonzero).mean()),
                         retained_probability=model['retained_probability'],
                         statistics=measurements))
audit = dict(seed=seed, blocks_per_configuration=blocks, configurations=len(rows),
             method='independently implemented Monte Carlo; five-standard-error check',
             exact_probability_checks_passed=True, results=rows)
(base/'zero_state_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(dict(configurations=len(rows), all_checks_passed=True,
                     maximum_absolute_z=max(v['absolute_z'] for row in rows
                                            for v in row['statistics'].values()))))
