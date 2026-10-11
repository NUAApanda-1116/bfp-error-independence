"""Prescribed latent-loading sensitivity and independent Monte Carlo validation.

Two fixed codecs, tau in {0,.3,.6,.9}; all cases are retained. GH 41/61
convergence is checked. --from-saved regenerates figure/section and hashes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import time
import numpy as np
from scipy.fft import dstn
import elliptic2d_application as elliptic
import latent_elliptic_application as latent
import poisson_application as codec
import run_revision as core

OUT=Path(__file__).resolve().parent
TAUS=(0.,.3,.6,.9)
ORDERS=(41,61)
MESH=16
# Increased after the initial coverage audit. The original 20,000 vectors
# remain the prefix of this deterministic stream; the added vectors are
# retained in the same paired, fixed-seed design.
TRIALS=200000
BOOTSTRAP=1000
CONFIG_IDS=('b32_contiguous_uniform','b128_contiguous_uniform')

def selected_configs():
    all_configs={c['id']:c for c in elliptic.configurations(MESH)}
    return [all_configs[key] for key in CONFIG_IDS]

def artifacts(rows, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Single-panel quantitative robustness figure. The population ratios
    # isolate conditional covariance; markers/intervals use independent vectors.
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    with plt.rc_context({'font.family':['Arial','DejaVu Sans'],'font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,
            'pdf.fonttype':42,'svg.fonttype':'none',
            'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(figsize=(7.2,2.65))
        for size,color in ((32,'#174c72'),(128,'#b65c12')):
            rr=[r for r in rows if int(r['block_size'])==size]
            rr.sort(key=lambda r:r['tau'])
            x=[r['tau'] for r in rr]
            ratio=[r['exact_to_independent_mse_ratio'] for r in rr]
            ax.plot(x,ratio,color=color,lw=1.3,label=f'{size} cells: conditional diagonal')
            ax.plot(x,[r['exact_to_unconditional_diagonal_mse_ratio'] for r in rr],
                    color=color,lw=1.3,ls='--',label=f'{size} cells: unconditional diagonal')
            y=np.array([r['mc_to_independent_mse_ratio'] for r in rr])
            low=np.array([r['mc_to_independent_mse_ratio_low'] for r in rr])
            high=np.array([r['mc_to_independent_mse_ratio_high'] for r in rr])
            ax.errorbar(x,y,yerr=[y-low,high-y],fmt='o',color=color,
                        markersize=3,capsize=2,lw=1.)
        ax.axhline(1.,color='0.45',lw=1.)
        ax.set_xticks(TAUS)
        ax.set_xlabel(r'Latent loading $\tau$')
        ax.set_ylabel('Full / diagonal energy MSE')
        ax.set_ylim(.97,1.82)
        ax.grid(alpha=.15)
        ax.legend(frameon=False,fontsize=7,ncol=2,loc='upper left')
        fig.tight_layout()
        fig.savefig(OUT/'latent_correlation_sweep.pdf',metadata={'CreationDate':None,'ModDate':None},bbox_inches='tight')
        fig.savefig(OUT/'latent_correlation_sweep.svg',bbox_inches='tight')
        fig.savefig(OUT/'latent_correlation_sweep.png',dpi=600,bbox_inches='tight')
        fig.savefig(OUT/'latent_correlation_sweep.tiff',dpi=600,bbox_inches='tight')
        plt.close(fig)
    coverage=sum(bool(r['exact_in_mc_interval']) for r in rows)
    maximum=max(abs(r['relative_prediction_error']) for r in rows)
    change=max(r['quadrature_relative_change'] for r in rows)
    section=r"""\paragraph{Sensitivity to the common source factor.}
For two fixed contiguous codecs ($|B|=32,128$), we also prescribe
$\tau\in\{0,0.3,0.6,0.9\}$ in $b_i=a_i\exp(Z_i+\tau\phi_iY)$.
Each setting uses __TRIALS__ iid source vectors,
paired across codecs, and $1000$ whole-vector bootstrap replicates.
Gauss--Hermite orders $41$ and $61$ differ by at most __CHANGE__ in the
full and comparison energy predictions. The nominal intervals contain
__COVERAGE__ of eight population predictions, with maximum RMS discrepancy
__ERROR__\%. This is an empirical interval check rather than a repeated-sampling
coverage claim; all cases are retained in the saved records.
Figure~\ref{fig:latent-sensitivity} shows that removing
conditional block covariances still underestimates MSE throughout this
range. For length $128$, the full/conditional-diagonal ratio changes from
$1.47$ to $1.41$. Removing the latent mean covariances as well increases
the ratio to $1.73$ at $\tau=0.9$. The two losses of covariance are thus
distinct, even in this one-factor family.
\begin{figure}[tb]
\centering
\includegraphics[width=\linewidth]{revision/experiments/latent_correlation_sweep.pdf}
\caption{Common-factor sensitivity at seven bits per source value.
Solid curves retain the full latent mean response in the diagonal comparison;
dashed curves also remove unconditional off-diagonal covariances. Markers
and nominal $95\%$ intervals use independent source vectors.}
\label{fig:latent-sensitivity}
\end{figure}
"""
    section=section.replace('__TRIALS__',f'${TRIALS:,}$'.replace(',',r'\,')).replace('__CHANGE__',f'${change:.2g}$').replace('__COVERAGE__',str(coverage)).replace('__ERROR__',f'{100*maximum:.2f}')
    import re
    section=re.sub(r'\$([0-9.]+)e([+-][0-9]+)\$',lambda m:'$'+m[1]+r'\times10^{'+str(int(m[2]))+'}$',section)
    (OUT.parent/'latent_sensitivity.tex').write_text(section,encoding='utf-8')
    manifest.update(exact_interval_contains_count=coverage,total_cases=len(rows),
        maximum_relative_rms_prediction_error=maximum,
        maximum_quadrature_relative_change=change)
    files=sorted(p for p in OUT.glob('latent_correlation_sweep*') if p.is_file() and not p.name.endswith('_manifest.json'))
    manifest['files']={p.name:elliptic.sha(p) for p in files}
    manifest['section_sha256']=elliptic.sha(OUT.parent/'latent_sensitivity.tex')
    (OUT/'latent_correlation_sweep_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Sensitivity: {coverage}/8 intervals contain population, max RMS difference {100*maximum:.3f}%, GH change {change:.3e}',flush=True)

def run():
    start=time.time()
    configs=selected_configs()
    snapshot=OUT/'latent_correlation_sweep_numerical_source.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    manifest=dict(date='2026-10-11 Asia/Shanghai',mesh=MESH,taus=TAUS,
        quadrature_orders=ORDERS,trials_per_tau=TRIALS,
        initial_trials_per_tau=20000,extension_trials_per_tau=TRIALS-20000,
        bootstrap_replicates=BOOTSTRAP,
        configurations=configs,design='All four loadings and two codecs fixed before predictions and sampling; sample count was increased from 20,000 to 200,000 after the initial coverage audit; no deletion or seed search.',
        independent_unit='Complete source vector; codecs paired within each loading.',
        rng="core.rng_for('latent_sensitivity','20261011','heldout',tau)",
        numerical_source_sha256=elliptic.sha(snapshot),
        dependencies={p:elliptic.sha(OUT/p) for p in ('elliptic2d_application.py','latent_elliptic_application.py','poisson_application.py','run_revision.py')},
        stage='Before predictions or sampling')
    (OUT/'latent_correlation_sweep_prespecification.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    h,green,eigenvalues,_=elliptic.geometry(MESH)
    scales,phi=elliptic.source_scales(MESH,'hotspot'),latent.loading(MESH)
    rows=[]
    for tau in TAUS:
        earlier=latent.quadrature(configs,mesh=MESH,tau=tau,order=ORDERS[0])
        predictions=latent.quadrature(configs,mesh=MESH,tau=tau,order=ORDERS[1])
        energies=np.empty((TRIALS,len(configs)))
        rng=core.rng_for('latent_sensitivity','20261011','heldout',tau)
        digest=hashlib.sha256()
        for begin in range(0,TRIALS,1000):
            count=min(1000,TRIALS-begin)
            y=rng.normal(size=count)
            inputs=scales*np.exp(rng.normal(size=(count,MESH*MESH))+tau*y[:,None]*phi)
            digest.update(inputs.tobytes())
            for j,cfg in enumerate(configs):
                error,_,_=codec.bounded_quantize(inputs,cfg)
                modes=dstn(error.reshape(-1,MESH,MESH),type=1,norm='ortho',axes=(1,2))
                energies[begin:begin+count,j]=h*h*np.sum(modes*modes/eigenvalues,axis=(1,2))
        np.savez_compressed(OUT/f'latent_correlation_sweep_tau{tau:.1f}.npz',energy=energies,codec_ids=np.array(CONFIG_IDS))
        for j,(pred,old) in enumerate(zip(predictions,earlier)):
            change=max(abs(pred[k]/old[k]-1) for k in ('exact_energy_mse','independent_energy_mse','unconditional_diagonal_energy_mse'))
            if change>1e-7: raise ArithmeticError(f'GH convergence failed tau={tau}: {change}')
            values=energies[:,j]
            mean=float(values.mean())
            rms=math.sqrt(mean/pred['reference_expected_energy'])
            lo,hi=codec.bootstrap_rms(values,pred['reference_expected_energy'],('latent_sensitivity',tau,pred['id']),BOOTSTRAP)
            row=dict(mesh=MESH,tau=tau,quadrature_order=ORDERS[1],codec=pred['id'],block_size=pred['block_size'],
                exact_relative_rms=pred['exact_relative_rms'],independent_relative_rms=pred['independent_relative_rms'],
                unconditional_diagonal_relative_rms=pred['unconditional_diagonal_relative_rms'],
                exact_to_independent_mse_ratio=pred['exact_energy_mse']/pred['independent_energy_mse'],
                exact_to_unconditional_diagonal_mse_ratio=pred['exact_energy_mse']/pred['unconditional_diagonal_energy_mse'],
                exact_energy_mse=pred['exact_energy_mse'],reference_expected_energy=pred['reference_expected_energy'],
                independent_energy_mse=pred['independent_energy_mse'],mc_energy_mse=mean,
                mc_relative_rms=rms,mc_relative_rms_low=float(lo),mc_relative_rms_high=float(hi),
                mc_to_independent_mse_ratio=mean/pred['independent_energy_mse'],
                mc_to_independent_mse_ratio_low=lo*lo*pred['reference_expected_energy']/pred['independent_energy_mse'],
                mc_to_independent_mse_ratio_high=hi*hi*pred['reference_expected_energy']/pred['independent_energy_mse'],
                exact_in_mc_interval=bool(lo<=pred['exact_relative_rms']<=hi),
                relative_prediction_error=rms/pred['exact_relative_rms']-1,
                mc_energy_standard_error=float(values.std(ddof=1)/math.sqrt(TRIALS)),
                quadrature_relative_change=change,omitted_energy_mse_bound=pred['omitted_energy_mse_bound'],
                input_sha256=digest.hexdigest(),trials=TRIALS)
            rows.append(row)
        print(f'Completed tau={tau}: {TRIALS} new vectors, {len(configs)} codecs',flush=True)
    codec.write_csv(OUT/'latent_correlation_sweep.csv',rows)
    manifest.update(duration_seconds=time.time()-start,python=platform.python_version(),numpy=np.__version__,stage='Completed')
    artifacts(rows,manifest)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-saved',action='store_true')
    args=parser.parse_args()
    if args.from_saved:
        artifacts(codec.read_rows('latent_correlation_sweep.csv'),json.loads((OUT/'latent_correlation_sweep_manifest.json').read_text(encoding='utf-8')))
    else:run()
