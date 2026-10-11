"""Independent conditional-block Monte Carlo for the heavy-tail benchmark.

Exponent probabilities are analytical; conditional moments remain simulated.
The original unconditional experiment is preserved in benchmark_ordinary.csv.
"""
import csv,json,math,time
import numpy as np
import run_revision as core
OUT=core.OUT


def estimate(data,weights,n):
    es=eq=es2=scale2=var_es=0.
    for k,p in weights.items():
        a=data[k];S,Q,s2=a.T[:3]
        es+=p*S.mean();eq+=p*Q.mean();es2+=p*np.mean(S*S);scale2+=p*np.mean(s2)
        var_es+=p*p*S.var(ddof=1)/len(S)
    mu=es/n
    var=eq/n-mu*mu+var_es/(n*n)
    vs=es2-es*es+var_es
    return dict(mean=mu,var=var,inflation=vs/(n*var),mse_sum=es2,
                variance_ratio=12*var/scale2,cov=(vs-n*var)/(n*(n-1)))


def config(n):
    layers=core.exponent_layers('pareto4',n)
    ps=sum(l[1] for l in layers);weights={l[0]:l[1]/ps for l in layers}
    # Every retained rare stratum is sampled, rather than silently excluded by
    # an ordinary-bootstrap realization with no extreme blocks.
    alloc={k:max(100,round(30000*p)) for k,p in weights.items()}
    raw={lm:{} for lm in (2,4)}
    for k,p,la,lb,_,_ in layers:
        rng=core.rng_for('stratified_pareto',n,k)
        chunks={2:[],4:[]}
        batch=max(1,core.CHUNK_ELEMS//n)
        for start in range(0,alloc[k],batch):
            B=min(batch,alloc[k]-start);u=rng.random(B)
            logfm=np.logaddexp(la+np.log1p(-u),lb+np.log(u))/n
            tail=-np.expm1(logfm);mx=tail**(-.25)
            v=rng.random((B,n-1))
            x=np.column_stack((((1-v)+v*tail[:,None])**(-.25),mx))
            for lm in (2,4):
                e,s,xi=core.quantize(x,lm,'rne',rng)
                if np.any(xi!=k):raise ValueError('stratum mismatch')
                chunks[lm].append(np.column_stack((e.sum(1),(e*e).sum(1),s[:,0]**2,xi[:,0])))
        for lm in (2,4):raw[lm][k]=np.concatenate(chunks[lm])
    rows=[]
    for lm in (2,4):
        model=core.exact_model('pareto4',n,lm)
        point=estimate(raw[lm],weights,n)
        boot={key:[] for key in point};rng=core.rng_for('strat_boot',n,lm)
        for _ in range(core.BOOT):
            data={k:a[rng.integers(len(a),size=len(a))] for k,a in raw[lm].items()}
            for key,val in estimate(data,weights,n).items():boot[key].append(val)
        for key,vs in boot.items():
            lo,hi=np.percentile(vs,[2.5,97.5]);point[key+'_low']=lo;point[key+'_high']=hi
        fname=f'raw_pareto4_n{n}_L{lm}_stratified.npz'
        np.savez_compressed(OUT/fname,**{f'k{k}':a for k,a in raw[lm].items()})
        row=dict(distribution='pareto4',n=n,Lm=lm,rounding='rne',blocks=sum(alloc.values()),**point,
                 **{'exact_'+k:v for k,v in model.items() if k!='layers'},raw_file=fname,
                 estimator='conditional stratified',strata=len(weights),omitted_probability=1-ps)
        rows.append(row)
    # Since |e|<=s/2, E[e^2; xi>=K] <= E[M^2; M>=2^K]/(4*2^(2Lm)).
    # Union/max bound for Pareto4: E[M^2; M>=T]<=2n/T^2.  The lower omitted
    # stratum error is at most 2^(2(kmin-Lm))/4 times its probability.
    kmin=min(weights);kmax=max(weights);T=2.**(kmax+1)
    diag=dict(n=n,alloc=alloc,weights=weights,retained_probability=ps,
              omitted_second_moment_bound={lm:n/(2*2.**(2*lm)*T*T) for lm in (2,4)},
              upper_cutoff_exponent=kmax+1,lower_cutoff_exponent=kmin)
    return rows,diag


def main():
    started=time.time()
    with (OUT/'benchmark.csv').open(newline='',encoding='utf-8') as f:original=list(csv.DictReader(f))
    core.write_csv('benchmark_ordinary.csv',original)
    replacement=[];diags=[]
    for n in (32,128,512,4096):
        t=time.time();rows,diag=config(n);replacement+=rows;diags.append(diag)
        print(n,[(r['Lm'],r['inflation'],r['exact_inflation'],r['inflation_low'],r['inflation_high']) for r in rows],f'{time.time()-t:.1f}s',flush=True)
    final=[r for r in original if r['distribution']!='pareto4']+replacement
    core.write_csv('benchmark.csv',final)
    (OUT/'stratification.json').write_text(json.dumps(dict(duration_seconds=time.time()-started,bootstrap_replicates=core.BOOT,
        bootstrap='percentile; resample independent conditional blocks within each stratum, exact probabilities fixed',
        allocation='max(100, round(30000*P(xi=k))) conditional blocks per retained stratum',
        retained_probability_threshold=1e-15,diagnostics=diags),indent=2),encoding='utf-8')
    # Refresh controls using an independent exact n=1 prediction, and remove
    # numerical pseudo-covariance in the exponential high-precision output.
    core.weighted_experiment();core.large_exponential()
    print('COMPLETE',time.time()-started,flush=True)


if __name__=='__main__':main()
