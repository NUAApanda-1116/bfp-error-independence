"""Reproducible BFP revision experiments (uniform, unbounded mantissa grid).

This is an idealized BFP model, not a simulation of MXFP4/E2M1.  All computed
objects use fixed deterministic weights and exact (binary64) accumulation of
the input quantization error; accumulation rounding is not included.
"""
from pathlib import Path
import csv, hashlib, json, math, time, sys, platform
import numpy as np
from scipy import special
import scipy
import mpmath as mp

OUT = Path(__file__).resolve().parent
MASTER_SEED = 20261010
BOOT = 200
CHUNK_ELEMS = 600_000
NAMES = {"gaussian": "Gaussian", "lognormal": "Lognormal", "halfnormal": "Half-normal",
         "relu": "ReLU", "pareto4": "Pareto(4)", "exponential": "Exponential", "beta": "Beta(2,5)"}


def rng_for(*labels):
    h = hashlib.sha256("|".join(map(str, labels)).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, *np.frombuffer(h[:16], dtype="<u4").tolist()]))


def tail_moments(name, x):
    """Positive-magnitude tail mass, first moment, second moment (atoms handled)."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    if name in ("gaussian", "halfnormal", "relu"):
        c = 1.0 if name == "relu" else 2.0
        ph = np.exp(-x*x/2) / math.sqrt(2*math.pi)
        q = special.ndtr(-x)
        p = c*q
        if name == "relu":
            p = np.where(x == 0, 1.0, p)
        return p, c*ph, c*(x*ph+q)
    if name == "lognormal":
        z = np.log(np.maximum(x, np.finfo(float).tiny))
        return special.ndtr(-z), math.exp(.5)*special.ndtr(1-z), math.exp(2)*special.ndtr(2-z)
    if name == "pareto4":
        t = np.maximum(x, 1)
        return t**-4, (4/3)*t**-3, 2*t**-2
    if name == "exponential":
        p = np.exp(-x)
        return p, (x+1)*p, (x*x+2*x+2)*p
    if name == "beta":
        t = np.clip(x, 0, 1)
        return special.betaincc(2,5,t), (2/7)*special.betaincc(3,5,t), (3/28)*special.betaincc(4,5,t)
    raise ValueError(name)


def cutoff(name):
    return {"gaussian": 9, "halfnormal": 9, "relu": 9, "lognormal": 2300,
            "pareto4": 4000, "exponential": 34, "beta": 1}[name]


def residual_integrals(name, s, upper, mode="rne"):
    """Integrate e and e^2 against the original measure over |X|<upper.

    Cells use analytical truncated raw moments; no fitted or MC response
    function is used.  The atom at zero is in the first zero-output cell.
    For untruncated reduced responses omitted tails are bounded by s/2 times
    omitted probability (mean), s^2/4 times it (second moment).
    """
    if name in ('gaussian','halfnormal','relu') and upper == cutoff(name) and s < .05:
        # High-resolution normal response.  The 8-term midpoint Euler--
        # Maclaurin expansion avoids materializing tens of millions of cells
        # in strata with a very small ReLU maximum. At s<=.05 its first omitted
        # term is <1e-27; the total normal tail mass beyond 9 is <3e-19.
        atom_factor=.5 if name=='relu' else 1.
        if mode=='sr':return 0.,atom_factor*s*s/6
        first=0.
        if name!='gaussian':
            h=mp.mpf(s);f0=mp.sqrt(2/mp.pi)
            coeff=mp.mpf(1)
            for j in range(1,9):
                if j>1:coeff*=2*j-3
                first+=float(mp.bernpoly(2*j,mp.mpf('.5'))/mp.factorial(2*j)*(-1)**(j-1)*coeff*f0*h**(2*j))
        return atom_factor*first,atom_factor*s*s/12
    if name == 'exponential' and upper == cutoff(name):
        # Infinite geometric cell sum, evaluated at high precision to retain
        # the high-resolution residual after cancellation of raw moments.
        with mp.workdps(50):
            h=mp.mpf(s);t=mp.exp(-h);z=mp.exp(-h/2);d=-mp.expm1(-h)
            if mode == 'rne':
                H=h/(2*mp.sinh(h/2))
                return float(H-1),float(2-2*H)
            # E[(X-floor(X/h)h)(h-(X-floor(X/h)h))].
            theta1=1-h*t/d
            theta2=2-(h*h+2*h)*t/d
            return 0.,float(h*theta1-theta2)
    if name == 'exponential':
        with mp.workdps(50):
            h=mp.mpf(s);mass=-mp.expm1(-mp.mpf(upper))
            H=h/(2*mp.sinh(h/2))
            if mode=='rne':return float(mass*(H-1)),float(mass*(2-2*H))
            return 0.,float(mass*(h/mp.tanh(h/2)-2))
    if mode == "sr":
        centers = np.arange(int(math.ceil(upper/s))) * s
        lo, hi = centers, np.minimum(centers+s, upper)
    else:
        centers = np.arange(int(math.ceil(upper/s+.5))) * s
        lo, hi = np.maximum(centers-s/2, 0), np.minimum(centers+s/2, upper)
    ok = hi > lo
    centers, lo, hi = centers[ok], lo[ok], hi[ok]
    p0, p1, p2 = tail_moments(name, lo)
    q0, q1, q2 = tail_moments(name, hi)
    mass, m1, m2 = p0-q0, p1-q1, p2-q2
    if mode == "sr":
        # Var(Q(X)-X | X)=(X-lo)(hi_grid-X), with hi_grid=lo+s.
        second = np.sum((2*lo+s)*m1-m2-lo*(lo+s)*mass)
        return 0.0, float(max(second, 0.0))
    first = float(np.sum(centers*mass-m1))
    second = float(np.sum(centers**2*mass-2*centers*m1+m2))
    if name == "gaussian":
        first = 0.0
    return first, max(second, 0.0)


def exponent_layers(name, n):
    out = []
    for k in range(-30, 50):
        a, b = 2.0**k, 2.0**(k+1)
        sa, sb = float(tail_moments(name, a)[0]), float(tail_moments(name,b)[0])
        la = n*math.log1p(-sa) if sa < 1 else -math.inf
        lb = n*math.log1p(-sb) if sb < 1 else -math.inf
        p = math.exp(lb)*(-math.expm1(la-lb)) if lb > -math.inf else 0.0
        if p > 1e-15:
            out.append((k,p,la,lb,sa,sb))
    return out


def exact_model(name, n, lm, mode="rne",compute_reduced=True):
    layers=[]
    for k,p,la,lb,sa,sb in exponent_layers(name,n):
        a,b,s=2.0**k,2.0**(k+1),2.0**(k-lm)
        fa,fb=1-sa,1-sb
        ia,qa=residual_integrals(name,s,a,mode)
        ib,qb=residual_integrals(name,s,b,mode)
        u=ia/fa if fa>0 else 0.0
        v=ib/fb
        ua=qa/fa if fa>0 else 0.0
        vb=qb/fb
        r=math.exp(la-lb) if la>-math.inf else 0.0
        d=-math.expm1(la-lb) if la>-math.inf else 1.0
        m=(v-r*u)/d
        q=(vb-r*ua)/d
        c=-r*(v-u)**2/d**2
        if name=='exponential':c=0.0 # periodic residual + exponential memorylessness
        if mode=="sr":
            m=c=0.0
        rm,rq=residual_integrals(name,s,cutoff(name),mode) if compute_reduced else (m,q)
        layers.append(dict(k=k,p=p,m=m,q=q,c=c,rm=rm,rq=rq,s=s))
    # The atom at zero has its own exponent state.  It contributes no raw
    # error moments but does contribute to the mixture variance of the means.
    # Normalizing only positive-maximum states would condition the population
    # on a nonzero block, a material error for small ReLU blocks.
    zero_state_probability=0.5**n if name=='relu' else 0.0
    ps=sum(t['p'] for t in layers)+zero_state_probability
    mu=sum(t['p']*t['m'] for t in layers)/ps
    e2=sum(t['p']*t['q'] for t in layers)/ps
    between=(sum(t['p']*(t['m']-mu)**2 for t in layers)
             +zero_state_probability*mu*mu)/ps
    cond=sum(t['p']*t['c'] for t in layers)/ps
    var=e2-mu*mu
    cov=between+cond
    rmu=sum(t['p']*t['rm'] for t in layers)/ps
    re2=sum(t['p']*t['rq'] for t in layers)/ps
    rb=(sum(t['p']*(t['rm']-rmu)**2 for t in layers)
        +zero_state_probability*rmu*rmu)/ps
    rv=re2-rmu*rmu
    return dict(mean=mu,var=var,cov=cov,between=between,conditional=cond,
                inflation=1+(n-1)*cov/var, drop_conditional_inflation=1+(n-1)*between/var,
                mse_sum=n*var+n*(n-1)*cov+n*n*mu*mu,
                reduced_inflation=1+(n-1)*rb/rv, reduced_mean=rmu,
                variance_ratio=12*var/(sum(t['p']*t['s']**2 for t in layers)/ps),
                retained_probability=ps,zero_state_probability=zero_state_probability,layers=layers)


def draw(name, shape, rng):
    if name=="gaussian": return rng.normal(size=shape)
    if name=="halfnormal": return np.abs(rng.normal(size=shape))
    if name=="relu": return np.maximum(rng.normal(size=shape),0)
    if name=="lognormal": return rng.lognormal(size=shape)
    if name=="pareto4": return rng.pareto(4,size=shape)+1
    if name=="exponential": return rng.exponential(size=shape)
    if name=="beta": return rng.beta(2,5,size=shape)
    raise ValueError(name)


def quantize(x,lm,mode,rng,block_size=None):
    n=x.shape[1]
    bs=n if block_size is None else block_size
    z=x.reshape(len(x),n//bs,bs)
    mx=np.max(np.abs(z),axis=2,keepdims=True)
    xi=np.floor(np.log2(np.maximum(mx,np.finfo(float).tiny)))
    s=np.where(mx>0,np.exp2(xi-lm),0)
    u=z/np.where(s>0,s,1)
    if mode=="rne": q=np.rint(u)
    else:
        lo=np.floor(u)
        q=lo+(rng.random(u.shape)<u-lo)
    return ((q-u)*s).reshape(x.shape), s.reshape(len(x),-1), xi.reshape(len(x),-1).astype(int)


def estimate(a,n):
    S,Q,s2=a.T[:3]
    mu=S.mean()/n
    var=Q.mean()/n-mu*mu
    vs=S.var(ddof=1)
    return dict(mean=mu,var=var,inflation=vs/(n*var),mse_sum=np.mean(S*S),
                variance_ratio=12*var/np.mean(s2),cov=(vs-n*var)/(n*(n-1)))


def bootstrap(a,n,labels):
    point=estimate(a,n)
    rng=rng_for('bootstrap',*labels)
    vals={key:[] for key in point}
    for _ in range(BOOT):
        b=a[rng.integers(len(a),size=len(a))]
        for key,val in estimate(b,n).items(): vals[key].append(val)
    for key,vs in vals.items():
        lo,hi=np.percentile(vs,[2.5,97.5])
        point[key+'_low']=float(lo); point[key+'_high']=float(hi)
    return point


def mc_config(name,n,B,lms,modes):
    rng=rng_for('input',name,n)
    streams={(lm,mode):rng_for('rounding',name,n,lm,mode) for lm in lms for mode in modes}
    raw={(lm,mode):[] for lm in lms for mode in modes}
    batch=max(1,CHUNK_ELEMS//n)
    for start in range(0,B,batch):
        x=draw(name,(min(batch,B-start),n),rng)
        for lm in lms:
            for mode in modes:
                e,s,k=quantize(x,lm,mode,streams[lm,mode])
                raw[lm,mode].append(np.column_stack((e.sum(1),(e*e).sum(1),s[:,0]**2,k[:,0])))
    rows=[]
    for (lm,mode),chunks in raw.items():
        a=np.concatenate(chunks)
        model=exact_model(name,n,lm,mode)
        row=dict(distribution=name,n=n,Lm=lm,rounding=mode,blocks=B,**bootstrap(a,n,(name,n,lm,mode)))
        row.update({'exact_'+k:v for k,v in model.items() if k!='layers'})
        layers=[]
        for layer in model['layers']:
            sel=a[a[:,3]==layer['k']]
            l=dict(distribution=name,n=n,Lm=lm,rounding=mode,**layer,blocks=len(sel))
            if len(sel)>1:
                S,Q=sel.T[:2]; m=layer['m']
                # Unbiased conditional-covariance estimate using the analytic
                # conditional mean, independent of this Monte Carlo sample.
                cvals=((S-n*m)**2-(Q-2*m*S+n*m*m))/(n*(n-1))
                l.update(mc_mean=S.mean()/n,mc_mean_se=S.std(ddof=1)/(n*math.sqrt(len(sel))),
                         mc_conditional=float(cvals.mean()),mc_conditional_se=float(cvals.std(ddof=1)/math.sqrt(len(sel))))
            layers.append(l)
        fname=f'raw_{name}_n{n}_L{lm}_{mode}.npz'
        np.savez_compressed(OUT/fname,stats=a,columns=np.array(['S','Q','s2','xi']))
        row['raw_file']=fname
        rows.append((row,layers))
    return rows


def write_csv(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def weighted_experiment(n=128,B=50000,lm=2):
    rng=rng_for('weighted_input')
    weights={'positive':np.ones(n),'balanced':np.tile([1.,-1.],n//2),
             'near_cancel':np.tile([1.,-0.9],n//2)}
    methods=[('shared_rne',n,'rne'),('split16_rne',16,'rne'),('independent_rne',1,'rne'),('shared_sr',n,'sr')]
    res={(method,w):[] for method,_,_ in methods for w in weights}
    qs={method:[] for method,_,_ in methods}
    means={method:[] for method,_,_ in methods}
    for start in range(0,B,max(1,CHUNK_ELEMS//n)):
        b=min(CHUNK_ELEMS//n,B-start)
        x=draw('lognormal',(b,n),rng)
        for method,bs,mode in methods:
            e,_,_=quantize(x,lm,mode,rng_for('weighted_round',method,start),bs)
            qs[method].extend((e*e).sum(1)/n)
            means[method].extend(e.sum(1)/n)
            for w,values in weights.items():res[method,w].extend(e@values)
    rows=[]
    for method,bs,mode in methods:
        if bs==1:
            md=exact_model('lognormal',1,lm,mode,compute_reduced=False)
            mu,ve,cov=md['mean'],md['var'],0.
        else:
            md=exact_model('lognormal',bs,lm,mode)
            mu,ve,cov=md['mean'],md['var'],md['cov']
        for w,values in weights.items():
            a=np.asarray(res[method,w]);brng=rng_for('weighted_boot',method,w)
            point=dict(mean=float(a.mean()),variance=float(a.var(ddof=1)),mse=float(np.mean(a*a)))
            boot={key:[] for key in point}
            for _ in range(BOOT):
                b=a[brng.integers(len(a),size=len(a))]
                boot['mean'].append(b.mean());boot['variance'].append(b.var(ddof=1));boot['mse'].append(np.mean(b*b))
            for key,val in boot.items():
                lo,hi=np.percentile(val,[2.5,97.5]);point[key+'_low']=float(lo);point[key+'_high']=float(hi)
            wb=values.reshape(n//bs,bs)
            pv=(ve-cov)*np.sum(values*values)+cov*np.sum(wb.sum(1)**2)
            pb=mu*np.sum(values)
            rows.append(dict(distribution='lognormal',n=n,Lm=lm,blocks=B,method=method,weights=w,
                             weight_sum=float(np.sum(values)),weight_square_sum=float(np.sum(values**2)),
                             **point,predicted_mean=pb,predicted_variance=pv,predicted_mse=pv+pb*pb,
                             prediction_source='exact single-element integration' if bs==1 else 'exact finite-n conditional moments'))
    np.savez_compressed(OUT/'raw_weighted.npz',**{f'{m}_{w}':np.asarray(a) for (m,w),a in res.items()})
    write_csv('weighted.csv',rows)
    return rows


def atomic_counterexample():
    rows=[]
    for n in (2,4,8,32):
        mu=e2=es=es2=0.
        for j in range(n+1):
            p=math.comb(n,j)*.5**n
            step=2.0**(math.floor(math.log2(2.05 if j else 1.3))-2)
            el=np.rint(1.3/step)*step-1.3;eh=np.rint(2.05/step)*step-2.05
            S=(n-j)*el+j*eh;Q=(n-j)*el*el+j*eh*eh
            es+=p*S;es2+=p*S*S;e2+=p*Q/n
        mu=es/n;ve=e2-mu*mu;vs=es2-es*es
        rows.append(dict(n=n,Lm=2,mean=mu,var=ve,cov=(vs-n*ve)/(n*(n-1)),
                         inflation=vs/(n*ve),mse_sum=es2))
    write_csv('atomic_counterexample.csv',rows)
    return rows


def large_exponential():
    """100-digit finite-n laws: n=floor(exp(2^j)), with exact exponent weights.

    The integral is over finitely many rounding cells inside each exponent bin;
    tail bin bounds and precision comparison are included in the CSV output.
    """
    rows=[]
    for precision in (100,160):
        mp.mp.dps=precision
        for j in (3,4,5,6,7,8):
            a0=mp.mpf(2)**j
            with mp.workdps(220):n_integer=int(mp.floor(mp.exp(mp.mpf(2)**j)))
            n=mp.mpf(n_integer);lm=2
            vals=[];oracle=[]
            def integrals(s,T):
                first=second=mp.mpf(0)
                for idx in range(int(mp.ceil(T/s+mp.mpf('.5')))):
                    center=idx*s;lo=max(center-s/2,mp.mpf(0));hi=min(center+s/2,T)
                    if hi<=lo:continue
                    mass=mp.exp(-lo)-mp.exp(-hi)
                    x1=(lo+1)*mp.exp(-lo)-(hi+1)*mp.exp(-hi)
                    x2=(lo*lo+2*lo+2)*mp.exp(-lo)-(hi*hi+2*hi+2)*mp.exp(-hi)
                    first+=center*mass-x1
                    second+=center*center*mass-2*center*x1+x2
                return first,second
            for k in range(j-4,j+9):
                a,b=mp.mpf(2)**k,mp.mpf(2)**(k+1);s=mp.mpf(2)**(k-lm)
                fa,fb=-mp.expm1(-a),-mp.expm1(-b)
                la=n*mp.log1p(-mp.exp(-a));lb=n*mp.log1p(-mp.exp(-b))
                p=mp.exp(lb)*(-mp.expm1(la-lb))
                if p<mp.mpf('1e-90'):continue
                ia,qa=integrals(s,a);ib,qb=integrals(s,b)
                u,v=ia/fa,ib/fb;r=mp.exp(la-lb);d=-mp.expm1(la-lb)
                m=(v-r*u)/d;q=(qb/fb-r*qa/fa)/d;c=mp.mpf(0)
                vals.append((p,m,q,c))
                oracle.append((p,s/(2*mp.sinh(s/2))))
            ps=sum(p for p,_,_,_ in vals);mu=sum(p*m for p,m,_,_ in vals)/ps
            ve=sum(p*q for p,_,q,_ in vals)/ps-mu*mu
            between=sum(p*(m-mu)**2 for p,m,_,_ in vals)/ps
            cond=sum(p*c for p,_,_,c in vals)/ps
            I=1+(n-1)*(between+cond)/ve
            EH=sum(p*H for p,H in oracle)/ps
            VB=sum(p*(H-EH)**2 for p,H in oracle)/ps
            Ioracle=1+(n-1)*VB/(1-EH*EH)
            constant=mp.exp(-1)*(1-mp.exp(-1))*mp.mpf(2)**(-2*lm-2)
            asym=constant*mp.log(n)**2*n**(1-mp.mpf(2)**(-lm-1))
            rows.append(dict(precision=precision,j=j,log_n=str(a0),two_power_j=str(a0),actual_log_n=mp.nstr(mp.log(n),40),n=str(n_integer),
                             inflation=mp.nstr(I,40),inflation_over_n=mp.nstr(I/n,40),
                             asymptotic=mp.nstr(asym,40),ratio=mp.nstr(I/asym,40),
                             variance=mp.nstr(ve,40),conditional=mp.nstr(cond,40),
                             between=mp.nstr(between,40),omitted_probability=mp.nstr(1-ps,10)))
            rows[-1]['closed_form_relative_error']=mp.nstr(abs(I/Ioracle-1),10)
    write_csv('exponential_large_n.csv',rows)
    return rows


def main():
    started=time.time();allrows=[];layerrows=[]
    configs=[]
    for name in ('gaussian','lognormal','halfnormal','relu','pareto4','exponential'):
        for n,B in ((32,100000),(128,40000),(512,20000),(4096,6000)):
            configs.append((name,n,B,(2,4),('rne','sr') if n==128 and name in ('gaussian','lognormal','halfnormal','relu') else ('rne',)))
    for name in ('beta','exponential'):
        for n in (2,4,8):configs.append((name,n,150000,(2,),('rne',)))
    configs.extend([('beta',32,100000,(2,),('rne',)),('beta',128,40000,(2,),('rne',))])
    for name,n,B,lms,modes in configs:
        t=time.time()
        for row,layers in mc_config(name,n,B,lms,modes):allrows.append(row);layerrows.extend(layers)
        write_csv('benchmark.csv',allrows);write_csv('conditional_layers.csv',layerrows)
        print(f'{name:12} n={n:5} B={B:6} models={len(lms)*len(modes)} time={time.time()-t:.1f}s',flush=True)
    weighted=weighted_experiment();atoms=atomic_counterexample();large=large_exponential()
    manifest=dict(master_seed=MASTER_SEED,rng='NumPy PCG64; SeedSequence with SHA256 labels',
                  bootstrap_replicates=BOOT,bootstrap_unit='independent blocks',confidence_level=.95,
                  format='idealized uniform-grid BFP; xi=floor(log2 max|X|), step=2^(xi-Lm); no saturation',
                  rounding='RNE or independent Bernoulli SR conditional on X',
                  scope='input quantization error; binary64 accumulation; not OCP MXFP4/E2M1',
                  benchmark_configurations=len(allrows),weighted_configurations=len(weighted),
                  n_blocks_by_n={2:150000,4:150000,8:150000,32:100000,128:40000,512:20000,4096:6000},
                  duration_seconds=time.time()-started,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                  platform=platform.platform(),input_distributions=NAMES,
                  weighted_weights={'positive':'w_i=1','balanced':'alternating +1,-1',
                                    'near_cancel':'alternating +1,-0.9'},
                  reduction_precision='100 and 160 decimal digits for very large exponential n',
                  timestamp='2026-10-10 Asia/Shanghai')
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'COMPLETE {len(allrows)} configurations, {time.time()-started:.1f}s',flush=True)


if __name__=='__main__':main()
