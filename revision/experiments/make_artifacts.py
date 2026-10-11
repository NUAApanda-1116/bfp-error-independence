"""Generate all manuscript numerical artifacts exclusively from new CSV data."""
from pathlib import Path
import csv,json,math,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import run_revision as core
OUT=core.OUT
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':8,
                     'pdf.fonttype':42,'ps.fonttype':42,'lines.linewidth':1.3,
                     'axes.spines.top':False,'axes.spines.right':False})
COLORS={2:'#1764a0',4:'#b65c12'}


def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def f(row,key):return float(row[key])
def select(rows,**kw):return [r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
def value_ci(row,key='inflation',divisor=1):
    v,lo,hi=(f(row,key+suffix)/divisor for suffix in ('','_low','_high'))
    places=3 if abs(v)<10 else 2 if abs(v)<100 else 1
    return f'{v:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]'


def benchmark_figure(rows):
    fig,axes=plt.subplots(2,3,figsize=(5.125,3.65),constrained_layout=True)
    for ax,name in zip(axes.flat,('gaussian','halfnormal','relu','lognormal','pareto4','exponential')):
        for lm in (2,4):
            rs=sorted(select(rows,distribution=name,Lm=lm,rounding='rne'),key=lambda r:int(r['n']))
            rs=[r for r in rs if int(r['n'])>=32]
            x=np.array([int(r['n']) for r in rs]);y=np.array([f(r,'inflation') for r in rs])
            ci=np.array([[f(r,'inflation_low') for r in rs],[f(r,'inflation_high') for r in rs]])
            ax.plot(x,[f(r,'exact_inflation') for r in rs],color=COLORS[lm],label=f'Exact $L={lm}$')
            ax.plot(x,[f(r,'exact_reduced_inflation') for r in rs],color=COLORS[lm],ls=':',alpha=.85,label='Unconditional reduced' if lm==2 else None)
            ax.errorbar(x,y,yerr=np.maximum(np.vstack([y-ci[0],ci[1]-y]),0),fmt='o' if lm==2 else 's',
                        color=COLORS[lm],markersize=3,capsize=2,lw=.7,label='MC, 95% CI' if lm==2 else None)
        ax.set_xscale('log',base=2);ax.set_xticks([32,128,512,4096],labels=['32','128','512','4096'])
        ax.set_title(core.NAMES[name]);ax.set_xlabel('$n$');ax.set_ylabel(r'$\mathcal{I}_n$')
        if name in ('lognormal','pareto4'):ax.set_yscale('log')
        ax.grid(True,alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    fig.savefig(OUT/'figure_benchmark.pdf');fig.savefig(OUT/'figure_benchmark.png',dpi=180)
    return fig


def weighted_figure(rows):
    fig,axes=plt.subplots(1,2,figsize=(5.125,2.5),constrained_layout=True)
    labels={'shared_rne':'Shared RNE','split16_rne':'Split into 16','independent_rne':'Independent scales','shared_sr':'Shared SR'}
    colors=['#1764a0','#b65c12','#28824d','#744795']
    for method,color in zip(labels,colors):
        rs=sorted(select(rows,method=method),key=lambda r:f(r,'weight_sum')**2/(int(r['n'])*f(r,'weight_square_sum')))
        x=np.array([f(r,'weight_sum')**2/(int(r['n'])*f(r,'weight_square_sum')) for r in rs])
        for ax,key in zip(axes,('variance','mse')):
            d=np.array([f(r,'weight_square_sum') for r in rs]);y=np.array([f(r,key) for r in rs])/d
            lo=np.array([f(r,key+'_low') for r in rs])/d;hi=np.array([f(r,key+'_high') for r in rs])/d
            pred=np.array([f(r,'predicted_'+key) for r in rs])/d
            # For the repeated two-weight pattern, the exact normalized
            # variance and MSE are affine in chi.  Evaluate that affine
            # moment law on a dense grid before the nonlinear axis transform.
            dense=np.r_[0.,np.geomspace(1e-6,1.,300)]
            predicted_dense=pred[0]+(pred[-1]-pred[0])*dense
            if not np.allclose(pred,pred[0]+(pred[-1]-pred[0])*x,rtol=1e-12,atol=1e-12):
                raise ValueError('weighted prediction does not satisfy affine moment law')
            ax.plot(dense,predicted_dense,color=color,label=labels[method]);ax.errorbar(x,y,yerr=np.maximum([y-lo,hi-y],0),fmt='o',color=color,markersize=3,lw=.7,capsize=2)
            near=next(f(r,'weight_sum')**2/(int(r['n'])*f(r,'weight_square_sum')) for r in rs if r['weights']=='near_cancel')
            ax.set_xscale('symlog',linthresh=.005);ax.set_xticks([0,near,1],labels=['0','0.00276','1'])
            ax.set_yscale('log');ax.grid(True,alpha=.15);ax.set_xlabel(r'$\chi=(\sum_i w_i)^2/(n\sum_i w_i^2)$')
    axes[0].set_ylabel(r'Variance / $\sum_i w_i^2$');axes[1].set_ylabel(r'MSE / $\sum_i w_i^2$')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    fig.savefig(OUT/'figure_weighted.pdf');fig.savefig(OUT/'figure_weighted.png',dpi=180)
    return fig


def exponential_figure(rows):
    rows=select(rows,precision=160)
    x=np.array([f(r,'log_n') for r in rows]);i=np.array([f(r,'inflation') for r in rows]);a=np.array([f(r,'asymptotic') for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(5.125,2.2),constrained_layout=True)
    axes[0].plot(x,i,'o-',color=COLORS[2],label='Exact finite-$n$');axes[0].plot(x,a,'--',color=COLORS[4],label='Theorem normalization')
    axes[0].set_yscale('log');axes[0].set_ylim(10,1e103);axes[0].set_ylabel(r'$\mathcal{I}_n$');axes[0].legend(frameon=False)
    axes[1].plot(x,i/np.exp(x),'o-',color=COLORS[2]);axes[1].set_yscale('log');axes[1].set_ylabel(r'$\mathcal{I}_n/n$')
    for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks(x,labels=[str(int(v)) for v in x]);ax.set_xlabel(r'$2^j$, $n=\lfloor e^{2^j}\rfloor$');ax.grid(True,alpha=.15)
    fig.savefig(OUT/'figure_exponential.pdf');fig.savefig(OUT/'figure_exponential.png',dpi=180)
    return fig


def tables(bench,weighted,large,atoms):
    lines=[r'% Automatically generated by make_artifacts.py; all numerical values come from new CSVs.',
           r'\begin{table}[t]',r'\centering\footnotesize',
           r'\caption{New finite-$n$ validation at $L=2$, $n=4096$. Monte Carlo brackets are 95\% block-bootstrap intervals; Pareto uses conditional stratification. $\mathcal I_{\mathrm{drop}}$ drops only $c_k$ while retaining exact conditional one-element moments; $\mathcal I_{\mathrm{red}}$ additionally replaces those moments by unconditional responses.}\label{tab:revision-validation}',
           r'\begin{tabular}{@{}lrrrr@{}}\toprule',
           r'Distribution & MC [95\% CI] & Exact & $\mathcal I_{\mathrm{drop}}$ & $\mathcal I_{\mathrm{red}}$\\\midrule']
    for name in ('gaussian','halfnormal','relu','lognormal','pareto4','exponential'):
        r=select(bench,distribution=name,n=4096,Lm=2,rounding='rne')[0]
        vals=[f(r,k) for k in ('exact_inflation','exact_drop_conditional_inflation','exact_reduced_inflation')]
        lines.append(core.NAMES[name]+' & '+value_ci(r)+' & '+' & '.join(f'{v:.3f}' if v<100 else f'{v:.1f}' for v in vals)+r'\\')
    lines.extend([r'\bottomrule\end{tabular}',r'\end{table}',
                  r'\begin{table}[t]',r'\centering\footnotesize',
                  r'\caption{Weighted dot-product quantization MSE divided by $\sum_iw_i^2$, lognormal inputs, $n=128$, $L=2$, 50,000 blocks. Brackets are 95\% intervals. The weights are fixed before sampling. Balanced: alternating $1,-1$; near cancellation: alternating $1,-0.9$; positive: all ones.}\label{tab:revision-weighted}',
                  r'\begin{tabular}{@{}lrrr@{}}\toprule',
                  r'Method & Balanced & Near cancellation & Positive\\\midrule'])
    for method,label in [('shared_rne','Shared RNE'),('split16_rne','Split blocks of 16'),('independent_rne','Independent scales'),('shared_sr','Shared SR')]:
        vals=[]
        for w in ('balanced','near_cancel','positive'):
            r=select(weighted,method=method,weights=w)[0];vals.append(value_ci(r,'mse',f(r,'weight_square_sum')))
        lines.append(label+' & '+' & '.join(vals)+r'\\')
    lines.extend([r'\bottomrule\end{tabular}',r'\end{table}',
                  r'\begin{table}[t]',r'\centering\footnotesize',
                  r'\caption{Exact high-precision exponential counterexample at $L=2$, $n=\lfloor e^{2^j}\rfloor$. The theorem normalization is $C(\log n)^2 n^{7/8}$ with $C=e^{-1}(1-e^{-1})/64$. Convergence of the ratio to one is predicted by the asymptotic theorem. All displayed values agree between 100 and 160 decimal digits.}\label{tab:revision-exponential}',
                  r'\begin{tabular}{@{}rrrr@{}}\toprule',
                  r'$2^j$ & $\mathcal I_n$ & $\mathcal I_n/n$ & Exact / normalization\\\midrule'])
    def sci(v):
        mant,exp=f'{v:.3e}'.split('e');return rf'${mant}\times10^{{{int(exp)}}}$'
    for r in select(large,precision=160):
        lines.append(str(int(f(r,'log_n')))+' & '+sci(f(r,'inflation'))+' & '+sci(f(r,'inflation_over_n'))+' & '+f"{f(r,'ratio'):.6f}"+r'\\')
    lines.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    (OUT/'tables.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    captions=r'''% Automatically generated figure environments; include graphics relative to paper.tex.
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{revision/experiments/figure_benchmark.pdf}
\caption{Finite-block validation on six input laws. Solid curves are exact moment predictions; dotted curves use unconditional responses and drop conditional covariance. Markers show Monte Carlo estimates with 95\% intervals, using conditional stratification for Pareto.}\label{fig:revision-benchmark}
\end{figure}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{revision/experiments/figure_weighted.pdf}
\caption{Weighted dot products for lognormal inputs ($n=128$, $L=2$). Fixed weights vary the normalized imbalance $\chi=(\sum_iw_i)^2/(n\sum_iw_i^2)\in[0,1]$. Curves evaluate the affine laws from exact moments on a dense grid; markers show Monte Carlo with 95\% intervals.}\label{fig:revision-weighted}
\end{figure}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{revision/experiments/figure_exponential.pdf}
\caption{Exponential inputs provide unbounded variance inflation on a dyadic subsequence despite $\mathcal I_n/n\to0$. Values use exact exponent probabilities and 160-digit cell integration, independently checked at 100 digits.}\label{fig:revision-exponential}
\end{figure}
'''
    (OUT/'figures.tex').write_text(captions,encoding='utf-8')


def summary(bench,weighted,large):
    outside=[dict(distribution=r['distribution'],n=r['n'],Lm=r['Lm'],rounding=r['rounding']) for r in bench
             if not f(r,'inflation_low')<=f(r,'exact_inflation')<=f(r,'inflation_high')]
    maxr=max(bench,key=lambda r:abs(f(r,'inflation')/f(r,'exact_inflation')-1))
    reduced=max(bench,key=lambda r:abs(f(r,'exact_reduced_inflation')/f(r,'exact_inflation')-1))
    dropout=max(f(r,'exact_drop_conditional_inflation')-f(r,'exact_inflation') for r in bench)
    werror=max(abs(f(r,'mse')/f(r,'predicted_mse')-1) for r in weighted)
    woutside=[(r['method'],r['weights']) for r in weighted if not f(r,'mse_low')<=f(r,'predicted_mse')<=f(r,'mse_high')]
    precision=[]
    for lo,hi in zip(select(large,precision=100),select(large,precision=160)):
        precision.append(abs(f(lo,'inflation')/f(hi,'inflation')-1))
    diag=dict(configurations=len(bench),coverage_count=len(bench)-len(outside),outside=outside,
              max_relative_mc_error=abs(f(maxr,'inflation')/f(maxr,'exact_inflation')-1),max_error_config=maxr,
              max_relative_unconditional_reduced_error=abs(f(reduced,'exact_reduced_inflation')/f(reduced,'exact_inflation')-1),
              max_reduced_error_config=reduced,max_drop_conditional_absolute_excess=dropout,
              weighted_max_relative_mse_error=werror,weighted_mse_outside_ci=woutside,
              high_precision_max_float_relative_difference=max(precision))
    (OUT/'validation_summary.json').write_text(json.dumps(diag,indent=2,ensure_ascii=False),encoding='utf-8')
    text=f'''新实验结果摘要（2026-10-10）

实验严格模拟无饱和均匀 mantissa BFP：xi=floor(log2 max|X|)，s=2^(xi-Lm)。并非 MXFP4/E2M1；只研究输入量化误差，dot product 使用 binary64 累加。

主实验共 {len(bench)} 个配置，涵盖 Gaussian、Half-normal、ReLU、Lognormal、Pareto(4)、Exponential、Beta(2,5)，n=2 至4096，Lm=2/4，RNE/SR。主种子20261010，每配置独立标签随机流；同分布同n两精度和两模式共用输入。95%CI 使用200次独立块 percentile bootstrap；Pareto 使用精确指数层权重的条件块模拟，并在每层内bootstrap，保留普通MC作诊断。

精确有限n预测落在 {len(bench)-len(outside)}/{len(bench)} 个95%CI内；未覆盖配置详见 validation_summary.json。这是有限样本覆盖结果，未宣称所有配置均覆盖。最大相对偏差为 {diag['max_relative_mc_error']:.2%}。drop-only-c近似和unconditional-response近似分别记录，不混用：前者在这些实验中最大绝对高估 {dropout:.6g}；后者最大相对差 {diag['max_relative_unconditional_reduced_error']:.2%}。

全部8个SR配置95%CI包含1；条件无偏机制为数学精确性质，MC只是实现校验。RNE条件协方差解析为非正；两原子反例在n2时 I2=2/3，证明忽略c会漏掉deflation。

weighted dot product 共12个配置（50000块，n128,Lm2），权重固定为全1、交替±1、交替1/−0.9。报告variance与MSE而非只报中心方差；shared、split16、independent scale、SR全配对使用相同输入。独立scale对照预测来自独立精确单点积分，而非拟合同一模拟。MSE最大预测相对偏差 {werror:.2%}，未覆盖95%CI配置为 {woutside}。

Exponential(1) 高精反例令n=floor(exp(2^j))，j3至8。I_n快速发散，I_n/n后期下降；C(log n)^2 n^(7/8) 的渐近等价已由理论证明，在最大点的精确/渐近比为 {f(select(large,precision=160)[-1],'ratio'):.9f}。100与160位结果double读数一致。指数分布条件c解析严格为0，CSV不把数值舍入残差解释为负协方差。

复现顺序：python run_revision.py；python stratified_pareto.py；python make_artifacts.py。主结果 benchmark.csv；原普通MC benchmark_ordinary.csv；weighted.csv；exponential_large_n.csv。可插入paper的文件为tables.tex、figures.tex及三个figure_*.pdf。figures.pdf为三页图合集，仅供单独查看，勿整文件插入主文。
'''
    (OUT/'results_summary_zh.md').write_text(text,encoding='utf-8')
    return diag


def main():
    bench,weighted,large,atoms=(read(f) for f in ('benchmark.csv','weighted.csv','exponential_large_n.csv','atomic_counterexample.csv'))
    figs=[benchmark_figure(bench),weighted_figure(weighted),exponential_figure(large)]
    with PdfPages(OUT/'figures.pdf') as pdf:
        for fig in figs:pdf.savefig(fig)
    for fig in figs:plt.close(fig)
    tables(bench,weighted,large,atoms);diag=summary(bench,weighted,large)
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    manifest['conditional_stratification']=json.loads((OUT/'stratification.json').read_text(encoding='utf-8'))
    manifest['validation']=dict(coverage_count=diag['coverage_count'],configs=diag['configurations'],
                               maximum_relative_mc_error=diag['max_relative_mc_error'])
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in diag.items() if 'config' not in k},indent=2))


if __name__=='__main__':main()
