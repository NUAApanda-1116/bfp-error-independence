新实验结果摘要（2026-10-10）

实验严格模拟无饱和均匀 mantissa BFP：xi=floor(log2 max|X|)，s=2^(xi-Lm)。并非 MXFP4/E2M1；只研究输入量化误差，dot product 使用 binary64 累加。

主实验共 64 个配置，涵盖 Gaussian、Half-normal、ReLU、Lognormal、Pareto(4)、Exponential、Beta(2,5)，n=2 至4096，Lm=2/4，RNE/SR。主种子20261010，每配置独立标签随机流；同分布同n两精度和两模式共用输入。95%CI 使用200次独立块 percentile bootstrap；Pareto 使用精确指数层权重的条件块模拟，并在每层内bootstrap，保留普通MC作诊断。

精确有限n预测落在 61/64 个95%CI内；未覆盖配置详见 validation_summary.json。这是有限样本覆盖结果，未宣称所有配置均覆盖。最大相对偏差为 4.66%。drop-only-c近似和unconditional-response近似分别记录，不混用：前者在这些实验中最大绝对高估 0.0064746；后者最大相对差 5.81%。

全部8个SR配置95%CI包含1；条件无偏机制为数学精确性质，MC只是实现校验。RNE条件协方差解析为非正；两原子反例在n2时 I2=2/3，证明忽略c会漏掉deflation。

weighted dot product 共12个配置（50000块，n128,Lm2），权重固定为全1、交替±1、交替1/−0.9。报告variance与MSE而非只报中心方差；shared、split16、independent scale、SR全配对使用相同输入。独立scale对照预测来自独立精确单点积分，而非拟合同一模拟。MSE最大预测相对偏差 1.48%，未覆盖95%CI配置为 [('split16_rne', 'near_cancel')]。

Exponential(1) 高精反例令n=floor(exp(2^j))，j3至8。I_n快速发散，I_n/n后期下降；C(log n)^2 n^(7/8) 的渐近等价已由理论证明，在最大点的精确/渐近比为 0.999999550。100与160位结果double读数一致。指数分布条件c解析严格为0，CSV不把数值舍入残差解释为负协方差。

复现顺序：python run_revision.py；python stratified_pareto.py；python make_artifacts.py。主结果 benchmark.csv；原普通MC benchmark_ordinary.csv；weighted.csv；exponential_large_n.csv。可插入paper的文件为tables.tex、figures.tex及三个figure_*.pdf。figures.pdf为三页图合集，仅供单独查看，勿整文件插入主文。
