# 复现包：块浮点共享指数下舍入误差的独立性假设

对应论文 **The Independence Assumption for Rounding Errors in Block Floating Point:
Failure of Conditional Symmetry and Its Effect on Summation Error Accumulation**
（Yongkang Xiong）。

本目录包含产生论文全部数值的脚本与历史输出。代码与论文分开授权：论文文本
CC BY 4.0，本目录代码 MIT（见 `LICENSE`）。

---

## 1. 环境与运行

```
Python >= 3.9
numpy                 # 全部实验脚本
matplotlib            # 仅 make_figures.py 需要
```

```bash
pip install -r requirements.txt
python3 tools/check_paper_numbers.py      # 先做一次归档输出 ↔ 论文表格的核对
bash run_all.sh                           # 重跑全部实验（结果写入 outputs/rerun/）
```

`run_all.sh` 不会覆盖 `outputs/` 里的历史输出（那些是论文数字的原始记录），
新结果一律写到 `outputs/rerun/`。

---

## 2. 脚本 ↔ 论文结果 ↔ 归档输出

| 脚本（`src/`） | 论文中对应结果 | 归档输出（`outputs/`） | 随机种子 |
|---|---|---|---|
| `bfp_error_pilot.py` (v1) | 初版；§4.2 估计量修正史的背景 | — | 20260913 |
| `bfp_error_pilot_v2.py` | `tab:bias`（偏置 z 值）、`tab:var`（方差比）、图 1–2 | `out_v2.txt` | 20260914 |
| `bfp_error_pilot_v3.py` | `tab:modes`（四种舍入规则敏感性）、图 5；其表 B 给出 §4.2 被否定的负膨胀因子 −17.487 / −6.797 | `rerun/out_v3.txt`（原始终端输出未保存，重跑复现，见第 3 节） | 20260915 |
| `bfp_error_pilot_v4.py` | §4.2 第三轮估计量（分组均值 ± 标准误）；其 docstring 记录了被否定的 −17.487 | `out_v4.txt` | 20260916 |
| `bfp_error_pilot_v5.py` | `tab:infl2`、`tab:infl4`（$\mathcal I_n$ 随 $n$ 的走势与局部斜率）、图 3 | `out_v5.txt` | 20260917 |
| `bfp_error_pilot_v6.py` | `tab:sr`、`tab:srfull`（随机舍入的可证伪检验）、图 4 | `out_v6.txt` | 20260918 |
| `bfp_error_pilot_v7.py` | `tab:cond`（条件分解直接测量）、`tab:model` 实测列、§5.5 三条路径与 $\mathbb E[c]/\mathrm{Cov}$ | `out_v7.txt` | 20260920 |
| `bfp_error_pilot_v8.py` | `tab:bounded`、`tab:explain` 的 $\mathbb P(\xi=\text{mode})$、§5.8 | `out_v8.txt`（作者机上 75.1 s） | 20260921 / 20260922 |
| `bfp_error_pilot_v9.py` | `tab:largen`（两种子、$\rho_n$ 在大 $n$ 的稳定性） | `out_v9.txt`（作者机上 100.9 s） | 20260930 / 20260931 |

高精度求值与独立复核（`src/exact/`，结果为运行时打印，无归档文件）：

| 脚本 | 论文中对应结果 |
|---|---|
| `model_pop.py` | §3.7 remark：总体 $\mathcal I_n$ 曲线（峰值 $5.9\times10^4$；$n=10^6,10^7$ 的 $3.61\times10^4$、$9.2\times10^3$）。用生存函数之差 + 两遍方差，避免 $1-\mathrm{CDF}$ 的相消 |
| `pareto_asymptotic.py` | 同上，60 位十进制精度，可把 $n$ 推到 $10^{30}$ |
| `lattice_check2.py` | §3.4：单位区间 Gauss–Legendre 求积得到高精度 $\delta(s),\gamma(s)$；格恒等式数值检验 |
| `check_lattice_identity.py` | 上者的早期版本（保留以记录推导过程） |
| `audit_logic.py` | 第四轮独立复核（Test A/C）；其 Test B 精度不足，精确求值见前两个脚本 |
| `make_figures.py` | 图 1–5（数据取自 v2–v6 的历史输出，硬编码在脚本内） |

---

## 3. 已知缺口（投稿/发布前请注意）

1. **`tab:modes` 的原始终端输出没有保存**（对应脚本 `src/bfp_error_pilot_v3.py` 在）。
   2026-09-17 用原种子 `20260915` 在另一台机器（Python 3.12 / NumPy 2.5.3）重跑，
   结果已存入 `outputs/rerun/out_v3.txt`，与论文数字**逐位一致**：
   `tab:modes` 的对数正态行（$-25.33$、$-24.68$、$-422.71$、$-0.98$）与
   §4.2 的两个负膨胀因子（$-17.487$ 在 student-$t_3$、$n=512$；
   $-6.797$ 在 half-normal、$n=256$）均可在该文件中查到。
   重跑命令：`python3 src/bfp_error_pilot_v3.py > outputs/rerun/out_v3.txt`。
2. **各表来自不同批次的运行**。例如 `tab:infl2` 的对数正态 $n=4096$ 是
   $176.996$（v5），而 `tab:cond`/`tab:model` 同一配置是 $175.7679$（v7）——
   两者估计量与分组不同，论文 §5.5 已说明这一点，但读代码时需要留意：
   **不要跨表比较小数点后第二位的差异**。
3. **`make_figures.py` 中的数值是硬编码的**，它不重新跑实验，只重画图。
   改数据需先改脚本内的数组。
4. **v7 的耗时未记录**：配置数（72）与每配置块数（$20\times1000$）都比 v8/v9
   大，作者机上量级为分钟到十几分钟，首次运行请留足时间。
5. 本包**未包含**论文正文与 SIAM 类文件，那些在上级目录的 `siam_adapt/`。

---

## 4. 归属（provenance）

各文件来自作者工作目录的不同位置，此处汇总为一个自包含目录：

```
src/bfp_error_pilot.py … v6.py   ← /home/apanda/project/LLM/math_pilot/
src/bfp_error_pilot_v7.py … v9.py ← /home/apanda/project/paper/
src/exact/*.py                    ← /home/apanda/project/paper/repro/
src/make_figures.py               ← /home/apanda/project/LLM/paper/
outputs/out_v*.txt                ← /home/apanda/project/paper/repro/
outputs/rerun/out_v3.txt          ← 2026-09-17 重跑复现（见第 3 节第 1 条）
```

`src/` 与 `outputs/` 下的文件内容未做任何修改（`rerun/out_v3.txt` 是新增的重跑记录）。

## 5. 关于随机种子

脚本里的种子写成 `20260913`、`20260914` … 这样的日期形式，但它们只是**逐次实验的
标签**，不是日期（例如 `src/bfp_error_pilot_v9.py` 用的 `20260931` 并不是合法日期）。
种子按工作顺序手工递增，缺失的号段表示当天没有跑实验。数值结果只依赖种子本身，
与它是否对应真实日期无关。
