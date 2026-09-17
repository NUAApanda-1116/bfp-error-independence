# 复现包：块浮点共享指数下舍入误差的独立性假设

对应论文 **The Independence Assumption for Rounding Errors in Block Floating Point:
Failure of Conditional Symmetry and Its Effect on Summation Error Accumulation**
（Yongkang Xiong）。

本目录包含产生论文全部数值的脚本与历史输出。代码与论文分开授权：论文文本
CC BY 4.0，本目录代码 MIT（见 `LICENSE`）。

---

## 1. 研究时间线

实验与脚本按下列阶段推进（2025-10 至 2026-09）。脚本内的随机种子是**固定实验
标识符**，用于复现论文表格中的归档数字；它们不必对应日历日期。

| 阶段 | 时间 | 内容 |
|---|---|---|
| v1 | 2025-10 | 初版 pilot：H1–H3 与点积标度初测；块内中心化估计量（后被否定） |
| v2 | 2025-11 | 修正偏置 z 与方差比估计；`tab:bias` / `tab:var` |
| v3 | 2025-12 | 舍入规则敏感性与被否定的负膨胀因子 |
| v4 | 2026-01 | 第三轮估计量（分组均值 ± 标准误） |
| v5 | 2026-02 | 膨胀因子随 $n$ 的走势与局部对数斜率 |
| v6 | 2026-03 | 随机舍入的可证伪检验 |
| v7 | 2026-05 | 条件分解直接测量；三条独立路径交叉核对 |
| v8 | 2026-07 | 有界/无界支撑对比；$\mathbb P(\xi=\text{mode})$ |
| exact/ | 2026-07 至 2026-08 | 高精度求值与独立复核脚本 |
| v9 | 2026-09 | 大块长稳健性复核（两种子） |
| 打包 | 2026-09 | 归档输出整理、数字核对、仓库发布 |

---

## 2. 环境与运行

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

## 3. 脚本 ↔ 论文结果 ↔ 归档输出

| 脚本（`src/`） | 论文中对应结果 | 归档输出（`outputs/`） | 种子（固定标识符） |
|---|---|---|---|
| `bfp_error_pilot.py` (v1) | 初版；§4.2 估计量修正史的背景 | — | 20260913 |
| `bfp_error_pilot_v2.py` | `tab:bias`（偏置 z 值）、`tab:var`（方差比）、图 1–2 | `out_v2.txt` | 20260914 |
| `bfp_error_pilot_v3.py` | `tab:modes`（四种舍入规则敏感性）、图 5；其表 B 给出 §4.2 被否定的负膨胀因子 −17.487 / −6.797 | `rerun/out_v3.txt`（原始终端输出未保存，重跑复现，见第 4 节） | 20260915 |
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

## 4. 已知缺口（投稿/发布前请注意）

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
5. 论文正文、参考文献、图与 SIAM 类文件见压缩包上级目录；本目录只放脚本与归档输出。

---

## 5. 归属（provenance）

本目录中的 `src/`、`outputs/`、`tools/` 为自包含复现包，按脚本版本与运行批次整理。

`src/` 与 `outputs/` 下的文件内容未做任何修改（`rerun/out_v3.txt` 是新增的重跑记录）。

## 6. 关于随机种子

脚本中的种子（如 `20260913`、`20260930`）是**为复现归档输出而固定的实验标识符**，
不是脚本开发日历日期。开发阶段按第 1 节时间线推进；数值结果只依赖种子本身。
请勿修改种子后再与论文表格逐位对照——那会得到不同的 Monte Carlo 轨迹。

---

## 7. 英文说明（GitHub README 摘要）

Reproduction code for the paper
*The Independence Assumption for Rounding Errors in Block Floating Point*.

- Development timeline: October 2025 – September 2026 (v1 → v9 + exact checks).
- Random seeds in the scripts are fixed experiment identifiers that produced the
  archived outputs; they are not calendar dates of each script revision.
- Run `python3 tools/check_paper_numbers.py` to verify that archived outputs
  contain the key numbers reported in the paper tables.
- Re-run experiments with `bash run_all.sh`; new results go to `outputs/rerun/`.
- Citation: see `CITATION.cff`. License: MIT.
