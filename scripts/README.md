# 复现包：块浮点共享指数下舍入误差的独立性假设

对应论文 **The Independence Assumption for Rounding Errors in Block Floating Point:
Failure of Conditional Symmetry and Its Effect on Summation Error Accumulation**
（Yongkang Xiong）。

本目录包含产生论文全部数值的脚本与历史输出。代码与论文分开授权：论文文本
CC BY 4.0，本目录代码 MIT（见 `LICENSE`）。

---

## 1. 脚本版本

实验脚本按开发顺序编号 v1–v9（`bfp_error_pilot.py` 记为 v1），另有 `src/exact/`
下用于高精度求值与独立复核的脚本。较早的版本被保留，因为它们记录了估计量
的修正过程与若干被否定的猜想（见第 4 节）。

脚本内的随机种子是**固定实验标识符**，用于复现论文表格中的归档数字；
它们不是脚本的开发日期。

| 脚本（`src/`） | 内容 |
|---|---|---|
| v1 | 初版 pilot：H1–H3 与点积标度初测；块内中心化估计量（后被否定） |
| v2 | 修正偏置 z 与方差比估计；`tab:bias` / `tab:var` |
| v3 | 舍入规则敏感性与被否定的负膨胀因子 |
| v4 | 第三轮估计量（分组均值 ± 标准误） |
| v5 | 膨胀因子随 $n$ 的走势与局部对数斜率 |
| v6 | 随机舍入的可证伪检验 |
| v7 | 条件分解直接测量；三条独立路径交叉核对 |
| v8 | 有界/无界支撑对比；$\mathbb P(\xi=\text{mode})$ |
| `src/exact/` | 高精度求值与独立复核脚本 |
| v9 | 大块长稳健性复核（两种子） |

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
| `bfp_error_pilot_v8.py` | `tab:bounded`、`tab:explain` 的 $\mathbb P(\xi=\text{mode})$、§5.9 | `out_v8.txt`（作者机上 75.1 s） | 20260921 / 20260922 |
| `bfp_error_pilot_v9.py` | `tab:largen`（两种子、$\rho_n$ 在大 $n$ 的稳定性） | `out_v9.txt`（作者机上 100.9 s） | 20260930 / 20260923 |

高精度求值与独立复核（`src/exact/`，结果为运行时打印，无归档文件）：

| 脚本 | 论文中对应结果 |
|---|---|
| `model_pop.py` | §3.6 remark：总体 $\mathcal I_n$ 曲线（峰值 $5.9\times10^4$；$n=10^6,10^7$ 的 $3.61\times10^4$、$9.2\times10^3$）。用生存函数之差 + 两遍方差，避免 $1-\mathrm{CDF}$ 的相消 |
| `pareto_asymptotic.py` | 同上，60 位十进制精度，可把 $n$ 推到 $10^{30}$ |
| `lattice_check2.py` | §3.4：单位区间 Gauss–Legendre 求积得到高精度 $\delta(s),\gamma(s)$；格恒等式数值检验 |
| `check_lattice_identity.py` | 上者的早期版本（保留以记录推导过程） |
| `audit_logic.py` | 第四轮独立复核（Test A/C）；其 Test B 精度不足，精确求值见前两个脚本 |
| `make_figures.py` | 图 1–5 的原始版本（单图 PDF，数据取自 v2–v6 的历史输出，硬编码在脚本内） |
| `make_figures_compact.py` | 论文实际使用的 3 个图：把图 1+2 叠成 `figures/fig_biasvar.pdf`、图 4+5 并成 `figures/fig_srmodes.pdf`，图 3 沿用 `figures/fig3_inflation.pdf`；用 `make_figures.py` 的同一批硬编码数据重画，不重跑实验 |

论文（`paper.tex`）实际引用的图文件为 `figures/fig_biasvar.pdf`、
`figures/fig3_inflation.pdf`、`figures/fig_srmodes.pdf`；`figures/fig1_*.pdf`、
`fig2_*.pdf`、`fig4_*.pdf`、`fig5_*.pdf` 是它们的单幅来源，不直接进入排版。

---

## 3b. 表号对照（压缩排版前后）

2026-09-18 的压缩排版（28 → 23 页，见 `CHANGE_LOG.md` §七）改动了部分表号。
下表左侧的编号在论文中**已不存在**，读到本文档或 `src/` 注释里的旧表号时按下表换算：

| 旧表号（本文档 / 脚本注释） | 论文现表号 | 说明 |
|---|---|---|
| `tab:bias` | —— | $z$ 值矩阵移入 §5.1 正文文字，数值未变 |
| `tab:var` | `tab:var` | 未变 |
| `tab:infl2` | `tab:infl` 的 $L_m=2$ 行 | 两宽度合并为一表 |
| `tab:infl4` | `tab:infl` 的 $L_m=4$ 行 | 同上 |
| `tab:modes` | `tab:modes`（移入附录） | 数据未变，位置从 §5.4 移到附录 |
| `tab:sr` | `tab:srfull`（$n=4096$ 行） | 其余 $n=32,512$ 行在正文文字中给出 |
| `tab:srfull` | `tab:srfull` | 未变（现只列 $n=4096$） |
| `tab:modes` | `tab:modes` | 表与数据均未变，**仍在 §5.4 原位**，未移入附录 |
| `tab:full_a/b`、`tab:full4_a/b` | —— | 四张完整数据表已撤出论文，完整网格见 `outputs/`；附录只有 "Data availability" 一节，不再列出 `tab:full` |
| `tab:bounded`、`tab:largen`、`tab:cond`、`tab:model`、`tab:explain` | 同名 | 列数或行数有删减，数值未变；另注 `tab:bounded` 的 $\mathbb P(\xi=\text{mode})$ 取 $n=32$ 的值（表头 "in $n=4096$" 只修饰模型列） |

图号：图 1+2 已合成 `fig_biasvar`，图 4+5 已合成 `fig_srmodes`，图 3 未变。

---

## 4. 已知缺口（投稿/发布前请注意）

1. **`tab:modes` 的原始终端输出没有保存**（对应脚本 `src/bfp_error_pilot_v3.py` 在）。
   2026-09-17 用原种子 `20260915` 在另一台机器（Python 3.12 / NumPy 2.5.3）重跑，
   结果已存入 `outputs/rerun/out_v3.txt`，与论文数字**逐位一致**：
   `tab:modes` 的对数正态行（$-25.33$、$-24.68$、$-422.71$、$-0.98$）与
   §4.2 的两个负膨胀因子（$-17.487$ 在 student-$t_3$、$n=512$；
   $-6.797$ 在 half-normal、$n=256$）均可在该文件中查到。
   重跑命令：`python3 src/bfp_error_pilot_v3.py > outputs/rerun/out_v3.txt`。
2. **各表来自不同批次的运行**。例如 `tab:infl` 的对数正态 $n=4096$ 是
   $176.996$（v5），而 `tab:cond`/`tab:model` 同一配置是 $175.7679$（v7）——
   两者估计量与分组不同，论文 §5.5 已说明这一点，但读代码时需要留意：
   **不要跨表比较小数点后第二位的差异**。另注：压缩排版后表号有所变动——
   `tab:infl2`+`tab:infl4` 合并为 `tab:infl`，原 `tab:bias` 的数值移入正文文字，
   四张 `tab:full_a/b`、`tab:full4_a/b` 已从论文撤出（完整网格见 `outputs/`）。
   数值本身未改动。
3. **`make_figures.py` 与 `make_figures_compact.py` 中的数值都是硬编码的**，
   它们不重新跑实验，只重画图。改数据需先改脚本内的数组。
   后者被 `make_figures.py` 导入复用同一批数组，因此两个脚本不会出现数值分歧。
4. **v7 的耗时未记录**：配置数（72）与每配置块数（$20\times1000$）都比 v8/v9
   大，作者机上量级为分钟到十几分钟，首次运行请留足时间。
5. 论文正文、参考文献、图与 SIAM 类文件见压缩包上级目录；本目录只放脚本与归档输出。

---

## 5. 归属（provenance）

本目录中的 `src/`、`outputs/`、`tools/` 为自包含复现包，按脚本版本与运行批次整理。
`src/` 与 `outputs/` 下的**数值内容**未做任何修改（`rerun/out_v3.txt` 是新增的重跑记录）。

归档输出（`outputs/`）是论文表格数字的原始记录。`tab:modes` 与 §4.2 被否决的
负膨胀因子所对应的初始终端输出没有保留，只有用原种子重跑的记录
（`outputs/rerun/out_v3.txt`），这是本复现包唯一的一处 provenance 缺口。

`out_v9.txt` 已用原种子在另一台机器（Python 3.11.9 / NumPy 2.3.5）独立重跑复核，
全部数值逐位一致，仅耗时不同（203.1 s 对原记录的 100.9 s）。
`out_v8.txt` 与其余输出未做独立重跑，仍为原始归档。

## 5b. 参考文献页码元数据的一处外部错误

参考文献 `fan2018`（FPL 2018）的页码在 **Crossref、OpenAlex、Semantic Scholar
三个库里都显示为 `287-2877`**，明显不可能。这不是引用错误，而是 IEEE 为本次会议
提交的元数据本身有问题。

诊断过程与结论：

1. 取该会议 DOI 前缀 `10.1109/FPL.2018.00001`–`00099` 的 93 条记录，**全部**满足
   "end 字段 = start 字段的字符串 + 一个后缀数字"；
2. 该后缀数字恰好等于 *跨度 − 1*（跨度 = 真实起止页之差）。验证：

   | 后缀 | 条数 | 解码跨度 | 例子 |
   |---|---|---|---|
   | `0` | 7 | 1 页 | `[Title page i]`（起页 1） |
   | `3` | 19 | 4 页 | 起页 26 |
   | `4` | 18 | 5 页 | 起页 16 |
   | `7` | 27 | 8 页 | 起页 8 |

   即 IEEE 的导出流程在生成 end 字段时，把"起页字符串"与"页数−1"直接**字符串拼接**
   而没有做加法，产出了一个长达 4 位的假页号。

3. 本条目：起页 `287`，后缀 `7` → 跨度 8 页 → 真实页码 **287–294**。
   独立佐证：作者自存版 `refs/fan2018_Imperial-authorcopy.pdf` **恰好 8 页**。

因此 `references.bib` 中写作 `pages = {287--294}`，而非数据库给出的 `287-2877`。
同类错误也见于 IEEE 其他 2018 年会议，引用前宜按上述规律自查。

## 6. 关于随机种子与仓库历史

脚本中的种子（如 `20260913`、`20260930`）是**为复现归档输出而固定的实验标识符**，
用于区分不同的 Monte Carlo 轨迹；它们不是脚本的开发日期，也不编码任何时间信息。
其中 `20260931` 甚至不是合法日期，这恰好说明它们只是生成器标识符。数值结果只依赖
种子本身，请勿修改种子后再与论文表格逐位对照。

本仓库的提交历史不反映脚本的开发过程：全部文件是在整理完成后一次性提交的。
脚本版本的先后顺序见第 1 节。

---

## 7. 英文说明（GitHub README 摘要）

Reproduction code for the paper
*The Independence Assumption for Rounding Errors in Block Floating Point*.

- Script versions are numbered v1 → v9 plus the `exact/` checks, in development order.
- The git history does not reflect the development process: the files were committed
  together after the package was assembled.
- Random seeds in the scripts are fixed experiment identifiers that produced the
  archived outputs; they are not calendar dates of each script revision.
- The only provenance gap is the missing initial terminal output for `tab:modes` and
  the rejected negative inflation factors; `outputs/rerun/out_v3.txt` is a re-run with
  the original seed.
- Run `python3 tools/check_paper_numbers.py` to verify that archived outputs
  contain the key numbers reported in the paper tables.
- Re-run experiments with `bash run_all.sh`; new results go to `outputs/rerun/`.
- Citation: see `CITATION.cff`. License: MIT.
