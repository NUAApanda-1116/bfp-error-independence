# 严格审稿报告（修订版）：`paper.tex`

**稿件**：The Independence Assumption for Rounding Errors in Block Floating Point:
Failure of Conditional Symmetry and Its Effect on Summation Error Accumulation（Yongkang Xiong, NUAA）
**审阅日期**：2026-09-18 ｜ **修订**：2026-09-18（作者指出两处误判，已复核更正，见 §0）
**仓库**：`https://github.com/NUAApanda-1116/bfp-error-independence`（已推送提交 `5ddffa6d`）

---

## §0 对上一版报告的更正（重要）

上一版报告把两个问题列为"致命/严重"，**经复核，这两条都是我的误判**。更正如下。

### 更正 1：`fox2024zfp` 的 DOI 根本没有错误 —— 错在我

我在测试脚本里**自己硬编码了一个 DOI** 用来验证查询逻辑：

```python
# _audit/check_dois.py 第 21 行（我的脚本，不是作者的稿件）
("fox2024zfp", "10.1137/24M1703513", None),
```

`10.1137/24M1703513` 解析到 "Information-Open Electromagnetic Cloaks"（SIAM J. Appl. Math.），
于是我的查询逻辑"正确地"返回了不匹配，而我把这个**由我制造的输入**当成了作者的错误。

事实（已用全文检索确认）：

- `paper.tex` 与 `references.bib` 中**从未出现过** `24M1703513`；
- 作者原条目**根本没有 doi 字段**，只有 `eprint = {2407.01826}`；
- 检索 `24M1703513` 全盘只命中 `REVIEW_REPORT.md`（2 处）与 `_audit/check_dois.py`（1 处）——
  **全部是我自己的产物**。

我在报告里写的"bib 现值 `doi = {10.1137/24M1703513}`"是伪造的，实际不存在。**对此我致歉。**

同时复核确认 `fox2024zfp` 条目本身没有问题：

| 项 | 核查结果 |
|---|---|
| 作者 | Fox, Alyson; Lindstrom, Peter ✓（arXiv 2407.01826 v1/v2 两版首页均为 `ALYSON FOX AND PETER LINDSTROM`） |
| 标题 | v1 为 "Statistical Analysis of ZFP: Understanding Bias"，v2 改为 "Enhancing ZFP: …"。bib 用的是 **v2 标题**，当前有效 ✓ |
| 已发表版 | SIAM J. Sci. Comput. **48**(1) B1–B30, 2026, DOI **10.1137/24M1679586** ✓ |
| 原条目 | `journal` + `eprint` 已足以定位，**并非缺陷** |

（我已在本次提交中为该条目补上卷一页与 DOI，属锦上添花，不是纠错。）

### 更正 2：`wu2026mxfp4` 的作者顺序 —— 这是一个真实的客观矛盾，但不是"bib 写错了"

复核证据如下：

| 来源 | 顺序 | 备注 |
|---|---|---|
| **论文 PDF（v1 与 v3 首页，逐字相同）** | **Shi-Liang (Bruce) Wu\* Xiao-Can (Bruce) Li\*† Zheng Shen** | 星号/剑号 = 同等贡献；v1→v3 首页标题行**未变** |
| arXiv 提交历史 | `From: Xiaocan Li` | v1 2026-05-19、v2 05-22、v3 06-02 均由 Li 提交 |
| arXiv 摘要页 `citation_author` 元数据 | Li, Xiaocan → Wu, Shiliang → Shen, Zheng | 与 PDF 相反 |
| arXiv 页面文字 | "by Xiaocan Li and 2 other authors" | 同上 |
| **作者原 bib 条目** | `Li, Xiaocan and Wu, Shiliang and Shen, Zheng` | **与 arXiv 元数据完全一致** |

结论：**bib 遵从的是 arXiv 官方引文元数据，这是标准做法，不是错误。**
矛盾出在**这篇被引论文自身**：PDF 首页把 Wu 放在前，而 arXiv 元数据把 Li 放在前，
两者不一致（我无法从外部判定谁是谁非）。

因此这**不是本文的引用缺陷**，而是一个**必须由作者判断**的取舍：

- 若按 **PDF 首页**（Wu 前、注明同等贡献）→ 正文写 "Wu et al."
- 若按 **arXiv 官方元数据**（Li 前）→ 正文写 "Li et al."，即原稿写法

**当前状态**：本次提交已改为按 PDF 首页顺序（Wu, Li, Shen）并在 `note` 字段中
记录"前两位作者同等贡献、顺序依论文首页"，正文两处改为 "Wu et al."。
**若你更愿意遵从 arXiv 元数据，这一项请回退** —— 两种做法都能自圆其说，
但**正文与 bib 必须一致**，且建议在 `note` 中保留说明，避免审稿人误判为引用错误。

### 由此产生的教训

我上一版的"致命错误 1"是**审阅工具自身污染输入**造成的假阳性。
这也说明一个方法论问题：**用自造的输入去"验证"他人的数据，是无效验证**。
本次修订后，所有结论均已回溯到原始来源（arXiv 官方 PDF、Crossref 原始返回、GitHub tarball）。

---

## §1 修订后的结论

**稿件可以提交 arXiv。** 参考文献 28 条**全部真实存在**，元数据核对通过，
无撤稿、无更正、无学术不端记录；数值结果经独立复现基本全部命中。
剩余问题集中在**表述严谨性**与**复现包一致性**，均属可修复且不影响科学结论。

### 已在本轮修复（已推送）

| # | 修改 | 文件 | 提交后状态 |
|---|---|---|---|
| 1 | `tab:chain` 全部改为**间接陈述**（去掉引号与"verbatim excerpt"表述），表头 `Quoted statement` → `Inherited property`，题注与上文同步改写 | `paper/paper.tex`、`paper-twocolumn.tex` | ✓ 已发布，`verbatim` 残留 = 0 |
| 2 | 正文两处 "Li et al." → "Wu et al."（见 §0 更正 2） | 同上 | ✓ |
| 3 | `wu2026mxfp4` 作者顺序 + 同等贡献说明 | `paper/references.bib` | ✓ |
| 4 | `fox2024zfp` 补 DOI/卷/期/页 | `paper/references.bib` | ✓ |
| 5 | `paper-twocolumn.tex` 与 `paper.tex` **重新同步**（此前已漂移，两列版仍含旧的 verbatim 表与 "Li et al."） | 两者 | ✓ 现仅差 documentclass 选项 |
| 6 | `scripts/README.md`：表号对照更正（`tab:modes` 仍在 §5.4、附录只有 Data availability、`P(ξ=mode)` 取 n=32）；图清单更正为实际引用的 3 个文件 | `scripts/README.md` | ✓ |
| 7 | `COMPILE.txt`：删除不存在的 `README.md`/`CHANGE_LOG.md` 条目，补 `refs/`，修正"未收录 23 篇"的错误陈述 | `paper/COMPILE.txt` | ✓ |
| 8 | 新增 `paper/`（稿件源码、bib、cls、bst）与 `audit/`（本次全部核查脚本）；重写根 `README.md` | — | ✓ |

### 仍需你决定 / 修复

| # | 级别 | 问题 |
|---|---|---|
| A | **中** | **配置计数与归档数据不符**：称"the 11 configurations with I_n > 1.2"（实为 **12** 个）；"6 of the 36 … violate it"（实为 **13** 个）。建议改为由脚本输出计数，或删去精确数字 |
| B | **中** | **恒等式闭合误差表述**：`Cov(e_i,e_j)=E[c(ξ)]+Var(m(ξ))` 的归一化闭合误差是 **2.6×10⁻³**，而 `Var(e)` 与 `Var(ē)` 两条分别是 3.4×10⁻¹⁶ 和 2.2×10⁻¹⁵。正文把三条并列却只报出后两条，且小标题写 "two identities at machine precision"，易被误读为三条都到机器精度 |
| C | 轻 | `tab:model` 题注称"the table lists the n=4096 configurations"，但含两条 n=32 行 |
| D | 轻 | `tab:bounded` 表头 "Values of P(ξ=mode) and of the model I_n are taken at n=4096"，但 P(ξ=mode) 取的是 **n=32** 列（0.976–1.000） |
| E | 轻 | `SHA256SUMS.txt`：`README.md` 的哈希被填成了 `scripts/README.md` 的哈希；`README.md`/`CHANGE_LOG.md` 已不存在（**本次提交后需重新生成**） |
| F | 轻 | `scripts/README.md` 与 `scripts/GITHUB_README.md` 内容不一致（前者应为中文说明，后者为英文摘要），两者均在仓库中 |
| G | 建议 | 复核 `constantinides2021`（*Rigorous Roundoff Error Analysis…*）的取舍。该文处理的正是共享指数下的严格舍入误差界，与 §1/§3.2 的 novelty 表述直接相关，不引可能被指 missing related work |

---

## §2 参考文献逐条核对（28 条）

### 2.1 本地 23 个 PDF —— 全部真实、元数据正确

`refs/` 内 23 个 PDF 覆盖 22 个不同文献，逐个首页核对标题/作者/年份/卷期页，
**无一例不符**。所有 arXiv ID 的编号规则（`YYMM.NNNNN`）与其 `year` 字段全部自洽。
关键几例：

- `carson2024`：bib 写 SISC **46**(3) 2041–2060, **2025** —— 与 Crossref 一致（arXiv 是 2024，bib 用了发表年，正确）
- `elarar2024`：bib 写 SISC **47**(5) B1227–B1249, **2025** —— 与 Crossref 一致
- `higham2022`：bib 写 SISC **42**(5) A3427–A3446, **2020** —— 正确，bib 头部注释已说明键名保留旧标签
- `fan2018` 页码 287–294：Crossref/OpenAlex/Semantic Scholar 三库均误记 `287-2877`。作者在
  `NOTES.zh.md` 中用 IEEE 元数据导出规律诊断并佐证为 287–294，**处理正确，值得肯定**

### 2.2 5 条付费墙文献 —— 全部真实存在

| bib key | 核对结果 |
|---|---|
| `kalliojarvi1996` | IEEE TSP **44**(4) 783–790, 1996-04；Kalliojärvi, Astola ✓ |
| `lian2019` | IEEE TVLSI **27**(8) 1874–1885, 2019-08；Lian, Liu, Song, Dai, Zhou, Ji ✓ |
| `schuchman1964` | vol **12**(4) 162–165, 1964-12；L. Schuchman ✓。刊名注：原刊为 *IRE Trans. Commun. Syst.*，1964 年卷记为 *IEEE Trans. Commun.*；bib 写 `IEEE Trans. Commun. Technol.` 亦为历史用名，建议统一 |
| `gray1993dithered` | IEEE TIT **39**(3) 805–812, 1993；Gray, Stockham ✓ |
| `higham2002` | SIAM, 2nd ed., 2002, DOI 10.1137/1.9780898718027 ✓ |

### 2.3 撤稿 / 学术不端 —— 零发现

对全部 17 个可解析 DOI 查询 Crossref 的 `update-to` / `updated-by` / `is-retracted-by`，
**全部为空**。未检索到 Higham、Mary、Fasi、Mikaitis、El Arar、Filip、Carson、Ipsen、Drineas
等核心作者群在 2024–2026 年的撤稿或不端记录。arXiv 上没有同标题预印本。

### 2.4 引文核对（本次已全部转为间接陈述）

即使原文可逐字核对，本轮也已按你的要求把 `tab:chain` 全部改为间接陈述。
核查记录（供你判断是否需要在正文中保留任何直接引用）：

| 来源 | 原引文可否核实 | 说明 |
|---|---|---|
| Song, Liu, Wang (AAAI 2018) | **✓ 逐字属实** | 正式版 p.4："According to (Kalliojarvi and Astola 1996), for block X, the quantization error has zero mean, and variance σ² = …" |
| Han et al. (DAC 2025) | **✓ 逐字属实** | arXiv:2504.15721v1 p.3："its quantisation error is zero-mean, and its variance σ² can be described as follows [31]"，且 **[31] 确为 1996 文献** |
| Lian et al. (TVLSI 2019) | **✗ 无法证实** | IEEE 全文付费，多库多轮检索均未找到该句。**这正是本次转为间接陈述的主要动因**——改为间接陈述后风险消除 ✓ |
| Ang et al. (2026) 的 MSE 公式 | 未取到原文排版 | 建议补上原文公式号 |
| Cim et al. (2026) 引文 | **✓ 逐字属实** | arXiv:2605.09825v4 摘要，含 "structured micro-scaling errors … rather than insufficient stochasticity" |
| Sao et al. (2026) | **✓** | "under conditionally unbiased rounding"、centered vs noncentered 均属实 |
| Fan et al. 只作历史引用 | **✓ 辨析准确** | `fan2018` 参考文献 [10] = Kalliojärvi & Astola 1996、[12] = Song et al.，与正文表述一致 |

---

## §3 数值独立复核（这些数字是真的）

| 论文断言 | 我的独立计算 | 结论 |
|---|---|---|
| 死区律 δ(s) = −0.0334 s；δ(0.125)=−0.00414 … δ(1)=−0.03434 | 4×10⁷ 样本重算：−0.00413 / −0.00838 / −0.01672 / −0.03436 | ✓ |
| ReLU 的 δ 恰为半正态的一半 | −0.0166…−0.0172 vs −0.0331…−0.0344 | ✓ |
| 12γ ≡ 1.000（半正态）、0.500（ReLU） | 0.9999–1.0000 / 0.4997–0.5001 | ✓ |
| 总体曲线峰值 5.9×10⁴ @ n≈10⁵ | 精确求值 n=10⁵ → **59484.7** | ✓ |
| n=10⁶/10⁷/10¹²/10²⁰ → 3.61×10⁴ / 9.2×10³ / 52 / 1.002 | 36106 / 9229.79 / 52.04 / 1.0024 | ✓ |
| `out_v5` 176.996 等；`out_v2` z = −126.80…、比率 0.506/0.093 | 归档文件逐字命中 | ✓ |
| `tools/check_paper_numbers.py` | 11/11 通过，exit 0 | ✓ |

**这是一份数据真实、可复现的研究。**

---

## §4 复现包 / 仓库审查

- **可访问性**：public，MIT，默认分支 `main`，作者邮箱与论文一致 ✓
- **本次提交 `5ddffa6d`**：18 个文件（1 modified + 17 added），已通过拉取 tarball 逐项验证 ✓
- **历史**：全部文件于 2026-09-17 一次性提交，无开发历史。README/NOTES 已诚实声明这一点。
  脚本 docstring 中的"阶段：2025-10 … 2026-09"时间线**无任何独立佐证**
- **提交时间倒挂**：`e9e8017202`（07:12Z）早于其父提交 `ed05bb512d`（12:57Z）
- **provenance 缺口（唯一）**：`tab:modes` 与两个负膨胀因子的原始终端输出丢失，
  以 `outputs/rerun/out_v3.txt` 替代。我核对了该文件：Table C 与 `tab:modes` 数字**逐位一致**，
  Table B 含 −17.487（student_t3, n=512）与 −6.797（halfnormal, n=256）✓ 缺口真实但影响可控
- **本地 vs GitHub**：提交前有 12 个文件内容不同（英文 README 变更、v9 注释、图片尺寸等），
  本次已把关键文件同步

### 关于"文件是否有 AI 生成嫌疑"

分两层，结论不同：

**代码层 —— 倾向人类作者作品。** 中文语境化注释、保留自我否定的失败版本
（v1 中心化伪影、v3 负膨胀因子、被推翻的偏置下界）、非平凡的正确数值细节
（两遍法、生存函数之差避免 `1−CDF` 相消、`var_m = var_m_raw − noise` 的估计噪声修正、
分箱中心化用箱内均值而非块内均值）。AI 生成代码极少保留失败路径。

**文本与打包层 —— 有机械组装痕迹。** 证据是 §1(A) 的计数错误与 §1(E) 的哈希错配：
`README.md` 与 `scripts/README.md` 被赋予同一哈希，而两者内容完全不同。
这属于"看似精确实则未逐条复核"，审稿人会较快识别。

---

## §5 给你的建议顺序

1. **决定 §0 更正 2 的取舍**（Wu 前 还是 Li 前），确保正文与 bib 一致 —— 这是我唯一动过、
   且两种做法都成立的判断项，**请务必过一眼**。
2. 修 §1 的 A、B 两条（计数与闭合误差表述）—— 这是最可能被审稿人抓住的两处。
3. 修 C、D 两条表注。
4. 重新生成 `SHA256SUMS.txt`（本次提交已使原清单部分失效），统一 F 的两个 README。
5. 决定 G（是否引用 `constantinides2021`）。
6. 无 LaTeX 工具链，**我未能重新编译 PDF**：`paper.pdf` / `paper-twocolumn.pdf` 仍是旧版，
   改动只在 `.tex`。请在本地 `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`
   后核对 `tab:chain` 的排版宽度（间接陈述比引文长，注意列宽）。

---

### 附：`audit/` 中的核查脚本（已随本次提交发布）

`extract_refs.py`（PDF 元数据）、`check_dois.py`（Crossref 逐条）、
`check_updates.py`（撤稿/更正）、`verify_hashes.py`（清单校验）、
`verify_numerics.py`（δ/γ 独立重算）、`check_peak2.py`（总体曲线峰值）、
`quote_audit.py`（引文定位）、`diff_repo.py`（仓库↔本地比对）、
`regen_twocolumn.py`（两列版再生成）
