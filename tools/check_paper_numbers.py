#!/usr/bin/env python3
"""核对归档输出（outputs/*.txt）是否包含论文表格里引用的关键数字。

纯标准库实现，不需要 numpy，因此可以在任何机器上跑：

    python3 tools/check_paper_numbers.py

它做的是"存在性核对"：论文每个表里的代表性数字，是否确实出现在对应的
历史输出文件里。这不能替代重新运行实验（见 README 的"复现路径"一节），
但能在不装 numpy 的环境下抓住"表格数字与归档输出对不上"的问题。

退出码：0 = 全部通过；1 = 有缺失。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "outputs")

# (论文中的位置, 归档输出, 必须出现的数字串)
CHECKS = [
    ("tab:bias   块均值偏置 z 值", "out_v2.txt",
     ["126.80", "60.57", "39.48", "1.63"]),
    ("tab:var    方差比（实测/经典）", "out_v2.txt",
     ["0.506", "0.093", "0.808", "0.999"]),
    ("tab:infl2  方差膨胀因子 Lm=2", "out_v5.txt",
     ["176.996", "14.989", "6.132", "1.013"]),
    ("tab:infl4  方差膨胀因子 Lm=4", "out_v5.txt",
     ["165.025", "1.834", "1.300"]),
    ("tab:sr     SR 可证伪检验", "out_v6.txt",
     ["176.051", "6.111", "14.871", "0.988"]),
    ("tab:cond   条件分解直接测量", "out_v7.txt",
     ["175.7679", "6.1631", "14.8804"]),
    ("tab:model  退化模型实测列", "out_v7.txt",
     ["176.4593", "6.3155", "15.0719"]),
    ("tab:bounded 有界 vs 无界", "out_v8.txt",
     ["41.8848", "2886.3007", "41.4477", "2887.2895"]),
    ("tab:largen 大块长 rho_n", "out_v9.txt",
     ["0.70383", "0.74229", "0.69575", "15.77"]),
    ("tab:modes  舍入规则敏感性", os.path.join("rerun", "out_v3.txt"),
     ["-25.33", "-24.68", "-422.71", "14.270"]),
    ("§4.2 被否定的负膨胀因子", os.path.join("rerun", "out_v3.txt"),
     ["-17.487", "-6.797"]),
]


def main():
    failures = 0
    for label, fname, needles in CHECKS:
        if fname is None:
            print(f"[缺口] {label:<34} 无归档输出（脚本 src/bfp_error_pilot_v3.py 可重跑）")
            continue
        path = os.path.join(OUT, fname)
        if not os.path.exists(path):
            print(f"[缺失] {label:<34} 找不到 {fname}")
            failures += 1
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        missing = [n for n in needles if n not in text]
        if missing:
            print(f"[不符] {label:<34} {fname} 缺少 {missing}")
            failures += 1
        else:
            print(f"[通过] {label:<34} {fname}  ({len(needles)} 个数字全部命中)")

    print()
    if failures:
        print(f"{failures} 项未通过：归档输出与论文表格不一致，请检查。")
        return 1
    print(f"全部通过：{len(CHECKS)} 组代表性数字均可在归档输出中找到。")
    print("注：tab:modes 与 §4.2 的原始终端输出未保存，核对的是 tools/../outputs/rerun/out_v3.txt，")
    print("    即用原种子 20260915 重跑的复现结果（见 README 第 3 节）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
