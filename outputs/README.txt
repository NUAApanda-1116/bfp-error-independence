本目录保存论文数字的原始终端输出（未做任何编辑），供核对用。

  out_v2.txt  bfp_error_pilot_v2.py  → tab:bias, tab:var
  out_v4.txt  bfp_error_pilot_v4.py  → §4.2 第三轮估计量
  out_v5.txt  bfp_error_pilot_v5.py  → tab:infl2, tab:infl4
  out_v6.txt  bfp_error_pilot_v6.py  → tab:sr, tab:srfull
  out_v7.txt  bfp_error_pilot_v7.py  → tab:cond, tab:model
  out_v8.txt  bfp_error_pilot_v8.py  → tab:bounded（末行给出该次运行耗时）
  out_v9.txt  bfp_error_pilot_v9.py  → tab:largen

缺 out_v3.txt（tab:modes 与 §4.2 负膨胀因子的原始终端输出未保存）。

  替代文件：rerun/out_v3.txt —— 2026-09-17 用原种子 20260915 重跑复现，
  与论文数字逐位一致（详见 ../README.md 第 3 节第 1 条）。

重跑结果请写到 outputs/rerun/，不要覆盖这里的文件。

核对脚本：python3 tools/check_paper_numbers.py
