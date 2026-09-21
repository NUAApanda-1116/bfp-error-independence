#!/usr/bin/env bash
# 重跑全部实验。结果写入 outputs/rerun/，不覆盖 outputs/ 中的历史输出。
#
#   bash run_all.sh             # 全部
#   bash run_all.sh fast        # 只跑快的高精度脚本（几秒到几十秒）
set -uo pipefail
cd "$(dirname "$0")"

OUT=outputs/rerun
mkdir -p "$OUT"

run() {                       # run <输出名> <工作目录> <脚本> [参数...]
  local name="$1" dir="$2" script="$3"; shift 3
  printf '==> %-28s ' "$name"
  local t0=$SECONDS
  if (cd "$dir" && python3 "$script" "$@") > "$OUT/$name" 2>&1; then
    echo "完成（$((SECONDS - t0)) s）-> $OUT/$name"
  else
    echo "失败（$((SECONDS - t0)) s），见 $OUT/$name"
  fi
}

if [ "${1:-all}" = "fast" ]; then
  run exact_model_pop.txt      src/exact model_pop.py
  run exact_pareto_60dig.txt   src/exact pareto_asymptotic.py
  run exact_lattice.txt        src/exact lattice_check2.py
  run exact_lattice_old.txt    src/exact check_lattice_identity.py
  run exact_audit.txt          src/exact audit_logic.py
  run exact_verify.txt         src/exact verify_numerics.py
  exit 0
fi

# 蒙特卡洛：v7 最慢，放最后
run out_v2.txt  src bfp_error_pilot_v2.py
run out_v3.txt  src bfp_error_pilot_v3.py
run out_v4.txt  src bfp_error_pilot_v4.py
run out_v5.txt  src bfp_error_pilot_v5.py
run out_v6.txt  src bfp_error_pilot_v6.py
run out_v8.txt  src bfp_error_pilot_v8.py
run out_v9.txt  src bfp_error_pilot_v9.py
run out_v7.txt  src bfp_error_pilot_v7.py

# 图（需要 matplotlib；数据硬编码在脚本内）
printf '==> %-28s ' "figures"
if (cd src && python3 make_figures.py) > "$OUT/make_figures.log" 2>&1; then
  echo "完成 -> src/figures/"
else
  echo "失败，见 $OUT/make_figures.log"
fi
