"""Targeted scan for the population peak of I_n (Pareto(4), Lm=2) — the paper's 5.9e4 near n~1e5."""
import time
src = open(r"E:\paper\scripts\src\exact\model_pop.py", encoding="utf-8").read()
src = src.split('print("=== 与论文表')[0]
ns = {}
exec(compile(src, "stub", "exec"), ns)
model_I = ns["model_I"]

pts = [30000, 50000, 70000, 100000, 130000, 160000, 200000, 300000, 500000,
       1000000, 3000000, 10000000, 10**12, 10**20]
print(f"{'n':>12}{'I_n':>16}")
best = (0.0, 0)
for n in pts:
    I, Em, vm, A = model_I("pareto4", 2, n)
    if I > best[0]:
        best = (I, n)
    print(f"{n:>12}{I:>16.6g}")
print(f"\npeak in this scan: {best[0]:.4g} at n={best[1]:.4g}")
print("paper claims: peak 5.9e4 near n~1e5; 3.61e4 at n=1e6; 9.2e3 at n=1e7; 52 at 1e12; 1.002 at 1e20")
