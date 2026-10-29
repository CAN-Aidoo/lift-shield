"""Analysis exactly as PROTOCOL.md; writes results/summary.json and prints tables. Usage: python analyze.py test|confirm"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
from statsmodels.stats.multitest import multipletests

which = sys.argv[1] if len(sys.argv) > 1 else "test"
d = pd.read_csv(f"results/{which}.csv")
for c in ("ok", "held", "projected", "declined", "exceed"): d[c] = d[c].astype(str) == "True"
d["fail"] = ~d["ok"]
main = d[d.r_max == 1.2]
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def hboot(sub, base, col, B=10000):
    """Hierarchical bootstrap of mean(method - B0): resample seeds, then objects."""
    seeds = sorted(sub.seed.unique()); objs = sorted(sub.obj.unique())
    M = sub.pivot_table(index="obj", columns="seed", values=col).loc[objs, seeds].values
    b = base.set_index("obj").loc[objs, col].values
    out = np.empty(B)
    for k in range(B):
        s = rng.integers(0, len(seeds), len(seeds)); o = rng.integers(0, len(objs), len(objs))
        out[k] = (M[o][:, s].mean(1) - b[o]).mean()
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


S = {}
rs = sorted(main.r.unique())
for r in rs:
    base = main[(main.method == "B0") & (main.r == r)].sort_values("obj")
    for meth in sorted(main.method.unique()):
        sub = main[(main.method == meth) & (main.r == r)]
        k = int(sub.fail.sum()); n = len(sub)
        row = dict(n=n, T_mean=round(sub["T"].mean(), 4), Frel_mean=round(sub.F_rel.mean(), 4), fail=k, fail_rate=round(k / n, 4),
                   fail_ci=wilson(k, n), slip_max_mm=round(1000 * sub.slip.max(), 2), slip_med_mm=round(1000 * sub.slip.median(), 3),
                   projected=round(sub.projected.mean(), 3), declined=round(sub.declined.mean(), 3))
        if meth != "B0":
            po = sub.groupby("obj")[["T", "F_rel", "fail"]].mean().loc[base.obj]
            for col in ("T", "F_rel"):
                diff = po[col].values - base[col].values
                row[f"d{col}_mean"] = round(float(diff.mean()), 4)
                row[f"p_{col}"] = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                row[f"ci_{col}"] = hboot(sub, base, col) if sub.seed.nunique() > 1 else None
            a = (po.fail.values > 0); bb = base.fail.values
            n10 = int((a & ~bb).sum()); n01 = int((~a & bb).sum())
            row["objfail_more_than_B0"] = n10; row["objfail_less_than_B0"] = n01
            row["p_mcnemar"] = float(binomtest(n10, n10 + n01).pvalue) if n10 + n01 else 1.0
        S[f"{meth}@r{r}"] = row
# Holm within families: Q1 (B2, B4g, B5; T and F_rel; all r), Q3 handled descriptively, Q2 McNemar (B2, B4g, B5)
for fam, meths, cols in (("Q1", ("B2", "B4g", "B5"), ("p_T", "p_F_rel")), ("Q2", ("B2", "B4g", "B5"), ("p_mcnemar",))):
    keys = [(f"{m}@r{r}", c) for m in meths for r in rs for c in cols if f"{m}@r{r}" in S]
    adj = multipletests([S[k][c] for k, c in keys], method="holm")[1]
    for (k, c), a in zip(keys, adj): S[k][c + "_holm"] = float(a)
# Q4: certificate on shielded, certified (not declined) episodes
Q4 = {}
sh = d[d.method.isin(["B0S", "B4g", "B4t", "B5"]) & ~d.declined]
for (rmax, inside), g in sh.groupby([sh.r_max, sh.r <= sh.r_max]):
    Q4[f"rmax{rmax}_{'r<=rmax' if inside else 'r>rmax'}"] = dict(
        n=len(g), frac_exceed=round(g.exceed.mean(), 4), max_excess_mm=round(1000 * (g.slip - g.bound).max(), 3),
        n_slip_over_tol=int((g.slip > 0.005).sum()), n_drop=int((~g.held).sum()))
for (meth, rmax), g in d[d.method.isin(["B0S", "B5"])].groupby(["method", "r_max"]):
    Q4[f"{meth}_rmax{rmax}"] = dict(T_mean=round(g["T"].mean(), 4), Frel_mean=round(g.F_rel.mean(), 4), fail=int(g.fail.sum()),
                                   n=len(g), declined=round(g.declined.mean(), 3))
S["Q4"] = Q4
json.dump(S, open(f"results/summary_{which}.json", "w"), indent=1)
t = pd.DataFrame({k: v for k, v in S.items() if "@" in k}).T
print(t[["n", "T_mean", "Frel_mean", "fail", "slip_max_mm", "projected", "declined"]].to_string())
print(json.dumps(Q4, indent=1))
