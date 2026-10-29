"""Confirmation set (r ~ U[1.0, 1.3] per object): pooled over objects, same tests as the test set."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
d = pd.read_csv("results/confirm.csv"); d["fail"] = d.ok.astype(str) != "True"
base = d[d.method == "B0"].set_index("obj").sort_index(); out = {}
for m, g in d.groupby("method"):
    po = g.groupby("obj")[["T", "F_rel", "fail"]].mean().loc[base.index]
    row = dict(n=len(g), T_mean=round(g["T"].mean(), 4), Frel_mean=round(g.F_rel.mean(), 4), fail=int(g.fail.sum()),
               slip_max_mm=round(1000 * g.slip.max(), 2), projected=round((g.projected.astype(str) == "True").mean(), 3))
    if m != "B0":
        for c in ("T", "F_rel"):
            diff = po[c].values - base[c].values; row[f"d{c}"] = round(float(diff.mean()), 4)
            row[f"p_{c}"] = float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0
        a = po.fail.values > 0; b = base.fail.values; n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
        row.update(objfail_more=n10, objfail_less=n01, p_mcnemar=float(binomtest(n10, n10 + n01).pvalue) if n10 + n01 else 1.0)
    out[m] = row
# failures of B2 by r tercile
g = d[d.method == "B2"]; out["B2_fail_by_r"] = {str(k): int(v) for k, v in g.groupby(pd.cut(g.r, [1.0, 1.1, 1.2, 1.3]), observed=False).fail.sum().items()}
json.dump(out, open("results/summary_confirm.json", "w"), indent=1); print(json.dumps(out, indent=1))
