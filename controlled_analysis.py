"""Addendum A1 analysis: deltas vs reference with bootstrap significance; heatmap figure and full tables."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from controlled import ROWS
plt.rcParams.update({"font.family": "serif", "font.size": 7})
METH = ["B0", "B0S", "B2", "B4g", "B4t", "B5"]
UNPAIRED = {"E1", "E2"}           # different objects (same seed, different ranges): unpaired bootstrap


def prep(df, tol=0.005):
    df = df.copy(); df["fail"] = (df.ok.astype(str) != "True").astype(float); df["ep_time"] = df.t_probe + 0.6 + df["T"] + 0.3
    return df


ref = prep(pd.read_csv("results/test.csv")); ref = ref[ref.r_max == 1.2]
METRICS = [("fail", "failed lifts (%)", 100), ("T", "lift time (ms)", 1000), ("F_rel", "grip / weight (%)", 100), ("ep_time", "episode time (s)", 1)]


def per_obj(df, meth, col):
    return df[df.method == meth].groupby("obj")[col].mean().sort_index().values


def boot(a, b, paired, B=10000, rng=np.random.default_rng(0)):
    n = len(a)
    if paired:
        idx = rng.integers(0, n, (B, n)); dd = (a[idx] - b[idx]).mean(1)
    else:
        dd = a[rng.integers(0, n, (B, n))].mean(1) - b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    return float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))


def analyse():
    out = {"REF": {}}
    for meth in METH:
        out["REF"][meth] = {c: float(per_obj(ref, meth, c).mean() * s) for c, _, s in METRICS}
    for row in ROWS:
        p = f"results/controlled/{row}.csv"
        if not os.path.exists(p): continue
        df = prep(pd.read_csv(p)); out[row] = {}
        for meth in METH:
            o = {}
            for c, _, s in METRICS:
                a = per_obj(df, meth, c); b = per_obj(ref, meth, c)
                lo, hi = boot(a, b, row not in UNPAIRED)
                o[c] = float(a.mean() * s); o["d_" + c] = float((a.mean() - b.mean()) * s)
                o["ci_" + c] = [lo * s, hi * s]; o["sig_" + c] = bool(lo > 0 or hi < 0)
            o["fails"] = int(df[df.method == meth].fail.sum()); o["n"] = int((df.method == meth).sum())
            o["max_slip_mm"] = float(1000 * df[df.method == meth].slip.max())
            o["declined"] = float(df[df.method == meth].declined.astype(str).eq("True").mean())
            out[row][meth] = o
    json.dump(out, open("results/controlled_summary.json", "w"), indent=1)
    return out


def heat_ax(ax, out, metric, title, fmt="{:+.1f}", afmt="{:.1f}", labels=True, worse_is_up=True):
    rows = [r for r in ROWS if r in out]
    order = [r for r in rows if r[0] == "E" and not r.endswith("L")] + [r for r in rows if r[0] == "S"] + [r for r in rows if r[0] == "T"] + [r for r in rows if r.endswith("L")]
    D = np.array([[out[r][m]["d_" + metric] for m in METH] for r in order])
    lim = max(1e-9, np.nanmax(np.abs(D)))
    Z = np.vstack([np.zeros(len(METH)), np.sign(D) * np.sqrt(np.abs(D) / lim)])
    cm = plt.get_cmap("RdYlGn_r" if worse_is_up else "RdYlGn")
    ax.imshow(Z, cmap=cm, norm=TwoSlopeNorm(0, -1.15, 1.15), aspect="auto")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), len(METH), 1, color="0.94", zorder=2))
    for j, m in enumerate(METH):
        ax.text(j, 0, afmt.format(out["REF"][m][metric]), ha="center", va="center", fontsize=5.8, color="0.25", zorder=3)
    for i, r in enumerate(order, start=1):
        for j, m in enumerate(METH):
            o = out[r][m]; star = "*" if o["sig_" + metric] else ""; z = abs(Z[i, j])
            col = "white" if z > 0.75 else "black"
            dv = o["d_" + metric]
            txt = "0" if abs(dv) < 1e-9 else ((("{:+.2f}" if abs(dv) < 0.5 and "0f" in fmt else fmt).format(dv)) + star)
            ax.text(j, i - 0.14, txt, ha="center", va="center", fontsize=5.5, fontweight="bold" if star else "normal", color=col)
            ax.text(j, i + 0.26, "(" + afmt.format(o[metric]) + ")", ha="center", va="center", fontsize=4.2, color="0.85" if z > 0.75 else "0.35")
    ax.set_yticks(range(len(order) + 1))
    ax.set_yticklabels(["Reference"] + [f"{r}  {ROWS[r][0]}" for r in order] if labels else [], fontsize=5.6)
    ax.set_xticks(range(len(METH))); ax.set_xticklabels(METH, fontsize=6.2); ax.xaxis.tick_top()
    nE = sum(r[0] == "E" and not r.endswith("L") for r in order); nS = sum(r[0] == "S" for r in order); nT = sum(r[0] == "T" for r in order)
    for y in (0.5, nE + 0.5, nE + nS + 0.5, nE + nS + nT + 0.5):
        ax.axhline(y, color="white", lw=2.2)
    ax.set_title(title, fontsize=7, pad=13)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)


def heat(out, metric, title, fname, fmt="{:+.1f}", afmt="{:.1f}", worse_is_up=True):
    n = len([r for r in ROWS if r in out]) + 1
    fig, ax = plt.subplots(figsize=(3.45, 0.28 * n + 0.6))
    heat_ax(ax, out, metric, title, fmt, afmt, worse_is_up=worse_is_up)
    fig.tight_layout(pad=0.2); fig.savefig(f"paper/{fname}.pdf"); plt.close(fig)


def heat_main(out):
    n = len([r for r in ROWS if r in out]) + 1
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 0.25 * n + 0.75), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    heat_ax(ax[0], out, "fail", "(a) failed lifts (%), change", labels=True)
    heat_ax(ax[1], out, "F_rel", "(b) grip (% of weight), change", fmt="{:+.0f}", afmt="{:.0f}", labels=False)
    heat_ax(ax[2], out, "ep_time", "(c) episode time (s), change", fmt="{:+.2f}", afmt="{:.2f}", labels=False)
    fig.tight_layout(pad=0.2, w_pad=0.6); fig.savefig("paper/fig_ctrl_main.pdf"); plt.close(fig)


def table(out):
    rows = [r for r in ROWS if r in out]; lines = []
    for r in rows:
        cells = []
        for m in METH:
            o = out[r][m]
            cells.append(f"{o['fails']}/{o['n']} & {o['T']:.0f} & {o['F_rel']:.0f}")
        lines.append(f"{r} & " + " & ".join(cells) + r"\\")
    refc = " & ".join(f"{int(round(out['REF'][m]['fail']*0 + (ref[ref.method==m].fail.sum())))}/{int((ref.method==m).sum())} & {out['REF'][m]['T']:.0f} & {out['REF'][m]['F_rel']:.0f}" for m in METH)
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{m}}}" for m in METH)
    sub = " & ".join(["fail & $T$ & $F$"] * len(METH))
    return (r"""\begin{table*}[!htbp]\centering\scriptsize
\caption{\textbf{Controlled study, absolute values.} For every row and method: failed lifts (slip above the row's tolerance, or a
drop) out of all lifts (40 objects $\times$ 4 values of $r$, $\times$ 5 seeds for learned methods), mean lift time $T$ (ms) and mean
grip $F/mg$ (\%). Row descriptions are in Table~\ref{tab:changes}.}\label{tab:controlled}
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}l""" + "rrr" * len(METH) + r"""@{}}\toprule
 & """ + head + r"""\\
 & """ + sub + r"""\\\midrule
REF & """ + refc + r"""\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table*}""")


if __name__ == "__main__":
    out = analyse()
    heat_main(out)
    heat(out, "fail", "Change in failed lifts (percentage points)", "fig_ctrl_fail")
    heat(out, "T", "Change in lift time (ms)", "fig_ctrl_time", fmt="{:+.0f}", afmt="{:.0f}")
    heat(out, "F_rel", "Change in grip (% of object weight)", "fig_ctrl_grip", fmt="{:+.1f}", afmt="{:.0f}")
    heat(out, "ep_time", "Change in episode time (s)", "fig_ctrl_ep", fmt="{:+.2f}", afmt="{:.2f}")
    open("paper/tab_controlled.tex", "w").write(table(out))
    print("rows:", [r for r in out if r != "REF"])
