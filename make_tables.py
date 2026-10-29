"""LaTeX tables for the long-form paper, all computed from committed result files."""
import json, sys, numpy as np, pandas as pd
from lift_env import sample_object, G, model_grip, T0
import lift_env as L
rng = np.random.default_rng(11); OBJS = [sample_object(rng) for _ in range(40)]
d = pd.read_csv("results/test.csv"); d["fail"] = d.ok.astype(str) != "True"; m = d[d.r_max == 1.2]
B = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, 100 * (c - h)), 100 * (c + h)


def per_object():
    rows = []
    b0 = m[(m.method == "B0") & (m.r == 1.0)].set_index("obj")
    for i, o in enumerate(OBJS):
        Fb = o["m"] * G / (2 * o["mu"]); bb = b0.loc[i]
        f2 = m[(m.method == "B2") & (m.obj == i)].fail.sum()
        g5 = m[(m.method == "B5") & (m.obj == i)].F_rel.mean()
        rows.append(f"{i} & {1000*o['m']:.0f} & {o['mu']:.2f} & {bb.F_stuck/Fb:.3f} & {int(bb.probes)} & {bb.t_probe:.1f} & {bb.F_rel:.2f} & {g5:.2f} & {f2}/20 \\\\")
    body = "\n".join(rows)
    return r"""\begin{table*}[!htbp]\centering\footnotesize
\caption{\textbf{Per-object test results.} $m$ and $\mu_s$ are the object's mass and static friction; $F_{\rm stuck}/F_b$ is the probed
breakaway grip relative to the true quasi-static breakaway grip $F_b=mg/(2\mu_s)$ (it is $\ge1$ by construction and at most
$1/0.9$ up to sensing noise); probes and probe time are from the B0 episode at $r=1$. Grip columns are $F/mg$ (mean over $r$ and seeds).
B2 failures count lifts with slip $>5$\,mm over 5 seeds $\times$ 4 values of $r$.}\label{tab:perobj}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}rrrrrrrrr@{}}\toprule
Obj & $m$ (g) & $\mu_s$ & $F_{\rm stuck}/F_b$ & probes & probe time (s) & B0 grip & B5 grip & B2 failures\\\midrule
""" + body + r"""
\bottomrule\end{tabular}\end{table*}"""


def per_seed():
    rows = []
    for meth in ("B2", "B4g", "B4t", "B5"):
        for s in range(5):
            g = m[(m.method == meth) & (m.seed == s)]
            aT = np.log(g["T"] / T0).mean() / L.ST
            mg = np.array([model_grip(fs, t) for fs, t in zip(g.F_stuck, g["T"])])
            aF = (np.log(g.F / mg) / L.SF).mean()
            fails = " & ".join(str(int(g[g.r == r].fail.sum())) for r in (1.0, 1.1, 1.2, 1.3))
            rows.append(f"{meth} & {s} & {g['T'].mean():.3f} & {g.F_rel.mean():.3f} & {aT:+.2f} & {aF:+.2f} & {fails} & {1000*g.slip.max():.1f} \\\\")
        rows.append(r"\midrule")
    body = "\n".join(rows[:-1])
    return r"""\begin{table}[!htbp]\centering\footnotesize
\caption{\textbf{Per-seed test results.} $\bar a_T$, $\bar a_F$: effective mean duration and grip actions after any shield
projection ($\bar a_F>0$ means a grip above the model's for the chosen lift time). Failures out of 40 lifts at each $r$.}\label{tab:perseed}
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lcccrrccccr@{}}\toprule
 & seed & $T$ (s) & $F/mg$ & $\bar a_T$ & $\bar a_F$ & \multicolumn{4}{c}{failures at $r=$} & max slip\\
 & & & & & & 1.0 & 1.1 & 1.2 & 1.3 & (mm)\\\midrule
""" + body + r"""
\bottomrule\end{tabular}\end{table}"""


def per_r():
    S = json.load(open("results/summary_test.json")); rows = []
    for r in (1.0, 1.1, 1.2, 1.3):
        rows.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{$r={r}$}}}}\\")
        for meth in ("B0", "B0S", "B2", "B4g", "B4t", "B5"):
            v = S[f"{meth}@r{r}"]; lo, hi = wilson(v["fail"], v["n"])
            dT = "--" if meth == "B0" else f"{v['dT_mean']:+.3f}"
            ciT = "" if (meth == "B0" or not v.get("ci_T")) else f" [{v['ci_T'][0]:+.3f}, {v['ci_T'][1]:+.3f}]"
            dF = "--" if meth == "B0" else f"{v['dF_rel_mean']:+.3f}"
            ciF = "" if (meth == "B0" or not v.get("ci_F_rel")) else f" [{v['ci_F_rel'][0]:+.3f}, {v['ci_F_rel'][1]:+.3f}]"
            pm = "--" if meth == "B0" else (f"{v.get('p_mcnemar_holm', v['p_mcnemar']):.1g}" if v['objfail_more_than_B0'] + v['objfail_less_than_B0'] else "1")
            rows.append(f"{meth} & {v['fail']}/{v['n']} & [{lo:.1f}, {hi:.1f}] & {dT}{{\\scriptsize{ciT}}} & {dF}{{\\scriptsize{ciF}}} & {v['objfail_more_than_B0'] if meth!='B0' else '--'} & {pm} & {v['slip_max_mm']:.2f}\\\\")
    return r"""\begin{table*}[!htbp]\centering\footnotesize
\caption{\textbf{Test set, every method at every $r$} ($r_{\max}=1.2$). Failure counts with Wilson 95\% intervals (\%);
$\Delta T$ and $\Delta F/mg$ against B0 with hierarchical bootstrap 95\% intervals (seeds, then objects); ``obj.\ worse'' counts
objects that failed at least once under the method but never under B0, with the Holm-corrected exact McNemar $p$ (B0S and B4t are
outside the pre-registered Holm families and show the raw $p$). All lift-time and grip differences have Holm-corrected Wilcoxon
$p<10^{-10}$.}\label{tab:perr}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrllrrr@{}}\toprule
Method & failures & Wilson CI (\%) & $\Delta T$ (s) & $\Delta F/mg$ & obj.\ worse & $p$ & max slip (mm)\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table*}"""


def confirm():
    S = json.load(open("results/summary_confirm.json")); rows = []
    for meth in ("B0", "B0S", "B2", "B4g", "B5"):
        v = S[meth]; lo, hi = wilson(v["fail"], v["n"])
        extra = "-- & -- & --" if meth == "B0" else f"{v['dT']:+.3f} & {v['dF_rel']:+.3f} & {v['objfail_more']} ($p$={v['p_mcnemar']:.1g})"
        rows.append(f"{meth} & {v['T_mean']:.3f} & {v['Frel_mean']:.3f} & {v['fail']}/{v['n']} [{lo:.1f}, {hi:.1f}] & {extra} & {v['slip_max_mm']:.2f}\\\\")
    band = S["B2_fail_by_r"]
    return (r"\begin{table*}[!htbp]\centering\footnotesize" + "\n"
            + r"\caption{\textbf{Confirmation set} (40 fresh objects, $r\sim U[1.0,1.3]$ per object, run once after the test results were committed). Failures with Wilson 95\% intervals (\%). B2's failures by $r$ band: " + ", ".join(f"{k}: {v}" for k, v in band.items()) + r".}\label{tab:confirm}" + "\n"
            + r"\setlength{\tabcolsep}{5pt}\begin{tabular}{@{}lcclllll@{}}\toprule" + "\n"
            + r" & $T$ (s) & $F/mg$ & failures [Wilson \%] & $\Delta T$ (s) & $\Delta F/mg$ & obj.\ worse & max slip (mm)\\\midrule" + "\n"
            + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}\end{table*}")


def rmax():
    S = json.load(open("results/summary_test.json"))["Q4"]; rows = []
    for meth in ("B0S", "B5"):
        for rm in (1.0, 1.2, 1.4):
            v = S[f"{meth}_rmax{rm}"]
            g = d[(d.method == meth) & (d.r_max == rm)]
            fr = " & ".join(str(int(g[g.r == r].fail.sum())) for r in (1.0, 1.1, 1.2, 1.3))
            rows.append(f"{meth} & {rm} & {v['T_mean']:.3f} & {v['Frel_mean']:.3f} & {fr} & {1000*g.slip.max():.2f}\\\\")
    cert = []
    for k, v in S.items():
        if k.startswith("rmax"):
            cert.append(f"{k.replace('rmax','').replace('_',' & ').replace('r<=rmax','$r\\le r_{{\\max}}$').replace('r>rmax','$r>r_{{\\max}}$')} & {v['n']} & {100*v['frac_exceed']:.1f} & {v['max_excess_mm']:.2f} & {v['n_slip_over_tol']}\\\\")
    return r"""\begin{table}[!htbp]\centering\footnotesize
\caption{\textbf{Mismatched $r_{\max}$.} Top: B0S and B5 (trained with $r_{\max}=1.2$) evaluated with the shield set to
$r_{\max}\in\{1.0,1.2,1.4\}$; failures out of 40 (B0S) or 200 (B5) lifts per $r$. Bottom: the certificate over all
shielded methods (B0S, B4g, B4t, B5), split by whether the true $r$ is within the assumed bound; ``exceed'' is the share of
lifts whose slip exceeds the bound (dominated by elastic creep, deviation D1).}\label{tab:rmax}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccccr@{}}\toprule
 & $r_{\max}$ & $T$ (s) & $F/mg$ & \multicolumn{4}{c}{failures at $r=$} & max slip\\
 & & & & 1.0 & 1.1 & 1.2 & 1.3 & (mm)\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}

\vspace{6pt}
\begin{tabular}{@{}llrrrr@{}}\toprule
$r_{\max}$ & true $r$ & lifts & exceed (\%) & max excess (mm) & $>5$\,mm\\\midrule
""" + "\n".join(sorted(cert)) + r"""
\bottomrule\end{tabular}\end{table}"""



def allow():
    from lift_env import shield_grip
    rms = (1.0, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3); lines = []
    for T in (0.37, 0.5, 1.0):
        for sl in (3.5, 10, 30):
            vals = " & ".join(f"{max(0.0, 100*(1-shield_grip(0.1, T, 1.0, rm, sl/1000)/model_grip(1.0, T))):.2f}" for rm in rms)
            lines.append(f"{T:.2f} & {sl:g} & {vals}\\\\")
    return (r"""\begin{tabular}{@{}cc""" + "r" * len(rms) + r"""@{}}\toprule
$T$ (s) & allowance (mm) & """ + " & ".join(f"$r_{{\\max}}={rm}$" for rm in rms) + r"""\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}""")

if __name__ == "__main__":
    for f in sys.argv[1:]:
        open(f"paper/tab_{f}.tex", "w").write(globals()[f]())
    print("ok")
