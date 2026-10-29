"""All P7 figures from committed results."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from lift_env import minjerk, lift_bound, shield_grip, model_grip, G
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C = {"B0": "#6b6b6b", "B2": "#d1495b", "B4g": "#2e86ab", "B4t": "#8fb8de", "B5": "#1b4965"}
d = pd.read_csv("results/test.csv"); d["fail"] = d.ok.astype(str) != "True"; m = d[d.r_max == 1.2]

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

# teaser: failure rate vs r (left), lift time and grip (right)
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.7), gridspec_kw=dict(width_ratios=[1.2, 1]))
rs = [1.0, 1.1, 1.2, 1.3]
for meth, lab in (("B0", "model (B0)"), ("B2", "residual (B2)"), ("B5", "shielded residual (B5)")):
    k = [m[(m.method == meth) & (m.r == r)].fail.sum() for r in rs]; n = [len(m[(m.method == meth) & (m.r == r)]) for r in rs]
    lo, hi = zip(*[wilson(a, b) for a, b in zip(k, n)]); p = np.array(k) / np.array(n)
    ax[0].errorbar(rs, 100 * p, yerr=[100 * (p - lo), 100 * (np.array(hi) - p)], marker="o", ms=3, lw=1.2, capsize=2, color=C[meth], label=lab)
ax[0].axvspan(1.2, 1.32, color="0.92", lw=0); ax[0].text(1.26, 3, "$r>r_{\\max}$", ha="center", fontsize=7, color="0.4")
ax[0].set_xlabel("static/kinetic ratio $r$"); ax[0].set_ylabel("failed lifts (%)"); ax[0].set_xticks(rs); ax[0].set_xlim(0.97, 1.32)
ax[0].legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.2); ax[0].set_ylim(-1, 30)
meths = ["B0", "B2", "B4g", "B4t", "B5"]
T = [m[m.method == x]["T"].mean() for x in meths]; F = [m[m.method == x].F_rel.mean() for x in meths]
for i, x in enumerate(meths):
    off = {"B0": (-12, 4), "B2": (-14, -3), "B4t": (5, -3), "B4g": (5, 2), "B5": (-14, 2)}[x]
    ax[1].scatter(T[i], F[i], s=18, color=C[x], zorder=3, ec="white", lw=0.4); ax[1].annotate(x, (T[i], F[i]), xytext=off, textcoords="offset points", fontsize=6.5)
ax[1].set_xlabel("lift time $T$ (s)"); ax[1].set_ylabel("grip / weight $F/mg$"); ax[1].set_xlim(0.3, 1.1)
fig.tight_layout(pad=0.3); fig.savefig("paper/fig_teaser.pdf")

# method diagram
fig, ax = plt.subplots(figsize=(7.1, 1.25)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 20)
boxes = [(0, "Weight\nestimate\n$\\pm$10%"), (17, "Reactive probe\ngrip $\\times$0.9\nuntil it slips\n$\\Rightarrow F_{\\rm stuck}$"),
         (34.4, "Model lift\n$T_0$ = 1 s\n$F_{\\rm stuck}(g+a_{pk})/g$"), (51.6, "Linear residual\n$T\\,e^{a_T}$\n$F\\,e^{0.25a_F}$"),
         (68.8, "Shield (Prop. 1)\nbound $\\leq$ 3.5 mm?\nelse raise grip\nor slow down"), (86, "Min-jerk lift\n10 cm\nhold 0.3 s")]
for x, t in boxes:
    ax.add_patch(FancyBboxPatch((x, 2), 14.2, 16, boxstyle="round,pad=0.3", fc="#eef3f8" if "Shield" not in t else "#d7e8f3", ec="#1b4965", lw=0.8))
    ax.text(x + 7.1, 10, t, ha="center", va="center", fontsize=5.9)
for x0, x1 in ((14.6, 16.6), (31.6, 34.0), (49.0, 51.2), (66.2, 68.4), (83.4, 85.6)):
    ax.annotate("", (x1, 10), (x0, 10), arrowprops=dict(arrowstyle="->", lw=0.8))
fig.savefig("paper/fig_method.pdf", bbox_inches="tight", pad_inches=0.02)

# the cliff: grip saved by a slip allowance vs r_max, plus a bound curve
Fs = 1.0; fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6))
rr = np.linspace(1.0, 1.3, 31)
for s, ls in ((0.0035, "-"), (0.01, "--"), (0.03, ":")):
    ax[0].plot(rr, [100 * (1 - shield_grip(0.1, 0.37, Fs, r, s) / model_grip(Fs, 0.37)) for r in rr], ls, color="#1b4965", lw=1.1, label=f"{s*1000:g} mm")
ax[0].set_xlabel("assumed $r_{\\max}$"); ax[0].set_ylabel("grip saved (%)"); ax[0].legend(frameon=False, fontsize=6, title="allowed slip", title_fontsize=6)
fr = np.linspace(0.9, 1.02, 60); Fm = model_grip(Fs, 0.37)
for r, c in ((1.0, "#8fb8de"), (1.1, "#2e86ab"), (1.2, "#1b4965")):
    ax[1].plot(100 * fr, [1000 * lift_bound(f * Fm, 0.37, Fs, r) for f in fr], color=c, lw=1.1, label=f"$r_{{\\max}}={r}$")
ax[1].axhline(3.5, color="0.5", lw=0.6, ls="--"); ax[1].set_ylim(0, 40)
ax[1].set_xlabel("grip (% of no-onset grip)"); ax[1].set_ylabel("slip bound (mm)"); ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout(pad=0.3); fig.savefig("paper/fig_cliff.pdf")
print("figs ok")
