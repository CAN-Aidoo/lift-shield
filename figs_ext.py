"""Figures for the long-form paper (Addendum A results and new views of the main results)."""
import json, sys, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import lift_env as L
from lift_env import sample_object, G, DT
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 7.5, "legend.fontsize": 6.5})
C = {"B0": "#6b6b6b", "B0S": "#a0a0a0", "B2": "#d1495b", "B4g": "#2e86ab", "B4t": "#8fb8de", "B5": "#1b4965"}
rng = np.random.default_rng(11); OBJS = [sample_object(rng) for _ in range(40)]
TR = json.load(open("results/extended/traces.json"))


def load(i, meth):
    tr = np.load(f"results/extended/trace_obj{i}_{meth}.npy"); res = TR["episodes"][f"{i}_{meth}"]
    n = len(tr); nl = int(round(res["T"] / DT)); s = n - nl - 300          # lift starts after probe, ramp, settle
    w0 = s - 100; seg = tr[w0:]; t = (np.arange(len(seg)) - 100) * DT
    pv = seg[:, 2]; acc = np.gradient(pv, DT)
    rel = seg[:, 2] - seg[:, 3]                                             # palm minus rod velocity (rod slides down)
    slip = seg[:, 6] - seg[100, 6]
    return t, acc, rel, seg[:, 4], seg[:, 5], slip, res


def anatomy(i=None):
    i = i if i is not None else TR["objects"][0]; o = OBJS[i]
    fig, ax = plt.subplots(3, 2, figsize=(7.1, 4.7), sharex="col")
    for c, meth in enumerate(("B2", "B5")):
        t, acc, rel, sl, F, slip, res = load(i, meth)
        Fs = res["F_stuck"]; Fc = res["F"]
        ax[0, c].plot(t, G + acc, color="k", lw=1, label="demand $g+a(t)$")
        ax[0, c].axhline(2 * o["mu"] * Fc / o["m"], color="#2e86ab", lw=1, ls="-", label=r"static capacity $2\mu_sF/m$")
        ax[0, c].axhline(2 * o["mu"] / 1.2 * Fc / o["m"], color="#d1495b", lw=1, ls="--", label=r"kinetic capacity $2\mu_kF/m$")
        ax[0, c].axhline(G * Fc / Fs, color="#1b4965", lw=0.8, ls=":", label=r"certified $gF/F_{\rm stuck}$")
        ax[0, c].set_ylabel("m/s$^2$"); ax[0, c].set_title(f"{meth}: grip {res['F_rel']:.2f} mg, lift {res['T']:.2f} s")
        ax[1, c].plot(t, 1000 * rel, color="#444", lw=0.9)
        ax[1, c].fill_between(t, 0, 1, where=sl > 0, color="#d1495b", alpha=0.18, transform=ax[1, c].get_xaxis_transform(), lw=0)
        ax[1, c].set_ylabel("rel. speed (mm/s)")
        ax[2, c].plot(t, 1000 * slip, color=C[meth], lw=1.2)
        ax[2, c].axhline(5, color="0.5", lw=0.6, ls="--"); ax[2, c].text(t[-1], 5.6, "tolerance 5 mm", ha="right", fontsize=6, color="0.4")
        ax[2, c].set_ylabel("slip (mm)"); ax[2, c].set_xlabel("time from lift start (s)")
        for a in ax[:, c]: a.axvspan(0, res["T"], color="0.94", lw=0, zorder=-5)
    h, l = ax[0, 0].get_legend_handles_labels(); h.append(Patch(color="#d1495b", alpha=0.25)); l.append("kinetic friction active")
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, -0.005))
    ax[2, 1].set_ylim(ax[2, 0].get_ylim())
    fig.tight_layout(pad=0.3, rect=(0, 0.05, 1, 1)); fig.savefig("paper/fig_anatomy.pdf"); plt.close(fig)


def gallery():
    objs = TR["objects"]; n = len(objs); cols = 7; rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(7.1, 1.45 * rows), sharex=True, sharey=True)
    for k, i in enumerate(objs):
        a = ax.flat[k]
        for meth in ("B0", "B2", "B5"):
            t, *_, slip, res = load(i, meth)
            a.plot(t, 1000 * slip, color=C[meth], lw=1)
        a.axhline(5, color="0.6", lw=0.5, ls="--"); a.set_title(f"object {i}\n{OBJS[i]['m']*1000:.0f} g, $\\mu_s$ = {OBJS[i]['mu']:.2f}", fontsize=5.8); a.set_xlim(-0.1, 0.7)
        a.set_ylim(-2, 60)
    for a in ax.flat[n:]: a.axis("off")
    ax.flat[0].legend(["B0", "B2", "B5"], frameon=False, fontsize=5.5, loc="upper left")
    fig.supxlabel("time from lift start (s)", fontsize=7); fig.supylabel("slip (mm)", fontsize=7)
    fig.tight_layout(pad=0.3); fig.savefig("paper/fig_gallery.pdf"); plt.close(fig)


def training():
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 1.8), sharey=True)
    for c, (mode, lab) in enumerate((("residual", "B2: trained without shield"), ("residualS", "B5: trained with shield"))):
        for s in range(5):
            lg = json.load(open(f"weights/log_{mode}_s{s}.json"))
            ax[c].plot([x["episodes"] for x in lg], [x["f_center"] for x in lg], color=C["B2" if c == 0 else "B5"], alpha=0.35 + 0.13 * s, lw=1, label=f"seed {s}")
        ax[c].set_title(lab); ax[c].set_xlabel("training episodes"); ax[c].axhline(-1.1, color="0.5", lw=0.6, ls=":")
    ax[0].set_ylabel("reward of current policy\n(6 training objects)"); ax[0].text(1560, -1.25, "model controller", fontsize=6, color="0.4", ha="right")
    ax[1].legend(frameon=False, ncol=2, fontsize=5.8, loc="lower right")
    fig.tight_layout(pad=0.3); fig.savefig("paper/fig_training.pdf"); plt.close(fig)




def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def teaser():
    from lift_env import shield_grip, model_grip
    d = pd.read_csv("results/test.csv"); d["fail"] = d.ok.astype(str) != "True"; m = d[d.r_max == 1.2]
    fig = plt.figure(figsize=(7.16, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.55, 1, 0.85, 0.95], wspace=0.42, left=0.005, right=0.985, bottom=0.2, top=0.86)
    sub = gs[0].subgridspec(1, 3, wspace=0.03); ims = ["hero_residual_start", "hero_residual_end", "hero_residualS_end"]
    labs = ["before lift", "B2 (no shield)\nslip 21 mm", "B5 (shield)\nslip 0.1 mm"]
    for k in range(3):
        a = fig.add_subplot(sub[k]); im = plt.imread(f"paper/{ims[k]}.png"); h, w = im.shape[:2]
        a.imshow(im[int(0.08 * h):int(0.72 * h), int(0.3 * w):int(0.72 * w)]); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
        a.set_xlabel(labs[k], fontsize=6.3, color=["0.2", C["B2"], C["B5"]][k])
        if k == 0: a.set_title("(a) same object, r = 1.2", fontsize=7, loc="left")
    a = fig.add_subplot(gs[1]); rs = [1.0, 1.1, 1.2, 1.3]
    for meth, lab in (("B0", "model B0"), ("B2", "residual B2"), ("B5", "shielded B5")):
        k = [m[(m.method == meth) & (m.r == r)].fail.sum() for r in rs]; n = [len(m[(m.method == meth) & (m.r == r)]) for r in rs]
        lo, hi = zip(*[wilson(x, y) for x, y in zip(k, n)]); p = np.array(k) / np.array(n)
        a.errorbar(rs, 100 * p, yerr=[100 * (p - lo), 100 * (np.array(hi) - p)], marker="o", ms=2.5, lw=1.1, capsize=1.5, color=C[meth], label=lab)
    a.axvspan(1.2, 1.32, color="0.93", lw=0); a.text(1.26, 25, "beyond\n$r_{\\max}$", ha="center", fontsize=5.8, color="0.4")
    a.set_xlim(0.97, 1.32); a.set_ylim(-1, 30); a.set_xticks(rs); a.set_xlabel("static/kinetic ratio $r$"); a.set_ylabel("lifts slipping > 5 mm (%)")
    a.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1.2); a.set_title("(b) safety", fontsize=7, loc="left")
    a = fig.add_subplot(gs[2]); meths = ["B0", "B2", "B4t", "B4g", "B5"]
    T = [m[m.method == x]["T"].mean() for x in meths]; F = [m[m.method == x].F_rel.mean() for x in meths]
    off = {"B0": (-12, 5), "B2": (-13, -4), "B4t": (4, -8), "B4g": (4, 1), "B5": (-13, 2)}
    for i, x in enumerate(meths):
        a.scatter(T[i], F[i], s=16, color=C[x], zorder=3, ec="white", lw=0.4); a.annotate(x, (T[i], F[i]), xytext=off[x], textcoords="offset points", fontsize=6)
    a.set_xlabel("lift time (s)"); a.set_ylabel("grip / weight"); a.set_xlim(0.2, 1.1); a.set_ylim(0.95, 1.42); a.set_title("(c) speed vs grip", fontsize=7, loc="left")
    a = fig.add_subplot(gs[3]); rr = np.linspace(1.0, 1.3, 31)
    for s, ls in ((0.0035, "-"), (0.01, "--"), (0.03, ":")):
        a.plot(rr, [100 * (1 - shield_grip(0.1, 0.37, 1.0, r, s) / model_grip(1.0, 0.37)) for r in rr], ls, color=C["B5"], lw=1, label=f"{s*1000:g} mm")
    a.set_xlabel("assumed $r_{\\max}$"); a.set_ylabel("grip saved (%)"); a.legend(frameon=False, fontsize=5.6, title="slip allowed", title_fontsize=5.6)
    a.set_title("(d) value of allowing slip", fontsize=7, loc="left")
    fig.savefig("paper/fig_teaser.pdf"); plt.close(fig)




def bound_explainer():
    """Analytic illustration of Proposition 1 at T = 0.37 s for a unit-breakaway object."""
    from lift_env import minjerk, model_grip, lift_bound
    T = 0.37; Fs = 1.0; _, a = minjerk(T); t = np.arange(len(a)) * DT
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.9))
    for k, (f, title) in enumerate(((1.0, "no-onset grip: bound 0"), (0.97, "3% less grip, $r_{\\max}=1$"), (0.97, "3% less grip, $r_{\\max}=1.2$"))):
        rm = 1.0 if k == 1 else 1.2
        F = model_grip(Fs, T) * f; cap = G * F / Fs; kin = G * F / (rm * Fs)
        A = ax[k]; A.plot(t, G + a, color="k", lw=1, label="demand $g+a(t)$")
        A.axhline(cap, color="#2e86ab", lw=1, label="certified static $gF/F_{\\rm stuck}$")
        A.axhline(kin, color="#d1495b", lw=1, ls="--", label="certified kinetic $gF/(r_{\\max}F_{\\rm stuck})$")
        over = np.nonzero(G + a > cap)[0]
        if len(over):
            t0 = over[0]; v = 0; vs = []
            for x in (G + a[t0:]) - kin: v = max(0, v + x * DT); vs.append(v)
            vs = np.array(vs); act = vs > 0
            A.fill_between(t[t0:], kin, G + a[t0:], where=act, color="#d1495b", alpha=0.18, lw=0)
            A.axvline(t[t0], color="0.5", lw=0.6, ls=":")
        b = lift_bound(F, T, Fs, rm) * 1000
        A.set_title(f"({'abc'[k]}) {title}", fontsize=7, loc="left"); A.text(0.98, 0.9, f"bound {b:.1f} mm", transform=A.transAxes, ha="right", fontsize=6.5,
                                                                          color=C["B2"] if b > 3.5 else C["B5"])
        A.set_xlabel("time in lift (s)"); A.set_ylim(5, 15.5)
    ax[0].set_ylabel("m/s$^2$ (per unit mass)")
    h, l = ax[0].get_legend_handles_labels(); h.append(Patch(color="#d1495b", alpha=0.25)); l.append("relative acceleration while slipping")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1)); fig.savefig("paper/fig_bound.pdf"); plt.close(fig)


def overview():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.1, 2.0)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 26)
    def box(x, y, w, h, title, body, fc="#eef3f8", ec="#1b4965"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=6.6, fontweight="bold")
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=5.1, linespacing=1.4)
    box(0.5, 1, 19.5, 24, "Reference configuration", "40 test objects\n$r\\in\\{1.0,1.1,1.2,1.3\\}$\n$r_{\\max}=1.2$, tolerance 5 mm\nmargin 1.5 mm, probe step 0.9\nES: 20 iterations, 5 seeds")
    box(23, 1, 21.5, 24, "Six methods", "B0 model, B0S model + shield\nB2 residual (no shield)\nB4g / B4t: B2 + test-time shield\n(grip or time projection)\nB5 residual trained with shield")
    box(47.8, 1, 34, 24, "19 single changes (Addendum A)", "E1-E9 environment and sensing: heavier, slippery,\nnoisy sensor, wrong weight, probe step, lift height,\nsoft pads, kinetic switch speed\nS1-S6 shield: $r_{\\max}$, tolerance, margin\nT1-T4 training: no friction drop, wider\naction, grip-heavy reward, 2x budget", fc="#d7e8f3")
    box(85.3, 1, 14.4, 24, "Measured", "failed lifts\nlift time\ngrip\nepisode time\npaired bootstrap\nvs reference")
    for x0, x1 in ((20.3, 22.7), (44.8, 47.5), (82.1, 85.0)):
        ax.annotate("", (x1, 13), (x0, 13), arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(45.8, 15.5, "$\\times$", fontsize=10, ha="center")
    fig.savefig("paper/fig_overview.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def validation():
    v = json.load(open("results/validation.json")); rows = pd.DataFrame(v["V3_rows"])
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.0))
    Ts = sorted(v["V2_lift_breakaway_ratio"], key=float); mu = [v["V2_lift_breakaway_ratio"][t]["mean"] for t in Ts]
    lo = [v["V2_lift_breakaway_ratio"][t]["min"] for t in Ts]; hi = [v["V2_lift_breakaway_ratio"][t]["max"] for t in Ts]
    x = [float(t) for t in Ts]
    ax[0].errorbar(x, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], marker="o", ms=3, color=C["B5"], capsize=2, lw=1)
    ax[0].axhline(1, color="0.5", lw=0.6, ls="--"); ax[0].set_xlabel("lift time $T$ (s)"); ax[0].set_ylabel("breakaway grip / quasi-static")
    ax[0].set_title("(a) V2: model is conservative", loc="left")
    fin = rows[(rows.bound > 0) & ~rows.dropped]; zero = rows[rows.bound == 0]
    ax[1].scatter(1000 * fin.bound, 1000 * fin.slip, s=6, color=C["B4g"], label=f"bound > 0, held (n={len(fin)})")
    ax[1].scatter(np.full(len(zero), 0.012), (1000 * zero.slip).clip(lower=0.012), s=6, color=C["B5"], marker="x", lw=0.6, label=f"bound = 0 (n={len(zero)})")
    lim = [0.01, 300]; ax[1].plot(lim, lim, color="0.4", lw=0.7, ls="--"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].set_xlabel("bound (mm)"); ax[1].set_ylabel("measured slip (mm)")
    ax[1].legend(frameon=False, fontsize=5.6, loc="upper left"); ax[1].set_title("(b) V3: bound vs slip", loc="left")
    m = sorted(v["V4_creep_vs_margin"], key=float)
    ax[2].bar(range(len(m)), [v["V4_creep_vs_margin"][k]["median_mm"] for k in m], color=C["B4t"], width=0.6, label="median")
    ax[2].scatter(range(len(m)), [v["V4_creep_vs_margin"][k]["max_mm"] for k in m], color=C["B5"], s=10, zorder=3, label="max")
    ax[2].set_xticks(range(len(m))); ax[2].set_xticklabels(m); ax[2].set_xlabel("grip margin over need"); ax[2].set_ylabel("creep (mm)")
    ax[2].legend(frameon=False, fontsize=5.8); ax[2].set_title("(c) V4: elastic creep", loc="left")
    fig.tight_layout(pad=0.3); fig.savefig("paper/fig_validation.pdf"); plt.close(fig)


def e7case():
    """A certified lift that fails: friction drops during elastic (pre-sliding) motion (row E7, lift height 20 cm)."""
    import controlled as Cc
    meta = json.load(open("results/extended/e7_case.json")); tr = np.load("results/extended/trace_E7_B5.npy")
    Cc.setup("E7"); o = Cc.objects("E7")[meta["obj"]]
    import pandas as pd
    d = pd.read_csv("results/controlled/E7.csv"); res = d[(d.method == "B5") & (d.obj == meta["obj"]) & (d.r == 1.2) & (d.seed == meta["seed"])].iloc[0]
    n = len(tr); nl = int(round(res["T"] / DT)); s = n - nl - 300; seg = tr[s - 100:]; t = (np.arange(len(seg)) - 100) * DT
    acc = np.gradient(seg[:, 2], DT); rel = seg[:, 2] - seg[:, 3]; sl = seg[:, 4]; slip = seg[:, 6] - seg[100, 6]
    Cc.setup("REF") if "REF" in Cc.ROWS else None
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.85))
    ax[0].plot(t, G + acc, color="k", lw=1, label="demand")
    ax[0].axhline(2 * o["mu"] * res["F"] / o["m"], color="#2e86ab", lw=1, label="true static capacity")
    ax[0].axhline(G * res["F"] / res["F_stuck"], color="#1b4965", lw=0.8, ls=":", label="certified")
    ax[0].axhline(2 * o["mu"] / 1.2 * res["F"] / o["m"], color="#d1495b", lw=1, ls="--", label="kinetic capacity")
    ax[0].set_ylabel("m/s$^2$"); ax[0].legend(frameon=False, fontsize=5.5, loc="lower left"); ax[0].set_title("(a) demand never reaches static capacity", loc="left")
    ax[1].plot(t, 1000 * rel, color="#444", lw=0.9); ax[1].axhline(20, color="#d1495b", lw=0.7, ls="--"); ax[1].text(t[-1], 22, "switch threshold 20 mm/s", ha="right", fontsize=5.8, color="#d1495b")
    ax[1].fill_between(t, 0, 1, where=sl > 0, color="#d1495b", alpha=0.18, transform=ax[1].get_xaxis_transform(), lw=0)
    ax[1].set_ylabel("rel. speed (mm/s)"); ax[1].set_title("(b) pre-sliding creep triggers the drop", loc="left")
    ax[2].plot(t, 1000 * slip, color=C["B5"], lw=1.2); ax[2].axhline(5, color="0.5", lw=0.6, ls="--"); ax[2].set_ylabel("slip (mm)")
    ax[2].set_title(f"(c) certified lift slips {1000*res['slip']:.1f} mm", loc="left")
    for a in ax: a.set_xlabel("time from lift start (s)"); a.axvspan(0, res["T"], color="0.94", lw=0, zorder=-5)
    fig.tight_layout(pad=0.3); fig.savefig("paper/fig_e7case.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in sys.argv[1:]: globals()[f]()
    print("ok")
