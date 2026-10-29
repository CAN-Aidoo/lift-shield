"""Rig validation V1-V3 (Paper 7). python3 validate.py -> results/validation.json"""
import json, os, sys
import numpy as np
from lift_env import *

rng = np.random.default_rng(7)
out = {}

def first_slip_grip(make, run, lo, hi, thr):
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if run(make(), mid) > thr: lo = mid
        else: hi = mid
    return hi

# V1: static breakaway grip vs m g / (2 mu), r = 1
v1 = []
for i in range(20):
    o = sample_object(rng); th = o["m"] * G / (2 * o["mu"])
    def run(rig, F): s0 = rig.slip(); rig.set_grip(F); return rig.slip() - s0
    Fb = first_slip_grip(lambda: LiftRig(o["m"], o["mu"]), run, 0.5 * th, 1.5 * th, MOVE_THR)
    v1.append(Fb / th)
out["V1_ratio_sim_over_theory"] = dict(mean=float(np.mean(v1)), min=float(np.min(v1)), max=float(np.max(v1)), n=len(v1))

# V2: breakaway grip during a lift vs m (g + a_peak) / (2 mu), r = 1
v2 = {}
for T in (0.25, 0.35, 0.5, 1.0):
    rs = []
    for i in range(8):
        o = sample_object(rng); th = o["m"] * (G + peak_acc(T)) / (2 * o["mu"])
        def run(rig, F, T=T): rig.set_grip(F); s0 = rig.slip(); rig.lift(F, T); return rig.slip() - s0
        Fb = first_slip_grip(lambda: LiftRig(o["m"], o["mu"]), run, 0.5 * th, 1.5 * th, 0.001)
        rs.append(Fb / th)
    v2[str(T)] = dict(mean=float(np.mean(rs)), min=float(np.min(rs)), max=float(np.max(rs)), n=len(rs))
out["V2_lift_breakaway_ratio"] = v2

# V3: Proposition 3 bound vs measured lift slip, F_stuck from the probe search, r in [1, 1.2], r_max = 1.2
v3 = []
for i in range(120):
    o = sample_object(rng); r = rng.uniform(1.0, 1.2)
    rig = LiftRig(o["m"], o["mu"], ratio=r)
    F_stuck, F_mov, n = probe(rig, o["m"] * rng.uniform(0.9, 1.1), rng)
    probe_slip = rig.slip()
    T = float(rng.uniform(0.25, 1.0))
    need = F_stuck * (G + peak_acc(T)) / G
    F = float(np.clip(need * rng.uniform(0.6, 1.4), 0.1, FMAX))
    b = lift_bound(F, T, F_stuck, RMAX)
    rig.set_grip(F); s0 = rig.slip(); rig.lift(F, T); s = rig.slip() - s0
    v3.append(dict(m=o["m"], mu=o["mu"], r=r, T=T, F=F, F_stuck=F_stuck, bound=b, slip=s, dropped=not rig.held(),
                   probe_slip=probe_slip, probes=n))
B = np.array([d["bound"] for d in v3]); S = np.array([d["slip"] for d in v3])
fin = B < 0.07
out["V3"] = dict(n=len(v3), probe_slip_max_mm=float(1000 * max(d["probe_slip"] for d in v3)),
                 probes_median=float(np.median([d["probes"] for d in v3])), n_bound_finite=int(fin.sum()), n_violations=int(np.sum(S[fin] > B[fin] + 1e-4)),
                 max_excess_mm=float(1000 * np.max(S[fin] - B[fin])), n_lift_slip_over_tol_when_bound_le_tol=int(np.sum((B <= S_TOL) & (S > S_TOL))), n_zero_bound=int(np.sum(B == 0)),
                 n_slip_when_zero_bound=int(np.sum((B == 0) & (S > 0.0005))))
# V4: creep of a well-held lift (r = 1) against static grip margin F / [m (g + a_peak) / (2 mu)]
v4 = {}
for marg in (1.1, 1.25, 1.5, 2.0):
    cr = []
    for i in range(12):
        o = sample_object(rng); T = float(rng.uniform(0.25, 1.0))
        rig = LiftRig(o["m"], o["mu"]); F = min(FMAX, marg * o["m"] * (G + peak_acc(T)) / (2 * o["mu"]))
        rig.set_grip(F); s0 = rig.slip(); rig.lift(F, T); cr.append(1000 * (rig.slip() - s0))
    v4[str(marg)] = dict(median_mm=float(np.median(cr)), max_mm=float(np.max(cr)), n=len(cr))
out["V4_creep_vs_margin"] = v4
os.makedirs("results", exist_ok=True)
json.dump(dict(out, V3_rows=v3), open("results/validation.json", "w"), indent=1)
print(json.dumps(out, indent=1))
