"""Addendum A1: controlled one-change-at-a-time study. Usage: python controlled.py ROW [ROW ...]  (or ALL_ENV / ALL_TRAIN)
Each row changes one design choice relative to the reference (the test set) and reruns all six methods on the same
40 objects and episode seeds. Writes results/controlled/<ROW>.csv."""
import sys, os, csv, numpy as np
from multiprocessing import Pool
import lift_env as L
from lift_env import run_episode, sample_object, LinearPolicy

ROWS = {  # row: (description, module overrides, xml replacements, weight tag)
    "E1": ("heavier objects 0.2-0.5 kg", dict(M_RANGE=(0.2, 0.5)), {}, ""),
    "E2": ("slippery objects mu 0.2-0.4", dict(MU_RANGE=(0.2, 0.4)), {}, ""),
    "E3": ("slip-monitor noise 1.5 mm", dict(SIG=0.0015), {}, ""),
    "E4": ("weight error +-25%", dict(M_ERR=0.25), {}, ""),
    "E5": ("coarser probe step 0.8", dict(SEARCH=0.8), {}, ""),
    "E6": ("finer probe step 0.95", dict(SEARCH=0.95), {}, ""),
    "E7": ("lift height 20 cm", dict(H=0.20), {}, ""),
    "E8": ("softer pads solref 0.006", {}, {'solref="0.003 1"': 'solref="0.006 1"'}, ""),
    "E9": ("earlier kinetic switch 5 mm/s", dict(V_SLIP=0.005), {}, ""),
    "S1": ("shield r_max = 1.0", dict(RMAX=1.0), {}, ""),
    "S2": ("shield r_max = 1.1", dict(RMAX=1.1), {}, ""),
    "S3": ("shield r_max = 1.4", dict(RMAX=1.4), {}, ""),
    "S4": ("slip tolerance 2 mm", dict(S_TOL=0.002), {}, ""),
    "S5": ("slip tolerance 10 mm", dict(S_TOL=0.010), {}, ""),
    "S6": ("no sensing margin", dict(MARGIN=0.0), {}, ""),
    "R0L": ("reference, lag-free switch", dict(LAGFIX=True), {}, ""),
    "E7L": ("lift height 20 cm, lag-free switch", dict(H=0.20, LAGFIX=True), {}, ""),
    "E9L": ("switch 5 mm/s, lag-free switch", dict(V_SLIP=0.005, LAGFIX=True), {}, ""),
    "T1": ("trained on r = 1.0 only", {}, {}, "_T1"),
    "T2": ("wider time action (scale 1.5)", dict(ST=1.5), {}, "_T2"),
    "T3": ("grip weight lambda = 10", {}, {}, "_T3"),
    "T4": ("double training budget", {}, {}, "_T4"),
}
METHODS = {"B0": ("model", None, "grip"), "B0S": ("modelS", None, "grip"), "B2": ("residual", "residual", "grip"),
           "B4g": ("residualS", "residual", "grip"), "B4t": ("residualS", "residual", "time"), "B5": ("residualS", "residualS", "grip")}
BASE = {k: getattr(L, k) for k in ("M_RANGE", "MU_RANGE", "SIG", "M_ERR", "SEARCH", "H", "V_SLIP", "RMAX", "S_TOL", "MARGIN", "ST", "LAGFIX")}
XML0 = L.XML


def setup(row):
    _, ov, xr, _ = ROWS[row]
    for k, v in BASE.items(): setattr(L, k, v)
    for k, v in ov.items(): setattr(L, k, v)
    x = XML0
    for a, b in xr.items(): x = x.replace(a, b)
    L.XML = x


def job(a):
    row, meth, seed, i, obj, r = a
    setup(row)
    mode, wname, axis = METHODS[meth]
    pol = LinearPolicy(np.load(f"weights/w_{wname}_s{seed}{ROWS[row][3]}.npy")) if wname else None
    res = run_episode(dict(obj, ratio=r), np.random.default_rng(5000 + i), mode, pol, shield_axis=axis)
    return dict(row=row, method=meth, seed=seed, obj=i, m=obj["m"], mu=obj["mu"], r=r, **res)


def objects(row):
    setup(row); rng = np.random.default_rng(11)
    return [sample_object(rng) for _ in range(40)]


def main():
    rows = sys.argv[1:]
    if rows == ["ALL_ENV"]: rows = [r for r in ROWS if r[0] in "ES" and not r.endswith("L")]
    if rows == ["ALL_TRAIN"]: rows = [r for r in ROWS if r[0] == "T"]
    os.makedirs("results/controlled", exist_ok=True)
    for row in rows:
        objs = objects(row)
        jobs = [(row, m, s, i, o, r) for i, o in enumerate(objs) for r in (1.0, 1.1, 1.2, 1.3) for m in METHODS
                for s in (range(5) if METHODS[m][1] else [0])]
        with Pool(int(os.environ.get("NPROC", 2))) as p:
            out = p.map(job, jobs, chunksize=8)
        with open(f"results/controlled/{row}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print(row, ROWS[row][0], len(out), flush=True)


if __name__ == "__main__":
    main()
