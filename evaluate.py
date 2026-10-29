"""Test-set (and confirmation-set) evaluation exactly as PROTOCOL.md. Usage: python evaluate.py test|confirm"""
import sys, csv, numpy as np
from multiprocessing import Pool
from lift_env import run_episode, sample_object, LinearPolicy

SEEDS = range(5)
METHODS = {  # id: (mode, weights, shield axis)
    "B0": ("model", None, "grip"), "B0S": ("modelS", None, "grip"), "B2": ("residual", "residual", "grip"),
    "B4g": ("residualS", "residual", "grip"), "B4t": ("residualS", "residual", "time"), "B5": ("residualS", "residualS", "grip")}


def job(a):
    meth, seed, i, obj, r, rmax, eseed = a
    mode, wname, axis = METHODS[meth]
    pol = LinearPolicy(np.load(f"weights/w_{wname}_s{seed}.npy")) if wname else None
    res = run_episode(dict(obj, ratio=r), np.random.default_rng(eseed), mode, pol, r_max=rmax, shield_axis=axis)
    return dict(method=meth, seed=seed, obj=i, m=obj["m"], mu=obj["mu"], r=r, r_max=rmax, **res)


def main():
    which = sys.argv[1]
    jobs = []
    if which == "test":
        rng = np.random.default_rng(11); objs = [sample_object(rng) for _ in range(40)]
        for i, o in enumerate(objs):
            for r in (1.0, 1.1, 1.2, 1.3):
                for meth in METHODS:
                    for s in (SEEDS if METHODS[meth][1] else [0]):
                        jobs.append((meth, s, i, o, r, 1.2, 5000 + i))
                for rmax in (1.0, 1.4):
                    jobs.append(("B0S", 0, i, o, r, rmax, 5000 + i))
                    for s in SEEDS: jobs.append(("B5", s, i, o, r, rmax, 5000 + i))
    else:
        rng = np.random.default_rng(47); objs = [dict(sample_object(rng), r=float(rng.uniform(1.0, 1.3))) for _ in range(40)]
        for i, o in enumerate(objs):
            for meth in ("B0", "B0S", "B2", "B4g", "B5"):
                for s in (SEEDS if METHODS[meth][1] else [0]):
                    jobs.append((meth, s, i, dict(m=o["m"], mu=o["mu"]), o["r"], 1.2, 7000 + i))
    with Pool(2) as p:
        rows = p.map(job, jobs, chunksize=8)
    with open(f"results/{which}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(which, len(rows), "episodes")


if __name__ == "__main__":
    main()
