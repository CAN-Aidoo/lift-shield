"""Addendum A2-A4. Usage: python extended.py safetymap|tightness|traces"""
import sys, os, csv, json, numpy as np, mujoco
from multiprocessing import Pool
import lift_env as L
from lift_env import run_episode, sample_object, LinearPolicy, LiftRig, probe, model_grip, lift_bound, G

rng0 = np.random.default_rng(11); OBJS = [sample_object(rng0) for _ in range(40)]
AT = np.linspace(-1, 0, 7); AF = np.linspace(-1, 1, 9)


class Fixed:
    def __init__(self, a): self.a = np.array(a, float)
    def __call__(self, o): return self.a


def sm_job(a):
    aT, aF, i, r, shield = a
    res = run_episode(dict(OBJS[i], ratio=r), np.random.default_rng(5000 + i), "residualS" if shield else "residual", Fixed([aF, aT]))
    return dict(aT=aT, aF=aF, obj=i, r=r, shield=shield, **res)


def tight_job(a):
    f, T, r, i = a
    o = OBJS[i]; rng = np.random.default_rng(5000 + i)
    rig = LiftRig(o["m"], o["mu"], ratio=r)
    Fs, Fm, n = probe(rig, o["m"] * rng.uniform(0.9, 1.1), rng)
    F = model_grip(Fs, T) * f
    rig.set_grip(F); s0 = rig.slip(); rig.lift(F, T); slip = rig.slip() - s0
    return dict(f=f, T=T, r=r, obj=i, F_stuck=Fs, F=F, F_rel=F / (o["m"] * G), slip=slip, held=rig.held(),
                bound=lift_bound(F, T, Fs, r), n_breakaway=rig.n_breakaway)


TRACE = None
_mj1 = LiftRig.mj1
def traced_mj1(self):
    _mj1(self)
    if TRACE is not None:
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._rod, self._vel, 0)
        TRACE.append((self.palm_z, float(self.data.xpos[self._rod][2]), self.palm_v, float(self._vel[5]),
                      int(self.sliding), float(self.data.ctrl[0]), self.offset() - getattr(self, "off0", self.offset())))
LiftRig.mj1 = traced_mj1


def trace_episode(i, r, meth, seed):
    global TRACE
    TRACE = []
    wname = {"B2": "residual", "B5": "residualS", "B0": None}[meth]
    mode = {"B2": "residual", "B5": "residualS", "B0": "model"}[meth]
    pol = LinearPolicy(np.load(f"weights/w_{wname}_s{seed}.npy")) if wname else None
    res = run_episode(dict(OBJS[i], ratio=r), np.random.default_rng(5000 + i), mode, pol)
    tr = np.array(TRACE); TRACE = None
    return res, tr


def write(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    what = sys.argv[1]; os.makedirs("results/extended", exist_ok=True); n = int(os.environ.get("NPROC", 2))
    if what == "safetymap":
        jobs = [(aT, aF, i, r, s) for aT in AT for aF in AF for i in range(40) for r in (1.0, 1.2) for s in (False, True)]
        with Pool(n) as p: out = p.map(sm_job, jobs, chunksize=16)
        write(out, "results/extended/safetymap.csv")
    elif what == "tightness":
        jobs = [(f, T, r, i) for f in (0.85, 0.88, 0.91, 0.94, 0.97, 1.0) for T in (0.37, 0.5, 1.0) for r in (1.0, 1.1, 1.2) for i in range(40)]
        with Pool(n) as p: out = p.map(tight_job, jobs, chunksize=16)
        write(out, "results/extended/tightness.csv")
    elif what == "traces":
        import pandas as pd
        d = pd.read_csv("results/test.csv")
        fails = d[(d.method == "B2") & (d.seed == 2) & (d.r == 1.2) & (d.ok.astype(str) != "True")].obj.tolist()
        meta = {}
        for i in fails:
            for meth, seed in (("B2", 2), ("B5", 2), ("B0", 0)):
                res, tr = trace_episode(i, 1.2, meth, seed)
                np.save(f"results/extended/trace_obj{i}_{meth}.npy", tr)
                meta[f"{i}_{meth}"] = {k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in res.items()}
        json.dump(dict(objects=fails, episodes=meta), open("results/extended/traces.json", "w"), indent=1)
    print(what, "done")
