"""Antithetic ES for the lift residual (settings as Paper 5). Usage: python train_es.py MODE SEED  (MODE residual|residualS)"""
import sys, os, json, time, pickle, numpy as np
from lift_env import run_episode, sample_object, score, LinearPolicy, OBS_DIM

POP_PAIRS = 6; N_OBJ = 6; SIGMA = 0.1; LR = 0.06; ITERS = 20


def evaluate(mode, w, objs, seeds):
    pol = LinearPolicy(w)
    return float(np.mean([score(run_episode(o, np.random.default_rng(s), mode, pol)) for o, s in zip(objs, seeds)]))


def main():
    mode, seed = sys.argv[1], int(sys.argv[2])
    os.makedirs("weights", exist_ok=True)
    rng = np.random.default_rng(seed)
    w = np.zeros(2 * OBS_DIM); m = np.zeros_like(w); v = np.zeros_like(w); log = []; t0 = time.time()
    for it in range(ITERS):
        objs = [dict(sample_object(rng), ratio=float(rng.uniform(1.0, 1.2))) for _ in range(N_OBJ)]
        seeds = [int(x) for x in rng.integers(0, 2**31, N_OBJ)]
        eps = rng.normal(size=(POP_PAIRS, w.size))
        fp = np.array([evaluate(mode, w + SIGMA * e, objs, seeds) for e in eps])
        fm = np.array([evaluate(mode, w - SIGMA * e, objs, seeds) for e in eps])
        f0 = evaluate(mode, w, objs, seeds)
        allf = np.concatenate([fp, fm]); ranks = allf.argsort().argsort() / (allf.size - 1) - 0.5
        g = ((ranks[:POP_PAIRS] - ranks[POP_PAIRS:])[:, None] * eps).sum(0) / (POP_PAIRS * SIGMA)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w = w + LR * (m / (1 - 0.9 ** (it + 1))) / (np.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
        log.append(dict(it=it, f_center=f0, f_mean=float(allf.mean()), episodes=(it + 1) * (2 * POP_PAIRS + 1) * N_OBJ,
                        wall=time.time() - t0))
        print(mode, seed, it, round(f0, 3), round(float(allf.mean()), 3), round(time.time() - t0), flush=True)
    np.save(f"weights/w_{mode}_s{seed}.npy", w)
    json.dump(log, open(f"weights/log_{mode}_s{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
