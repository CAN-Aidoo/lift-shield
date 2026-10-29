"""Exploratory (not pre-registered): how much grip does a slip allowance buy? For each test object (F_stuck from the
B0 episode at r = 1.0) and lift time T, compare the no-onset grip F_stuck (g + a_pk) / g with the certified grip at
s_lim = 3.5 mm (r_max = 1.2), and the certified grip at larger allowances."""
import json, numpy as np, pandas as pd
from lift_env import shield_grip, model_grip, G
d = pd.read_csv("results/test.csv"); b = d[(d.method == "B0") & (d.r == 1.0)]
out = {}
for T in (0.37, 0.5, 1.0):
    for s in (0.0035, 0.01, 0.03):
        ratios = [shield_grip(0.1, T, Fs, 1.2, s) / model_grip(Fs, T) for Fs in b.F_stuck]
        out[f"T{T}_s{s*1000:g}mm"] = dict(mean=round(float(np.mean(ratios)), 4), min=round(float(np.min(ratios)), 4), max=round(float(np.max(ratios)), 4))
json.dump(out, open("results/explore_allowance.json", "w"), indent=1); print(json.dumps(out, indent=1))

# grip saved by a 3.5 mm allowance at T = 0.37 s as a function of the assumed friction drop r_max
cliff = {}
Fs = float(b.F_stuck.iloc[0])
for rm in (1.0, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3):
    cliff[str(rm)] = round(shield_grip(0.1, 0.37, Fs, rm, 0.0035) / model_grip(Fs, 0.37), 4)
out["cliff_T0.37_s3.5mm_vs_rmax"] = cliff
json.dump(out, open("results/explore_allowance.json", "w"), indent=1); print(cliff)
