# PROTOCOL: Lift-with-slip shield (Paper 7)

Frozen 29 Oct 2026, before any policy training. Changes after this date go in **Deviations** at the end, dated.

## Task

A parallel gripper holds a rod of unknown mass m ~ U[0.05, 0.2] kg and unknown friction mu_s ~ U[0.4, 0.8] and must
lift it H = 10 cm. Goal: lift fast, with low grip, and let the rod slip no more than S_TOL = 5 mm relative to the
fingers during the lift. Kinetic friction is mu_k = mu_s / r with r unknown to every controller.

Episode:
1. **Weight estimate.** m_est = m * U[0.9, 1.1] (wrist force sensing, +-10%).
2. **Probe.** With the palm still, the grip starts at m_est g / (2 * 0.3) and is multiplied by 0.9 each step (100 ms
   ramp, 300 ms hold) until a slip monitor (noise sigma 0.5 mm) reads more than 2.5 mm. On motion the grip snaps to
   2.5x the grip that moved within 20 ms and holds 0.3 s. F_stuck is the last grip at which the rod did not move.
3. **Lift.** The controller chooses a grip F and duration T. The grip ramps to F over 0.3 s and settles 0.3 s, then
   the palm follows a minimum-jerk profile of height H in T (peak acceleration 5.7735 H / T^2), then holds 0.3 s.
4. **Outcome.** Slip = change in palm-minus-rod offset over the lift phase. Success (ok) = rod still held
   (offset change < 7 cm) and slip <= S_TOL.

## Rig (MuJoCo 3, dt = 1 ms), modelling choices fixed during development

- Palm on a z slide joint driven kinematically (position and velocity imposed every step), gravity-compensated.
- Pads: spheres, condim 4, solref 0.003, impratio 10, noslip 20.
- Rod constrained to a vertical slide joint. A free rod rocked about the grasp axis after slip and then crept; the
  paper studies axial slip only, and says so.
- Stick-slip emulation: pad friction switches mu_s -> mu_k when |v_rod - v_palm| > 0.02 m/s and back below
  0.002 m/s (as in Paper 5, thresholds raised to keep slow elastic creep from triggering kinetic friction).
- Reactive probe (above) so that probing never lets the rod run away when r > 1.

## Validation of the rig (development only; `validate.py`, `results/validation.json`)

- V1 static breakaway grip / (m g / 2 mu): mean 0.989 (range 0.986-1.014, n = 20).
- V2 breakaway grip during a lift / (m (g + a_pk) / 2 mu): 0.890, 0.930, 0.952, 0.983 at T = 0.25, 0.35, 0.5, 1.0 s.
  The quasi-static model over-predicts the needed grip at short T, i.e. it is conservative.
- V3 120 random lifts (r ~ U[1, 1.2], r_max = 1.2, T ~ U[0.25, 1] s, grip = 0.6-1.4x the true quasi-static need): 0 slips when the bound was 0
  (n = 67); 39 exceedances of a non-zero bound, max 0.196 mm (below MARGIN = 1.5 mm); 0 lifts with slip > S_TOL when
  the bound was <= S_TOL. Probe slip max 5.3 mm, median 8 probes.
- V4 elastic creep under hold: 0.09-0.19 mm at every grip margin 1.1-2.0.

## Shield (Proposition 3)

With F_stuck as a lower bound on the static breakaway grip at rest, slip cannot begin while
g + a(t) <= g F / F_stuck. From the first instant it can, the relative acceleration is bounded by
a_bar(t) = (g + a(t)) - g F / (r_max F_stuck); integrating with velocity clipped at 0 over the rest-lift-rest profile
gives a bound on slip. A command is certified if its bound <= s_lim = S_TOL - MARGIN = 3.5 mm.
- **Grip projection:** smallest F >= F_cmd (<= FMAX = 30 N) that is certified, by bisection.
- **Time projection:** shortest T >= T_cmd (<= 2 s) that is certified at F_cmd.
- If no certified command exists the shield **declines** and the lift uses the model controller's command
  (with grip max(F_model, F_cmd)).
- r_max = 1.2 unless stated.

## Controllers

- Model: T = T0 = 1.0 s, F = F_stuck (g + a_pk(T)) / g (quasi-static Coulomb, r = 1).
- Residual: linear policy, 6 features (log F_stuck / m_est g, log F_moved / F_stuck, m_est / 0.2, probes / 10,
  log(F_stuck / FMAX) / 4, bias), output tanh, 2 actions: T = clip(T0 exp(1.0 a_T), 0.25, 2.0),
  F = clip(F_model(T) exp(0.25 a_F), 0.1, FMAX).
- Reward: -10 if not ok, else -T / T0 - F_rel / 10 with F_rel = F / (m g).

## Methods

| ID | Controller | Shield at training | Shield at test |
|---|---|---|---|
| B0 | model | - | - |
| B0S | model | - | grip |
| B2 | residual | no | no |
| B4g | residual (B2 weights) | no | grip |
| B4t | residual (B2 weights) | no | time |
| B5 | residual | grip | grip |

Training: antithetic ES as Paper 5 (6 pairs, 6 objects per iteration, sigma 0.1, Adam lr 0.06, 20 iterations,
1560 episodes per run); objects with r ~ U[1.0, 1.2], r_max = 1.2; seeds 0-4 for B2 and B5. Weights start at 0
(the model controller). Development objects use seed 123 only.

## Evaluation sets

- **Test:** object seed 11, 40 objects, episode seeds 5000 + i, each object at r in {1.0, 1.1, 1.2, 1.3}
  (1.3 is outside r_max). All methods.
- **r_max sweep (Q4):** B0S and B5 (trained at 1.2) on the test set with shield r_max in {1.0, 1.4}.
- **Confirmation:** object seed 47, 40 objects, r ~ U[1.0, 1.3] drawn with the same seed, episode seeds 7000 + i;
  run once, after the analysis code and the test-set results are committed. Methods B0, B0S, B2, B4g, B5.

## Questions and outcomes

- **Q1 (speed and grip).** Does a residual lift faster than the model, and how much of that survives the shield?
  Outcomes: lift time T, relative grip F_rel, success. Test B2, B4g, B5 vs B0 per r.
- **Q2 (safety vs r).** Failure rate (drop or slip > 5 mm) per method and r; is the unshielded residual's failure
  rate higher at larger r?
- **Q3 (which axis).** Grip projection (B4g) vs time projection (B4t) on the same commands: T, F_rel, success,
  fraction projected and declined.
- **Q4 (certificate).** For every shielded episode: fraction with slip > bound, max exceedance, and number with
  slip > S_TOL among certified lifts, at r <= r_max and at r > r_max; effect of r_max = 1.0 / 1.2 / 1.4.

## Statistics

- Learned methods summarised per object as the mean over 5 seeds.
- Paired Wilcoxon signed-rank tests vs B0 over the 40 objects, for T and F_rel (all episodes) at each r;
  Holm correction within each question family.
- Failure rates: exact counts with Wilson 95% intervals (per seed-episode); McNemar-style paired exact test vs B0
  on per-object any-failure.
- Hierarchical bootstrap 95% CIs (resample seeds, then objects; 10,000 draws) for mean differences.
- Only measured results are reported; every number in the paper is produced by a script in this repository.

## Deviations

- **D1 (29 Oct 2026).** The pre-registered exceedance metric (slip > bound) is dominated by elastic creep: every
  certified lift had a bound of exactly 0 and measured 0.05-0.49 mm of creep, so "exceedance" is 100% at r_max = 1.2.
  We report it as registered and add the size of the excess against MARGIN (1.5 mm) and the count of lifts over S_TOL.
- **D2 (29 Oct 2026).** The confirmation set draws r per object from U[1.0, 1.3], so the per-r tests do not apply; it is
  analysed pooled over objects with the same tests (`analyze_confirm.py`), plus failures by r band.
- **D3 (29 Oct 2026, exploratory).** Not pre-registered: how much grip a slip allowance buys as a function of r_max
  (`explore_allowance.py`). Reported and labelled as exploratory.
- Naming: the bound called "Proposition 3" in code comments is Proposition 1 in the paper.

## Addendum A: extended study (frozen 29 Oct 2026, after the confirmatory results above were committed)

Everything in this addendum is **exploratory**: it was designed after the test and confirmation results were known.
It is fixed here, before any of it is run, so that what we report is not selected after the fact.

### A1. Controlled one-change-at-a-time study
Reference configuration = the test set above (40 objects, seed 11, r in {1.0, 1.1, 1.2, 1.3}, r_max = 1.2, weights
from the main training). Each row changes **one** design choice and re-runs B0, B0S, B2, B4g, B4t, B5 on the same
objects and episode seeds. Rows:

| Row | Change | Retrain? |
|---|---|---|
| E1 | heavier objects, m ~ U[0.2, 0.5] kg | no |
| E2 | slippery objects, mu_s ~ U[0.2, 0.4] | no |
| E3 | slip-monitor noise 1.5 mm (3x) | no |
| E4 | weight estimate error +-25% | no |
| E5 | coarser probe, step 0.8 | no |
| E6 | finer probe, step 0.95 | no |
| E7 | lift height 20 cm | no |
| E8 | softer pads, solref 0.006 | no |
| E9 | earlier kinetic switch, v_slip 0.005 m/s | no |
| S1 | shield r_max = 1.0 | no |
| S2 | shield r_max = 1.1 | no |
| S3 | shield r_max = 1.4 | no |
| S4 | slip tolerance 2 mm | no |
| S5 | slip tolerance 10 mm | no |
| S6 | no sensing margin (MARGIN = 0) | no |
| T1 | trained on r = 1.0 only | B2, B5 retrained (5 seeds) |
| T2 | wider time action (duration scale 1.5) | retrained |
| T3 | grip weight lambda = 10 | retrained |
| T4 | double training budget (40 iterations) | retrained |

Metrics per method: failure rate (% of lifts with slip > tolerance, or drop), mean lift time, mean grip F/mg,
mean episode time. Changes are tested against the reference with a paired bootstrap over the 40 objects
(10,000 resamples of per-object means over r and seeds); * marks a 95% interval excluding 0.

### A2. Action-space safety map
Fixed residual actions on a grid a_T in {-1, -0.83, ..., 0} (7 values) x a_F in {-1, -0.75, ..., 1} (9 values),
40 test objects, r in {1.0, 1.2}, without and with the grip shield. Report failure rate, grip and lift time per cell.

### A3. Bound tightness
Deliberately under-gripped lifts: grip = model grip x {0.85, 0.88, 0.91, 0.94, 0.97, 1.0}, T in {0.37, 0.5, 1.0} s,
r in {1.0, 1.1, 1.2}, 40 test objects; Proposition 1 evaluated with r_max = true r. Report predicted vs measured slip.

### A4. Anatomy of failures
Full-state traces (palm and rod height, relative velocity, friction state, grip, bound) for B2 and B5 on the same
objects; the object list is every test object where B2 seed 2 fails at r = 1.2 (seed 2 has the most failures).

### A5. Renders and training curves
MuJoCo renders of one lift per method (illustration only) and the ES training logs already saved.

Code refactor allowed before running: module constants used as default arguments are read at call time instead
(no change in behaviour; checked by re-running test-set episodes and requiring identical results).

### A6. Diagnostic rows (added 29 Oct 2026, after rows E7 and E9 were seen, before running these)
Rows E7 and E9 produced failures of certified lifts. Traces show the emulator's kinetic switch fired at a true relative
speed of 12 mm/s: the switch compares the rod's velocity from the previous step with the palm's commanded velocity for
the current step, which adds a*dt (about 8 mm/s at E7's peak acceleration) to the measured relative speed. This lag is part
of the frozen rig and is kept for every other result; in the reference configuration it never caused a switch during a
shielded lift (0 of 2,560). Diagnostic rows E7L and E9L repeat E7 and E9 with the lag removed (palm velocity from the same
step as the rod velocity). They are reported next to E7 and E9, not instead of them.
- Added 29 Oct 2026 after seeing E7L (0 shielded failures): row R0L repeats the reference configuration with the lag
  removed, to measure how much of the main-study results depend on the lag. Reported as a diagnostic next to the main
  results; the main results stay as registered.
