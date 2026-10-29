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

(none yet)
