"""Gentle fast lift with slip (Paper 7).

A vertical rod is held at its centre of mass between two spherical pads (condim 4), as in the Paper 5 sliding rig,
so gravity acts along the rod axis. The palm rides a vertical slide joint whose position and velocity are imposed
every step (heavy, gravity-compensated body), so it follows the planned trajectory and its acceleration a(t) is imposed. The rod is supported only by friction. Task: lift the rod by H = 0.10 m, quickly
and with a low grip, while the rod slips relative to the fingers by no more than S_TOL.

The rod may only translate along its axis (a slide joint): rotation about the grasp axis is suppressed so that
the study isolates axial slip (in a free rod, slip excites a rocking pivot, which is Paper 1's task, not this one).

Stick-slip: sliding friction of all contact geoms is mu_s while the rod is stuck relative to the palm and mu_s / r
while it slides (switch at |v_rel| > v_slip, back at |v_rel| < v_stick), as in Paper 5.

Physics (two pads, normal force F each, Coulomb):
  stuck while  m (g + a) <= 2 mu_s F
  sliding:     m a_rel = m (g + a) - 2 mu_k F     (a_rel = downward acceleration of the rod relative to the palm)

Proposition 3 (lift travel bound). Let F_stuck be a grip at which the rod was observed stuck with the palm at rest
(so 2 mu_s F_stuck >= m g) and r = mu_s / mu_k <= r_max. Then for any planned grip F(t) and palm acceleration a(t),
the relative slip is at most the travel of
    a_bar(t) = (g + a(t)) - g F(t) / (r_max F_stuck),   v' = max(0, v + a_bar dt),   x' = x + v dt,
started from rest, provided slip cannot start while (g + a(t)) <= g F(t) / F_stuck (static capacity >= demand).
Proof sketch: 2 mu_k F / m = (2 mu_s F / m) / r >= g F / (r_max F_stuck); while sliding, the true relative
acceleration is <= a_bar; while stuck it is 0 <= max(0, ...). The bound integrates a_bar from the first instant the
demand could exceed the static capacity, so it covers every breakaway time.
"""
import numpy as np, mujoco

G = 9.81
DT = 0.001
H = 0.10              # lift height (m)
S_TOL = 0.005         # allowed slip during the lift (m)
MARGIN = 0.0015       # sensing margin (3 sigma), as in Paper 5
SIG = 0.0005          # surrogate slip-sensor noise (m)
FMAX = 30.0
SEARCH = 0.9          # probe phase: grip multiplied by this each step until the rod moves
MOVE_THR = 0.0025     # probe phase: slip that counts as "moved" (m)
T0 = 1.0              # model controller's lift duration (s)
RMAX = 1.2            # shield's assumed bound on mu_s / mu_k

XML = """
<mujoco>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="20" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="palm" pos="0 0 0.5" gravcomp="1"><joint name="pz" type="slide" axis="0 0 1" damping="0"/><geom name="palmg" type="box" size="0.002 0.002 0.002" mass="1000" contype="0" conaffinity="0"/>
      <body name="lf"><joint name="lj" type="slide" axis="1 0 0" damping="300" range="0 0.03" limited="true"/>
        <geom name="lpad" type="sphere" size="0.006" pos="-0.0262 0 0" mass="0.02" condim="4" friction="{mu} 0.005 0.0001" solref="0.003 1" solimp="0.95 0.99 0.0005"/></body>
      <body name="rf"><joint name="rj" type="slide" axis="-1 0 0" damping="300" range="0 0.03" limited="true"/>
        <geom name="rpad" type="sphere" size="0.006" pos="0.0262 0 0" mass="0.02" condim="4" friction="{mu} 0.005 0.0001" solref="0.003 1" solimp="0.95 0.99 0.0005"/></body>
    </body>
    <body name="rod" pos="0 0 0.5"><joint name="rz" type="slide" axis="0 0 1"/>
      <geom name="rodg" type="box" size="0.01 0.01 0.08" mass="{m}" condim="4" friction="{mu} 0.005 0.0001" solref="0.003 1" solimp="0.95 0.99 0.0005"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="lgrip" joint="lj" gear="1" ctrlrange="0 40"/>
    <motor name="rgrip" joint="rj" gear="1" ctrlrange="0 40"/>
  </actuator>
</mujoco>
"""


def sample_object(rng):
    return dict(m=rng.uniform(0.05, 0.2), mu=rng.uniform(0.4, 0.8))


def minjerk(T, h=H):
    """Minimum-jerk lift of height h in time T, sampled at DT: returns (z, a) arrays (relative to start)."""
    n = max(2, int(round(T / DT)))
    s = np.arange(1, n + 1) / n
    z = h * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    a = h / T ** 2 * (60 * s - 180 * s ** 2 + 120 * s ** 3)
    return z, a


def peak_acc(T, h=H):
    return 5.7735 * h / T ** 2     # 10 / sqrt(3) h / T^2


class LiftRig:
    def __init__(self, m, mu, ratio=1.0, v_slip=0.02, v_stick=0.002):
        self.model = mujoco.MjModel.from_xml_string(XML.format(m=m, mu=mu))
        self.data = mujoco.MjData(self.model)
        M = self.model
        self.m, self.mu_s, self.ratio = m, mu, ratio
        self.mu_k = mu / ratio
        self.v_slip, self.v_stick = v_slip, v_stick
        self.sliding = False; self.n_breakaway = 0
        self._g = [M.geom(g).id for g in ("lpad", "rpad", "rodg")]
        self._rod = M.body("rod").id; self._vel = np.zeros(6)
        self._pq = M.joint("pz").qposadr[0]; self._pv = M.joint("pz").dofadr[0]
        self.palm_z = 0.5; self.palm_v = 0.0
        self.grip = FMAX; self.t_sim = 0.0; self.armed = False
        for j in ("lj", "rj"):
            self.data.qpos[M.joint(j).qposadr[0]] = 0.0262 - 0.01 - 0.006 + 0.0002
        self.data.ctrl[:] = [FMAX, FMAX]
        self.step(0.8); self.off0 = self.offset(); self.armed = True

    def offset(self):
        """Palm height minus rod height (grows when the rod slips down relative to the fingers)."""
        return self.palm_z - float(self.data.xpos[self._rod][2])

    def slip(self):
        return self.offset() - self.off0

    def held(self):
        return self.slip() < 0.07

    def mj1(self):
        # the palm is driven kinematically: its position and velocity are imposed every step
        self.data.qpos[self._pq] = self.palm_z - 0.5; self.data.qvel[self._pv] = self.palm_v
        if self.ratio != 1.0 and self.armed:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._rod, self._vel, 0)
            v = abs(self._vel[5] - self.palm_v)
            if not self.sliding and v > self.v_slip:
                self.sliding = True; self.n_breakaway += 1
                self.model.geom_friction[self._g, 0] = self.mu_k
            elif self.sliding and v < self.v_stick:
                self.sliding = False
                self.model.geom_friction[self._g, 0] = self.mu_s
        mujoco.mj_step(self.model, self.data)
        self.data.qpos[self._pq] = self.palm_z - 0.5; self.data.qvel[self._pv] = self.palm_v

    def step(self, secs, grip=None):
        for _ in range(int(round(secs / DT))):
            if grip is not None: self.data.ctrl[:] = grip
            self.palm_v = 0.0
            self.mj1()
        self.t_sim += secs

    def set_grip(self, F1, T=0.3):
        """Ramp grip to F1 over T with the palm still, then settle 0.3 s."""
        n = max(1, int(round(T / DT))); F0 = self.grip
        for k in range(n):
            self.data.ctrl[:] = F0 + (F1 - F0) * (k + 1) / n
            self.palm_v = 0.0; self.mj1()
        self.grip = F1; self.t_sim += T
        self.step(0.3, grip=F1)

    def lift(self, F, T):
        """Minimum-jerk lift of H in T at constant grip F (grip already set), then hold still 0.3 s."""
        z, a = minjerk(T)
        z0 = self.palm_z; zp = z0
        for k in range(len(z)):
            self.data.ctrl[:] = F
            znew = z0 + z[k]
            self.palm_v = (znew - zp) / DT; zp = znew
            self.palm_z = znew
            self.mj1()
        self.t_sim += T
        self.palm_v = 0.0
        self.step(0.3, grip=F)


def lift_bound(F, T, F_stuck, r_max):
    """Proposition 3 for one lift at constant grip F: 0.6 s at rest (grip ramp 0.3 s + settle 0.3 s; the ramp runs
    from a higher grip, so using F there is conservative), the minimum-jerk lift, then 0.3 s at rest.
    Slip cannot start while g + a <= g F / F_stuck; from the first instant it could, integrate a_bar."""
    _, al = minjerk(T)
    a = np.concatenate([np.zeros(600), al, np.zeros(300)])
    cap = G * F / F_stuck                      # lower bound on static capacity (per unit mass)
    over = np.nonzero(G + a > cap)[0]
    if len(over) == 0:
        return 0.0
    abar = (G + a[over[0]:]) - G * F / (r_max * F_stuck)
    v = x = 0.0
    for ab in abar:
        v = max(0.0, v + ab * DT); x += v * DT
    return x


def probe(rig, m_est, rng=None, sig=SIG):
    """Search phase with the palm still: step the grip down by SEARCH from the mu = 0.3 grip until the rod moves.
    Each step ramps to the new grip and holds 0.3 s while a slip monitor watches; if slip since the step began
    exceeds MOVE_THR the grip snaps to 2.5x that grip within 20 ms (reactive re-clamp), so a probe never lets the rod run away.
    Returns F_stuck (last grip with no motion; FMAX if the first step moved), F_moved, number of probes."""
    F = min(FMAX, m_est * G / (2 * 0.3))
    F_stuck = FMAX; n = 0
    while True:
        n += 1; s0 = rig.slip(); F0 = rig.grip; moved = False
        steps = [F0 + (F - F0) * (k + 1) / 100 for k in range(100)] + [F] * 300
        for Fk in steps:
            rig.data.ctrl[:] = Fk; rig.palm_v = 0.0; rig.mj1()
            noise = rng.normal(0, sig) if rng is not None else 0.0
            if rig.slip() - s0 + noise > MOVE_THR:
                moved = True; break
        rig.t_sim += 0.4
        if moved or F < 0.05:
            Fs = min(FMAX, 2.5 * F)                  # snap to 2.5x the grip that moved (ample static margin)
            rig.grip = Fs
            for k in range(20):                      # 20 ms snap
                rig.data.ctrl[:] = Fs; rig.palm_v = 0.0; rig.mj1()
            rig.step(0.3, grip=Fs)
            return F_stuck, F, n
        rig.grip = F; F_stuck = F; F *= SEARCH


# ----------------------------------------------------------------------------------------------- episode
SF, ST = 0.25, 1.0          # residual scales: grip x exp(0.25 a), duration x exp(1.0 a)
T_MIN, T_MAX = 0.25, 2.0
OBS_DIM = 6


def model_grip(F_stuck, T):
    """Model controller: quasi-static Coulomb with r = 1 and F_stuck as the breakaway grip at rest."""
    return min(FMAX, F_stuck * (G + peak_acc(T)) / G)


def shield_grip(F, T, F_stuck, r_max, s_lim):
    """Grip projection: smallest grip >= F whose Proposition 3 bound is <= s_lim (None if FMAX is not enough)."""
    if lift_bound(F, T, F_stuck, r_max) <= s_lim:
        return F
    if lift_bound(FMAX, T, F_stuck, r_max) > s_lim:
        return None
    lo, hi = F, FMAX
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if lift_bound(mid, T, F_stuck, r_max) <= s_lim: hi = mid
        else: lo = mid
    return hi


def shield_time(F, T, F_stuck, r_max, s_lim):
    """Time projection: shortest duration >= T (<= T_MAX) whose bound is <= s_lim at grip F (None if none)."""
    if lift_bound(F, T, F_stuck, r_max) <= s_lim:
        return T
    if lift_bound(F, T_MAX, F_stuck, r_max) > s_lim:
        return None
    lo, hi = T, T_MAX
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if lift_bound(F, mid, F_stuck, r_max) <= s_lim: hi = mid
        else: lo = mid
    return hi


def features(F_stuck, F_moved, m_est, n_probe):
    w = m_est * G
    return np.array([np.log(F_stuck / w), np.log(F_moved / F_stuck), m_est / 0.2, n_probe / 10.0,
                     np.log(F_stuck / FMAX) / 4.0, 1.0])


class LinearPolicy:
    def __init__(self, w=None):
        self.w = np.zeros((2, OBS_DIM)) if w is None else w.reshape(2, OBS_DIM)

    def __call__(self, o):
        return np.tanh(self.w @ o)


def run_episode(obj, rng, mode="model", policy=None, r_max=RMAX, shield_axis="grip"):
    """mode: model | modelS | residual | residualS (shield in the loop). Returns a result dict."""
    rig = LiftRig(obj["m"], obj["mu"], ratio=obj.get("ratio", 1.0))
    m_est = obj["m"] * rng.uniform(0.9, 1.1)                 # weight from wrist sensing, +-10%
    F_stuck, F_moved, n_probe = probe(rig, m_est, rng)
    t_probe = rig.t_sim
    T_m = T0; F_m = model_grip(F_stuck, T_m)
    if mode.startswith("residual"):
        a = np.clip(policy(features(F_stuck, F_moved, m_est, n_probe)), -1, 1)
        T_cmd = float(np.clip(T_m * np.exp(ST * a[1]), T_MIN, T_MAX))
        F_cmd = float(np.clip(model_grip(F_stuck, T_cmd) * np.exp(SF * a[0]), 0.1, FMAX))
    else:
        T_cmd, F_cmd = T_m, F_m
    s_lim = S_TOL - MARGIN
    bound0 = lift_bound(F_cmd, T_cmd, F_stuck, r_max)
    projected = declined = False
    if mode.endswith("S"):
        if shield_axis == "grip":
            Fs = shield_grip(F_cmd, T_cmd, F_stuck, r_max, s_lim)
            if Fs is None: declined = True
            else: projected = Fs > F_cmd + 1e-9; F_cmd = Fs
        else:
            Ts = shield_time(F_cmd, T_cmd, F_stuck, r_max, s_lim)
            if Ts is None: declined = True
            else: projected = Ts > T_cmd + 1e-9; T_cmd = Ts
    if declined:                                              # no certified lift: fall back to the model's lift
        T_cmd, F_cmd = T_m, max(F_m, F_cmd)
    bound = lift_bound(F_cmd, T_cmd, F_stuck, r_max)
    rig.set_grip(F_cmd); s0 = rig.slip(); rig.lift(F_cmd, T_cmd)
    slip = rig.slip() - s0
    held = rig.held()
    return dict(T=T_cmd, F=F_cmd, F_rel=F_cmd / (obj["m"] * G), slip=slip, held=held,
                ok=bool(held and slip <= S_TOL), probes=n_probe, t_probe=t_probe, F_stuck=F_stuck,
                bound=bound, bound_before=bound0, projected=projected, declined=declined,
                exceed=bool(slip > bound), n_breakaway=rig.n_breakaway)


def score(res, lam=1.0):
    """Training reward: failure (drop or slip > S_TOL) costs 10; otherwise faster and gentler is better.
    Lift time in units of T0, grip in units of the object's weight m g."""
    if not res["ok"]:
        return -10.0
    return -res["T"] / T0 - lam * res["F_rel"] / 10.0
