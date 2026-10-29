def run_episode(obj, rng, mode="model", policy=None, r_max=None, shield_axis="grip"):
    """mode: model | modelS | residual | residualS (shield in the loop). Returns a result dict."""
    r_max = RMAX if r_max is None else r_max
    rig = LiftRig(obj["m"], obj["mu"], ratio=obj.get("ratio", 1.0))
    m_est = obj["m"] * rng.uniform(1 - M_ERR, 1 + M_ERR)     # weight from wrist sensing, +-10%
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
