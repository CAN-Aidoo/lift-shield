def probe(rig, m_est, rng=None, sig=None):
    """Search phase with the palm still: step the grip down by SEARCH from the mu = 0.3 grip until the rod moves.
    Each step ramps to the new grip and holds 0.3 s while a slip monitor watches; if slip since the step began
    exceeds MOVE_THR the grip snaps to 2.5x that grip within 20 ms (reactive re-clamp), so a probe never lets the rod run away.
    Returns F_stuck (last grip with no motion; FMAX if the first step moved), F_moved, number of probes."""
    sig = SIG if sig is None else sig
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
