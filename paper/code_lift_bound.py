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
