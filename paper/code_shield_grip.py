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
