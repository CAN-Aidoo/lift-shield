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
