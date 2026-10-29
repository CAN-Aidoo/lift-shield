def score(res, lam=None):
    lam = LAM if lam is None else lam
    """Training reward: failure (drop or slip > S_TOL) costs 10; otherwise faster and gentler is better.
    Lift time in units of T0, grip in units of the object's weight m g."""
    if not res["ok"]:
        return -10.0
    return -res["T"] / T0 - lam * res["F_rel"] / 10.0
