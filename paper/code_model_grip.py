def model_grip(F_stuck, T):
    """Model controller: quasi-static Coulomb with r = 1 and F_stuck as the breakaway grip at rest."""
    return min(FMAX, F_stuck * (G + peak_acc(T)) / G)
