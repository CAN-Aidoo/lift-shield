def features(F_stuck, F_moved, m_est, n_probe):
    w = m_est * G
    return np.array([np.log(F_stuck / w), np.log(F_moved / F_stuck), m_est / 0.2, n_probe / 10.0,
                     np.log(F_stuck / FMAX) / 4.0, 1.0])
