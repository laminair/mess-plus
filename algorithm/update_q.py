

def update_Q(Q: float, Q_tracked: float, alpha: float, user_satisfaction: float, sample_has_feedback: bool):
    Q = max(0.0, Q + alpha - user_satisfaction)

    if sample_has_feedback:
        Q_tracked = max(0.0, Q_tracked + alpha - user_satisfaction)

    return Q, Q_tracked