import numpy as np
from .utils import EPS

def trp_from_bandpowers(P_base: np.ndarray,
                        P_task: np.ndarray,
                        mode: str = "logratio") -> np.ndarray:
    """
    Compute TRP given baseline and task band powers.

    Modes:
    - "logratio", "log-ratio", "log_ratio", "ln", "ln-ratio":
        ln(P_task / P_base)      # natural log ratio (paper-style TRP)
    - "log10", "log10-ratio":
        log10(P_task / P_base)   # base-10 log ratio
    - "db", "decibel":
        10 * log10(P_task / P_base)
    - "ratio", "r":
        (P_task - P_base) / P_base    # relative change; baseline = 0

    Notes:
    - A small EPS is added for numerical stability.
    - All operations are elementwise and broadcast-friendly.
    """
    m = (mode or "").lower()

    # Canonicalize synonyms
    if m in {"logratio", "log-ratio", "log_ratio", "lg-ratio"}:
        return np.log10((P_task + EPS) / (P_base + EPS))

    if m in {"ln", "ln-ratio"}:
        return np.log((P_task + EPS) / (P_base + EPS))

    if m in {"log10", "log10-ratio", "log10_ratio"}:
        return np.log10((P_task + EPS) / (P_base + EPS))

    if m in {"db", "decibel"}:
        return 10.0 * np.log10((P_task + EPS) / (P_base + EPS))

    if m in {"ratio", "r"}:
        return (P_task - P_base) / (P_base + EPS)

    # Backward-compatible alias: treat "log" as natural log-ratio
    if m == "log":
        return np.log((P_task + EPS) / (P_base + EPS))

    raise ValueError("mode must be one of {'ratio','db','log','logratio','log10'}.")