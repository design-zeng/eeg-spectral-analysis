import numpy as np
from .utils import EPS

def trp_from_bandpowers(P_base: np.ndarray, P_task: np.ndarray, mode: str = "log"):
    if mode == "ratio":
        return (P_task - P_base) / (P_base + EPS)
    elif mode == "db":
        return 10.0 * (np.log10(P_task + EPS) - np.log10(P_base + EPS))
    elif mode == "log":
        return np.log10(P_task + EPS) - np.log10(P_base + EPS)
    else:
        raise ValueError("mode must be 'ratio' or 'db'")
