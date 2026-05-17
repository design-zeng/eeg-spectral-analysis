import numpy as np
from .features import band_mask
from .utils import EPS, trapz_compat

def faa_from_psd(psd: np.ndarray, freqs: np.ndarray, ch_names, left="F3", right="F4", alpha=(8.0,13.0), use_db=False):
    idx = {c:i for i,c in enumerate(ch_names)}
    if left not in idx or right not in idx:
        return float("nan")
    li, ri = idx[left], idx[right]
    m = band_mask(freqs, alpha[0], alpha[1])
    P_L = trapz_compat(psd[li, m], freqs[m])
    P_R = trapz_compat(psd[ri, m], freqs[m])
    if use_db:
        return float(10*np.log10(P_R + EPS) - 10*np.log10(P_L + EPS))
    else:
        return float(np.log(P_R + EPS) - np.log(P_L + EPS))
