import numpy as np
from scipy.integrate import simpson
from typing import Dict, Tuple
from .utils import EPS

def band_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return (freqs >= float(fmin)) & (freqs <= float(fmax))

def bandpower(psd: np.ndarray, freqs: np.ndarray, bands: Dict[str, Tuple[float,float]], relative=False, total_range=(1.0,45.0)):
    out = {}
    if relative:
        tm = band_mask(freqs, total_range[0], total_range[1])
        denom = simpson(psd[:, tm], freqs[tm], axis=1) + EPS
    for name, (fmin,fmax) in bands.items():
        m = band_mask(freqs, fmin, fmax)
        p = simpson(psd[:, m], freqs[m], axis=1)
        if relative:
            p = p / denom
        out[name] = p
    return out

def spectral_entropy(psd: np.ndarray, freqs: np.ndarray, fmin=1.0, fmax=45.0, log_base=np.e):
    m = band_mask(freqs, fmin, fmax)
    P = psd[:, m]
    Psum = np.sum(P, axis=1, keepdims=True) + EPS
    p = P / Psum
    ent = -np.sum(p * np.log(p + EPS), axis=1)
    if log_base != np.e:
        ent = ent / np.log(log_base)
    return ent

def spectral_moments(psd: np.ndarray, freqs: np.ndarray, fmin=1.0, fmax=45.0):
    m = band_mask(freqs, fmin, fmax)
    f = freqs[m]
    P = psd[:, m]
    Pw = np.sum(P, axis=1, keepdims=True) + EPS
    mu = np.sum(P * f, axis=1) / np.sum(P, axis=1)
    var = np.sum(P * (f - mu[:,None])**2, axis=1) / Pw.squeeze()
    skew = np.sum(P * (f - mu[:,None])**3, axis=1) / (Pw.squeeze() * (np.sqrt(var)+EPS)**3)
    kurt = np.sum(P * (f - mu[:,None])**4, axis=1) / (Pw.squeeze() * (var+EPS)**2)
    return {"centroid": mu, "variance": var, "skewness": skew, "kurtosis": kurt}

def spectral_edge(psd: np.ndarray, freqs: np.ndarray, percent=0.95, fmin=1.0, fmax=45.0):
    m = band_mask(freqs, fmin, fmax)
    f = freqs[m]
    P = psd[:, m]
    cum = np.cumsum(P, axis=1)
    total = cum[:, -1][:, None] + EPS
    thresh = percent * total
    idxs = np.argmax(cum >= thresh, axis=1)
    return f[idxs]

def median_frequency(psd: np.ndarray, freqs: np.ndarray, fmin=1.0, fmax=45.0):
    return spectral_edge(psd, freqs, percent=0.5, fmin=fmin, fmax=fmax)
