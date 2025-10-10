# src/eegspec/plot_trp.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Group TRP line plots (lower & upper alpha), with logging and CLI integration.

Features
- Inputs: multiple JSONs (--summary repeatable) or a root directory (--summary-dir + --summary-glob)
- Modes: ratio | db | log | log10 | logratio (logratio = natural log ratio)
- Layouts: hemi (LH/RH × 5 regions) | collapsed (merge hemispheres → 5 regions)
- Task merges via --merge "New=Old1,Old2,..." (wildcards OK), --exclude-condition to filter
- Midline exclusion (--exclude-midline)
- Y axis fixed to 2 decimals
- Designed to be called from eegspec.cli subcommand (plot-trp)
"""

from __future__ import annotations
import json, re, fnmatch, logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .base import BaseApp
from .utils import EPS, parse_locs_file, builtin_montage_path  # reuse project utilities


# ------------------------ constants & helpers ------------------------
LOWER_KEYS = {"lower","lower_alpha","lower-alpha","alpha_low","alpha_lower","8-10","8_10","low_alpha","alpha8_10"}
UPPER_KEYS = {"upper","upper_alpha","upper-alpha","alpha_high","alpha_upper","10-12","10_12","high_alpha","alpha10_12"}
REGIONS   = ["frontal","central","temporal","parietal","occipital"]
HEMS      = ["LH","RH"]


def canon_mode(m: Optional[str]) -> Optional[str]:
    """Normalize mode synonyms."""
    if m is None: return None
    m = m.lower()
    if m in ("logratio","log-ratio","log_ratio","lnratio","ln-ratio"): return "logratio"
    if m in ("log","ln"): return "log"
    if m in ("log10","log10-ratio","log10_ratio"): return "log10"
    if m in ("db","decibel"): return "db"
    if m in ("ratio","r"): return "ratio"
    return m


def hemisphere_from_label(lbl: str) -> str:
    """Infer hemisphere by odd/even suffix; 'z' as midline."""
    if lbl.lower().endswith("z"):  # midline
        return "MID"
    m = re.search(r"(\d+)$", lbl)
    if m:
        return "LH" if (int(m.group(1)) % 2 == 1) else "RH"
    # fallback: treat as MID if no numeric hint
    return "MID"


def region_from_label(lbl: str) -> str:
    """Heuristic 10-10 prefix → region."""
    u = lbl.upper()
    if u.startswith(("FP","AF","F")): return "frontal"
    if u.startswith(("FC",)) and re.search(r"(1|2|3|4)$", u): return "frontal"
    if u.startswith(("FC","C")): return "central"
    if u.startswith(("FT","T","TP")): return "temporal"
    if u.startswith(("CP","P")): return "parietal"
    if u.startswith(("PO","O")): return "occipital"
    return "central"


def detect_band_keys(band_dict: Dict[str, List[float]],
                     cli_lower: Optional[str],
                     cli_upper: Optional[str]) -> Tuple[str,str]:
    """Pick band keys from CLI if provided, else infer from common aliases."""
    if cli_lower and cli_upper:
        return cli_lower, cli_upper
    lower = next((k for k in band_dict.keys() if k.lower() in LOWER_KEYS), None)
    upper = next((k for k in band_dict.keys() if k.lower() in UPPER_KEYS), None)
    if not lower or not upper:
        raise ValueError("Cannot find lower/upper alpha keys. Use --lower-key/--upper-key.")
    return lower, upper


def sem(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standard error with NaN robustness."""
    arr = np.asarray(arr, float)
    n = np.sum(~np.isnan(arr), axis=axis, keepdims=True)
    s = np.nanstd(arr, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n>1, s/np.sqrt(n), 0.0).squeeze()


# ------------------------ per-file aggregation ------------------------
def aggregate_by_region_by_subject(values_by_channel: Dict[str, List[float]],
                                   labels: List[str], hemis: List[str], regions: List[str],
                                   exclude_midline: bool):
    """Average across channels within each region; keep subject axis."""
    max_len = 0
    for v in values_by_channel.values():
        max_len = max(max_len, len(v))
    agg_ch = {hem: {reg: [] for reg in REGIONS} for hem in HEMS}
    for lbl, hem, reg in zip(labels, hemis, regions):
        if exclude_midline and hem == "MID": continue
        if hem not in HEMS: continue
        vec = values_by_channel.get(lbl, [])
        arr = np.asarray(vec, float)
        if arr.size < max_len:
            arr = np.concatenate([arr, np.full(max_len - arr.size, np.nan)])
        agg_ch[hem][reg].append(arr)
    out = {hem: {reg: np.array([]) for reg in REGIONS} for hem in HEMS}
    for hem in HEMS:
        for reg in REGIONS:
            if len(agg_ch[hem][reg]) == 0: continue
            M = np.vstack(agg_ch[hem][reg]).T  # (subjects, channels_in_region)
            out[hem][reg] = np.nanmean(M, axis=1)
    return out


def aggregate_from_vectors(vec: np.ndarray,
                           labels: List[str], hemis: List[str], regions: List[str],
                           exclude_midline: bool):
    """Group per-channel single values into region arrays (channels)."""
    out = {hem: {reg: [] for reg in REGIONS} for hem in HEMS}
    for val, lbl, hem, reg in zip(vec, labels, hemis, regions):
        if exclude_midline and hem == "MID": continue
        if hem not in HEMS: continue
        out[hem][reg].append(float(val))
    for hem in HEMS:
        for reg in REGIONS:
            out[hem][reg] = np.asarray(out[hem][reg], float)
    return out


# ------------------------ merge / filter ------------------------
def parse_merge_specs(specs: List[str], available: List[str]) -> Dict[str, List[str]]:
    """Parse rules like New=Old1,Old2 (wildcards OK)."""
    mapping = {}
    for s in specs or []:
        if "=" not in s:
            raise ValueError(f"--merge '{s}' needs '=' (e.g., New=Old1,Old2)")
        new, src = s.split("=", 1)
        new = new.strip()
        patterns = [p.strip() for p in src.split(",") if p.strip()]
        picked = []
        for pat in patterns:
            if any(ch in pat for ch in "*?[]"):
                picked += [c for c in available if fnmatch.fnmatch(c, pat)]
            else:
                if pat in available: picked.append(pat)
        picked = [c for c in available if c in set(picked)]  # dedup keep order
        if not picked:
            raise ValueError(f"--merge '{s}' matched nothing in {available}")
        mapping[new] = picked
    return mapping


def merge_condition_vectors(vec_dict: Dict[str, np.ndarray], sources: List[str]) -> np.ndarray:
    """Average multiple condition vectors per subject (NaN-aligned)."""
    if not sources: return np.array([])
    vecs = [np.asarray(vec_dict.get(k, np.array([])), float) for k in sources]
    max_len = max((v.size for v in vecs), default=0)
    if max_len == 0: return np.array([])
    stack = []
    for v in vecs:
        if v.size < max_len:
            v = np.concatenate([v, np.full(max_len - v.size, np.nan)])
        stack.append(v)
    M = np.vstack(stack)  # (n_src, subjects)
    return np.nanmean(M, axis=0)


def apply_merges(one_band_data, conditions: List[str],
                 merge_map: Dict[str, List[str]], include_unmerged: bool):
    """Return merged data and new conditions list."""
    new_data = {hem:{reg:{} for reg in REGIONS} for hem in HEMS}
    used = set()
    for new_name, srcs in merge_map.items():
        for hem in HEMS:
            for reg in REGIONS:
                vec_dict = {cond: one_band_data[hem][reg].get(cond, np.array([])) for cond in srcs}
                new_vec  = merge_condition_vectors(vec_dict, srcs)
                new_data[hem][reg][new_name] = new_vec
        used.update(srcs)
    if include_unmerged:
        for cond in conditions:
            if cond in used: continue
            for hem in HEMS:
                for reg in REGIONS:
                    new_data[hem][reg][cond] = one_band_data[hem][reg].get(cond, np.array([]))
        new_conditions = list(merge_map.keys()) + [c for c in conditions if c not in used]
    else:
        new_conditions = list(merge_map.keys())
    return new_data, new_conditions


def apply_exclusions(conditions: List[str], patterns: List[str]) -> List[str]:
    if not patterns: return conditions
    out = []
    for c in conditions:
        if any(fnmatch.fnmatch(c, pat) for pat in patterns): continue
        out.append(c)
    return out


# ------------------------ transforms ------------------------
def apply_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    """Elementwise numeric transform."""
    if transform == "none":  return arr
    if transform == "db":    return 10.0 * np.log10(np.maximum(arr, EPS))
    if transform == "log":   return np.log(np.maximum(arr, EPS))       # natural log
    if transform == "log10": return np.log10(np.maximum(arr, EPS))
    raise ValueError(f"Unknown transform: {transform}")


# ------------------------ collapse hemispheres ------------------------
def _avg_vectors_align(vecs: List[np.ndarray]) -> np.ndarray:
    if not vecs: return np.array([])
    max_len = max((v.size for v in vecs), default=0)
    if max_len == 0: return np.array([])
    stack = []
    for v in vecs:
        a = np.asarray(v, float)
        if a.size < max_len:
            a = np.concatenate([a, np.full(max_len - a.size, np.nan)])
        stack.append(a)
    M = np.vstack(stack)
    return np.nanmean(M, axis=0)


def collapse_hemis(one_band_data):
    """dict[hem][reg][cond]->dict[reg][cond], averaging LH&RH."""
    out = {reg:{} for reg in REGIONS}
    for reg in REGIONS:
        all_conds = set(one_band_data["LH"][reg].keys()) | set(one_band_data["RH"][reg].keys())
        for cond in all_conds:
            v_l = one_band_data["LH"][reg].get(cond, np.array([]))
            v_r = one_band_data["RH"][reg].get(cond, np.array([]))
            out[reg][cond] = _avg_vectors_align([v_l, v_r])
    return out


# ------------------------ plotting ------------------------
def _fix_y_2dec(ax):
    """Force 2 decimals on Y axis."""
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))


def plot_lines_hemi(one_band_data, conditions, ymode, transform, out_png, logger: logging.Logger):
    x_labels = [f"LH-{r}" for r in REGIONS] + [f"RH-{r}" for r in REGIONS]
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for cond in conditions:
        means, errs = [], []
        for hem in HEMS:
            for reg in REGIONS:
                vals = one_band_data[hem][reg].get(cond, np.array([]))
                vals = apply_transform(vals, transform)
                m = float(np.nanmean(vals)) if vals.size>0 else np.nan
                e = float(sem(vals)) if vals.size>1 else 0.0
                means.append(m); errs.append(e)
        ax.errorbar(x, np.asarray(means), yerr=np.asarray(errs), marker='o', linestyle='-', linewidth=1.5,
                    capsize=3, label=cond)
    baseline = 1.0 if ymode == "ratio" else 0.0
    y_label_map = {"ratio":"TRP (ratio)","db":"TRP (dB)","log":"TRP (log)",
                   "log10":"TRP (log10-ratio)","logratio":"TRP (log-ratio)"}
    ax.axhline(baseline, color="k", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel(y_label_map.get(ymode, f"TRP ({ymode})"))
    _fix_y_2dec(ax)
    ax.legend(frameon=False, ncol=2); ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_png}")


def plot_lines_collapsed(collapsed_data, conditions, ymode, transform, out_png, custom_ylabel, logger: logging.Logger):
    x_labels = [f"{r}" for r in REGIONS]
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for cond in conditions:
        means, errs = [], []
        for reg in REGIONS:
            vals = collapsed_data[reg].get(cond, np.array([]))
            vals = apply_transform(vals, transform)
            m = float(np.nanmean(vals)) if vals.size>0 else np.nan
            e = float(sem(vals)) if vals.size>1 else 0.0
            means.append(m); errs.append(e)
        ax.errorbar(x, np.asarray(means), yerr=np.asarray(errs), marker='o', linestyle='-', linewidth=1.5,
                    capsize=3, label=cond)
    baseline = 1.0 if ymode == "ratio" else 0.0
    ax.axhline(baseline, color="k", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel(f"TRP Alpha {custom_ylabel} ({ymode})")
    _fix_y_2dec(ax)
    ax.legend(frameon=False, ncol=2); ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_png}")


# ------------------------ IO & combine ------------------------
def _list_summary_paths(summaries: List[str], summary_dir: Optional[str], summary_glob: str) -> List[Path]:
    paths: List[Path] = []
    for s in summaries or []:
        p = Path(s);
        if p.exists(): paths.append(p)
    if summary_dir:
        root = Path(summary_dir)
        paths += [p for p in root.rglob(summary_glob) if p.is_file()]
    # de-duplicate & sort
    uniq, seen = [], set()
    for p in sorted(paths):
        k = p.as_posix()
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq


def _concat_1d(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    xs = [np.asarray(a).ravel() for a in list_of_arrays if np.asarray(a).size > 0]
    return np.concatenate(xs) if xs else np.array([])


# ------------------------ public entry ------------------------
def plot_trp_entry(*,
                   summary: List[str],
                   summary_dir: Optional[str],
                   summary_glob: str,
                   montage: Optional[str],
                   out_dir: str,
                   layout: str,
                   mode: Optional[str],
                   transform: str,
                   exclude_midline: bool,
                   lower_key: Optional[str],
                   upper_key: Optional[str],
                   merges: List[str],
                   only_merged: bool,
                   exclude_patterns: List[str],
                   log_kwargs: Dict):
    """
    Programmatic entry (used by eegspec.cli). All args are pure-Python (no argparse here).
    """
    app = BaseApp(**(log_kwargs or {}))
    L = app.logger

    files = _list_summary_paths(summary, summary_dir, summary_glob)
    if not files:
        L.error("No input summaries found."); raise SystemExit(2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    L.info(f"Found {len(files)} summary file(s).")

    # montage labels
    montage_path = montage or builtin_montage_path()
    labels = parse_locs_file(montage_path)
    hemis  = [hemisphere_from_label(lbl) for lbl in labels]
    regions= [region_from_label(lbl) for lbl in labels]
    if exclude_midline:
        keep = [i for i,h in enumerate(hemis) if h != "MID"]
        labels = [labels[i] for i in keep]
        hemis  = [hemis[i]  for i in keep]
        regions= [regions[i]for i in keep]
        L.info(f"Excluded midline; kept {len(labels)} channels.")

    # mode / transform
    # use first file's json mode unless CLI overrides
    first = json.loads(files[0].read_text(encoding="utf-8"))
    json_mode_first = canon_mode(first.get("mode","ratio"))
    ymode = canon_mode(mode) or json_mode_first
    if transform != "auto":
        xform = transform
    else:
        if json_mode_first == "ratio" and ymode in ("db","log","log10","logratio"):
            xform = "db" if ymode == "db" else ("log10" if ymode == "log10" else "log")
        else:
            xform = "none"
    L.info(f"Axis mode={ymode}; numeric transform={xform}; files={len(files)}")

    # combined containers
    lower_lists = {hem:{reg:{} for reg in REGIONS} for hem in HEMS}
    upper_lists = {hem:{reg:{} for reg in REGIONS} for hem in HEMS}
    all_conditions = set()

    def _append(dst_lists, src_dict):
        for hem in HEMS:
            for reg in REGIONS:
                for cond, vec in src_dict[hem][reg].items():
                    dst_lists[hem][reg].setdefault(cond, []).append(np.asarray(vec, float))

    # per-file loop
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        low_one = {hem:{reg:{} for reg in REGIONS} for hem in HEMS}
        up_one  = {hem:{reg:{} for reg in REGIONS} for hem in HEMS}

        if "by_channel" in d:
            by_ch = d["by_channel"]
            conditions = apply_exclusions(list(by_ch.keys()), exclude_patterns)
            if not conditions:
                L.warning(f"{p.name}: all conditions excluded.");
                continue
            lkey, ukey = detect_band_keys(by_ch[conditions[0]], lower_key, upper_key)
            for cond in conditions:
                bd = by_ch[cond]
                # lower
                vals_ch = {lbl: bd.get(lkey, {}).get(lbl, []) for lbl in labels}
                agg = aggregate_by_region_by_subject(vals_ch, labels, hemis, regions, exclude_midline=False)
                for hem in HEMS:
                    for reg in REGIONS:
                        low_one[hem][reg][cond] = agg[hem][reg]
                # upper
                vals_ch = {lbl: bd.get(ukey, {}).get(lbl, []) for lbl in labels}
                agg = aggregate_by_region_by_subject(vals_ch, labels, hemis, regions, exclude_midline=False)
                for hem in HEMS:
                    for reg in REGIONS:
                        up_one[hem][reg][cond] = agg[hem][reg]
            all_conditions |= set(conditions)
            L.debug(f"{p.name}: by_channel with keys {lkey}/{ukey}")

        elif "bands" in d:
            bands = d["bands"]
            conditions = apply_exclusions(list(bands.keys()), exclude_patterns)
            if not conditions:
                L.warning(f"{p.name}: all conditions excluded.");
                continue
            lkey, ukey = detect_band_keys(bands[conditions[0]], lower_key, upper_key)

            # optional channel_order (may be subset)
            ch_order = d.get("channel_order")
            if ch_order is not None:
                name_to_idx = {name:i for i,name in enumerate(ch_order)}
                idxs = [name_to_idx[lbl] for lbl in labels if lbl in name_to_idx]
                if not idxs:
                    raise ValueError(f"{p.name}: None of montage labels in channel_order")
                need_len = max(idxs) + 1
            else:
                idxs = list(range(len(labels)))
                need_len = len(labels)

            for cond in conditions:
                bd = bands[cond]
                # lower
                vfull = np.asarray(bd.get(lkey, []), float)
                if vfull.size < need_len:
                    raise ValueError(f"{p.name}:{cond}/{lkey} len={vfull.size} < {need_len}")
                vec = vfull[idxs]
                reg_arrays = aggregate_from_vectors(vec, labels, hemis, regions, exclude_midline=False)
                for hem in HEMS:
                    for reg in REGIONS:
                        low_one[hem][reg][cond] = np.array([np.nanmean(reg_arrays[hem][reg])], float)
                # upper
                vfull = np.asarray(bd.get(ukey, []), float)
                if vfull.size < need_len:
                    raise ValueError(f"{p.name}:{cond}/{ukey} len={vfull.size} < {need_len}")
                vec = vfull[idxs]
                reg_arrays = aggregate_from_vectors(vec, labels, hemis, regions, exclude_midline=False)
                for hem in HEMS:
                    for reg in REGIONS:
                        up_one[hem][reg][cond] = np.array([np.nanmean(reg_arrays[hem][reg])], float)
            all_conditions |= set(conditions)
            L.debug(f"{p.name}: bands with keys {lkey}/{ukey}")
        else:
            raise ValueError(f"{p} must contain 'by_channel' or 'bands'.")

        _append(lower_lists, low_one)
        _append(upper_lists, up_one)

    # stitch lists → arrays
    lower_data = {hem:{reg:{} for reg in REGIONS} for hem in HEMS}
    upper_data = {hem:{reg:{} for reg in REGIONS} for hem in HEMS}
    conditions = sorted(all_conditions)
    for hem in HEMS:
        for reg in REGIONS:
            for cond in conditions:
                lower_data[hem][reg][cond] = _concat_1d(lower_lists[hem][reg].get(cond, []))
                upper_data[hem][reg][cond] = _concat_1d(upper_lists[hem][reg].get(cond, []))
    L.info(f"Combined subjects per condition: { {c:int(np.nanmax([lower_data['LH'][r].get(c, np.array([])).size for r in REGIONS] + [0])) for c in conditions} }")

    # merges
    merge_map = parse_merge_specs(merges, conditions)
    include_unmerged = not only_merged
    if merge_map:
        lower_data, new_conds = apply_merges(lower_data, conditions, merge_map, include_unmerged)
        upper_data, _         = apply_merges(upper_data, conditions, merge_map, include_unmerged)
        conditions = new_conds
        L.info(f"Merged conditions → {conditions}")

    # layout & render
    if layout == "hemi":
        out1 = str(Path(out_dir) / "lower_alpha_lines.png")
        out2 = str(Path(out_dir) / "upper_alpha_lines.png")
        plot_lines_hemi(lower_data, conditions, ymode, xform, out1, L)
        plot_lines_hemi(upper_data, conditions, ymode, xform, out2, L)
    else:
        lower_c = collapse_hemis(lower_data)
        upper_c = collapse_hemis(upper_data)
        out1 = str(Path(out_dir) / "lower_alpha_lines_collapsed.png")
        out2 = str(Path(out_dir) / "upper_alpha_lines_collapsed.png")
        plot_lines_collapsed(lower_c, conditions, ymode, xform, out1, "8–10 Hz", L)
        plot_lines_collapsed(upper_c, conditions, ymode, xform, out2, "10–12 Hz", L)

    L.info("TRP plotting finished.")
