import os, json, traceback
import numpy as np
from typing import Dict, Any, List, Tuple
from eegspec.base import BaseApp
from eegspec.utils import load_subject_tasks_json, list_subject_jsons, subject_id_from_path, resolve_channels, save_json
from eegspec.psd import compute_psd_welch
from eegspec.features import bandpower, spectral_entropy, spectral_moments, spectral_edge, median_frequency
from eegspec.iaf import estimate_iaf
from eegspec.faa import faa_from_psd
from eegspec.trp import trp_from_bandpowers
from eegspec.bands import DEFAULT_BANDS


def run_task_compute(subject_id: str, task_name: str, data_txc: np.ndarray, sfreq: float,
                     nperseg: int, noverlap: int, window: str,
                     ch_names: List[str], alpha_band: Tuple[float, float],
                     use_db_faa: bool, out_dir: str,
                     log_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    app = BaseApp(**log_kwargs)
    try:
        app.logger.info(f"[Task start] subject={subject_id} task={task_name} shape={data_txc.shape}")
        freqs, psd = compute_psd_welch(data_txc, sfreq=sfreq, nperseg=nperseg, noverlap=noverlap, window=window)
        std_bands = DEFAULT_BANDS
        bp_abs = bandpower(psd, freqs, std_bands, relative=False)
        bp_rel = bandpower(psd, freqs, std_bands, relative=True, total_range=(1.0, 45.0))
        ent = spectral_entropy(psd, freqs, fmin=1.0, fmax=45.0, log_base=np.e)
        moms = spectral_moments(psd, freqs, fmin=1.0, fmax=45.0)
        sef95 = spectral_edge(psd, freqs, percent=0.95, fmin=1.0, fmax=45.0)
        f50 = median_frequency(psd, freqs, fmin=1.0, fmax=45.0)
        iaf = estimate_iaf(psd, freqs, fmin=alpha_band[0], fmax=alpha_band[1], smooth=True)
        faa_val = faa_from_psd(psd, freqs, ch_names, left="F3", right="F4", alpha=alpha_band, use_db=use_db_faa)

        subj_dir = os.path.join(out_dir, "subjects", subject_id)
        os.makedirs(subj_dir, exist_ok=True)
        psd_path = os.path.join(subj_dir, f"psd_{task_name}.json")
        metrics_path = os.path.join(subj_dir, f"metrics_{task_name}.json")

        save_json({"freqs": freqs.tolist(), "psd": psd.tolist(), "channels": ch_names}, psd_path)

        metrics = {
            "subject": subject_id,
            "task": task_name,
            "bands_abs": {k: v.tolist() for k, v in bp_abs.items()},
            "bands_rel": {k: v.tolist() for k, v in bp_rel.items()},
            "entropy": ent.tolist(),
            "moments": {k: v.tolist() for k, v in moms.items()},
            "SEF95": sef95.tolist(),
            "F50": f50.tolist(),
            "IAF": iaf.tolist(),
            "FAA": faa_val,
            "alpha_band": list(alpha_band)
        }
        save_json(metrics, metrics_path)
        app.logger.info(f"[Task done] subject={subject_id} task={task_name} -> psd:{psd_path} metrics:{metrics_path}")
        return {"ok": True, "subject": subject_id, "task": task_name, "psd": psd_path, "metrics": metrics_path}
    except Exception as e:
        app.logger.error(f"[Task error] subject={subject_id} task={task_name}: {e}")
        app.logger.debug(traceback.format_exc())
        return {"ok": False, "subject": subject_id, "task": task_name, "error": str(e)}


def analyze_entry(input_path: str, sfreq: float, out_dir: str,
                  nperseg: int = 1024, noverlap: int = None, window: str = "hann",
                  channels_file: str = None,
                  n_channels: int = None,
                  alpha: str = "8,13", faa_db: bool = False,
                  trp_mode: str = "ratio", trp_baseline: str = "rest",
                  max_processors: int = 4,
                  log_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
    if log_kwargs is None:
        log_kwargs = dict(log_level="INFO", log_dir=None, log_prefix="", log_suffix="", log_percentage=None)
    app = BaseApp(**log_kwargs)
    os.makedirs(out_dir, exist_ok=True)

    try:
        subjects = list_subject_jsons(input_path)
        if not subjects:
            raise FileNotFoundError("No JSON files found under input path")
        app.logger.info(f"Found {len(subjects)} subject file(s)")
    except Exception as e:
        app.logger.error(f"Failed to enumerate input: {e}")
        raise

    schedule = []
    alpha_band = tuple(map(float, alpha.split(",")))
    for spath in subjects:
        sid = subject_id_from_path(spath)
        try:
            # Step 1: Load data (disable auto-convert here, we'll handle it based on user's n_channels)
            tasks = load_subject_tasks_json(spath, auto_convert_63_to_64=False)  # task -> (n_times, n_channels)
            
            # Step 2: Convert channels if user specified expected dimension
            # This happens BEFORE resolve_channels to ensure channel names match converted data
            if n_channels is not None:
                from eegspec.utils import convert_channels_if_needed
                tasks = convert_channels_if_needed(tasks, expected_n_channels=n_channels, logger=app.logger)
            
            # Step 3: Get channel count AFTER conversion (if any)
            sample_task = next(iter(tasks.values()))
            n_ch = sample_task.shape[1]  # n_channels after potential conversion
            
            # Step 4: Resolve channel names based on FINAL channel count
            # This ensures channels match the converted data
            ch_names = resolve_channels(channels_file, n_channels=n_ch)
            if len(ch_names) != n_ch:
                app.logger.warning(f"Channel count mismatch for {sid}: using placeholders Ch1..Ch{n_ch}")
            for tname, data in tasks.items():
                schedule.append((sid, tname, data, ch_names))
        except Exception as e:
            app.logger.error(f"Failed to parse subject {sid}: {e}")

    summary = {"subjects": {}, "alpha": list(alpha_band), "sfreq": sfreq, "nperseg": nperseg, "window": window}
    total = len(schedule)
    app.logger.info(f"Total tasks to run: {total} (max_processors={max_processors})")
    if total == 0:
        save_json(summary, os.path.join(out_dir, "summary.json"))
        return summary

    import concurrent.futures
    idx = 0
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processors) as ex:
        while idx < total and len(futures) < max_processors:
            sid, tname, data, ch = schedule[idx];
            idx += 1
            task_log_kwargs = dict(log_kwargs);
            task_log_kwargs["log_suffix"] = f"_{sid}_{tname}"
            fut = ex.submit(run_task_compute, sid, tname, data, sfreq, nperseg, noverlap, window, ch, alpha_band,
                            faa_db, out_dir, task_log_kwargs)
            futures.append(fut)
        done_count = 0
        while futures:
            for fut in concurrent.futures.as_completed(futures, timeout=None):
                try:
                    res = fut.result()
                except Exception as e:
                    app.logger.error(f"Worker crashed: {e}")
                    res = {"ok": False, "error": str(e)}
                if res.get("ok"):
                    sid = res["subject"];
                    tname = res["task"]
                    summary.setdefault("subjects", {}).setdefault(sid, {})[tname] = {k: res[k] for k in
                                                                                     ("psd", "metrics") if k in res}
                else:
                    app.logger.error(f"Task failed: {res}")
                done_count += 1
                futures.remove(fut)
                if idx < total:
                    sid, tname, data, ch = schedule[idx];
                    idx += 1
                    task_log_kwargs = dict(log_kwargs);
                    task_log_kwargs["log_suffix"] = f"_{sid}_{tname}"
                    futures.append(
                        ex.submit(run_task_compute, sid, tname, data, sfreq, nperseg, noverlap, window, ch, alpha_band,
                                  faa_db, out_dir, task_log_kwargs))
                app.logger.info(f"Progress: {done_count}/{total}")
                break

    summ_path = os.path.join(out_dir, "summary.json")

    # === Post-process: compute TRP vs baseline for each subject ===
    try:
        for sid, tasks_map in summary.get("subjects", {}).items():
            # Load bands_abs for each task
            band_abs_by_task = {}
            for tname, paths in tasks_map.items():
                try:
                    with open(paths["metrics"], "r", encoding="utf-8") as f:
                        m = json.load(f)
                    band_abs_by_task[tname] = {k: np.asarray(v) for k, v in m["bands_abs"].items()}
                except Exception as e:
                    app.logger.warning(f"TRP: skip {sid}/{tname}: {e}")

            if not band_abs_by_task:
                continue

            # Choose baseline
            base_name = trp_baseline
            if base_name is None:
                # heuristics
                for cand in ["rest", "baseline", "eyes_closed", "ec", "eo", "eyes_open"]:
                    if cand in band_abs_by_task:
                        base_name = cand;
                        break
                if base_name is None:
                    base_name = sorted(band_abs_by_task.keys())[0]
            if base_name not in band_abs_by_task:
                app.logger.warning(f"TRP: baseline '{base_name}' not found for {sid}, skip")
                continue

            P_base = band_abs_by_task[base_name]
            trp_out = {"baseline": base_name, "mode": trp_mode, "bands": {}}
            for tname, bands in band_abs_by_task.items():
                if tname == base_name:
                    continue
                trp_out["bands"][tname] = {}
                for band, P_task in bands.items():
                    try:
                        val = trp_from_bandpowers(P_base[band], P_task, mode=trp_mode)
                        trp_out["bands"][tname][band] = val.tolist()
                    except Exception as e:
                        app.logger.warning(f"TRP: fail {sid}/{tname}/{band}: {e}")

            # Save TRP file
            subj_dir = os.path.join(out_dir, "subjects", sid)
            os.makedirs(subj_dir, exist_ok=True)
            trp_path = os.path.join(subj_dir, f"trp_{base_name}.json")
            with open(trp_path, "w", encoding="utf-8") as f:
                json.dump(trp_out, f, ensure_ascii=False, separators=(",", ":"))
            summary["subjects"][sid]["TRP"] = trp_path
            app.logger.info(f"TRP written: {trp_path}")
    except Exception as e:
        app.logger.warning(f"TRP post-process error: {e}")

    save_json(summary, summ_path)
    app.logger.info(f"Summary written: {summ_path}")
    return summary
