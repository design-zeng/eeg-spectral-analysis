import argparse, sys, os, traceback

from eegspec.base import BaseApp
from eegspec.analyze import analyze_entry
from eegspec.plot_trp import plot_trp_entry
from eegspec.design_creativity import design_creativity_entry
from eegspec.statistical_analysis import run_statistical_analysis
from eegspec.dynamic_window import dynamic_window_entry


def main(argv=None):
    try:
        return _main_impl(argv)
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


def _main_impl(argv=None):
    p = argparse.ArgumentParser(prog="eegspec", description="EEG spectral analysis toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_logging_args(sp):
        sp.add_argument("--log-level", type=str, default="INFO")
        sp.add_argument("--log-dir", type=str, default=None)
        sp.add_argument("--log-prefix", type=str, default="")
        sp.add_argument("--log-suffix", type=str, default="")
        sp.add_argument("--log-percentage", type=float, default=None)

    sp = sub.add_parser("analyze", help="Analyze a folder of subject JSONs/MAT files or a single subject file (task-centric).")
    sp.add_argument("--input", required=True, help="Folder with many subject.json/.mat files OR a single subject.json/.mat file")
    sp.add_argument("--sfreq", type=float, required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--nperseg", type=int, default=1024)
    sp.add_argument("--noverlap", type=int, default=None)
    sp.add_argument("--window", type=str, default="hann")
    sp.add_argument("--channels-file", type=str, default=None, help="Plain text, .csv or .locs")
    sp.add_argument("--n-channels", type=int, default=None, help="Expected number of channels. If specified and differs from data, conversion will be performed (e.g., 63→64)")
    sp.add_argument("--alpha", type=str, default="8,13")
    sp.add_argument("--trp-baseline", type=str, default="1_rest", help="Baseline task name for TRP (default: 1_rest)")
    sp.add_argument("--trp-mode", type=str, default="logratio", help="TRP output mode")
    sp.add_argument("--faa-db", action="store_true")
    sp.add_argument("--max-processors", type=int, default=4)
    add_logging_args(sp)
    sp.set_defaults(func=cmd_analyze)

    # -------- NEW 'plot-trp' --------
    sp = sub.add_parser("plot-trp", help="Plot group TRP line charts from one or many TRP summaries.")
    sp.add_argument("--summary", action="append", default=[],
                    help="TRP summary JSON file (repeatable)")
    sp.add_argument("--summary-dir", default=None,
                    help="Directory to search for TRP JSONs recursively")
    sp.add_argument("--summary-glob", default="trp_1_rest.json",
                    help="Filename pattern under --summary-dir (default: trp_1_rest.json)")
    sp.add_argument("--montage", default=None,
                    help="Montage .locs file; default: built-in caps63")
    sp.add_argument("--out-dir", required=True,
                    help="Output directory for figures")
    sp.add_argument("--layout", default="collapsed", choices=["hemi", "collapsed"],
                    help="hemi: LH/RH split; collapsed: hemis merged")
    sp.add_argument("--mode", default="logratio",
                    choices=["ratio", "db", "log", "log10", "logratio", "log-ratio"],
                    help="Axis labeling / value family")
    sp.add_argument("--transform", default="auto",
                    choices=["auto", "none", "db", "log", "log10"],
                    help="Numeric transform strategy (auto recommended)")
    sp.add_argument("--exclude-midline", action="store_true",
                    help="Exclude 'z' electrodes")
    sp.add_argument("--lower-key", default=None, help="Lower alpha key (e.g., 8_10 or lower_alpha)")
    sp.add_argument("--upper-key", default=None, help="Upper alpha key (e.g., 10_12 or upper_alpha)")
    sp.add_argument("--merge", action="append", default=[],
                    help='Merging rule; can repeat. Example: --merge "idea generation=1_idea generation,2_idea generation,3_idea generation"')
    sp.add_argument("--only-merged", action="store_true",
                    help="Keep only merged groups and drop others")
    sp.add_argument("--exclude-condition", action="append", default=[],
                    help='Exclude conditions by wildcard; can repeat. Example: --exclude-condition "3_rest" or "*_rest"')
    add_logging_args(sp)
    sp.set_defaults(func=cmd_plot_trp)

    # -------- NEW 'design-creativity' --------
    sp = sub.add_parser("design-creativity", help="Design creativity analysis using wPLI connectivity and graph features (Strength, Betweenness) with classification (SVM, MLP, KNN). Matches MATLAB Connectivity_Analysis.m exactly.")
    sp.add_argument("--input", required=True, help="Input path: folder containing Data_Creativity_Sub_*.mat files (Creativity_EEG_Dataset) OR folder with subject.json/.mat files OR single file")
    sp.add_argument("--sfreq", type=float, default=500.0, help="Sampling frequency (Hz), default: 500.0")
    sp.add_argument("--out-dir", required=True, help="Output directory for results")
    sp.add_argument("--channels-file", type=str, default=None, help="Channel names file (.locs, .txt, .csv). If not provided, uses built-in caps63.locs")
    sp.add_argument("--n-channels", type=int, default=None, help="Expected number of channels. If specified and differs from data, conversion will be performed (e.g., 63→64)")
    sp.add_argument("--freq-range", type=str, default="8,13", help="Frequency range for filtering as 'low,high' (Hz), default: '8,13' (alpha band)")
    sp.add_argument("--threshold", type=float, default=0.2, help="Threshold for filtering weak connections in betweenness computation (default: 0.2)")
    sp.add_argument("--max-processors", type=int, default=4, help="Maximum number of parallel workers (default: 4)")
    sp.add_argument("--no-classification", action="store_true", help="Skip classification step if only features are needed")
    add_logging_args(sp)
    sp.set_defaults(func=cmd_design_creativity)

    # -------- statistical-analysis --------
    sp = sub.add_parser("statistical-analysis", help="Run ANOVA and pairwise comparisons on design creativity results")
    sp.add_argument("--results-dir", required=True, help="Directory containing subjects/ with design_creativity_*.json files")
    sp.add_argument("--out-dir", default=None, help="Output directory (default: same as results-dir)")
    sp.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    sp.add_argument("--pairwise", type=str, default="tukey", choices=["tukey", "bonferroni"])
    add_logging_args(sp)
    sp.set_defaults(func=cmd_statistical_analysis)

    # -------- segment-protocol (design protocol segmentation pipeline) --------
    sp = sub.add_parser(
        "segment-protocol",
        help="Segment continuous EEG-derived microstate labels into subtasks (hardness → cluster → merge).",
    )
    sp.add_argument("--sfreq", type=float, required=True, help="Sampling rate (Hz)")
    sp.add_argument("--out-dir", required=True, help="Output directory for segmentation.json")
    sp.add_argument("--labels-npy", type=str, default=None, help="Path to (T,) int microstate labels .npy")
    sp.add_argument("--eeg-npy", type=str, default=None, help="Path to (T, C) float EEG .npy; requires microstate_analysis")
    sp.add_argument(
        "--line",
        type=str,
        choices=["none", "A", "B"],
        default="none",
        help="Preset: A=high-sensitivity (macro W=1), B=coarse (macro W=5, stride=5); default=manual config",
    )
    sp.add_argument("--group-sec", type=float, default=5.0, help="Fine group length in seconds")
    sp.add_argument(
        "--macro-window-size",
        type=int,
        default=None,
        help="Macro window in fine groups (default: preset A=1, B=5, none=1)",
    )
    sp.add_argument(
        "--macro-window-stride",
        type=int,
        default=None,
        help="Macro stride in fine groups (default: same as macro window, or 1 for preset none)",
    )
    sp.add_argument("--n-clusters", type=int, default=4, help="KMeans cluster count")
    sp.add_argument("--min-subtask-sec", type=float, default=10.0, help="Merge subtasks shorter than this (s)")
    sp.add_argument("--smooth-window-w", type=int, default=3, help="Sliding majority half-window scale")
    sp.add_argument("--min-state-duration-ms", type=float, default=80.0, help="Merge label runs shorter than this")
    sp.add_argument("--lambda-reg", type=float, default=0.25, help="Reserved regularization weight")
    sp.add_argument("--n-microstates", type=int, default=4, help="K for microstate_analysis bridge")
    sp.add_argument("--ms-n-runs", type=int, default=25, help="k-means modified runs (bridge)")
    add_logging_args(sp)
    sp.set_defaults(func=cmd_segment_protocol)

    # -------- dynamic-window (microstate-driven dynamic spectral windows) --------
    sp = sub.add_parser(
        "dynamic-window",
        help="Compute spectral metrics on dynamic windows: microstate runs or protocol subtasks.",
    )
    sp.add_argument("--mode", choices=["run", "subtask"], required=True,
                    help="run=A方案：正则化微状态run作窗；subtask=B方案：protocol subtask作窗")
    sp.add_argument("--sfreq", type=float, required=True, help="Sampling rate (Hz)")
    sp.add_argument("--out-dir", required=True, help="Output directory for dynamic_windows/metrics JSON")
    sp.add_argument(
        "--labels-npy",
        default=None,
        help="Path to (T,) int microstate labels .npy. If omitted, labels are fitted from EEG and saved.",
    )
    sp.add_argument("--eeg-npy", default=None, help="Path to (T, C) EEG .npy")
    sp.add_argument("--input", default=None, help="Task-centric JSON/MAT subject file or folder; alternative to --eeg-npy")
    sp.add_argument("--task", default=None, help="Task name to read from --input (default: first task)")
    sp.add_argument("--channels-file", type=str, default=None, help="Plain text, .csv or .locs")
    sp.add_argument("--n-channels", type=int, default=None, help="Expected channel count for --input")
    sp.add_argument("--nperseg", type=int, default=1024, help="Requested Welch segment length")
    sp.add_argument("--noverlap", type=int, default=None, help="Requested Welch overlap")
    sp.add_argument("--window", type=str, default="hann", help="Welch window function")
    sp.add_argument("--alpha", type=str, default="8,13", help="Alpha band for IAF/FAA")
    sp.add_argument("--faa-db", action="store_true", help="Compute FAA as dB difference")
    sp.add_argument("--min-window-sec", type=float, default=1.0,
                    help="Drop dynamic windows shorter than this duration")
    sp.add_argument("--smooth-window-w", type=int, default=3, help="Microstate label smoothing scale")
    sp.add_argument("--min-state-duration-ms", type=float, default=80.0,
                    help="Merge microstate runs shorter than this duration before windowing")
    sp.add_argument("--line", choices=["none", "A", "B"], default="B",
                    help="Only for --mode subtask: protocol preset")
    sp.add_argument("--group-sec", type=float, default=5.0,
                    help="Only for --mode subtask: fine group length")
    sp.add_argument("--macro-window-size", type=int, default=None,
                    help="Only for --mode subtask: macro window in fine groups")
    sp.add_argument("--macro-window-stride", type=int, default=None,
                    help="Only for --mode subtask: macro stride in fine groups")
    sp.add_argument("--n-clusters", type=int, default=4,
                    help="Only for --mode subtask: KMeans cluster count")
    sp.add_argument("--min-subtask-sec", type=float, default=10.0,
                    help="Only for --mode subtask: merge subtasks shorter than this")
    sp.add_argument("--n-microstates", type=int, default=4,
                    help="When --labels-npy is omitted: K for microstate fitting")
    sp.add_argument("--ms-n-runs", type=int, default=25,
                    help="When --labels-npy is omitted: modified k-means runs")
    sp.add_argument("--ms-n-std", type=int, default=3,
                    help="When --labels-npy is omitted: GFP peak std threshold")
    sp.add_argument("--ms-distance", type=int, default=10,
                    help="When --labels-npy is omitted: fit_back distance")
    sp.add_argument("--ms-all-samples", action="store_true",
                    help="When --labels-npy is omitted: fit maps from all samples instead of GFP peaks")
    sp.add_argument("--save-psd", action="store_true", help="Also write per-window PSD arrays")
    add_logging_args(sp)
    sp.set_defaults(func=cmd_dynamic_window)

    args = p.parse_args(argv)
    return args.func(args)


def cmd_analyze(args):
    app = BaseApp(log_level=args.log_level, log_dir=args.log_dir, log_prefix=args.log_prefix,
                  log_suffix=args.log_suffix, log_percentage=args.log_percentage)
    app.logger.info("Analyze entry")
    try:
        analyze_entry(
            input_path=args.input,
            sfreq=args.sfreq,
            out_dir=args.out_dir,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            window=args.window,
            channels_file=args.channels_file,
            n_channels=args.n_channels,
            alpha=args.alpha,
            faa_db=args.faa_db,
            trp_baseline=args.trp_baseline,
            trp_mode=args.trp_mode,
            max_processors=args.max_processors,
            log_kwargs=dict(log_level=args.log_level, log_dir=args.log_dir, log_prefix=args.log_prefix,
                            log_suffix=args.log_suffix, log_percentage=args.log_percentage),
        )
        app.logger.info(f"Wrote summary to {os.path.join(args.out_dir, 'summary.json')}")
    except Exception as e:
        app.logger.error(f"Analyze failed: {e}")
        app.logger.debug(traceback.format_exc())
        raise


def cmd_plot_trp(args):
    app = BaseApp(log_level=args.log_level, log_dir=args.log_dir, log_prefix=args.log_prefix,
                  log_suffix=args.log_suffix, log_percentage=args.log_percentage)
    app.logger.info("Plot-TRP entry")
    try:
        plot_trp_entry(
            summary=args.summary,
            summary_dir=args.summary_dir,
            summary_glob=args.summary_glob,
            montage=args.montage,
            out_dir=args.out_dir,
            layout=args.layout,
            mode=args.mode,
            transform=args.transform,
            exclude_midline=args.exclude_midline,
            lower_key=args.lower_key,
            upper_key=args.upper_key,
            merges=args.merge,
            only_merged=args.only_merged,
            exclude_patterns=args.exclude_condition,
            log_kwargs=dict(log_level=args.log_level, log_dir=args.log_dir, log_prefix=args.log_prefix,
                            log_suffix=args.log_suffix, log_percentage=args.log_percentage),
        )
        app.logger.info(f"TRP figures written to {args.out_dir}")
    except Exception as e:
        app.logger.error(f"Plot-TRP failed: {e}")
        app.logger.debug(traceback.format_exc())
        raise


def cmd_design_creativity(args):
    app = BaseApp(log_level=args.log_level, log_dir=args.log_dir, log_prefix=args.log_prefix,
                  log_suffix=args.log_suffix, log_percentage=args.log_percentage)
    app.logger.info("Design Creativity entry")
    try:
        # Parse frequency range
        freq_range_parts = args.freq_range.split(",")
        if len(freq_range_parts) != 2:
            raise ValueError(f"Invalid freq-range format: {args.freq_range}. Expected 'low,high' (e.g., '8,13')")
        freq_range = (float(freq_range_parts[0].strip()), float(freq_range_parts[1].strip()))
        
        design_creativity_entry(
            input_path=args.input,
            sfreq=args.sfreq,
            out_dir=args.out_dir,
            channels_file=args.channels_file,
            n_channels=args.n_channels,
            freq_range=freq_range,
            threshold=args.threshold,
            max_processors=args.max_processors,
            run_classification=not args.no_classification,
            log_kwargs=dict(log_level=args.log_level, log_dir=args.log_dir, log_prefix=args.log_prefix,
                            log_suffix=args.log_suffix, log_percentage=args.log_percentage),
        )
        app.logger.info(f"Design creativity analysis written to {args.out_dir}")
    except Exception as e:
        app.logger.error(f"Design Creativity failed: {e}")
        app.logger.debug(traceback.format_exc())
        raise


def cmd_segment_protocol(args):
    from microstate_analysis.protocol_segmentation.runner import run_segment_protocol_main

    app = BaseApp(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_prefix=args.log_prefix,
        log_suffix=args.log_suffix,
        log_percentage=args.log_percentage,
    )
    try:
        out_json = run_segment_protocol_main(
            sfreq=float(args.sfreq),
            out_dir=args.out_dir,
            labels_npy=args.labels_npy,
            eeg_npy=args.eeg_npy,
            line=args.line,
            group_sec=args.group_sec,
            macro_window_size=args.macro_window_size,
            macro_window_stride=args.macro_window_stride,
            n_clusters=args.n_clusters,
            min_subtask_sec=args.min_subtask_sec,
            smooth_window_w=args.smooth_window_w,
            min_state_duration_ms=args.min_state_duration_ms,
            lambda_reg=args.lambda_reg,
            n_microstates=args.n_microstates,
            ms_n_runs=args.ms_n_runs,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    app.logger.info(f"Wrote {out_json}")


def cmd_dynamic_window(args):
    app = BaseApp(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_prefix=args.log_prefix,
        log_suffix=args.log_suffix,
        log_percentage=args.log_percentage,
    )
    try:
        summary = dynamic_window_entry(
            out_dir=args.out_dir,
            sfreq=args.sfreq,
            labels_npy=args.labels_npy,
            mode=args.mode,
            eeg_npy=args.eeg_npy,
            input_path=args.input,
            task=args.task,
            channels_file=args.channels_file,
            n_channels=args.n_channels,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            window=args.window,
            alpha=args.alpha,
            faa_db=args.faa_db,
            min_window_sec=args.min_window_sec,
            smooth_window_w=args.smooth_window_w,
            min_state_duration_ms=args.min_state_duration_ms,
            line=args.line,
            group_sec=args.group_sec,
            macro_window_size=args.macro_window_size,
            macro_window_stride=args.macro_window_stride,
            n_clusters=args.n_clusters,
            min_subtask_sec=args.min_subtask_sec,
            n_microstates=args.n_microstates,
            ms_n_runs=args.ms_n_runs,
            ms_n_std=args.ms_n_std,
            ms_distance=args.ms_distance,
            ms_peaks_only=not args.ms_all_samples,
            save_psd=args.save_psd,
        )
    except Exception as e:
        app.logger.error(f"Dynamic-window failed: {e}")
        app.logger.debug(traceback.format_exc())
        raise
    app.logger.info(
        f"Dynamic-window complete: mode={summary['mode']} "
        f"n_windows={summary['n_windows']} n_skipped={summary['n_skipped']} "
        f"out={args.out_dir}"
    )


def cmd_statistical_analysis(args):
    app = BaseApp(log_level=args.log_level, log_dir=args.log_dir, log_prefix=args.log_prefix,
                  log_suffix=args.log_suffix, log_percentage=args.log_percentage)
    app.logger.info("Statistical analysis entry")
    try:
        run_statistical_analysis(
            results_dir=args.results_dir,
            out_dir=args.out_dir,
            alpha=args.alpha,
            pairwise_method=args.pairwise,
            log_kwargs=dict(log_level=args.log_level, log_dir=args.log_dir, log_prefix=args.log_prefix,
                            log_suffix=args.log_suffix, log_percentage=args.log_percentage),
        )
        app.logger.info("Statistical analysis complete")
    except Exception as e:
        app.logger.error(f"Statistical analysis failed: {e}")
        app.logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
