import argparse, sys, os, traceback
from eegspec.base import BaseApp
from eegspec.analyze import analyze_entry
from eegspec.plot_trp import plot_trp_entry
from eegspec.design_creativity import design_creativity_entry
from eegspec.statistical_analysis import run_statistical_analysis


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
