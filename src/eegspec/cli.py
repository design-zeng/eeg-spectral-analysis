import argparse, sys, os, traceback
from eegspec.base import BaseApp
from eegspec.analyze import analyze_entry
from eegspec.plot_trp import plot_trp_entry
from eegspec.design_creativity import design_creativity_entry


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
    sp = sub.add_parser("design-creativity", help="Design creativity analysis using wPLI connectivity and graph features (Strength, Betweenness) with classification (SVM, MLP, KNN).")
    sp.add_argument("--input", required=True, help="Folder with many subject.json/.mat files OR a single subject.json/.mat file")
    sp.add_argument("--sfreq", type=float, required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--channels-file", type=str, default=None, help="Plain text, .csv or .locs")
    sp.add_argument("--fmin", type=float, default=1.0, help="Minimum frequency for connectivity (default: 1.0 Hz)")
    sp.add_argument("--fmax", type=float, default=45.0, help="Maximum frequency for connectivity (default: 45.0 Hz)")
    sp.add_argument("--epoch-sec", type=float, default=2.0, help="Epoch length in seconds (default: 2.0)")
    sp.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction between epochs (default: 0.5)")
    sp.add_argument("--threshold", type=float, default=None, help="Threshold for binarizing connectivity in betweenness computation")
    sp.add_argument("--max-processors", type=int, default=4)
    sp.add_argument("--no-classification", action="store_true", help="Skip classification step")
    add_logging_args(sp)
    sp.set_defaults(func=cmd_design_creativity)

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
        design_creativity_entry(
            input_path=args.input,
            sfreq=args.sfreq,
            out_dir=args.out_dir,
            channels_file=args.channels_file,
            fmin=args.fmin,
            fmax=args.fmax,
            epoch_sec=args.epoch_sec,
            overlap=args.overlap,
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


if __name__ == "__main__":
    debug_args = [
        "analyze",
        "--input",
        r"D:\EEG-signals-respond-differently-to-three-modes-of-thinking-in-a-loosely-controlled-experiment\clean_data\sub_01.json",
        "--sfreq", "500",
        "--out-dir", r"D:\EEG\out",
        "--nperseg", "1024",
        "--noverlap", "512",
        "--window", "hann",
        "--trp-mode", "logratio",
        "--trp-baseline", "1_rest",
        "--alpha", "8,13",
        "--faa-db",
        "--max-processors", "8",
        "--log-level", "DEBUG",
        "--log-dir", r"D:\EEG\out\.logs",
        "--log-prefix", "run_",
        "--log-suffix", "_alpha",
        "--log-percentage", "0.8",
    ]
    plt_debug_args = [
        "plot-trp",
        "--summary-dir", r"D:\EEG\out\subjects",
        "--summary-glob", "trp_1_rest.json",
        "--out-dir", r"D:\EEG\out\group",
        "--layout", "collapsed",
        "--exclude-midline",
        "--mode", "logratio",
        "--lower-key", "lower_alpha", "--upper-key", "upper_alpha",
        "--merge", "idea generation=1_idea generation,2_idea generation,3_idea generation",
        "--merge", "idea rating=1_idea rating,2_idea rating,3_idea rating",
        "--merge", "idea evolution=1_idea evolution,2_idea evolution,3_idea evolution",
        "--exclude-condition", "3_rest",
        "--only-merged",
        "--log-level", "INFO",
        "--log-dir", r"D:\EEG\out\.logs",
        "--log-prefix", "plot_",
        "--log-suffix", "_trp",
    ]
    sys.exit(main(plt_debug_args))
