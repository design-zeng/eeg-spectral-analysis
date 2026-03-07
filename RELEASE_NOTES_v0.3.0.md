# Release v0.3.0

## Summary

This release brings full English localization, a new **statistical-analysis** CLI for ANOVA and post-hoc tests on design creativity features, and compatibility fixes.

## What's New

### Statistical Analysis CLI

Run ANOVA and Tukey HSD post-hoc comparisons on design creativity results:

```bash
eegspec statistical-analysis --results-dir /path/to/results [--out-dir /path] [--alpha 0.05] [--pairwise tukey|bonferroni]
```

- **Feature selection**: ANOVA F-test (p < 0.05)
- **ANOVA analysis**: F-statistics, p-values, partial eta squared for selected features
- **Pairwise comparisons**: Tukey HSD or Bonferroni correction

### English Localization

- All source code (docstrings, comments, log messages) translated to English
- Examples: `eeg_channel_conversion.py`, `example_usage.py`, `test_conversion.py`
- Documentation: `EEG_Channel_Conversion_Documentation.md`, examples `README.md`
- Replaced `examples/README_CN.md` with `examples/README.md` (English)

### Compatibility Fixes

- **Tukey HSD**: Fixed statsmodels `pairindices` attribute compatibility (uses `itertools.combinations` for pairwise group mapping)
- **MATLAB compatibility**: Strength ~1.0, Betweenness ~0.87 correlation with reference implementation

## Documentation

- **docs/PRECISION_GUARANTEE.md** — 64-bit floating-point precision guarantees for wPLI, Strength, Betweenness
- **docs/README.md** — Documentation index

## Installation

```bash
pip install -e .
```

Requires: Python ≥ 3.9, NumPy, SciPy, MNE, Matplotlib, NetworkX, scikit-learn, statsmodels.

## Full Changelog (v0.3.0)

- Add `eegspec statistical-analysis` subcommand
- Add `statistical_analysis.py` module (standalone ANOVA + Tukey)
- Add `statistics.py` (feature selection, ANOVA, pairwise comparisons)
- Fix Tukey HSD `pairindices` compatibility in `statistics.py`
- Translate `statistical_analysis.py` to English
- Translate all examples and channel conversion docs to English
- Add `docs/PRECISION_GUARANTEE.md`, `docs/README.md`
- Update `.gitignore` for analysis artifacts
- Remove `examples/README_CN.md` (replaced by English README)
