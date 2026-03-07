"""
Standalone statistical analysis module.

Reads features from design creativity analysis results and runs statistical analysis:
1. Feature selection (ANOVA F-test)
2. ANOVA analysis
3. Pairwise comparisons (Tukey HSD / Bonferroni)
"""

import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from eegspec.base import BaseApp
from eegspec.utils import save_json
from eegspec.design_creativity import prepare_classification_data, _ensure_float64_in_dict
from eegspec.statistics import statistical_analysis_pipeline


def run_statistical_analysis(
    results_dir: str,
    out_dir: Optional[str] = None,
    k: Optional[int] = None,
    alpha: float = 0.05,
    pairwise_method: str = 'tukey',
    task_mapping: Optional[Dict[str, str]] = None,
    log_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run statistical analysis on design creativity analysis results.

    Reads feature data from results_dir and performs:
    1. Feature selection (ANOVA F-test)
    2. ANOVA analysis
    3. Pairwise comparisons (Tukey HSD / Bonferroni)

    Parameters:
    -----------
    results_dir : str
        Directory containing design creativity results (must include subjects/ subdir)
    out_dir : Optional[str]
        Output directory. If None, uses results_dir
    k : Optional[int]
        Select top k features. If None, selects all features with p < alpha
    alpha : float
        Significance level (default: 0.05)
    pairwise_method : str
        Pairwise comparison method: 'tukey' or 'bonferroni' (default: 'tukey')
    task_mapping : Optional[Dict[str, str]]
        Mapping from task names to labels. If None, uses default mapping
    log_kwargs : Optional[Dict[str, Any]]
        Logging parameters

    Returns:
    --------
    Dict[str, Any]
        Statistical analysis results
    """
    if log_kwargs is None:
        log_kwargs = {}
    
    app = BaseApp(**log_kwargs)
    logger = app.logger
    
    if out_dir is None:
        out_dir = results_dir
    
    logger.info("=" * 80)
    logger.info("Standalone Statistical Analysis Module")
    logger.info("=" * 80)
    logger.info(f"Input directory: {results_dir}")
    logger.info(f"Output directory: {out_dir}")
    
    # Prepare data
    logger.info("Preparing classification data...")
    
    # Default task mapping if not provided
    if task_mapping is None:
        task_mapping = {
            'idg': 'IDG',
            'ide': 'IDE',
            'idr': 'IDR',
            'rst': 'RST',
            'idea generation': 'IDG',
            'idea evolution': 'IDE',
            'idea rating': 'IDR',
            'rest': 'RST',
        }
    
    try:
        X, y, feature_names = prepare_classification_data(
            results_dir,
            task_mapping=task_mapping,
        )
        logger.info(f"Data prepared: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Number of features: {len(feature_names)}")
        logger.info(f"Number of samples: {len(y)}")
        logger.info(f"Cognitive states: {np.unique(y, return_counts=True)}")
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    
    # Run statistical analysis
    logger.info("Running statistical analysis pipeline...")
    try:
        stat_results = statistical_analysis_pipeline(
            X, y,
            feature_names=feature_names,
            k=k,
            alpha=alpha,
            pairwise_method=pairwise_method,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    
    # Save results
    logger.info("Saving statistical analysis results...")
    os.makedirs(out_dir, exist_ok=True)
    
    stat_path = os.path.join(out_dir, "statistical_analysis_results.json")
    
    # Convert results for JSON serialization
    stat_results_json = {
        'feature_selection': {
            'selected_features': stat_results['feature_selection']['selected_features'],
            'selected_feature_names': stat_results['feature_selection']['selected_feature_names'],
            'f_scores': stat_results['feature_selection']['f_scores'],
            'p_values': stat_results['feature_selection']['p_values'],
            'selected_f_scores': stat_results['feature_selection']['selected_f_scores'],
            'selected_p_values': stat_results['feature_selection']['selected_p_values'],
        },
        'anova': stat_results['anova'],
        'pairwise_comparisons': stat_results['pairwise_comparisons'],
        'summary': stat_results['summary'],
    }
    
    save_json(stat_results_json, stat_path)
    logger.info(f"Statistical analysis results saved to: {stat_path}")
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Statistical Analysis Summary")
    logger.info("=" * 80)
    logger.info(f"Total features: {stat_results['summary']['n_total_features']}")
    logger.info(f"Selected features: {stat_results['summary']['n_selected_features']}")
    logger.info(f"Number of groups: {stat_results['summary']['n_groups']}")
    logger.info(f"Significance level: {stat_results['summary']['alpha']}")
    logger.info(f"Pairwise method: {stat_results['summary']['pairwise_method']}")
    logger.info(f"Selected features: {stat_results['feature_selection']['selected_feature_names']}")
    
    # Print ANOVA summary
    logger.info("\nANOVA Results Summary:")
    anova = stat_results['anova']
    for i, (name, f_stat, p_val) in enumerate(zip(
        anova['feature_names'],
        anova['f_statistics'],
        anova['p_values']
    )):
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        logger.info(f"  {name}: F={f_stat:.4f}, p={p_val:.6f} {significance}")
    
    # Print pairwise comparisons summary
    logger.info("\nPairwise Comparisons Summary:")
    pairwise = stat_results['pairwise_comparisons']
    for sig_pair in pairwise['significant_pairs']:
        logger.info(f"  {sig_pair['feature_name']}: {sig_pair['n_significant_pairs']} significant pair(s)")
        if sig_pair['n_significant_pairs'] > 0:
            logger.info(f"    Significant pairs: {', '.join(sig_pair['significant_pairs'])}")
    
    logger.info("=" * 80)
    
    return {
        'results_path': stat_path,
        'statistical_analysis': stat_results_json,
        'summary': stat_results['summary'],
    }

