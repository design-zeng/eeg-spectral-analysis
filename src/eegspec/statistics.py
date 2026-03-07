"""
Statistical analysis module for design creativity features.

This module implements:
1. Feature selection using ANOVA F-test
2. ANOVA analysis for selected features
3. Pair-wise comparisons using post-hoc tests (Tukey HSD)
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Any
from scipy.stats import f_oneway
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    pairwise_tukeyhsd = None

from sklearn.feature_selection import f_classif, SelectKBest


def select_features_anova(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k: Optional[int] = None,
    alpha: float = 0.05,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Select significant features using ANOVA F-test."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if logger:
        logger.info(f"[Feature Selection] Input: X shape={X.shape}, y shape={y.shape}, "
                    f"unique labels={np.unique(y)}, n_features={X.shape[1]}")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if len(feature_names) != X.shape[1]:
        raise ValueError(f"Number of feature names ({len(feature_names)}) must match "
                         f"number of features ({X.shape[1]})")

    f_scores, p_values = f_classif(X, y)
    f_scores = np.asarray(f_scores, dtype=np.float64)
    p_values = np.asarray(p_values, dtype=np.float64)

    if logger:
        logger.info(f"[Feature Selection] Computed F-statistics: "
                    f"F_range=[{np.min(f_scores):.6f}, {np.max(f_scores):.6f}], "
                    f"F_mean={np.mean(f_scores):.6f}, "
                    f"p_range=[{np.min(p_values):.6f}, {np.max(p_values):.6f}], "
                    f"significant_features (p<{alpha})={np.sum(p_values < alpha)}")

    selector = None
    if k is not None:
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
    else:
        selected_indices = np.where(p_values < alpha)[0]
        selected_indices = selected_indices[np.argsort(f_scores[selected_indices])[::-1]]

    if logger:
        logger.info(f"[Feature Selection] Selected {len(selected_indices)} features: "
                    f"indices={selected_indices.tolist()}")

    selected_feature_names = [feature_names[i] for i in selected_indices]
    selected_f_scores = f_scores[selected_indices]
    selected_p_values = p_values[selected_indices]

    if logger:
        logger.info(f"[Feature Selection] Selected features: {selected_feature_names}")
        logger.info(f"[Feature Selection] Selected F-scores: {selected_f_scores.tolist()}")
        logger.info(f"[Feature Selection] Selected p-values: {selected_p_values.tolist()}")

    return {
        'selected_features': selected_indices.tolist(),
        'selected_feature_names': selected_feature_names,
        'f_scores': f_scores.tolist(),
        'p_values': p_values.tolist(),
        'selected_f_scores': selected_f_scores.tolist(),
        'selected_p_values': selected_p_values.tolist(),
        'selector': selector if k is not None else None,
    }


def compute_anova(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute one-way ANOVA for selected features with partial eta squared."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    unique_labels = np.unique(y)
    n_groups = len(unique_labels)

    if logger:
        logger.info(f"[ANOVA] Input: X shape={X.shape}, y shape={y.shape}, "
                    f"n_groups={n_groups}, groups={unique_labels.tolist()}")

    if feature_indices is None:
        feature_indices = list(range(X.shape[1]))

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    anova_results = []
    group_means_list = []
    group_stds_list = []

    for idx in feature_indices:
        feature_data = X[:, idx]
        feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"

        groups = [feature_data[y == label] for label in unique_labels]
        f_stat, p_value = f_oneway(*groups)

        group_means = [np.mean(g, dtype=np.float64) for g in groups]
        group_stds = [np.std(g, dtype=np.float64, ddof=1) for g in groups]

        n_total = len(feature_data)
        df_between = n_groups - 1
        df_within = n_total - n_groups

        # Partial eta squared: SS_between / (SS_between + SS_within)
        # For one-way ANOVA: partial_eta_sq = F * df_between / (F * df_between + df_within)
        if not (np.isnan(f_stat) or np.isinf(f_stat)):
            partial_eta_sq = float((f_stat * df_between) / (f_stat * df_between + df_within))
        else:
            partial_eta_sq = float('nan')

        anova_results.append({
            'feature_index': int(idx),
            'feature_name': feature_name,
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'df_between': int(df_between),
            'df_within': int(df_within),
            'partial_eta_squared': partial_eta_sq,
        })

        group_means_list.append(group_means)
        group_stds_list.append(group_stds)

        if logger:
            logger.debug(f"[ANOVA] Feature {feature_name}: F={f_stat:.6f}, p={p_value:.6f}, "
                         f"eta2p={partial_eta_sq:.4f}, means={group_means}, stds={group_stds}")

    if logger:
        logger.info(f"[ANOVA] Completed ANOVA for {len(anova_results)} features")

    return {
        'feature_names': [r['feature_name'] for r in anova_results],
        'feature_indices': [r['feature_index'] for r in anova_results],
        'f_statistics': [r['f_statistic'] for r in anova_results],
        'p_values': [r['p_value'] for r in anova_results],
        'df_between': [r['df_between'] for r in anova_results],
        'df_within': [r['df_within'] for r in anova_results],
        'partial_eta_squared': [r['partial_eta_squared'] for r in anova_results],
        'group_means': group_means_list,
        'group_stds': group_stds_list,
        'group_labels': unique_labels.tolist(),
    }


def pairwise_comparisons(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    method: str = 'tukey',
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Perform pairwise comparisons using Tukey HSD or Bonferroni correction."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    unique_labels = np.unique(y)
    n_groups = len(unique_labels)

    if logger:
        logger.info(f"[Pairwise Comparisons] Input: X shape={X.shape}, y shape={y.shape}, "
                    f"n_groups={n_groups}, groups={unique_labels.tolist()}, method={method}")

    if method == 'tukey' and not HAS_STATSMODELS:
        if logger:
            logger.warning("[Pairwise Comparisons] statsmodels not available, falling back to Bonferroni correction")
        method = 'bonferroni'

    if feature_indices is None:
        feature_indices = list(range(X.shape[1]))

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    all_comparisons = []
    significant_pairs_summary = []

    for idx in feature_indices:
        feature_data = X[:, idx]
        feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        current_method = method

        if current_method == 'tukey':
            try:
                tukey_result = pairwise_tukeyhsd(
                    endog=feature_data,
                    groups=y,
                    alpha=alpha,
                )

                significant_pairs = []
                comparison_details = []

                # Extract group names - pairwise order matches (0,1),(0,2),...,(1,2),...
                unique_groups = tukey_result.groupsunique
                pair_indices = list(combinations(range(len(unique_groups)), 2))
                for i in range(len(tukey_result.pvalues)):
                    g1_idx, g2_idx = pair_indices[i] if i < len(pair_indices) else (0, 0)
                    group1 = unique_groups[g1_idx]
                    group2 = unique_groups[g2_idx]
                    p_val = float(tukey_result.pvalues[i])
                    mean_diff = float(tukey_result.meandiffs[i])
                    is_significant = bool(p_val < alpha)

                    comparison_details.append({
                        'group1': str(group1),
                        'group2': str(group2),
                        'mean_diff': mean_diff,
                        'p_value': p_val,
                        'significant': is_significant,
                    })

                    if is_significant:
                        significant_pairs.append(f"{group1}-{group2}")

                all_comparisons.append({
                    'feature_index': int(idx),
                    'feature_name': feature_name,
                    'method': 'tukey',
                    'comparisons': comparison_details,
                    'significant_pairs': significant_pairs,
                })
                significant_pairs_summary.append({
                    'feature_name': feature_name,
                    'n_significant_pairs': len(significant_pairs),
                    'significant_pairs': significant_pairs,
                })

                if logger:
                    logger.debug(f"[Pairwise Comparisons] Feature {feature_name}: "
                                 f"{len(significant_pairs)} significant pairs: {significant_pairs}")

            except Exception as e:
                if logger:
                    logger.warning(f"[Pairwise Comparisons] Tukey HSD failed for feature "
                                   f"{feature_name}: {e}")
                current_method = 'bonferroni'

        if current_method == 'bonferroni':
            from itertools import combinations
            from scipy.stats import ttest_ind

            groups = {label: feature_data[y == label] for label in unique_labels}
            comparison_details = []
            significant_pairs = []

            n_comparisons = n_groups * (n_groups - 1) // 2
            bonferroni_alpha = alpha / n_comparisons

            for group1, group2 in combinations(unique_labels, 2):
                data1 = groups[group1]
                data2 = groups[group2]
                t_stat, p_value = ttest_ind(data1, data2)
                mean_diff = np.mean(data1, dtype=np.float64) - np.mean(data2, dtype=np.float64)
                is_significant = bool(p_value < bonferroni_alpha)

                comparison_details.append({
                    'group1': str(group1),
                    'group2': str(group2),
                    'mean_diff': float(mean_diff),
                    'p_value': float(p_value),
                    'bonferroni_alpha': float(bonferroni_alpha),
                    'significant': is_significant,
                })

                if is_significant:
                    significant_pairs.append(f"{group1}-{group2}")

            all_comparisons.append({
                'feature_index': int(idx),
                'feature_name': feature_name,
                'method': 'bonferroni',
                'comparisons': comparison_details,
                'significant_pairs': significant_pairs,
            })
            significant_pairs_summary.append({
                'feature_name': feature_name,
                'n_significant_pairs': len(significant_pairs),
                'significant_pairs': significant_pairs,
            })

            if logger:
                logger.debug(f"[Pairwise Comparisons] Feature {feature_name}: "
                             f"{len(significant_pairs)} significant pairs: {significant_pairs}")

    if logger:
        logger.info(f"[Pairwise Comparisons] Completed pairwise comparisons "
                    f"for {len(all_comparisons)} features")

    return {
        'feature_names': [c['feature_name'] for c in all_comparisons],
        'feature_indices': [c['feature_index'] for c in all_comparisons],
        'comparisons': all_comparisons,
        'significant_pairs': significant_pairs_summary,
        'method': method,
        'alpha': float(alpha),
    }


def statistical_analysis_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k: Optional[int] = None,
    alpha: float = 0.05,
    pairwise_method: str = 'tukey',
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Complete statistical analysis pipeline:
    1. Feature selection using ANOVA F-test
    2. ANOVA analysis for selected features  (includes partial eta squared)
    3. Pair-wise comparisons using Tukey HSD / Bonferroni
    """
    if logger:
        logger.info(f"[Statistical Analysis] Starting complete pipeline: "
                    f"X shape={X.shape}, y shape={y.shape}, alpha={alpha}, k={k}")

    if logger:
        logger.info("[Statistical Analysis] Step 1/3: Feature selection using ANOVA F-test...")
    feature_selection = select_features_anova(
        X, y, feature_names=feature_names, k=k, alpha=alpha, logger=logger
    )

    if logger:
        logger.info("[Statistical Analysis] Step 2/3: ANOVA analysis for selected features...")
    anova_results = compute_anova(
        X, y,
        feature_indices=feature_selection['selected_features'],
        feature_names=feature_names,
        logger=logger,
    )

    if logger:
        logger.info("[Statistical Analysis] Step 3/3: Pairwise comparisons...")
    pairwise_results = pairwise_comparisons(
        X, y,
        feature_indices=feature_selection['selected_features'],
        feature_names=feature_names,
        alpha=alpha,
        method=pairwise_method,
        logger=logger,
    )

    if logger:
        logger.info(f"[Statistical Analysis] Pipeline complete: "
                    f"selected {len(feature_selection['selected_features'])} features")

    return {
        'feature_selection': feature_selection,
        'anova': anova_results,
        'pairwise_comparisons': pairwise_results,
        'summary': {
            'n_selected_features': len(feature_selection['selected_features']),
            'n_total_features': X.shape[1],
            'n_groups': len(np.unique(y)),
            'alpha': float(alpha),
            'pairwise_method': pairwise_method,
        },
    }
