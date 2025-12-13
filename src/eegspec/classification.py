"""
Classification module for design creativity state classification.

This module implements the classification methods described in the paper:
- SVM (Support Vector Machine)
- MLP (Multi-Layer Perceptron)
- KNN (K-Nearest Neighbors)

The paper achieved high classification accuracy (≥87%), with SVM performing best (~92%).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings


def train_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train and evaluate multiple classifiers (SVM, MLP, KNN) on the data.
    
    This function implements the classification pipeline described in the paper,
    which achieved high accuracy (≥87%) with SVM performing best (~92%).
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Labels of shape (n_samples,)
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    cv_folds : int
        Number of folds for cross-validation (default: 5)
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'svm': SVM classifier and results
        - 'mlp': MLP classifier and results
        - 'knn': KNN classifier and results
        - 'scaler': Fitted StandardScaler
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # === SVM Classifier ===
    # Paper mentions SVM achieved ~92% accuracy
    svm = SVC(kernel='rbf', random_state=random_state, probability=True)
    svm.fit(X_train_scaled, y_train)
    svm_train_score = svm.score(X_train_scaled, y_train)
    svm_test_score = svm.score(X_test_scaled, y_test)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    svm_cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    results['svm'] = {
        'classifier': svm,
        'train_accuracy': svm_train_score,
        'test_accuracy': svm_test_score,
        'cv_mean': np.mean(svm_cv_scores),
        'cv_std': np.std(svm_cv_scores),
        'cv_scores': svm_cv_scores.tolist(),
        'predictions': svm.predict(X_test_scaled),
        'y_test': y_test,
    }
    
    # === MLP Classifier ===
    # Multi-layer perceptron
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_train_score = mlp.score(X_train_scaled, y_train)
    mlp_test_score = mlp.score(X_test_scaled, y_test)
    
    mlp_cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    results['mlp'] = {
        'classifier': mlp,
        'train_accuracy': mlp_train_score,
        'test_accuracy': mlp_test_score,
        'cv_mean': np.mean(mlp_cv_scores),
        'cv_std': np.std(mlp_cv_scores),
        'cv_scores': mlp_cv_scores.tolist(),
        'predictions': mlp.predict(X_test_scaled),
        'y_test': y_test,
    }
    
    # === KNN Classifier ===
    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_train_score = knn.score(X_train_scaled, y_train)
    knn_test_score = knn.score(X_test_scaled, y_test)
    
    knn_cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    results['knn'] = {
        'classifier': knn,
        'train_accuracy': knn_train_score,
        'test_accuracy': knn_test_score,
        'cv_mean': np.mean(knn_cv_scores),
        'cv_std': np.std(knn_cv_scores),
        'cv_scores': knn_cv_scores.tolist(),
        'predictions': knn.predict(X_test_scaled),
        'y_test': y_test,
    }
    
    results['scaler'] = scaler
    results['X_test'] = X_test_scaled
    results['y_test'] = y_test
    
    return results


def get_classification_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of classification results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results dictionary from train_classifiers()
    
    Returns:
    --------
    Dict[str, Any]
        Summary dictionary with accuracy metrics for each classifier
    """
    summary = {}
    
    for name in ['svm', 'mlp', 'knn']:
        if name in results:
            res = results[name]
            summary[name] = {
                'test_accuracy': res['test_accuracy'],
                'cv_mean_accuracy': res['cv_mean'],
                'cv_std_accuracy': res['cv_std'],
            }
    
    return summary


def print_classification_report(results: Dict[str, Any], target_names: Optional[List[str]] = None):
    """
    Print detailed classification reports for all classifiers.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results dictionary from train_classifiers()
    target_names : Optional[List[str]]
        Names of the classes (e.g., ['IDG', 'IDE', 'IDR', 'RST'])
    """
    print("=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)
    
    for name in ['svm', 'mlp', 'knn']:
        if name not in results:
            continue
        
        res = results[name]
        print(f"\n{name.upper()} Classifier:")
        print(f"  Test Accuracy: {res['test_accuracy']:.4f} ({res['test_accuracy']*100:.2f}%)")
        print(f"  CV Mean Accuracy: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
        print(f"  CV Scores: {[f'{s:.4f}' for s in res['cv_scores']]}")
        
        print(f"\n  Classification Report:")
        print(classification_report(
            res['y_test'],
            res['predictions'],
            target_names=target_names,
        ))
        
        print(f"  Confusion Matrix:")
        cm = confusion_matrix(res['y_test'], res['predictions'])
        print(cm)
    
    print("\n" + "=" * 80)

