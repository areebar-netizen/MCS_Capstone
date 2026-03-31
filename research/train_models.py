#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG Model Training Script

This script trains machine learning models on EEG feature data with feature importance analysis.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import time
import warnings

from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Try to import XGBoost, but make it optional
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = e


def setup_environment() -> None:
    """Set up the environment by suppressing warnings."""
    warnings.filterwarnings('ignore', category=UserWarning)
    os.environ['PYTHONWARNINGS'] = 'ignore'


def load_data(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and prepare the dataset from CSV file."""
    df = pd.read_csv(csv_path)
    
    if 'Label' not in df.columns:
        # Create a new DataFrame with the label column added in one operation
        label_series = df.iloc[:, -1]
        df = pd.concat([df.iloc[:, :-1], label_series.rename('Label')], axis=1)
    
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    return df, X, y


def plot_feature_importance(importance, names, model_name, output_dir, top_n=20):
    """Plot feature importance for a given model."""
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    
    # Sort the DataFrame in descending order of feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    
    # Select top N features
    fi_df = fi_df.head(top_n)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f'Top {top_n} Features - {model_name}')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.tight_layout()
    
    # Save the figure
    filename = f"{output_dir}/feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fi_df

def analyze_feature_importance(model, X, y, feature_names, model_name, output_dir):
    """Analyze and plot feature importance using different methods."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # For tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi_df = plot_feature_importance(
            importances, feature_names, 
            f"{model_name} - Feature Importance",
            output_dir
        )
        results['feature_importance'] = fi_df
        fi_df.to_csv(f"{output_dir}/{model_name.lower().replace(' ', '_')}_feature_importance.csv", index=False)
    
    # Permutation importance
    try:
        result = permutation_importance(
            model, X, y, 
            n_repeats=10, 
            random_state=42, 
            n_jobs=-1
        )
        perm_importances = result.importances_mean
        perm_df = plot_feature_importance(
            perm_importances, feature_names, 
            f"{model_name} - Permutation Importance",
            output_dir
        )
        results['permutation_importance'] = perm_df
        perm_df.to_csv(f"{output_dir}/{model_name.lower().replace(' ', '_')}_permutation_importance.csv", index=False)
    except Exception as e:
        print(f"Error calculating permutation importance: {e}")
    
    return results

def perform_feature_selection(X: np.ndarray, y: np.ndarray, 
                            threshold: str = 'median') -> Tuple[np.ndarray, Any, List[str]]:
    """
    Perform feature selection using RandomForest feature importances.
    
    Returns:
        Tuple containing:
        - X_selected: Selected features
        - selector: Fitted SelectFromModel instance
        - selected_feature_names: List of selected feature names
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(rf, threshold=threshold)
    
    # Fit and transform
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices and names
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [f"feature_{i}" for i in selected_feature_indices]
    
    print(f"Selected {X_selected.shape[1]} out of {X.shape[1]} features")
    return X_selected, selector, selected_feature_names


def setup_cross_validation(y: np.ndarray) -> StratifiedKFold:
    """Set up stratified k-fold cross-validation."""
    n_splits = min(5, len(np.unique(y)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def evaluate_model(model, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> None:
    """Evaluate model using cross-validation and print classification report with per-class accuracy."""
    print("\nModel Evaluation:")
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    
    # Get detailed classification report
    report = classification_report(y, y_pred, 
                                  target_names=['relaxed', 'neutral', 'concentrating'],
                                  output_dict=True)
    
    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    class_names = ['relaxed', 'neutral', 'concentrating']
    for i, class_name in enumerate(class_names):
        accuracy = report[class_name]['precision']  # Using precision as per-class accuracy
        print(f"  {class_name.capitalize():12s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print overall metrics
    print(f"\nOverall Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    # Print full classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred, target_names=['relaxed', 'neutral', 'concentrating']))
    
    return y_pred

def train_random_forest(X: np.ndarray, y: np.ndarray, 
                       cv: StratifiedKFold) -> Dict[str, Any]:
    """
    Train and evaluate a RandomForest model with balanced class weights and regularization.
    
    Returns:
        Dictionary containing:
        - 'model': Trained model
        - 'grid': GridSearchCV results
        - 'cv_predictions': Cross-validated predictions
    """
    print("\nTraining RandomForest with balanced class weights and regularization...")
    
    # Calculate class distribution
    class_counts = np.bincount(y.astype(int))
    print("\nClass distribution:", 
          dict(zip(['relaxed', 'neutral', 'concentrating'], class_counts)))
    
    # Updated parameter grid with more regularization
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],  # Limit tree depth
        'min_samples_split': [5, 10, 20],  # Increased min samples to split
        'min_samples_leaf': [2, 4],  # Increased min samples at leaf
        'max_features': ['sqrt', 'log2'],  # Limit features considered
        'class_weight': ['balanced', 'balanced_subsample'],
        'bootstrap': [True],
        'oob_score': [True],  # Use out-of-bag samples for generalization estimate
        'max_samples': [0.7, 0.8]  # Use subset of samples for each tree
    }
    
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',  # Default to balanced if not in grid
        oob_score=True  # Enable out-of-bag estimates
    )
    
    grid = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring='f1_weighted',  # Better for imbalanced classes
        n_jobs=-1, 
        verbose=1,
        error_score='raise'
    )
    
    grid.fit(X, y)
    
    print("\nBest parameters:", grid.best_params_)
    print(f"CV F1 score: {grid.best_score_:.4f}")
    
    # Evaluate the best model
    best_model = grid.best_estimator_
    
    # Print OOB score if available
    if hasattr(best_model, 'oob_score_'):
        print(f"OOB Score: {best_model.oob_score_:.4f}")
    
    # Cross-validated evaluation
    y_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)
    
    # Get detailed classification report
    report = classification_report(y, y_pred, 
                                  target_names=['relaxed', 'neutral', 'concentrating'],
                                  output_dict=True)
    
    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    class_names = ['relaxed', 'neutral', 'concentrating']
    for i, class_name in enumerate(class_names):
        accuracy = report[class_name]['precision']  # Using precision as per-class accuracy
        print(f"  {class_name.capitalize():12s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print overall metrics
    print(f"\nOverall Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred, 
                              target_names=['relaxed', 'neutral', 'concentrating']))
    
    return {
        'model': best_model,
        'grid': grid,
        'cv_predictions': y_pred
    }


def train_xgboost(X: np.ndarray, y: np.ndarray, 
                       cv: StratifiedKFold, for_stacking: bool = False) -> Dict[str, Any]:
    """
    XGBoost training with optimized hyperparameters and early stopping.
    
    Args:
        X: Input features
        y: Target variable
        cv: Cross-validation strategy
        for_stacking: If True, returns a model without early stopping for stacking
        
    Returns:
        Dictionary containing:
        - 'model': Trained model
        - 'cv_predictions': Cross-validated predictions (None if for_stacking=True)
    """
    if not XGBOOST_AVAILABLE:
        print('XGBoost not available; skipping XGBoost training.')
        if 'XGBOOST_IMPORT_ERROR' in globals():
            print('XGBoost import error:', repr(XGBOOST_IMPORT_ERROR))
        return None
    
    if for_stacking:
        print("\nPreparing XGBoost model for stacking...")
    else:
        print("\nTraining XGBoost...")
    
    # Timer for tracking
    start_time = time.time()
    
    # Calculate class weights quickly
    unique, counts = np.unique(y, return_counts=True)
    scale_pos_weight = len(y) / (len(unique) * counts)
    
    # Base parameters
    params = {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'multi:softmax',
        'num_class': len(unique),
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist',
        'grow_policy': 'depthwise',
        'max_bin': 256,
        'enable_categorical': False
    }
    
    # Add early stopping only when not used in stacking
    if not for_stacking:
        params['early_stopping_rounds'] = 20
    
    print("Training with optimized settings...")
    
    if for_stacking:
        # For stacking, just return a simple model without cross-validation
        model = XGBClassifier(**params)
        model.fit(X, y, sample_weight=scale_pos_weight[y.astype(int)])
        return {'model': model, 'cv_predictions': None}
    else:
        # For standalone training, use cross-validation with early stopping
        best_model = None
        best_score = 0
        all_preds = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"  Fold {fold + 1}/{cv.n_splits}", end="\r")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model for this fold
            model = XGBClassifier(**params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                sample_weight=scale_pos_weight[y_train.astype(int)]
            )
            
            # Get validation predictions
            val_preds = model.predict(X_val)
            all_preds[val_idx] = val_preds
            
            # Keep track of best model
            fold_score = accuracy_score(y_val, val_preds)
            if fold_score > best_score or best_model is None:
                best_score = fold_score
                best_model = model
        
        # Now train final model on all data
        print("\nTraining final model on all data...")
        
        # Create validation split for early stopping
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        final_model = XGBClassifier(**params)
        
        final_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            verbose=False,
            sample_weight=scale_pos_weight[y_train_final.astype(int)]
        )
        
        # Get cross-validated predictions
        y_pred = all_preds
        
        elapsed_time = time.time() - start_time
        print(f"\nXGBoost training completed in {elapsed_time:.1f} seconds")
        print(f"Best fold accuracy: {best_score:.4f}")
        
        # Evaluation
        print("\nPer-Class Accuracy:")
        report = classification_report(y, y_pred, 
                                      target_names=['relaxed', 'neutral', 'concentrating'],
                                      output_dict=True)
        
        class_names = ['relaxed', 'neutral', 'concentrating']
        for i, class_name in enumerate(class_names):
            accuracy = report[class_name]['precision']
            print(f"  {class_name.capitalize():12s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\nOverall Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
        print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y, y_pred, 
                                  target_names=['relaxed', 'neutral', 'concentrating']))
        
        return {
            'model': final_model,
            'cv_predictions': y_pred
        }


def train_stacked_model(rf_model: Any, xgb_model: Any, 
                       X: np.ndarray, y: np.ndarray, 
                       cv: StratifiedKFold) -> Any:
    """
    Train a stacked model using RandomForest and XGBoost.
    
    Args:
        rf_model: Pre-trained RandomForest model
        xgb_model: Pre-trained XGBoost model
        X: Training features
        y: Training labels
        cv: Cross-validation strategy
        
    Returns:
        Trained StackingClassifier or None if XGBoost is not available
    """
    if xgb_model is None:
        print('Skipping stacked model training (XGBoost not available)')
        return None
        
    print("\nTraining Stacked Model...")
    
    # Define base models
    estimators = [
        ('random_forest', rf_model),
        ('xgboost', train_xgboost(X, y, cv, for_stacking=True)['model'])
    ]
    
    # Create and train the stacking classifier
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    stack.fit(X, y)
    scores = cross_val_score(stack, X, y, cv=cv, n_jobs=-1)
    print(f"CV accuracy: {scores.mean():.2f}% (+/- {scores.std() * 2:.2f}%)")
    
    return stack


def save_models(models: Dict[str, Any], output_dir: str) -> None:
    """Save trained models to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        if model is not None:
            path = os.path.join(output_dir, f'{name}.joblib')
            joblib.dump(model, path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"Saved {name}: {size_mb:.2f} MB")


def print_model_sizes(output_dir: str) -> None:
    """Print the sizes of all saved model files."""
    print("\nModel sizes:")
    for model_name in ['random_forest', 'xgboost', 'stacked_model']:
        path = os.path.join(output_dir, f'{model_name}.joblib')
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"{model_name}: {size_mb:.2f} MB")


def train_models(csv_path: str, output_dir: str = 'models') -> None:
    """
    Main function to train and evaluate models with enhanced class balancing.
    Uses enhanced features if available.
    
    Args:
        csv_path: Path to the CSV file containing training data
        output_dir: Directory to save trained models and results
    """
    print("\n" + "="*50)
    print("Starting model training process")
    print("="*50)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    feature_importance_dir = os.path.join(output_dir, 'feature_importance')
    os.makedirs(feature_importance_dir, exist_ok=True)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    df, X, y = load_data(csv_path)
    
    # Print class distribution
    class_counts = np.bincount(y.astype(int))
    class_dist = dict(zip(['relaxed', 'neutral', 'concentrating'], class_counts))
    print("\nClass distribution in training data:", class_dist)
    
    # Calculate and print class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    print("Class weights:", {cls: f"{w:.2f}" for cls, w in zip(['relaxed', 'neutral', 'concentrating'], class_weights)})
    
    # Check if this is enhanced features (already processed) or original features
    if X.shape[1] > 200:  # Likely original features, apply feature selection
        print("\nDetected original feature set, applying feature selection...")
        X_selected, selector, selected_feature_names = perform_feature_selection(X, y)
        
        # Save feature selector
        selector_path = os.path.join(output_dir, 'feature_selector.joblib')
        joblib.dump(selector, selector_path)
        print(f"\nSaved feature selector to {selector_path}")
    else:  # Already enhanced features
        print("\nDetected enhanced feature set, using as-is...")
        X_selected = X
        selected_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Save selected feature names
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        for i, name in enumerate(selected_feature_names):
            f.write(f"{i}: {name}\n")
    
    # Set up cross-validation
    print("\nSetting up cross-validation...")
    cv = setup_cross_validation(y)
    
    # Train models
    print("\n" + "="*50)
    print("Training RandomForest model")
    print("="*50)
    rf_result = train_random_forest(X_selected, y, cv)
    
    print("\n" + "="*50)
    print("Training XGBoost model")
    print("="*50)
    xgb_result = train_xgboost(X_selected, y, cv)
    
    # Train stacked model if both base models are available
    stack_model = None
    if rf_result is not None and xgb_result is not None:
        print("\n" + "="*50)
        print("Training Stacked model")
        print("="*50)
        stack_model = train_stacked_model(
            rf_result['model'], 
            xgb_result['model'],
            X_selected, y, cv
        )
    
    # Save all models
    models_to_save = {
        'random_forest': rf_result['model'] if rf_result else None,
        'xgboost': xgb_result['model'] if xgb_result else None,
        'stacked_model': stack_model
    }
    
    save_models(models_to_save, output_dir)
    print_model_sizes(output_dir)
    
    # Analyze feature importance for each model
    print("\n" + "="*50)
    print("Analyzing feature importance...")
    print("="*50)
    
    if rf_result is not None:
        print("\nAnalyzing RandomForest feature importance...")
        rf_importance = analyze_feature_importance(
            rf_result['model'], 
            X_selected, 
            y, 
            selected_feature_names, 
            "RandomForest",
            feature_importance_dir
        )
    
    if xgb_result is not None:
        print("\nAnalyzing XGBoost feature importance...")
        xgb_importance = analyze_feature_importance(
            xgb_result['model'], 
            X_selected, 
            y, 
            selected_feature_names, 
            "XGBoost",
            feature_importance_dir
        )
    
    print("\n" + "="*50)
    print("Training and analysis completed successfully!")
    print("="*50)
    print(f"\nResults saved to: {os.path.abspath(output_dir)}")
    print(f"Feature importance analysis saved to: {os.path.abspath(feature_importance_dir)}")


def main() -> None:
    """Main entry point for the training script."""
    if len(sys.argv) < 2:
        print('Usage: train_models.py <features_csv> [output_dir]')
        sys.exit(1)
        
    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    
    train_models(csv_path, output_dir)


if __name__ == '__main__':
    main()