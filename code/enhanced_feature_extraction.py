#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced EEG Feature Extraction Pipeline
Implements feature scaling, redundancy removal, and feature selection
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import the original feature extraction
from EEG_feature_extraction_adv import generate_feature_vectors_from_samples

def remove_highly_correlated_features(X, feature_names, threshold=0.95):
    """
    Remove highly correlated features to reduce redundancy.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        threshold: Correlation threshold above which to remove features
    
    Returns:
        X_reduced: Reduced feature matrix
        selected_names: Selected feature names
        removed_pairs: List of removed feature pairs
    """
    print(f"Removing highly correlated features (threshold: {threshold})...")
    
    # Calculate correlation matrix
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()
    
    # Find highly correlated pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    
    for i in range(len(upper_tri.columns)):
        for j in range(i+1, len(upper_tri.columns)):
            if upper_tri.iloc[i, j] > threshold:
                high_corr_pairs.append((
                    upper_tri.columns[i], 
                    upper_tri.columns[j], 
                    upper_tri.iloc[i, j]
                ))
    
    # Remove features with high correlations
    to_remove = set()
    for feat1, feat2, corr in high_corr_pairs:
        # Remove the feature that appears later in the list
        if feat1 not in to_remove and feat2 not in to_remove:
            to_remove.add(feat2)
    
    selected_features = [f for f in feature_names if f not in to_remove]
    X_reduced = df[selected_features].values
    
    print(f"Removed {len(to_remove)} highly correlated features")
    print(f"Remaining features: {X_reduced.shape[1]}")
    
    return X_reduced, selected_features, high_corr_pairs

def select_top_features(X, y, feature_names, n_features=100):
    """
    Select top N features based on Random Forest importance.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        n_features: Number of top features to select
    
    Returns:
        X_selected: Selected feature matrix
        selected_names: Selected feature names
        feature_importance: Feature importance scores
        original_indices: Indices of selected features from ORIGINAL space
    """
    print(f"Selecting top {n_features} features using Random Forest...")
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Get indices of top features
    indices = np.argsort(importance)[::-1][:n_features]
    
    # Select features using these indices
    selected_names = [feature_names[i] for i in indices]
    X_selected = X[:, indices]
    selected_importance = importance[indices]
    
    print(f"Selected {len(selected_names)} most important features")
    
    # Print top 10 features
    print("Top 10 most important features:")
    for i, (name, imp) in enumerate(zip(selected_names[:10], selected_importance[:10])):
        print(f"  {i+1:2d}. {name}: {imp:.4f}")
    
    return X_selected, selected_names, selected_importance, indices

def apply_feature_scaling(X, method='standard'):
    """
    Apply feature scaling to normalize feature ranges.
    
    Args:
        X: Feature matrix
        method: Scaling method ('standard' or 'minmax')
    
    Returns:
        X_scaled: Scaled feature matrix
        scaler: Fitted scaler
    """
    print(f"Applying {method} scaling to features...")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    X_scaled = scaler.fit_transform(X)
    
    print(f"Features scaled using {method} scaler")
    print(f"Mean after scaling: {X_scaled.mean():.6f}")
    print(f"Std after scaling: {X_scaled.std():.6f}")
    
    return X_scaled, scaler

def generate_enhanced_features_from_directory(directory_path, output_file, 
                                            n_features=100, 
                                            correlation_threshold=0.95,
                                            scaling_method='standard',
                                            cols_to_ignore=-1):
    """
    Generate enhanced features from directory with scaling, redundancy removal, and selection.
    
    Args:
        directory_path: Directory containing CSV files
        output_file: Output file for enhanced features
        n_features: Number of top features to select
        correlation_threshold: Threshold for removing correlated features
        scaling_method: Method for feature scaling
        cols_to_ignore: Columns to ignore from CSV files
    
    Returns:
        Enhanced feature matrix and feature names
    """
    print("="*60)
    print("ENHANCED EEG FEATURE EXTRACTION PIPELINE")
    print("="*60)
    
    # Step 1: Generate original features
    print("\nStep 1: Generating original features...")
    from EEG_generate_training_matrix import gen_training_matrix
    
    # Generate temporary features file
    temp_file = output_file.replace('.csv', '_temp.csv')
    gen_training_matrix(directory_path, temp_file, cols_to_ignore)
    
    # Load the generated features
    df = pd.read_csv(temp_file)
    
    if 'Label' not in df.columns:
        print("Error: No Label column found in generated features")
        return None, None
    
    # Separate features and labels
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    feature_names = df.drop('Label', axis=1).columns.tolist()
    
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    
    # Step 2: Apply feature scaling
    print("\nStep 2: Feature scaling...")
    X_scaled, scaler = apply_feature_scaling(X, scaling_method)
    
    # Step 3: Remove highly correlated features
    print("\nStep 3: Redundancy removal...")
    X_reduced, reduced_names, corr_pairs = remove_highly_correlated_features(
        X_scaled, feature_names, correlation_threshold
    )
    
    # Step 4: Feature selection
    print("\nStep 4: Feature selection...")
    # Capture the indices returned by the function
    X_selected, selected_names, importance, reduced_indices = select_top_features(
        X_reduced, y, reduced_names, n_features
    )
    
    # Map reduced indices back to original indices
    # We need to find which original features correspond to the selected reduced features
    original_to_reduced_mapping = {}
    for orig_idx, orig_name in enumerate(feature_names):
        for red_idx, red_name in enumerate(reduced_names):
            if orig_name == red_name:
                original_to_reduced_mapping[red_idx] = orig_idx
                break
    
    # Convert reduced indices to original indices
    original_indices = [original_to_reduced_mapping[red_idx] for red_idx in reduced_indices]
    
    # Step 5: Create final enhanced features dataframe
    print("\nStep 5: Creating final feature matrix...")
    final_df = pd.DataFrame(X_selected, columns=selected_names)
    final_df['Label'] = y
    
    # Save enhanced features
    final_df.to_csv(output_file, index=False)
    print(f"\nEnhanced features saved to: {output_file}")
    print(f"Final shape: {final_df.shape}")
    
    # Save preprocessing artifacts
    artifact_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    artifact_dir = os.path.join(artifact_dir, 'preprocessing_artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, os.path.join(artifact_dir, 'feature_scaler.joblib'))
    
    # Save feature selection info
    feature_info = {
        'selected_features': selected_names,
        'feature_importance': dict(zip(selected_names, importance)),
        'correlation_pairs': corr_pairs,
        'original_feature_count': len(feature_names),
        'after_correlation_removal': len(reduced_names),
        'final_feature_count': len(selected_names),
        'selected_indices': original_indices  # Save the mapped original indices
    }
    
    pd.to_pickle(feature_info, os.path.join(artifact_dir, 'feature_selection_info.pkl'))
    
    # Clean up temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"\nPreprocessing artifacts saved to: {artifact_dir}")
    print("="*60)
    print("ENHANCED FEATURE EXTRACTION COMPLETED")
    print("="*60)
    
    return X_selected, selected_names

def load_enhanced_features(csv_path):
    """
    Load enhanced features and return features and labels.
    
    Args:
        csv_path: Path to enhanced features CSV
    
    Returns:
        X: Feature matrix
        y: Labels
        feature_names: Feature names
    """
    df = pd.read_csv(csv_path)
    
    if 'Label' not in df.columns:
        raise ValueError("CSV file must contain 'Label' column")
    
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    feature_names = df.drop('Label', axis=1).columns.tolist()
    
    return X, y, feature_names

def load_preprocessing_artifacts(artifact_dir):
    """
    Load saved preprocessing artifacts (scaler and feature selection info).
    
    Args:
        artifact_dir: Directory containing saved artifacts
    
    Returns:
        scaler: Fitted StandardScaler
        feature_info: Dictionary with feature selection information
    """
    import joblib
    import pandas as pd
    
    scaler = joblib.load(os.path.join(artifact_dir, 'feature_scaler.joblib'))
    feature_info = pd.read_pickle(os.path.join(artifact_dir, 'feature_selection_info.pkl'))
    return scaler, feature_info

def apply_feature_pipeline(X, scaler, feature_info):
    """
    Apply the complete feature preprocessing pipeline.
    
    Args:
        X: Raw feature matrix
        scaler: Fitted StandardScaler
        feature_info: Dictionary with feature selection information
    
    Returns:
        X_processed: Scaled and feature-selected matrix
    """
    import pandas as pd
    
    # 1. Apply scaling
    X_scaled = scaler.transform(X)
    
    # 2. Apply feature selection using saved INDICES
    selected_indices = feature_info.get('selected_indices')
    
    if selected_indices is not None:
        # Ensure we have enough columns (handle edge cases)
        if X_scaled.shape[1] >= len(selected_indices):
            X_selected = X_scaled[:, selected_indices]
        else:
            raise ValueError(f"Input features ({X_scaled.shape[1]}) fewer than required indices ({len(selected_indices)})")
    else:
        # Fallback for old artifacts (will have the bug, but won't crash)
        print("Warning: 'selected_indices' not found in feature_info. Using fallback slicing.")
        selected_features = feature_info['selected_features']
        X_selected = X_scaled[:, :len(selected_features)]
        
    return X_selected

def main():
    """Main function for standalone execution."""
    if len(sys.argv) < 3:
        print('Usage: enhanced_feature_extraction.py <input_dir> <output_csv> [n_features] [correlation_threshold]')
        print('Example: enhanced_feature_extraction.py dataset/original_data enhanced_features.csv 100 0.95')
        sys.exit(1)
    
    directory_path = sys.argv[1]
    output_file = sys.argv[2]
    n_features = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    correlation_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
    
    generate_enhanced_features_from_directory(
        directory_path, output_file, n_features, correlation_threshold
    )

if __name__ == '__main__':
    main()
