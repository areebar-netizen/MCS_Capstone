#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Analysis and Visualization for EEG Features
This script loads the extracted features and creates comprehensive visualizations
to evaluate the quality of feature extraction for model training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_features(csv_path):
    """Load features from CSV file"""
    print(f"Loading features from {csv_path}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    if 'Label' in df.columns:
        features = df.drop('Label', axis=1)
        labels = df['Label']
    else:
        features = df
        labels = None
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    return features, labels

def analyze_feature_distribution(features, labels):
    """Analyze the distribution of features"""
    print("\n=== FEATURE DISTRIBUTION ANALYSIS ===")
    
    # Basic statistics
    print("Feature statistics:")
    print(f"Mean: {features.values.mean():.6f}")
    print(f"Std: {features.values.std():.6f}")
    print(f"Min: {features.values.min():.6f}")
    print(f"Max: {features.values.max():.6f}")
    
    # Check for missing values
    missing_values = features.isnull().sum().sum()
    print(f"Missing values: {missing_values}")
    
    # Check for infinite values
    infinite_values = np.isinf(features.values).sum()
    print(f"Infinite values: {infinite_values}")
    
    # Label distribution if available
    if labels is not None:
        print("\nLabel distribution:")
        label_counts = labels.value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = {0: 'Relaxed', 1: 'Neutral', 2: 'Concentrating'}.get(label, f'Class_{label}')
            print(f"  {label_name}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return features, labels

def visualize_feature_distributions(features, labels):
    """Create visualizations of feature distributions"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # 1. Feature distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot first few features to understand the scale
    for i in range(4):
        ax = axes[i//2, i%2]
        feature_name = features.columns[i]
        ax.hist(features.iloc[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution of {feature_name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/sachi/Documents/keystone/secondBrain/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation matrix (sample of features to avoid overcrowding)
    sample_features = features.iloc[:, :50]  # First 50 features
    correlation_matrix = sample_features.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix (First 50 Features)')
    plt.tight_layout()
    plt.savefig('/Users/sachi/Documents/keystone/secondBrain/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Box plots for different feature types
    feature_groups = {
        'Statistical': [col for col in features.columns if any(x in col for x in ['mean_', 'std_', 'min_', 'max_'])][:10],
        'Frequency': [col for col in features.columns if 'freq_' in col][:10],
        'Band Power': [col for col in features.columns if any(x in col for x in ['Delta_', 'Theta_', 'Alpha_', 'Beta_', 'Gamma_']) and 'pow' in col][:10],
        'Hjorth': [col for col in features.columns if 'hjorth' in col][:10]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, (group_name, group_features) in enumerate(feature_groups.items()):
        if group_features:
            ax = axes[idx//2, idx%2]
            # Select only features that exist
            existing_features = [f for f in group_features if f in features.columns]
            if existing_features:
                features[existing_features].boxplot(ax=ax)
                ax.set_title(f'{group_name} Features')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/sachi/Documents/keystone/secondBrain/feature_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Class separation visualization (if labels available)
    if labels is not None:
        # PCA for dimensionality reduction
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(10, 8))
        label_names = {0: 'Relaxed', 1: 'Neutral', 2: 'Concentrating'}
        colors = ['blue', 'green', 'red']
        
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                       c=colors[int(label)], label=label_names.get(label, f'Class_{label}'), 
                       alpha=0.6, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.title('PCA Visualization of Feature Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/Users/sachi/Documents/keystone/secondBrain/pca_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # t-SNE for non-linear visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features_scaled)
        
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                       c=colors[int(label)], label=label_names.get(label, f'Class_{label}'), 
                       alpha=0.6, s=50)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Feature Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/Users/sachi/Documents/keystone/secondBrain/tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_feature_importance(features, labels):
    """Analyze feature importance using statistical methods"""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    if labels is None:
        print("No labels available for feature importance analysis")
        return
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import f_classif
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    
    # Get top 20 important features
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    print("Top 20 most important features (Random Forest):")
    for i, idx in enumerate(indices):
        print(f"{i+1:2d}. {features.columns[idx]}: {importances[idx]:.4f}")
    
    # ANOVA F-test
    f_scores, p_values = f_classif(features, labels)
    f_indices = np.argsort(f_scores)[::-1][:20]
    
    print("\nTop 20 features by ANOVA F-score:")
    for i, idx in enumerate(f_indices):
        print(f"{i+1:2d}. {features.columns[idx]}: {f_scores[idx]:.2f} (p={p_values[idx]:.2e})")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(20), importances[indices])
    plt.xticks(range(20), [features.columns[i] for i in indices], rotation=45, ha='right')
    plt.title('Top 20 Feature Importance (Random Forest)')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('/Users/sachi/Documents/keystone/secondBrain/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_feature_quality(features, labels):
    """Evaluate the quality of extracted features"""
    print("\n=== FEATURE QUALITY EVALUATION ===")
    
    quality_issues = []
    recommendations = []
    
    # Check for zero-variance features
    zero_var_features = features.columns[features.var() == 0].tolist()
    if zero_var_features:
        quality_issues.append(f"Found {len(zero_var_features)} zero-variance features")
        recommendations.append("Remove zero-variance features as they provide no information")
    
    # Check for highly correlated features
    correlation_matrix = features.corr().abs()
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
    
    if high_corr_pairs:
        quality_issues.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95)")
        recommendations.append("Consider removing or combining highly correlated features")
    
    # Check feature scale distribution
    feature_ranges = features.max() - features.min()
    if feature_ranges.std() > feature_ranges.mean() * 2:
        quality_issues.append("Features have very different scales")
        recommendations.append("Apply feature scaling (StandardScaler or MinMaxScaler)")
    
    # Check class separation
    if labels is not None:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
        
        # Simple classifier to check feature quality
        lda = LinearDiscriminantAnalysis()
        scores = cross_val_score(lda, features, labels, cv=5)
        
        if scores.mean() < 0.7:
            quality_issues.append(f"Low classification accuracy ({scores.mean():.3f}) suggests poor feature separation")
            recommendations.append("Consider feature engineering or selection to improve class separation")
        
        print(f"Cross-validation accuracy with LDA: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Print quality assessment
    if quality_issues:
        print("\nQuality Issues Found:")
        for issue in quality_issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✅ No major quality issues detected")
    
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  💡 {rec}")
    
    return quality_issues, recommendations

def main():
    """Main function to run the feature analysis"""
    print("EEG Feature Analysis and Visualization")
    print("=" * 50)
    
    # Load features
    features_path = "../enhanced_features.csv"
    features, labels = load_features(features_path)
    
    # Analyze distributions
    features, labels = analyze_feature_distribution(features, labels)
    
    # Create visualizations
    visualize_feature_distributions(features, labels)
    
    # Analyze feature importance
    analyze_feature_importance(features, labels)
    
    # Evaluate feature quality
    quality_issues, recommendations = evaluate_feature_quality(features, labels)
    
    print("\n" + "=" * 50)
    print("Analysis complete! Visualizations saved to:")
    print("  - feature_distributions.png")
    print("  - correlation_matrix.png") 
    print("  - feature_boxplots.png")
    print("  - pca_visualization.png")
    print("  - tsne_visualization.png")
    print("  - feature_importance.png")

if __name__ == "__main__":
    main()
