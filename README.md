# MCS_Capstone

This project develops a personalized study optimization system using consumer-grade EEG technology (Muse 02 2016 headband) to provide real-time focus tracking and data-driven technique recommendations. By continuously monitoring brainwave patterns during study sessions, the system can detect when focus declines and increases, recommend optimal break timing, and identify which study techniques work best for each individual under different conditions. We aim to implement this system through a user-friendly app meant to promote studying and focus. Unlike generic productivity apps, our system adapts to each user's unique cognitive patterns, learning from their EEG data to provide increasingly personalized recommendations over time. We plan on making this data valuable to be user to access as well so that they can further study their own focus and attention patterns.

# secondBrain — Enhanced EEG Feature Extraction and Mental State Classification

A comprehensive toolkit for real‑time EEG signal processing, enhanced feature extraction, and classification of cognitive states (relaxed / neutral / concentrating). The pipeline includes **advanced feature preprocessing** with scaling, redundancy removal, and intelligent feature selection for optimal model performance.

---

## 🆕 Major Updates

### Enhanced Feature Extraction Pipeline
- **Feature Scaling**: StandardScaler normalization for consistent feature ranges
- **Redundancy Removal**: Automatic removal of highly correlated features (>0.95 threshold)
- **Intelligent Feature Selection**: Top 100 most important features using Random Forest importance
- **Improved Performance**: 96.53% accuracy with 100 features vs 854 original features
- **Per-Class Accuracy Reporting**: Detailed accuracy metrics for each mental state

### Updated Model Performance
- **RandomForest**: 95.72% overall (Relaxed: 97.81%, Neutral: 92.19%, Concentrating: 97.40%)
- **XGBoost**: 96.53% overall (Relaxed: 97.73%, Neutral: 93.68%, Concentrating: 98.32%)
- **Stacked Model**: ~96% accuracy with ensemble benefits

---

## Key Features

### Advanced Signal Processing
- **Automatic artifact removal** – FastICA with kurtosis thresholding (fallback to PCA denoising).
- **Missing data handling** – Forward‑fill interpolation of NaN values.
- **Comprehensive feature set**:
  - Band power (Delta, Theta, Alpha, Beta, Gamma) with statistics (mean, median, std, skew, kurtosis, RMS).
  - Hjorth parameters (activity, mobility, complexity).
  - Shannon entropy, covariance matrix, eigenvalues, log‑covariance.
  - FFT – top 10 frequency bins and full power spectrum.
  - Concentration heuristic: `Beta / (Theta + Alpha)`.
- **Windowing** – Sliding windows with configurable length and 50% overlap.

### Enhanced Feature Preprocessing
- **StandardScaler**: Normalizes all features to zero mean and unit variance
- **Correlation Analysis**: Identifies and removes redundant features (47 removed)
- **Feature Selection**: Selects top 100 most informative features using Random Forest importance
- **Pipeline Persistence**: Saves preprocessing artifacts for consistent test processing

### Machine Learning
- **Classifiers**: Random Forest, XGBoost, and a stacked ensemble.
- **Enhanced feature selection** – Top 100 features with importance ranking.
- **Hyperparameter tuning** – Grid search with cross‑validation, class‑weighting for imbalance.
- **Per-Class Accuracy Reporting** – Detailed metrics for each mental state.
- **Regularization**:
  - Random Forest: limited depth, higher min samples per split/leaf, bootstrap sampling.
  - XGBoost: `reg_alpha` (L1), `reg_lambda` (L2), `subsample`, `colsample_bytree`, early stopping.
- **Model persistence** – Saved as `.joblib` files for later use.

### Visualization
- **Offline timeline** – Coloured strip of predictions for a single file; play/pause/step controls.
- **Live GUI** – Real‑time display of predictions, confidence, and a scrolling coloured bar.
- **Raw signal logging** – Save incoming LSL samples to CSV while predicting.
- **Performance tracking** – Accuracy metrics saved to `visualization/` directory.

### Live Processing
- **LSL integration** – Connects to any EEG stream (e.g., Muse, OpenBCI) via Lab Streaming Layer.
- **In‑memory feature extraction** – Same feature code used on rolling buffer.
- **Automatic MuseLSL launcher** – Optional `--auto-stream` flag starts `muselsl stream`.

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/yourusername/secondBrain.git
cd secondBrain
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```
secondBrain/
├── code/                          # Core processing scripts
│   ├── EEG_feature_extraction_adv.py       # Core feature extraction + ICA cleaning
│   ├── enhanced_feature_extraction.py       # 🆕 Enhanced pipeline with scaling/selection
│   ├── EEG_generate_training_matrix.py     # Build feature matrix from raw CSVs
│   ├── train_models.py                      # Enhanced training with per-class accuracy
│   ├── predict_test.py                     # Batch prediction on test directory
│   ├── test_models.py                      # 🆕 Testing with preprocessing pipeline
│   └── check_features_numeric.py           # Utility: verify feature CSV is numeric
├── live_data/                     # Live acquisition and visualization
│   ├── live_predict.py            # Main LSL reader + live prediction GUI
│   ├── Stream.py                  # Handles LSL stream connection to MUSE device
│   ├── band_power.py              # Computes and visualizes band powers
│   ├── vis.py                     # Real-time EEG signal visualization
│   ├── bp.py                      # Band power calculation utilities
│   └── numeric_data.py            # Numeric data processing utilities
├── dataset/                       # Example raw EEG recordings
│   ├── original_data/            # Labelled training files (name-state-index)
│   ├── test/                     # Held‑out test files
│   └── our_data/                 # User data (ignored in training)
├── models_out/                   # Default output folder for trained models
├── visualization/                 # 🆕 Performance metrics and visualizations
├── preprocessing_artifacts/       # 🆕 Saved scalers and feature selection info
├── enhanced_features.csv         # 🆕 Generated optimized features
├── requirements.txt
└── README.md
```

---

## 🚀 Complete Workflow from Scratch

### Step 1: Enhanced Feature Extraction

**Generate enhanced features from raw data (Recommended)**
```bash
python code/enhanced_feature_extraction.py dataset/original_data enhanced_features.csv 100 0.95
```
**Parameters:**
- `dataset/original_data`: Input directory with raw EEG files
- `enhanced_features.csv`: Output file for optimized features
- `100`: Number of top features to select
- `0.95`: Correlation threshold for redundancy removal


### Step 2: Model Training

**Train with enhanced features **
```bash
python code/train_models.py enhanced_features.csv models_out
```



**What happens during training:**
- **Feature Detection**: Automatically detects enhanced vs basic features
- **Per-Class Accuracy**: Shows accuracy for Relaxed, Neutral, Concentrating
- **Model Training**: RandomForest, XGBoost, and Stacked ensemble
- **Performance Tracking**: Saves accuracy metrics to `visualization/`
- **Feature Importance**: Generates importance plots and rankings

### Step 3: Model Testing

**Test with enhanced preprocessing (Recommended)**
```bash
python3 live_data/live_predict.py --models models_out --model xgboost --csv-dir dataset/our_data --summary-out live_prediction_summary.csv
```

### Step 4: Visualization

**View model performance**
```bash
# Performance metrics are automatically saved to visualization/
ls visualization/
# - feature_distributions.png
# - correlation_matrix.png
# - pca_visualization.png
# - model_accuracy_summary.txt
```

---

## 📊 Model Performance Results

### Enhanced Feature Extraction Results
```
Original features: 854
After correlation removal: 807
After feature selection: 100
Training samples: 2,479
Class distribution: Balanced (819 relaxed, 830 neutral, 830 concentrating)
```

### Cross-Validation Performance

**RandomForest:**
```
Overall Accuracy: 95.72%
Relaxed: 97.81%
Neutral: 92.19%
Concentrating: 97.40%
Macro F1-Score: 95.72%
```

**XGBoost:**
```
Overall Accuracy: 96.53%
Relaxed: 97.73%
Neutral: 93.68%
Concentrating: 98.32%
Macro F1-Score: 96.53%
```

**Stacked Model:**
```
CV Accuracy: ~96%
Model Size: 5.61 MB
```

---

## 🧪 Testing and Evaluation

### Testing Pipeline
```bash
# Test models with preprocessing (Recommended)
python code/test_models.py dataset/test/10sec.csv models_out preprocessing_artifacts

# Batch prediction with optimized features
python3 live_data/live_predict.py --models models_out --model xgboost --csv-dir dataset/our_data --summary-out live_prediction_summary.csv

# Live testing with CSV directory
python3 live_data/live_predict.py --csv-dir dataset/test --models models_out --model xgboost
```

### Live Testing
```bash
# Live prediction with preprocessing
python live_data/live_predict.py --eeg --models models_out --model xgboost

# Mock mode with preprocessing
python live_data/live_predict.py --eeg --mock --replay dataset/test/10sec.csv --models models_out --model xgboost

# With automatic stream start
python live_data/live_predict.py --eeg --auto-stream --models models_out --model stacked_model
```

---

## 🔧 Advanced Usage

### Feature Extraction Pipeline

**feature_extraction.py (Optimized Pipeline)**
```bash
# Generate features from raw data
python code/enhanced_feature_extraction.py dataset/original_data enhanced_features.csv 100 0.95

# Custom feature selection with different parameters
python code/enhanced_feature_extraction.py dataset/original_data custom_features.csv 150 0.90

# Parameters:
# - 150: Select top 150 features
# - 0.90: Remove features with >0.90 correlation
```

### Model Training

```bash
# Train with optimized features (Recommended)
python code/train_models.py enhanced_features.csv models_out

# Training automatically:
# - Detects optimized vs basic features
# - Applies appropriate preprocessing
# - Shows per-class accuracy
# - Saves performance metrics to visualization/
```

---

## 📈 Performance Improvements

### Feature Quality Enhancements
- ✅ **Feature Scaling**: StandardScaler normalization
- ✅ **Redundancy Removal**: 47 highly correlated features eliminated
- ✅ **Feature Selection**: Top 100 most important features retained
- ✅ **Dimensionality Reduction**: 854 → 100 features (88% reduction)
- ✅ **Performance Gain**: Improved accuracy with fewer features

### Training Improvements
- ✅ **Per-Class Accuracy**: Detailed metrics for each mental state
- ✅ **Better Generalization**: Reduced overfitting through feature selection
- ✅ **Faster Training**: Fewer features = faster training and inference
- ✅ **Model Size**: Smaller models with better performance

---

## 🎯 Key Files and Their Purposes

### Core Processing
- **`EEG_feature_extraction_adv.py`**: Core feature extraction engine with ICA cleaning
- **`enhanced_feature_extraction.py`**: 🆕 Optimized pipeline with scaling and selection
- **`EEG_generate_training_matrix.py`**: Builds training matrix from raw CSV files

### Training and Testing
- **`train_models.py`**: Training with per-class accuracy reporting
- **`predict_test.py`**: 🆕 Batch prediction with preprocessing pipeline
- **`test_models.py`**: Testing with preprocessing pipeline

### Visualization and Live Processing
- **`live_predict.py`**: 🆕 Real-time LSL processing with optimized features
- **`save_accuracy_summary.py`**: 🆕 Automatic performance tracking

---

## 🐛 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **No EEG stream found** | Ensure device is broadcasting LSL (e.g., `muselsl stream`) |
| **Feature shape mismatch** | Ensure preprocessing artifacts exist in `preprocessing_artifacts/` |
| **Import errors** | Run from repository root directory |
| **No predictions generated** | Check if test data has enough samples (min 150 samples) |
| **Preprocessing failed** | Verify `preprocessing_artifacts/` directory contains `feature_scaler.joblib` and `feature_selection_info.pkl` |
| **Model expects different features** | Ensure you're using the same preprocessing pipeline as training |

### Pipeline Verification

```bash
# Check if preprocessing artifacts exist
ls -la preprocessing_artifacts/

# Verify feature dimensions
python3 -c "
from enhanced_feature_extraction import load_preprocessing_artifacts
scaler, info = load_preprocessing_artifacts('preprocessing_artifacts')
print(f'Scaler features: {scaler.n_features_in_}')
print(f'Selected features: {len(info[\"selected_features\"])}')
"
```

### Feature Extraction Issues
```bash
# Check feature extraction works
python -c "
from EEG_feature_extraction_adv import generate_feature_vectors_from_samples
try:
    vectors, headers = generate_feature_vectors_from_samples('dataset/test/10sec.csv', 150, 1.0)
    print(f'Success: {vectors.shape}')
except Exception as e:
    print(f'Error: {e}')
"
```

### Model Training Issues
```bash
# Verify training data
python -c "
import pandas as pd
df = pd.read_csv('enhanced_features.csv')
print(f'Data shape: {df.shape}')
print(f'Columns: {list(df.columns)[:5]}...')
if 'Label' in df.columns:
    print(f'Label distribution: {df.Label.value_counts().to_dict()}')
"
```

---

## 📋 Example End-to-End Workflow

```bash
# 1. Feature Extraction (Recommended)
python3 code/enhanced_feature_extraction.py dataset/original_data enhanced_features.csv 100 0.95

# 2. Train Models with Optimized Features
python3 code/train_models.py enhanced_features.csv models_out

# 3. Test Models with Preprocessing
python3 live_data/live_predict.py --models models_out --model xgboost --csv-dir dataset/original_data

# (wont work on 1sec cause window is set for 10sec)

# 4. Live Prediction
python3 live_data/live_predict.py --models models_out --model xgboost --eeg --auto-stream

# 5. Check Performance Metrics
cat visualization/model_accuracy_summary.txt
```

---

## 🎉 Success Indicators

When everything is working correctly, you should see:

1. **Feature Extraction**: 
   ```
   Feature Extraction COMPLETED
   Final shape: (2479, 101)
   Selected 100 most important features
   ```

2. **Model Training**:
   ```
   Per-Class Accuracy:
     Relaxed     : 97.81%
     Neutral     : 92.19%
     Concentrating: 97.40%
   Overall Accuracy: 95.72%
   ```

---

## 📚 Additional Resources

- **Feature Analysis**: Check `visualization/` for detailed feature importance plots
- **Model Artifacts**: Preprocessing artifacts saved in `preprocessing_artifacts/`
- **Performance Logs**: Training metrics saved in `visualization/model_accuracy_summary.txt`
- **Live Data**: Recordings saved during live sessions for later analysis

---

**The enhanced pipeline provides state-of-the-art EEG mental state classification with improved accuracy, reduced complexity, and comprehensive performance tracking.**
