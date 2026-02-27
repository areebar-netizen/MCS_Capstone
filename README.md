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
python3 code/enhanced_feature_extraction.py dataset/original_data enhanced_features.csv 100 0.95
```

**Arguments:**
- `dataset/original_data`: Training data directory
- `enhanced_features.csv`: Output features file
- `100`: Number of top features to select
- `0.95`: Correlation threshold for redundancy removal


### Step 2: Model Training

```bash
python3 code/train_models.py enhanced_features.csv models_out
```

**What happens:**
- Trains RandomForest, XGBoost, and Stacked models
- Saves models to `models_out/`
- Saves preprocessing artifacts to `preprocessing_artifacts/`
- Generates performance metrics in `visualization/`

### 4. Recording Your EEG Data

**Step 4.1: Start Muse Stream**
```bash
python3 -m muselsl stream
```

**Step 4.2: Record Your Sessions**
```bash
# Record 1-5 minutes for each mental state
python3 live_data/live_predict.py --eeg --models models_out --model xgboost --duration 1 --raw-out dataset/our_data/[your_name]_new/[your_name]_relaxed_1min.csv

python3 live_data/live_predict.py --eeg --models models_out --model xgboost --duration 2 --raw-out dataset/our_data/[your_name]_new/[your_name]_neutral_2min.csv

python3 live_data/live_predict.py --eeg --models models_out --model xgboost --duration 3 --raw-out dataset/our_data/[your_name]_new/[your_name]_concentrating_3min.csv
```

**Arguments:**
- `--eeg`: Use live EEG mode
- `--models models_out`: Model directory
- `--model xgboost`: Model type (xgboost/random_forest/stacked_model)
- `--duration X`: Recording duration in minutes (1-5)
- `--raw-out`: Output file for raw EEG data

**Repeat for different durations (1-5 minutes) and mental states (relaxed/neutral/concentrating)**

### 5. Process Recorded Data

```bash
# Process all your recorded files and save summary
python3 live_data/live_predict.py --models models_out --model xgboost --csv-dir dataset/our_data/[your_name]_new --summary-out dataset/[your_name]_results.csv
```

**Arguments:**
- `--csv-dir`: Directory with your recorded CSV files
- `--summary-out`: Output file for prediction summary

## 📊 Expected Performance

**Model Accuracy:**
- XGBoost: 96.53% (Relaxed: 97.73%, Neutral: 93.68%, Concentrating: 98.32%)
- RandomForest: 95.72% (Relaxed: 97.81%, Neutral: 92.19%, Concentrating: 97.40%)

**Features:**
- Original: 854 features
- After optimization: 100 features (88% reduction)

## 🔧 Troubleshooting

**Common Issues:**
- **No EEG stream**: Run `python3 -m muselsl stream` first
- **Import errors**: Run from repository root, activate venv
- **Model loading failed**: Ensure `models_out/` and `preprocessing_artifacts/` exist
- **Feature mismatch**: Re-run feature extraction and training

**Verification:**
```bash
# Check models exist
ls models_out/
ls preprocessing_artifacts/

# Test feature extraction
python3 -c "
from EEG_feature_extraction_adv import generate_feature_vectors_from_samples
vectors, headers = generate_feature_vectors_from_samples('dataset/test/10sec.csv', 150, 1.0)
print(f'Success: {vectors.shape}')
"
```

## 📝 Quick Reference Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Training (one-time)
python3 code/enhanced_feature_extraction.py dataset/original_data enhanced_features.csv 100 0.95
python3 code/train_models.py enhanced_features.csv models_out

# Recording
python3 -m muselsl stream
python3 live_data/live_predict.py --eeg --models models_out --model xgboost --duration 1 --raw-out dataset/our_data/name_new/name_relaxed_1min.csv

# Processing
python3 live_data/live_predict.py --models models_out --model xgboost --csv-dir dataset/our_data/name_new --summary-out dataset/name_results.csv
```

---

**System provides real-time EEG mental state classification with 96%+ accuracy using advanced feature extraction and ensemble machine learning.**
