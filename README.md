# MCS_Capstone

This project develops a personalized study optimization system using consumer-grade EEG technology (Muse 02 2016 headband) to provide real-time focus tracking and data-driven technique recommendations. By continuously monitoring brainwave patterns during study sessions, the system can detect when focus declines and increases, recommend optimal break timing, and identify which study techniques work best for each individual under different conditions. We aim to implement this system through a user-friendly app meant to promote studying and focus. Unlike generic productivity apps, our system adapts to each user's unique cognitive patterns, learning from their EEG data to provide increasingly personalized recommendations over time. We plan on making this data valuable to the user to access as well so that they can further study their own focus and attention patterns.
# secondBrain — EEG Feature Extraction and Mental State Classification

A comprehensive toolkit for real‑time EEG signal processing, feature extraction, and classification of cognitive states (relaxed / neutral / concentrating). The pipeline is designed to be **consistent from offline training to live deployment**: the same feature extraction, artifact removal, and model loading code is used everywhere.

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

### Machine Learning
- **Classifiers**: Random Forest, XGBoost, and a stacked ensemble.
- **Feature selection** – `SelectFromModel` with RandomForest importance (median threshold).
- **Hyperparameter tuning** – Grid search with cross‑validation, class‑weighting for imbalance.
- **Regularization**:
  - Random Forest: limited depth, higher min samples per split/leaf, bootstrap sampling.
  - XGBoost: `reg_alpha` (L1), `reg_lambda` (L2), `subsample`, `colsample_bytree`, early stopping.
- **Model persistence** – Saved as `.joblib` files for later use.

### Visualization
- **Offline timeline** – Coloured strip of predictions for a single file; play/pause/step controls.
- **Live GUI** – Real‑time display of predictions, confidence, and a scrolling coloured bar.
- **Raw signal logging** – Save incoming LSL samples to CSV while predicting.

### Live Processing
- **LSL integration** – Connects to any EEG stream (e.g., Muse, OpenBCI) via Lab Streaming Layer.
- **In‑memory feature extraction** – Same feature code used on the rolling buffer.
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

Project Structure
secondBrain/
├── code/                          # Core processing scripts
│   ├── EEG_feature_extraction_adv.py   # Feature extraction + ICA cleaning
│   ├── EEG_generate_training_matrix.py # Build feature matrix from raw CSVs
│   ├── train_models.py                # Train, tune, save models
│   ├── predict_test.py               # Batch prediction on test directory
│   ├── visualize_predictions.py      # Offline GUI for a single CSV
│   └── check_features_numeric.py     # Utility: verify feature CSV is numeric
├── live_data/                     # Live acquisition and visualization
│   ├── live_predict.py            # Main LSL reader + live prediction GUI
│   ├── Stream.py                  # Handles LSL stream connection to MUSE device
│   ├── band_power.py              # Computes and visualizes band powers (alpha, beta, delta, gamma, theta)
│   ├── vis.py                     # Real-time EEG signal visualization using pyqtgraph
│   ├── bp.py                      # Band power calculation utilities
│   └── numeric_data.py            # Numeric data processing utilities
├── dataset/                       # Example raw EEG recordings
│   ├── original_data/            # Labelled training files (name-state-index)
│   └── test/                     # Held‑out test files
├── models_out/                   # Default output folder for trained models
├── requirements.txt
└── README.md


### Training Models

train_models.py performs feature selection, hyperparameter tuning, and saves the best models.

```bash
python code/train_models.py features.csv models_out 
```
What happens inside:

Load data – Splits into X (features) and y (labels). If no Label column, the last column is treated as the label.
Feature selection – Trains a quick RandomForest, keeps features with importance above the median, saves the selector as feature_selector.joblib.
Random Forest – Grid search over:

max_depth, min_samples_split, min_samples_leaf, max_features, class_weight, max_samples.
Evaluation with f1_weighted and 5‑fold stratified CV.
XGBoost (if installed) –

Fixed regularisation: max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0.
Early stopping with 20 rounds on a validation split.
Sample weights for class imbalance.
Stacked model – Combines the tuned RandomForest and a fresh XGBoost using a StackingClassifier with a RandomForest meta‑learner.
Save models – random_forest.joblib, xgboost.joblib, stacked_model.joblib.
Feature importance analysis – Generates bar plots and CSV files in models_out/feature_importance/.

### Offline Prediction & Evaluation

Use predict_test.py to run a trained model on all CSV files in a test directory.

```bash
python code/predict_test.py \
    --models models_out \
    --model random_forest \
    --testdir dataset/test \
    --out prediction_summary.csv
```

Each file is processed with the same window parameters as training (150 samples, 1.0 s period, 0.5 s step).
Predictions are written to prediction_summary.csv with columns:

filename, n_windows, total_seconds, relaxed_seconds, neutral_seconds, concentrating_seconds, predicted_label, confidence.
Confidence is the mean of predict_proba maximum per window (or 1.0 if unavailable).

### Offline Visualisation

visualize_predictions.py opens a GUI that replays the per‑window predictions of a single CSV file.

```bash
python code/visualize_predictions.py \
    --models models_out \
    --model random_forest \
    --file dataset/test/10sec.csv
```
A coloured timeline (blue = relaxed, gray = neutral, green = concentrating) with a red vertical line showing the current position.
Large label panel, play/pause/step controls, and adjustable step speed.
Useful for inspecting how the model behaves over time.

### Live Prediction (EEG headset)

live_predict.py connects to an LSL EEG stream, performs real‑time filtering, feature extraction, and prediction, and visualises the results.

Basic usage:

```bash
python live_data/live_predict.py --models models_out --model stacked_model
```
With a Muse headset (automatic stream start):

```bash
python live_data/live_predict.py --eeg --auto-stream --raw-out live_recording.csv
```
Replay a pre‑recorded CSV (mock mode):

```bash
python live_data/live_predict.py --mock --replay dataset/test/10sec.csv
```
Important parameters:

--period – Window length in seconds (default 1.0).
--nsamples – Resampled points per window (default 150, must match training).
--min-buffer-sec – Minimum data before first prediction (default 1.5).
--raw-out – Save all incoming LSL samples to a CSV file.
--summary-out – Append predictions with timestamps to a CSV (can resume from previous runs).

### Live GUI features:

Top panel: raw EEG traces of the last few seconds (scrolls automatically).
Middle panel: coloured timeline of recent predictions (each bar = 0.5 s).
Bottom panel: large label showing current predicted state and confidence (if available).
Warning label for connection issues or poor signal quality (e.g., "Device may be incorrectly worn").

### Troubleshooting

Problem Likely solution
No EEG stream found Ensure the device is broadcasting LSL (e.g., muselsl stream).
Device may be incorrectly worn  Check electrode contacts, apply conductive gel, restart the acquisition.
No feature windows generated  Recording too short – reduce --period or increase --nsamples.
ICA did not converge  Fallback to PCA is automatic; the pipeline continues.
GUI does not start  Install PyQt5 / pyqtgraph: pip install PyQt5 pyqtgraph.
XGBoost import error  Install XGBoost: pip install xgboost (may require compilers).

### File Descriptions (Detailed)

File  Purpose
EEG_feature_extraction_adv.py Core engine. Implements forward‑fill, ICA/PCA cleaning, bandpass filtering, and all feature calculations (statistical, spectral, Hjorth, entropy, etc.). Provides two entry points: generate_feature_vectors_from_samples (disk) and generate_feature_vectors_from_matrix (memory).
EEG_generate_training_matrix.py Builds the training feature matrix from a directory of labelled raw EEG CSVs. Uses the extractor and appends labels.
train_models.py Trains RandomForest, XGBoost, and stacked models with feature selection and hyperparameter optimisation. Saves models + feature selector.
predict_test.py Batch‑mode prediction on a test folder. Produces per‑file summaries and overall statistics.
visualize_predictions.py  Offline GUI for a single prediction run. Plays back per‑window labels on a timeline.
check_features_numeric.py Quick sanity check: ensures all feature columns (except Label) are numeric.
live_predict.py Live LSL acquisition, real‑time filtering, feature extraction, prediction, and GUI visualisation. Includes mock/replay modes and raw data logging.

### Example End‑to‑End Workflow

```bash
# 1. Generate training features from raw recordings
python code/EEG_generate_training_matrix.py dataset/original_data features.csv

# 2. Train models (RandomForest, XGBoost, Stacked) with feature selection
python code/train_models.py features.csv models_out

# 3. Evaluate on test set and produce summary
python code/predict_test.py --models models_out --model stacked_model --testdir dataset/test --out summary.csv

# 4. Visually inspect predictions of a single test file
python code/visualize_predictions.py --models models_out --model stacked_model --file dataset/test/10sec.csv

# 5. Run live with Muse headset (auto‑start stream, save raw data, and show predictions)
python live_data/live_predict.py --eeg --auto-stream --raw-out session1.csv --summary-out live_summary.csv
```


