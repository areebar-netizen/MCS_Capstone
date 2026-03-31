# MCS_Capstone

This project develops a personalized study optimization system using consumer-grade EEG technology (Muse 02 2016 headband) to provide real-time focus tracking and data-driven technique recommendations. By continuously monitoring brainwave patterns during study sessions, the system can detect when focus declines and increases, recommend optimal break timing, and identify which study techniques work best for each individual under different conditions. We aim to implement this system through a user-friendly app meant to promote studying and focus. Unlike generic productivity apps, our system adapts to each user's unique cognitive patterns, learning from their EEG data to provide increasingly personalized recommendations over time. We plan on making this data valuable to be user to access as well so that they can further study their own focus and attention patterns.

# secondBrain — Enhanced EEG Feature Extraction and Mental State Classification

A comprehensive toolkit for real‑time EEG signal processing, enhanced feature extraction, and classification of cognitive states (relaxed / neutral / concentrating). The pipeline includes **advanced feature preprocessing** with scaling, redundancy removal, and intelligent feature selection for optimal model performance.

## Project Structure

```
secondBrain/
├── core_engine/
│   ├── __init__.py                       # Python package initialization
│   ├── EEG_feature_extraction_adv.py      # Advanced EEG feature extraction
│   ├── enhanced_feature_extraction.py    # Enhanced preprocessing pipeline
│   ├── EEG_generate_training_matrix.py   # Training data generation
│   ├── live_predict.py                  # Live prediction & recording pipeline
│   └── artifacts/                        # Model files (.joblib, .pkl, .txt)
├── data_pipeline/
│   ├── setup/                           # Hardware connection scripts
│   │   ├── ble_scan.py                  # Bluetooth scanning
│   │   ├── connectmuse.py               # Muse headset connection
│   │   └── Stream.py                    # LSL streaming
│   └── analysis/                        # Live processing & visualization
│       ├── vis.py                       # Data visualization
│       ├── band_power.py                # Band power analysis
│       └── bp.py                        # Band power utilities
├── research/                            # Training, testing, and EDA
│   ├── train_models.py                  # Model training script
│   ├── predict_test.py                  # Test prediction script
│   ├── check_features_numeric.py        # Feature validation
│   ├── feature_analysis.py              # Feature analysis and visualization
│   └── numeric_data.py                  # Numeric data processing utilities
├── webapp/                              # Django web application
│   ├── manage.py                        # Django management script
│   ├── secondBrain/                     # Django project directory
│   │   ├── __init__.py
│   │   ├── asgi.py                       # ASGI config
│   │   ├── settings.py                   # Django settings
│   │   ├── urls.py                       # URL routing
│   │   └── wsgi.py                       # WSGI config
│   └── secondBrain_App/                 # Django app
│       ├── __init__.py
│       ├── admin.py                      # Django admin
│       ├── apps.py                       # App config
│       ├── migrations/                   # Database migrations
│       ├── models.py                     # Database models
│       ├── tests.py                      # Unit tests
│       └── views.py                      # View logic
├── dataset/
│   ├── temp_logs/                       # Temporary CSV files and logs
│   └── test/                            # Test dataset
└── .env                                 # Environment variables (Django + AWS + DB)
```

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
---

## Database Setup (PostgreSQL)

### Step 1: Install PostgreSQL
```bash
# macOS
brew install postgresql
brew services start postgresql

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Step 2: Create Database
```bash
# Create database and user
sudo -u postgres psql
CREATE DATABASE secondbrain;
CREATE USER your_username WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE secondbrain TO your_username;
\q
```

### Step 3: Configure Environment
```bash
# Update .env in project root with your database credentials
DB_NAME=secondbrain
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

---

## Complete Workflow from Scratch

### Step 1: Enhanced Feature Extraction

**Generate enhanced features from raw data (Recommended)**
```bash
python3 core_engine/enhanced_feature_extraction.py dataset/original_data dataset/temp_logs/enhanced_features.csv 100 0.95
```

**Arguments:**
- `dataset/original_data`: Training data directory
- `dataset/temp_logs/enhanced_features.csv`: Output features file
- `100`: Number of top features to select
- `0.95`: Correlation threshold for redundancy removal


### Step 2: Model Training

```bash
python3 research/train_models.py dataset/temp_logs/enhanced_features.csv core_engine/artifacts
```

**What happens:**
- Trains RandomForest, XGBoost, and Stacked models
- Saves models to `core_engine/artifacts/`
- Saves preprocessing artifacts to `core_engine/artifacts/`
- Generates performance metrics in `dataset/temp_logs/`

### 4. Recording Your EEG Data

**Step 4.1: Start Muse Stream**
```bash
python3 -m muselsl stream
```

**Step 4.2: Record Your Sessions**
```bash
# Record 1-5 minutes for each mental state
python3 core_engine/live_predict.py --eeg --models core_engine/artifacts --model xgboost --duration 1 --raw-out dataset/our_data/[your_name]_new/[your_name]_relaxed_1min.csv

python3 core_engine/live_predict.py --eeg --models core_engine/artifacts --model xgboost --duration 2 --raw-out dataset/our_data/[your_name]_new/[your_name]_neutral_2min.csv

python3 core_engine/live_predict.py --eeg --models core_engine/artifacts --model xgboost --duration 3 --raw-out dataset/our_data/[your_name]_new/[your_name]_concentrating_3min.csv
```

**Arguments:**
- `--eeg`: Use live EEG mode
- `--models core_engine/artifacts`: Model directory
- `--model xgboost`: Model type (xgboost/random_forest/stacked_model)
- `--duration X`: Recording duration in minutes (1-5)
- `--raw-out`: Output file for raw EEG data

**Repeat for different durations (1-5 minutes) and mental states (relaxed/neutral/concentrating)**

### 5. Process Recorded Data

```bash
# Process all your recorded files and save summary
python3 core_engine/live_predict.py --models core_engine/artifacts --model xgboost --csv-dir dataset/our_data/[your_name]_new --summary-out dataset/temp_logs/[your_name]_results.csv
```

**Arguments:**
- `--csv-dir`: Directory with your recorded CSV files
- `--summary-out`: Output file for prediction summary

### 6. Django Webapp Setup

```bash
# Navigate to webapp directory
cd webapp

# Run database migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Start development server
python manage.py runserver
```

Access the webapp at: http://localhost:8000

## Expected Performance

**Model Accuracy:**
- XGBoost: 96.53% (Relaxed: 97.73%, Neutral: 93.68%, Concentrating: 98.32%)
- RandomForest: 95.72% (Relaxed: 97.81%, Neutral: 92.19%, Concentrating: 97.40%)

**Features:**
- Original: 854 features
- After optimization: 100 features (88% reduction)

## Troubleshooting

**Common Issues:**
- **No EEG stream**: Run `python3 -m muselsl stream` first
- **Import errors**: Run from repository root, activate venv
- **Model loading failed**: Ensure `core_engine/artifacts/` exists
- **Feature mismatch**: Re-run feature extraction and training

**Verification:**
```bash
# Check models exist
ls core_engine/artifacts/

# Test feature extraction
python3 -c "
from core_engine.EEG_feature_extraction_adv import generate_feature_vectors_from_samples
vectors, headers = generate_feature_vectors_from_samples('dataset/test/10sec.csv', 150, 1.0)
print(f'Success: {vectors.shape}')
"
```

## Quick Reference Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Training (one-time)
python3 core_engine/enhanced_feature_extraction.py dataset/original_data dataset/temp_logs/enhanced_features.csv 100 0.95
python3 research/train_models.py dataset/temp_logs/enhanced_features.csv core_engine/artifacts

# Recording
python3 -m muselsl stream
python3 core_engine/live_predict.py --eeg --models core_engine/artifacts --model xgboost --duration 1 --raw-out dataset/our_data/name_new/name_relaxed_1min.csv

# Processing
python3 core_engine/live_predict.py --models core_engine/artifacts --model xgboost --csv-dir dataset/our_data/name_new --summary-out dataset/temp_logs/name_results.csv
```

---

**System provides real-time EEG mental state classification with 96%+ accuracy using advanced feature extraction and ensemble machine learning.**
