# MCS_Capstone

This project develops a personalized study optimization system using consumer-grade EEG technology (Muse 02 2016 headband) to provide real-time focus tracking and data-driven technique recommendations. By continuously monitoring brainwave patterns during study sessions, the system can detect when focus declines and increases, recommend optimal break timing, and identify which study techniques work best for each individual under different conditions. We aim to implement this system through a user-friendly app meant to promote studying and focus. Unlike generic productivity apps, our system adapts to each user's unique cognitive patterns, learning from their EEG data to provide increasingly personalized recommendations over time. We plan on making this data valuable to be user to access as well so that they can further study their own focus and attention patterns.

# secondBrain — Enhanced EEG Feature Extraction, Mental State Classification, and Web Application

A comprehensive toolkit for real‑time EEG signal processing, enhanced feature extraction, classification of cognitive states (relaxed / neutral / concentrating), and a complete web application with user authentication, profile management, and AI-powered recommendations. The pipeline includes **advanced feature preprocessing** with scaling, redundancy removal, and intelligent feature selection for optimal model performance.

## Project Overview

### Core Features
- **Real-time EEG Processing**: Live brainwave monitoring and mental state classification
- **Focus Tracking**: Per-second focus scoring with detailed analytics
- **Personalized Recommendations**: AI-powered study technique suggestions
- **User Management**: Secure authentication with comprehensive profiling
- **Data Visualization**: Interactive dashboards and session history

### Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Django Web    │    │     Redis        │    │  Celery Worker  │
│   Server        │◄──►│   Message        │◄──►│  Real-time      │
│   (Frontend)    │    │     Broker       │    │  Processing     │
│                 │    │                  │    │  (Per-second)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                           │
         │                                           ▼
         │                                    ┌─────────────────┐
         │                                    │   PostgreSQL    │
         └───────────────────────────────────►│   (User Data)   │
                                              └─────────────────┘
```

## New Features (Latest Update)

### Real-Time EEG Pipeline
- **Per-Second Processing**: Real-time EEG data processing every second
- **Focus Scoring**: Live calculation (Relaxed: 0.3, Neutral: 0.6, Concentrating: 1.0)
- **Live Broadcasting**: Task state updates via Celery
- **CSV Streaming**: Thread-safe real-time data logging
- **Session Summaries**: PostgreSQL storage of session-level statistics

### OTP Authentication System
- **Email-based verification** with 6-digit OTP codes
- **Session management** for secure user access
- **Smart routing**: New users → Survey, Returning users → Dashboard
- **Console OTP** for development (free, no email setup required)

### PostgreSQL Integration
- **Complete database setup** with user profiles and recommendations
- **10-section comprehensive survey** stored in PostgreSQL
- **Real-time dashboard** displaying user data and focus metrics
- **AI recommendations** powered by Gemini with rich user context

### Enhanced Web Application
- **Django-based webapp** with modern UI
- **User profile management** with survey data
- **Focus tracking calendar** and session history
- **Brainwave visualization** and recommendation system

## Project Structure

```
secondBrain/
├── core_engine/
│   ├── __init__.py                       # Python package initialization
│   ├── EEG_feature_extraction_adv.py      # Advanced EEG feature extraction
│   ├── enhanced_feature_extraction.py    # Enhanced preprocessing pipeline
│   ├── EEG_generate_training_matrix.py   # Training data generation
│   ├── live_predict.py                  # Live prediction & recording pipeline
│   ├── recommendation.py                 # AI-powered recommendation engine
│   └── artifacts/                        # Model files (.joblib, .pkl, .txt)
│       ├── preprocessing_artifacts/      # Feature scaling & selection artifacts
│       │   ├── feature_scaler.joblib      # Feature scaling parameters
│       │   └── feature_selection_info.pkl  # Selected feature indices
│       ├── feature_importance/           # Feature analysis results
│       ├── random_forest.joblib        # RandomForest model
│       ├── xgboost.joblib              # XGBoost model
│       ├── stacked_model.joblib          # Stacked ensemble model
│       └── selected_features.txt        # Feature names list
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
│   ├── train_models.py                 # Model training pipeline
│   ├── predict_test.py                 # Prediction testing
│   ├── feature_analysis.py            # Feature analysis and EDA
│   └── model_evaluation/               # Model performance analysis
├── webapp/                             # Django web application
│   ├── secondBrain/                    # Django project configuration
│   │   ├── __init__.py                 # Celery app initialization
│   │   ├── celery.py                   # Celery configuration
│   │   ├── settings.py                 # Django settings
│   │   └── urls.py                    # Main URL routing
│   └── secondBrain_App/                # Main Django app
│       ├── models.py                   # Database models (User, Prediction, SessionSummary)
│       ├── views.py                    # API endpoints and business logic
│       ├── tasks.py                    # Celery tasks for async processing
│       ├── tasks_realtime.py           # Real-time EEG processing tasks
│       ├── urls.py                    # App-specific URL routing
│       ├── templates/                  # HTML templates
│       ├── static/                     # CSS, JavaScript, images
│       └── migrations/                 # Database migration files
├── dataset/                            # Data storage
│   ├── original_data/                 # Raw EEG recordings
│   ├── temp_logs/                    # Temporary processing files
│   └── our_data/                     # User session data
├── .env.example                        # Environment configuration template
└── requirements.txt                     # Python dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- Node.js 14+ (for frontend development)
- Muse 02 2016 EEG headset
- Environment configuration file (see `.env.example`)

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/areebar-netizen/MCS_Capstone.git
cd MCS_Capstone
```

2. **Create Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Database Setup**
```bash
# Start PostgreSQL
brew services start postgresql

# Create database
createdb -h localhost -U postgres secondbrain

# Run migrations
cd webapp
python manage.py makemigrations
python manage.py migrate
```

5. **Redis Setup**
```bash
# Install Redis
brew install redis

# Start Redis
brew services start redis
```

6. **Environment Configuration**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your database and email settings
# IMPORTANT: Update DB_PASSWORD, SECRET_KEY, and email settings
```

## Usage

### Development Setup

1. **Start Services** (3 terminals needed)

**Terminal 1: Django Server**
```bash
cd webapp
source ../.venv/bin/activate
python manage.py runserver
```

**Terminal 2: Celery Worker**
```bash
cd webapp
source ../.venv/bin/activate
celery -A secondBrain worker -l info
```

**Terminal 3: Redis** (if not running as service)
```bash
redis-server
```

2. **Access Application**
- Web Interface: http://localhost:8000
- Admin Panel: http://localhost:8000/admin

### Model Training (Required for First-Time Setup)

Before using the EEG system, you need to train the machine learning models:

1. **Generate Enhanced Features**
```bash
# Process raw EEG data and extract enhanced features
python3 core_engine/enhanced_feature_extraction.py dataset/original_data dataset/temp_logs/enhanced_features.csv 100 0.95
```

2. **Train Classification Models**
```bash
# Train RandomForest and XGBoost models
python3 research/train_models.py dataset/temp_logs/enhanced_features.csv core_engine/artifacts
```

3. **Verify Model Training**
```bash
# Check that model files are created
ls -la core_engine/artifacts/
# Should show: random_forest.joblib, xgboost.joblib, stacked_model.joblib, feature_selector.joblib, selected_features.txt, feature_importance/

# Check that preprocessing artifacts are created
ls -la core_engine/artifacts/preprocessing_artifacts/
# Should show: feature_scaler.joblib, feature_selection_info.pkl
```

### EEG Session Workflow

1. **Connect EEG Hardware**
```bash
# Start Muse streaming
python3 -m muselsl stream
```

2. **Start Session via Web Interface**
- Navigate to dashboard
- Click "Start EEG Session"
- Set duration (minutes)
- Monitor real-time focus

3. **View Results**
- Real-time focus scores updated every second
- Session summary saved to PostgreSQL
- CSV data saved to `dataset/our_data/`

### API Endpoints

### Authentication
- `GET /` - Email entry page
- `POST /send-otp/` - Generate and send OTP
- `POST /verify-otp/` - Verify OTP and authenticate

### Main Application
- `GET /dashboard/` - Main dashboard (requires authentication)
- `GET /onboarding/` - 10-section survey (requires authentication)
- `POST /onboarding/` - Save survey data (requires authentication)

### EEG Processing (Real-time)
- `POST /start_realtime_eeg/` - Start real-time EEG session
- `GET /realtime_eeg_status/` - Check session status
- `POST /stop_realtime_eeg/` - Stop session and get results
- `POST /api/predict/` - Direct prediction from EEG data rows
- `POST /upload_csv/` - Upload CSV file for batch prediction

### Data Services
- Real-time PostgreSQL integration
- AI-powered recommendations via Gemini
- Session tracking and analytics

## Model Performance

### Training Results
- **RandomForest**: 95.97% accuracy
- **XGBoost**: 97.10% accuracy
- **Feature Selection**: Top 50 features selected from 800+
- **Cross-validation**: 5-fold stratified validation

### Real-time Processing
- **Latency**: <100ms per prediction
- **Memory Usage**: <500MB for full pipeline
- **Scalability**: Supports concurrent user sessions

## Technical Details

### Feature Extraction Pipeline
1. **Signal Preprocessing**: Band-pass filtering (1-45Hz)
2. **Artifact Removal**: ICA-based blink and muscle noise removal
3. **Feature Engineering**: 
   - Band power features (delta, theta, alpha, beta, gamma)
   - Connectivity metrics (coherence, phase-locking)
   - Statistical features (variance, skewness, kurtosis)
4. **Feature Selection**: Recursive feature elimination with cross-validation

### Classification Models
- **RandomForest**: Ensemble of decision trees with bootstrapping
- **XGBoost**: Gradient boosting with regularization
- **Output Classes**: Relaxed (0), Neutral (1), Concentrating (2)

### Real-time Architecture
- **Message Queue**: Redis for task distribution
- **Background Processing**: Celery workers for async ML inference
- **Data Storage**: PostgreSQL for metadata, CSV for raw data
- **State Management**: Django sessions for user context

## Troubleshooting

### Common Issues

1. **EEG Connection Failed**
   - Ensure Muse headset is paired via Bluetooth
   - Check if `muselsl` is installed: `pip install muselsl`
   - Verify device is not connected to other apps

2. **Database Connection Error**
   - Check PostgreSQL service: `brew services list | grep postgresql`
   - Verify database credentials in `.env` file
   - Run migrations: `python manage.py migrate`

3. **Celery Task Not Processing**
   - Verify Redis is running: `redis-cli ping`
   - Check worker logs: `celery -A secondBrain worker -l info`
   - Restart services if needed

4. **Authentication Issues**
   - Check email configuration in settings
   - Verify OTP generation in development console
   - Clear browser cookies and session data

5. **Port Already in Use Error**
   - **Find process using port 8000**: `lsof -i :8000`
   - **Kill the process**: `kill -9 <PID>` (replace <PID> with actual process ID)
   - **Or use different port**: `python manage.py runserver 8001`
   - **Or find and stop Django server**: `ps aux | grep python | grep runserver`

### Performance Optimization

1. **Database Optimization**
   - Add indexes to frequently queried columns
   - Use connection pooling for high traffic
   - Consider read replicas for analytics queries

2. **ML Pipeline Optimization**
   - Cache preprocessing artifacts in memory
   - Use batch processing for multiple predictions
   - Implement model quantization for faster inference

