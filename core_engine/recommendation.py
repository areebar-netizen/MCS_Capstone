import pandas as pd
import uuid
import os
import warnings
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from google import genai
import sys
import django

# Add Django path to use Django models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'webapp'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'secondBrain.settings')
django.setup()

from secondBrain_App.models import UserProfile, Recommendation

warnings.filterwarnings("ignore")
load_dotenv()

# 1. DATABASE CONFIGURATION (PostgreSQL)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_SSLMODE = os.getenv("DB_SSLMODE", "auto").strip().lower()
if DB_SSLMODE == "auto":
    DB_SSLMODE = "disable" if DB_HOST in {"localhost", "127.0.0.1", "::1"} else "require"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# PostgreSQL Connection String with psycopg2
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine_kwargs = {"echo": False}
if DB_SSLMODE:
    engine_kwargs["connect_args"] = {"sslmode": DB_SSLMODE}
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine)

def get_db_session():
    return SessionLocal()

# 2. CORE LOGIC
def should_recommend(focus_state, focus_score, focus_drop_detected):
    RELAXED_THRESHOLD = 0.40
    NEUTRAL_THRESHOLD = 0.69
    return focus_drop_detected or (focus_state == 'relaxed' and focus_score < RELAXED_THRESHOLD) or (focus_state == 'neutral' and focus_score < NEUTRAL_THRESHOLD)

def basic_recommendation(user_data, session_data):
    """
    Basic recommendations for new users (<=5 sessions)
    """
    recommendations = []
    
    # Basic environment tweaks based on user profile
    sound_env = user_data.get('sound_environment', '')
    if sound_env == 'Complete Silence':
        recommendations.append("Try white noise or lo-fi music to maintain focus")
    elif sound_env == 'Soft Music' or sound_env == 'Instrumental Music':
        recommendations.append("Consider instrumental music to avoid lyrical distractions")
    elif sound_env == 'Cafe/Background Noise':
        recommendations.append("Try noise-canceling headphones for better focus")
    
    lighting = user_data.get('lighting_preference', '')
    if lighting == 'Bright Light':
        recommendations.append("Ensure proper task lighting to reduce eye strain")
    elif lighting == 'Dim Light':
        recommendations.append("Add a desk lamp for better visibility")
    
    phone_location = user_data.get('phone_location', '')
    if phone_location == 'In pocket' or phone_location == 'On desk':
        recommendations.append("Move your phone to another room to reduce distractions")
    
    session_length = user_data.get('session_length', '')
    if session_length == 'Extended (90+ min)':
        recommendations.append("Consider shorter sessions with regular breaks")
    elif session_length == 'Short (15-30 min)':
        recommendations.append("Try longer study sessions for deeper focus")
    
    return {
        'recommendation_category': 'basic',
        'message': f"Hi {user_data.get('name', 'there')}! Here are some quick tips: " + "; ".join(recommendations[:3]),
        'stimulus_name': 'basic_environment',
        'trigger_reason': 'new_user_setup'
    }

def generate_ai_recommendation(user_data, session_data, summary_data=None):
    """
    Unified AI prompt that uses the rich profile data from the 10-section survey.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Construct a high-context prompt using new survey fields
    context = f"""
    User Profile: {user_data.get('name', 'Student')}, Age {user_data.get('age', 'N/A')}, Level: {user_data.get('academic_level', 'N/A')}.
    Learning Style: {user_data.get('learning_style', 'N/A')}. Main Goals: {user_data.get('main_goals', 'N/A')}.
    Current Session: At {session_data.get('session_location', 'N/A')}, Stress: {session_data.get('stress_level_pre', 'N/A')}/10.
    Habits: Prefers {user_data.get('sound_environment', 'N/A')} and {user_data.get('lighting_preference', 'N/A')}.
    Distractions: Struggles with {user_data.get('distractions', 'N/A')}. Phone is {user_data.get('phone_location', 'N/A')}.
    Study Patterns: {user_data.get('session_length', 'N/A')} sessions, prefers {user_data.get('study_time_of_day', 'N/A')}.
    Caffeine: {'Yes' if user_data.get('consumes_caffeine') else 'No'}, {user_data.get('caffeine_timing', 'N/A')}.
    """
    
    prompt = f"You are an AI Study Coach. {context} The user is losing focus. Generate a 2-line fun recommendation and 3 specific study environment tweaks."
    
    response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
    return response.text

def save_recommendation(user_email, session_id, inference_id, recommendation_data):
    """
    Save recommendation to PostgreSQL database using Django models
    """
    try:
        recommendation = Recommendation.objects.create(
            user_id=user_email,  # Using email as user_id
            session_id=session_id,
            inference_id=inference_id,
            recommendation_category=recommendation_data.get('recommendation_category', 'ai'),
            stimulus_name=recommendation_data.get('stimulus_name', 'ai_coach'),
            trigger_reason=recommendation_data.get('trigger_reason', 'focus_drop'),
            message=recommendation_data.get('message', '')
        )
        return True
    except Exception as e:
        print(f"Error saving recommendation: {e}")
        return False

def get_user_session_count(user_email):
    """
    Get total number of sessions for a user to determine phase
    """
    db = get_db_session()
    try:
        result = db.execute(text("""
            SELECT COUNT(DISTINCT session_id) as session_count 
            FROM model_inference 
            WHERE user_id = :user_email
        """), {'user_email': user_email})
        
        session_count = result.scalar() or 0
        return session_count
    except Exception as e:
        print(f"Error getting session count: {e}")
        return 0
    finally:
        db.close()

def get_user_data_from_db(user_email):
    """
    Get user data from Django UserProfile model with human-readable text
    """
    try:
        user_profile = UserProfile.objects.get(email=user_email)
        
        # Helper functions to map indices to human-readable text
        def get_academic_level_text(level_id):
            levels = {
                0: 'High School',
                1: 'Undergraduate', 
                2: 'Graduate/Masters',
                3: 'PhD/Doctoral',
                4: 'Professional/Continuing Education',
                5: 'Other'
            }
            return levels.get(int(level_id), 'Not Set')
        
        def get_sleep_quality_text(quality_id):
            qualities = {
                0: 'Poor (frequently disrupted)',
                1: 'Fair (occasional issues)',
                2: 'Good (generally restful)',
                3: 'Excellent (consistently deep)'
            }
            return qualities.get(int(quality_id), 'Not Set')
        
        def get_alert_time_text(time_id):
            times = {
                0: 'Early Morning (5am-9am)',
                1: 'Mid-Morning to Afternoon (9am-5pm)',
                2: 'Evening to Late Night (5pm-12am)',
                3: 'Late Night (12am-5am)'
            }
            return times.get(int(time_id), 'Not Set')
        
        def get_learning_style_text(style_id):
            styles = {
                0: 'Visual Learner',
                1: 'Auditory Learner',
                2: 'Kinesthetic Learner',
                3: 'Reading/Writing Learner'
            }
            return styles.get(int(style_id), 'Not Set')
        
        def get_session_length_text(length_id):
            lengths = {
                0: 'Short (15-30 min)',
                1: 'Medium (30-60 min)',
                2: 'Long (60-90 min)',
                3: 'Extended (90+ min)'
            }
            return lengths.get(int(length_id), 'Not Set')
        
        def get_sound_environment_text(env_id):
            environments = {
                0: 'Complete Silence',
                1: 'White Noise',
                2: 'Soft Music',
                3: 'Nature Sounds',
                4: 'Cafe/Background Noise',
                5: 'Instrumental Music'
            }
            return environments.get(int(env_id), 'Not Set')
        
        def get_lighting_preference_text(pref_id):
            lighting = {
                0: 'Bright Light',
                1: 'Dim Light',
                2: 'Natural Light',
                3: 'Warm Light',
                4: 'Cool Light'
            }
            return lighting.get(int(pref_id), 'Not Set')
        
        def get_phone_location_text(location_id):
            locations = {
                0: 'In another room',
                1: 'In pocket',
                2: 'On desk',
                3: 'In backpack',
                4: 'Turned off'
            }
            return locations.get(int(location_id), 'Not Set')
        
        def get_study_time_text(time_id):
            times = {
                0: 'Early Morning',
                1: 'Morning',
                2: 'Afternoon', 
                3: 'Evening',
                4: 'Night'
            }
            return times.get(int(time_id), 'Not Set')
        
        def get_caffeine_timing_text(timing_id):
            timing = {
                0: 'Before studying',
                1: 'During studying',
                2: 'After studying',
                3: 'Between study sessions'
            }
            return timing.get(int(timing_id), 'Not Set')
        
        def get_exercise_frequency_text(freq_id):
            frequency = {
                0: 'Never',
                1: 'Rarely (1-2 times/week)',
                2: 'Sometimes (3-4 times/week)',
                3: 'Often (5-6 times/week)',
                4: 'Daily'
            }
            return frequency.get(int(freq_id), 'Not Set')
        
        def get_eating_timing_text(timing_id):
            timing = {
                0: 'Before studying',
                1: 'During studying',
                2: 'After studying',
                3: 'Between study sessions'
            }
            return timing.get(int(timing_id), 'Not Set')
        
        return {
            'name': user_profile.name,
            'age': user_profile.age,
            'academic_level': get_academic_level_text(user_profile.academic_level),
            'alert_time': get_alert_time_text(user_profile.alert_time),
            'sleep_hours': user_profile.sleep_hours,
            'sleep_quality': get_sleep_quality_text(user_profile.sleep_quality),
            'consumes_caffeine': user_profile.consumes_caffeine,
            'caffeine_types': user_profile.caffeine_types.strip('[]').replace("'", '').split(',') if user_profile.caffeine_types else [],
            'caffeine_servings': user_profile.caffeine_servings,
            'caffeine_timing': get_caffeine_timing_text(user_profile.caffeine_timing),
            'learning_style': get_learning_style_text(user_profile.learning_style),
            'study_subjects': user_profile.study_subjects.strip('[]').replace("'", '').split(',') if user_profile.study_subjects else [],
            'session_length': get_session_length_text(user_profile.session_length),
            'takes_breaks': user_profile.takes_breaks,
            'study_time_of_day': get_study_time_text(user_profile.study_time_of_day),
            'procrastination_level': user_profile.procrastination_level,
            'study_location': user_profile.study_location.strip('[]').replace("'", '').split(',')[0] if user_profile.study_location else 'Not Set',
            'sound_environment': get_sound_environment_text(user_profile.sound_environment),
            'lighting_preference': get_lighting_preference_text(user_profile.lighting_preference),
            'phone_location': get_phone_location_text(user_profile.phone_location),
            'distractions': user_profile.distractions.strip('[]').replace("'", '').split(',') if user_profile.distractions else [],
            'exercise_frequency': get_exercise_frequency_text(user_profile.exercise_frequency),
            'eating_timing': get_eating_timing_text(user_profile.eating_timing),
            'health_conditions': user_profile.health_conditions.strip('[]').replace("'", '').split(',') if user_profile.health_conditions else [],
            'main_goals': user_profile.main_goals.strip('[]').replace("'", '').split(',')[0] if user_profile.main_goals else 'Not Set',
            'study_effectiveness': user_profile.study_effectiveness
        }
    except UserProfile.DoesNotExist:
        print(f"User profile not found for {user_email}")
        return {}
    except Exception as e:
        print(f"Error getting user data: {e}")
        return {}

def main(user_email, session_id):
    """
    Main recommendation function using email as user identifier
    """
    db = get_db_session()
    try:
        # Load user data using Django models
        user_data = get_user_data_from_db(user_email)
        if not user_data:
            return "User profile not found."
        
        # Load inference data
        df_inf = pd.read_sql(text("SELECT * FROM model_inference WHERE user_id = :email AND session_id = :session ORDER BY inference_id DESC LIMIT 1"), 
                             engine, params={'email': user_email, 'session': session_id})
        
        if df_inf.empty:
            return "No inference data found."

        inference = df_inf.iloc[0]
        
        # Check if recommendation should be generated
        if not should_recommend(inference['focus_state'], inference['focus_score'], inference['focus_drop_detected']):
            return "No recommendation needed."
        
        # Get session count for phase logic
        total_sessions = get_user_session_count(user_email)
        
        # Prepare session data
        session_data = {
            'session_location': user_data.get('study_location', 'unknown'),
            'stress_level_pre': inference.get('stress_level', 5)
        }
        
        # Phase-based recommendation logic
        if total_sessions <= 5:
            # New user - basic recommendations
            recommendation_data = basic_recommendation(user_data, session_data)
        else:
            # Experienced user - AI recommendations
            ai_message = generate_ai_recommendation(user_data, session_data)
            recommendation_data = {
                'recommendation_category': 'ai',
                'message': ai_message,
                'stimulus_name': 'ai_coach',
                'trigger_reason': 'focus_drop'
            }
        
        # Save recommendation to database
        success = save_recommendation(user_email, session_id, inference['inference_id'], recommendation_data)
        
        if success:
            return recommendation_data['message']
        else:
            return "Failed to save recommendation."
            
    except Exception as e:
        print(f"Error in main recommendation function: {e}")
        return f"Error: {str(e)}"
    finally:
        db.close()

if __name__ == "__main__":
    # Test function
    result = main("test@example.com", "test_session_123")
    print(result)
