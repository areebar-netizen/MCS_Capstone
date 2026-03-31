import warnings
warnings.filterwarnings("ignore")
# 1. IMPORT NECESSARY LIBRARIES

import pandas as pd
import uuid
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()


# DATABASE CONNECTION

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")
API_KEY     = os.getenv("GEMINI_API_KEY")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine       = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)



# 2. USER DEFINED FUNCTIONS


def get_time_of_day():
    """Returns morning, afternoon, or evening based on current time."""
    hour = datetime.now().hour
    if hour < 12:
        return 'morning'
    elif hour < 17:
        return 'afternoon'
    else:
        return 'evening'

def should_recommend(focus_state, focus_score, focus_drop_detected):
    """
    Decides if a recommendation is needed at all.
    Returns True if intervention is required.
    """
    RELAXED_THRESHOLD = 0.40
    NEUTRAL_THRESHOLD = 0.69

    if focus_drop_detected:
        return True
    if focus_state == 'relaxed' and focus_score < RELAXED_THRESHOLD:
        return True
    if focus_state == 'neutral' and focus_score < NEUTRAL_THRESHOLD:
        return True
    return False

def build_available_stimuli(preferred_str, avoided_str):
    """
    Splits preferred and avoided stimulus strings into lists.
    Returns filtered available stimuli (preferred minus avoided).
    """
    preferred = [s.strip() for s in preferred_str.split(',')] if preferred_str else []
    avoided   = [s.strip() for s in avoided_str.split(',')]   if avoided_str   else []
    available = [s for s in preferred if s not in avoided]
    return available, avoided

def rotate_stimulus(available, last_stimulus):
    """
    Rotates to next available stimulus if last one is being repeated.
    """
    for stimulus in available:
        if stimulus != last_stimulus:
            return stimulus
    return available[0] if available else 'lo_fi'

def save_recommendation(db, user_id, session_id, inference_id,
                        category, stimulus, trigger, message):
    """
    Inserts a new recommendation record into the recommendation table.
    """
    rec_id = str(uuid.uuid4())
    now    = datetime.utcnow()

    query = text("""
        INSERT INTO recommendation (
            recommendation_id, user_id, session_id, inference_id,
            recommendation_category, stimulus_name, trigger_reason,
            action_started_at, message
        ) VALUES (
            :rec_id, :user_id, :session_id, :inference_id,
            :category, :stimulus, :trigger, :now, :message
        )
    """)

    db.execute(query, {
        'rec_id':        rec_id,
        'user_id':       user_id,
        'session_id':    session_id,
        'inference_id':  inference_id,
        'category':      category,
        'stimulus':      stimulus,
        'trigger':       trigger,
        'now':           now,
        'message':       message
    })
    db.commit()

    print(f"[SAVED] Recommendation: {stimulus} ({category}) for user {user_id}")
    return rec_id


# PHASE 1 — BASIC RECOMMENDATION (Sessions 1-5)
# Uses {A}: focus_state, focus_score, focus_drop_detected
#           preferred_stimulus_types, avoided_stimulus_types,
#           sleep_quality, primary_focus_goal, time_of_day
def basic_recommendation(df_inference, df_user_profile, user_id, session_id):
    """
    Phase 1 Cold Start recommendation.
    Called when total_sessions <= 5.
    Uses only user_profile and latest model_inference data.
    """
    print(f"\n[Phase 1] Basic recommendation for user {user_id}")

    # Get latest inference for this user and session
    inference = df_inference[
        (df_inference['user_id']    == user_id) &
        (df_inference['session_id'] == session_id)
    ].sort_values('inference_id').iloc[-1]

    focus_state         = inference['focus_state']
    focus_score         = float(inference['focus_score'])
    focus_drop_detected = bool(inference['focus_drop_detected'])

    # Check if recommendation is needed
    if not should_recommend(focus_state, focus_score, focus_drop_detected):
        print(f"[Phase 1] User is concentrating. No recommendation needed.")
        return None

    # Get user profile {A}
    user          = df_user_profile[df_user_profile['user_id'] == user_id].iloc[0]
    preferred_str = user['preferred_stimulus_types']
    avoided_str   = user['avoided_stimulus_types']
    sleep_quality = user['sleep_quality']

    available, avoided = build_available_stimuli(preferred_str, avoided_str)
    games      = [s for s in available if 'game' in s]
    music      = [s for s in available if 'game' not in s]
    poor_sleep = sleep_quality and sleep_quality.lower() == 'poor'

    result = None

    if focus_state == 'relaxed':
        # Strong drop — prefer game
        if games:
            result = {
                'category': 'game',
                'stimulus': games[0],
                'trigger':  'focus_drop',
                'message':  'Focus dropped! A quick game might help you reset.'
                            if not poor_sleep else
                            'Focus dropped and you may be tired. A short game might help!'
            }
        elif music:
            result = {
                'category': 'music',
                'stimulus': music[0],
                'trigger':  'focus_drop',
                'message':  f'Focus dropped. Try some {music[0]} to re-engage.'
            }

    elif focus_state == 'neutral':
        # Soft drift — prefer music
        if music:
            result = {
                'category': 'music',
                'stimulus': music[0],
                'trigger':  'low_focus',
                'message':  f'Focus drifting. {music[0]} might help you stay on track.'
                            if not poor_sleep else
                            f'You might be tired. Some {music[0]} could help you refocus.'
            }
        elif games:
            result = {
                'category': 'game',
                'stimulus': games[0],
                'trigger':  'low_focus',
                'message':  f'Try a quick {games[0]} to maintain your focus.'
            }

    # Fallback
    if not result:
        result = {
            'category': 'music',
            'stimulus': available[0] if available else 'lo_fi',
            'trigger':  'low_focus',
            'message':  'Some background music might help you focus.'
        }

    return result


# PHASE 2 — ADVANCED RECOMMENDATION (Sessions 6+)
# TODO: Uncomment and implement when enough data is collected
# Uses {A} + {B} + {C}
# def advanced_recommendation(df_inference, df_user_profile, df_user_summary,
#                              df_feedback, user_id, session_id):
#     """
#     Phase 2 Warm Start recommendation.
#     Called when total_sessions > 5.
#     Uses user_profile + model_inference + user_summary + user_feedback.
#
#     {B}: most_effective_stimulus, least_effective_stimulus,
#          optimal_focus_time_of_day, average_feedback_rating,
#          overall_sentiment_score, average_focus_score
#
#     {C}: helpfulness_rating, overall_rating, sentiment per stimulus
#     """
#     pass



def generate_basic_recommendation(df_user_profile, df_session, user_id, session_id):

    client = genai.Client(api_key=API_KEY)

    user = df_user_profile[df_user_profile['user_id'] == user_id].iloc[0]
    preferred_str = user['preferred_stimulus_types']
    avoided_str   = user['avoided_stimulus_types']
    # sleep_quality = user['sleep_quality']

    session = df_session[(df_session['user_id'] == user_id) & (df_session['session_id'] == session_id)].iloc[0]
    session_duration = session['session_duration']
    session_location = session['session_location']
    phone_present = session['phone_present']
    energy_level_pre = session['energy_level_pre']
    stress_level_pre = session['stress_level_pre']
    time_since_waking_hours = session['time_since_waking_hours']

    contents = """You are a AI-powered Study Optimization Advisor that analyzes brainwave state data, environmental factors, and behavioral patterns to provide personalized study recommendations. The user is currently distracted. User likes {0} but avoid {1}. Their session duration was {2} and session location was {3}. User rated their pre-session energy level as {4}, pre session stress level as {5} and time since waking up is {6}. What do you recommend? Generate 1-2 line fun recommendation. Provide Recommended Study Methods with 3-4 bullet points and what is an optimal study enviroment for this user """.format(preferred_str, avoided_str, session_duration,session_location, energy_level_pre, stress_level_pre, time_since_waking_hours)
    
    print(contents)

    response = client.models.generate_content(
    model="gemini-3-flash-preview", 
    contents=contents
    )

    return response.text


def generate_feedback_recommendation(df_user_profile, df_session, df_user_summary, df_feedback, user_id, session_id):

    client = genai.Client(api_key=API_KEY)

    user          = df_user_profile[df_user_profile['user_id'] == user_id].iloc[0]
    preferred_str = user['preferred_stimulus_types']
    avoided_str   = user['avoided_stimulus_types']

    session                 = df_session[(df_session['user_id'] == user_id) & (df_session['session_id'] == session_id)].iloc[0]
    session_location        = session['session_location']
    energy_level_pre        = session['energy_level_pre']
    stress_level_pre        = session['stress_level_pre']
    time_since_waking_hours = session['time_since_waking_hours']

    summary                   = df_user_summary[df_user_summary['user_id'] == user_id].iloc[0]
    total_sessions            = summary['total_sessions']
    average_focus_score       = summary['average_focus_score']
    most_effective_stimulus   = summary['most_effective_stimulus']
    least_effective_stimulus  = summary['least_effective_stimulus']
    optimal_focus_time_of_day = summary['optimal_focus_time_of_day']
    average_feedback_rating   = summary['average_feedback_rating']
    overall_sentiment_score   = summary['overall_sentiment_score']

    user_feedback = df_feedback[df_feedback['user_id'] == user_id]
    if not user_feedback.empty:
        last_feedback       = user_feedback.sort_values('feedback_id').iloc[-1]
        last_overall_rating = last_feedback['overall_rating']
        last_sentiment      = last_feedback['sentiment']
    else:
        last_overall_rating = 'N/A'
        last_sentiment      = 'N/A'

    contents = """You are an AI-powered Study Optimization Advisor. The user is currently distracted.
    USER: likes {preferred_str}, avoids {avoided_str}, at {session_location}.
    STATE: energy {energy_level_pre}/10, stress {stress_level_pre}/10, awake {time_since_waking_hours}hrs.
    HISTORY ({total_sessions} sessions): avg focus {average_focus_score}, best stimulus {most_effective_stimulus}, worst stimulus {least_effective_stimulus}, optimal time {optimal_focus_time_of_day}.
    FEEDBACK: avg rating {average_feedback_rating}/5, sentiment {overall_sentiment_score}, last rating {last_overall_rating}/5 ({last_sentiment}).

    RULES:
    - NEVER recommend {least_effective_stimulus}
    - PRIORITIZE {most_effective_stimulus}
    - If last_overall_rating <= 2, try something different
    - If overall_sentiment_score < 0.3, be more creative

    RESPOND WITH:
    1. 1-2 line fun personalized recommendation referencing their history
    2. Recommended Study Methods (3-4 bullet points)
    3. Optimal study environment for this user
    4. One line on what to avoid
    """.format(
            preferred_str             = preferred_str,
            avoided_str               = avoided_str,
            session_location          = session_location,
            energy_level_pre          = energy_level_pre,
            stress_level_pre          = stress_level_pre,
            time_since_waking_hours   = time_since_waking_hours,
            total_sessions            = total_sessions,
            average_focus_score       = average_focus_score,
            most_effective_stimulus   = most_effective_stimulus,
            least_effective_stimulus  = least_effective_stimulus,
            optimal_focus_time_of_day = optimal_focus_time_of_day,
            average_feedback_rating   = average_feedback_rating,
            overall_sentiment_score   = overall_sentiment_score,
            last_overall_rating       = last_overall_rating,
            last_sentiment            = last_sentiment
        )

    print(contents)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents
    )

    return response.text


# 3. MAIN

def main(user_id, session_id):
    db = SessionLocal()

    try:
        print(f"\n{'='*50}")
        print(f"Recommendation Engine Started")
        print(f"User: {user_id} | Session: {session_id}")
        print(f"{'='*50}")

        
        # 4. READ ALL SOURCE TABLES INTO DATAFRAMES
        
        df_user_profile = pd.read_sql(
            "SELECT * FROM user_profile",
            con=engine
        )

        df_inference = pd.read_sql(
            "SELECT * FROM model_inference",
            con=engine
        )

        df_user_summary = pd.read_sql(
            "SELECT * FROM user_summary",
            con=engine
        )

        df_session = pd.read_sql(
            "SELECT * FROM raw_session_data",
            con=engine
        )

        # NOTE: df_feedback and df_last_recommendation not needed for Phase 1
        #
        # df_feedback = pd.read_sql(
        #     """
        #     SELECT
        #         uf.user_id,
        #         uf.session_id,
        #         uf.overall_rating,
        #         uf.helpfulness_rating,
        #         uf.ease_of_use_rating,
        #         uf.recommendation_relevance,
        #         uf.sentiment,
        #         r.stimulus_name
        #     FROM user_feedback uf
        #     LEFT JOIN recommendation r
        #         ON uf.recommendation_id = r.recommendation_id
        #     """,
        #     con=engine
        # )

        rec_sql = """SELECT * FROM recommendation
            WHERE user_id = '{0}'
            AND session_id = '{1}'
            ORDER BY action_started_at DESC
            LIMIT 1""".format(user_id, session_id)

        df_last_recommendation = pd.read_sql(
            rec_sql,
            con=engine
            # params={'user_id': user_id, 'session_id': session_id}
        )

        print(f"\n[DATA] Tables loaded successfully")
        print(f"  user_profile rows    : {len(df_user_profile)}")
        print(f"  model_inference rows : {len(df_inference)}")
        print(f"  user_summary rows    : {len(df_user_summary)}")

        
        # 5. GET TOTAL SESSION COUNT
        
        user_summary_row = df_user_summary[df_user_summary['user_id'] == user_id]

        if user_summary_row.empty:
            total_sessions = 0
        else:
            total_sessions = int(user_summary_row.iloc[0]['total_sessions'])

        print(f"\n[SESSION COUNT] User {user_id} has {total_sessions} sessions")

        
        # 6. IF total_sessions <= 5 → BASIC RECOMMENDATION
        
        if total_sessions <= 5:
            print(f"[PHASE] Cold Start — using basic recommendation")
            result = basic_recommendation(
                df_inference,
                df_user_profile,
                user_id,
                session_id
            )

        
        # 7. ELSE → ADVANCED RECOMMENDATION (not active yet)
        
        else:
            # TODO: Switch to advanced_recommendation when Phase 2 is ready
            # For now, fall back to basic recommendation
            print(f"[PHASE] Warm Start not ready yet — falling back to basic recommendation")
            result = basic_recommendation(
                df_inference,
                df_user_profile,
                user_id,
                session_id
            )

        
        # SAVE RESULT
        
        if result:
            # Check for repeated stimulus and rotate if needed
            if not df_last_recommendation.empty:
                last_stimulus = df_last_recommendation.iloc[0]['stimulus_name']
                if last_stimulus == result['stimulus']:
                    user = df_user_profile[
                        df_user_profile['user_id'] == user_id
                    ].iloc[0]
                    available, _ = build_available_stimuli(
                        user['preferred_stimulus_types'],
                        user['avoided_stimulus_types']
                    )
                    result['stimulus'] = rotate_stimulus(available, last_stimulus)
                    result['category'] = 'game' if 'game' in result['stimulus'] else 'music'
                    result['message']  = f"Trying something different — {result['stimulus']} this time."

            # Get latest inference_id for saving
            inference = df_inference[
                (df_inference['user_id']    == user_id) &
                (df_inference['session_id'] == session_id)
            ].sort_values('inference_id').iloc[-1]

            save_recommendation(
                db,
                user_id      = user_id,
                session_id   = session_id,
                inference_id = inference['inference_id'],
                category     = result['category'],
                stimulus     = result['stimulus'],
                trigger      = result['trigger'],
                message      = result['message']
            )

            print(f"\n[RESULT] Stimulus  : {result['stimulus']} ({result['category']})")
            print(f"[RESULT] Trigger   : {result['trigger']}")
            print(f"[RESULT] Message   : {result['message']}")
            print("[RESULT]  Recommendation : {0}".format(generate_recommendation(df_user_profile, df_session, user_id, session_id)))

        else:
            print(f"\n[RESULT] No recommendation needed — user is focused!")

    finally:
        db.close()



# ENTRY POINT

if __name__ == "__main__":
    main(user_id="u-004", session_id="s-010")
    # db = SessionLocal()
    # df_user_profile = pd.read_sql(
    #         "SELECT * FROM user_profile",
    #         con=engine
    #     )
    # print(df_user_profile)
    # db.close()