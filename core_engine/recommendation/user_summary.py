# ============================================================
# user_summary.py
# Aggregates data from feedback, session, recommendation
# tables and populates user_summary table
# ============================================================

# ============================================================
# 1. IMPORT NECESSARY LIBRARIES
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# DATABASE CONNECTION
# ============================================================
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine       = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


# ============================================================
# 2. USER DEFINED FUNCTIONS
# ============================================================

def get_optimal_focus_time(df_session, df_inference, user_id):
    """
    Finds the time of day when user has the highest average focus score.
    Joins session start time with inference focus scores.
    Returns morning, afternoon, or evening.
    """
    user_sessions  = df_session[df_session['user_id'] == user_id][['session_id', 'session_start_time']]
    user_inference = df_inference[df_inference['user_id'] == user_id][['session_id', 'focus_score']]

    merged = user_sessions.merge(user_inference, on='session_id', how='inner')

    if merged.empty:
        return 'morning'

    def time_bucket(ts):
        hour = pd.to_datetime(ts).hour
        if hour < 12:   return 'morning'
        elif hour < 17: return 'afternoon'
        else:           return 'evening'

    merged['time_of_day'] = merged['session_start_time'].apply(time_bucket)

    best_time = (
        merged.groupby('time_of_day')['focus_score']
        .mean()
        .idxmax()
    )
    return best_time


def get_most_least_effective_stimulus(df_feedback, df_recommendation, user_id):
    """
    Joins feedback with recommendation to find which stimulus
    got the highest and lowest average overall_rating.
    Returns (most_effective, least_effective).
    """
    user_feedback = df_feedback[df_feedback['user_id'] == user_id]

    if user_feedback.empty:
        return None, None

    # Join feedback with recommendation to get stimulus_name
    merged = user_feedback.merge(
        df_recommendation[['recommendation_id', 'stimulus_name']],
        on='recommendation_id',
        how='left'
    )

    if merged.empty or merged['stimulus_name'].isnull().all():
        return None, None

    stimulus_ratings = (
        merged.groupby('stimulus_name')['overall_rating']
        .mean()
        .reset_index()
    )

    most_effective  = stimulus_ratings.loc[stimulus_ratings['overall_rating'].idxmax(), 'stimulus_name']
    least_effective = stimulus_ratings.loc[stimulus_ratings['overall_rating'].idxmin(), 'stimulus_name']

    return most_effective, least_effective


def get_stimulus_used(df_recommendation, user_id):
    """
    Returns comma-separated list of unique stimuli
    recommended to this user across all sessions.
    """
    user_recs = df_recommendation[df_recommendation['user_id'] == user_id]

    if user_recs.empty:
        return None

    stimuli = user_recs['stimulus_name'].dropna().unique().tolist()
    return ','.join(stimuli)


def get_overall_sentiment(df_feedback, user_id):
    """
    Computes overall sentiment score as:
    positive_count / total_feedback_count
    Returns score between 0.0 (all negative) and 1.0 (all positive).
    """
    user_feedback = df_feedback[df_feedback['user_id'] == user_id]

    if user_feedback.empty:
        return 0.0, 0, 0

    positive_count = len(user_feedback[user_feedback['sentiment'] == 'positive'])
    negative_count = len(user_feedback[user_feedback['sentiment'] == 'negative'])
    total          = len(user_feedback)

    sentiment_score = round(positive_count / total, 2) if total > 0 else 0.0

    return sentiment_score, positive_count, negative_count


def compute_user_summary(df_session, df_inference, df_feedback,
                          df_recommendation, user_id):
    """
    Aggregates all metrics for a single user.
    Returns a dict ready to INSERT into user_summary table.
    """
    print(f"\n[SUMMARY] Computing summary for user {user_id}")

    # ── Session metrics ──
    user_sessions  = df_session[df_session['user_id'] == user_id]
    total_sessions = len(user_sessions['session_id'].unique())
    total_focus_time_minutes = float(user_sessions['session_duration'].sum()) if not user_sessions.empty else 0.0
    first_session_date = user_sessions['session_start_time'].min().date() if not user_sessions.empty else None
    last_session_date  = user_sessions['session_start_time'].max().date() if not user_sessions.empty else None

    # ── Inference metrics ──
    user_inference     = df_inference[df_inference['user_id'] == user_id]
    average_focus_score = round(float(user_inference['focus_score'].mean()), 2) if not user_inference.empty else 0.0
    best_focus_score    = round(float(user_inference['focus_score'].max()), 2)  if not user_inference.empty else 0.0

    # ── Optimal focus time ──
    optimal_focus_time_of_day = get_optimal_focus_time(df_session, df_inference, user_id)

    # ── Stimulus metrics ──
    most_effective, least_effective = get_most_least_effective_stimulus(df_feedback, df_recommendation, user_id)
    stimulus_used = get_stimulus_used(df_recommendation, user_id)

    # ── Feedback metrics ──
    user_feedback = df_feedback[df_feedback['user_id'] == user_id]

    if not user_feedback.empty:
        average_feedback_rating     = int(round(user_feedback['overall_rating'].mean()))
        total_recommendations_rated = len(user_feedback)
        recommendation_feedback     = user_feedback.sort_values('feedback_id').iloc[-1]['sentiment']
    else:
        average_feedback_rating     = 0
        total_recommendations_rated = 0
        recommendation_feedback     = None

    # ── Sentiment metrics ──
    overall_sentiment_score, positive_count, negative_count = get_overall_sentiment(df_feedback, user_id)

    summary = {
        'user_id'                    : user_id,
        'total_sessions'             : total_sessions,
        'total_focus_time_minutes'   : total_focus_time_minutes,
        'average_focus_score'        : average_focus_score,
        'best_focus_score'           : best_focus_score,
        'optimal_focus_time_of_day'  : optimal_focus_time_of_day,
        'first_session_date'         : first_session_date,
        'last_session_date'          : last_session_date,
        'stimulus_used'              : stimulus_used,
        'recommendation_feedback'    : recommendation_feedback,
        'most_effective_stimulus'    : most_effective,
        'least_effective_stimulus'   : least_effective,
        'average_feedback_rating'    : average_feedback_rating,
        'positive_feedback_count'    : positive_count,
        'negative_feedback_count'    : negative_count,
        'overall_sentiment_score'    : overall_sentiment_score,
        'total_recommendations_rated': total_recommendations_rated
    }

    print(f"  total_sessions          : {total_sessions}")
    print(f"  total_focus_time_minutes: {total_focus_time_minutes}")
    print(f"  average_focus_score     : {average_focus_score}")
    print(f"  best_focus_score        : {best_focus_score}")
    print(f"  optimal_focus_time      : {optimal_focus_time_of_day}")
    print(f"  most_effective_stimulus : {most_effective}")
    print(f"  least_effective_stimulus: {least_effective}")
    print(f"  average_feedback_rating : {average_feedback_rating}")
    print(f"  overall_sentiment_score : {overall_sentiment_score}")

    return summary


def upsert_user_summary(db, summary):
    """
    INSERT or UPDATE user_summary for a given user.
    Uses INSERT ... ON DUPLICATE KEY UPDATE so it works
    whether the user already has a summary or not.
    """
    query = text("""
        INSERT INTO user_summary (
            user_id, total_sessions, total_focus_time_minutes,
            average_focus_score, best_focus_score,
            optimal_focus_time_of_day, first_session_date, last_session_date,
            stimulus_used, recommendation_feedback,
            most_effective_stimulus, least_effective_stimulus,
            average_feedback_rating, positive_feedback_count,
            negative_feedback_count, overall_sentiment_score,
            total_recommendations_rated
        ) VALUES (
            :user_id, :total_sessions, :total_focus_time_minutes,
            :average_focus_score, :best_focus_score,
            :optimal_focus_time_of_day, :first_session_date, :last_session_date,
            :stimulus_used, :recommendation_feedback,
            :most_effective_stimulus, :least_effective_stimulus,
            :average_feedback_rating, :positive_feedback_count,
            :negative_feedback_count, :overall_sentiment_score,
            :total_recommendations_rated
        )
        ON DUPLICATE KEY UPDATE
            total_sessions              = VALUES(total_sessions),
            total_focus_time_minutes    = VALUES(total_focus_time_minutes),
            average_focus_score         = VALUES(average_focus_score),
            best_focus_score            = VALUES(best_focus_score),
            optimal_focus_time_of_day   = VALUES(optimal_focus_time_of_day),
            last_session_date           = VALUES(last_session_date),
            stimulus_used               = VALUES(stimulus_used),
            recommendation_feedback     = VALUES(recommendation_feedback),
            most_effective_stimulus     = VALUES(most_effective_stimulus),
            least_effective_stimulus    = VALUES(least_effective_stimulus),
            average_feedback_rating     = VALUES(average_feedback_rating),
            positive_feedback_count     = VALUES(positive_feedback_count),
            negative_feedback_count     = VALUES(negative_feedback_count),
            overall_sentiment_score     = VALUES(overall_sentiment_score),
            total_recommendations_rated = VALUES(total_recommendations_rated)
    """)

    db.execute(query, summary)
    db.commit()
    print(f"[SAVED] Summary upserted for user {summary['user_id']}")


# ============================================================
# 3. MAIN
# ============================================================
def main():
    db = SessionLocal()

    try:
        print(f"\n{'='*50}")
        print(f"User Summary Aggregation Started")
        print(f"{'='*50}")

        # --------------------------------------------------------
        # 4. READ ALL SOURCE TABLES INTO DATAFRAMES
        # --------------------------------------------------------
        df_user_profile = pd.read_sql(
            "SELECT * FROM user_profile",
            con=engine
        )

        df_session = pd.read_sql(
            "SELECT * FROM raw_session_data",
            con=engine
        )

        df_inference = pd.read_sql(
            "SELECT * FROM model_inference",
            con=engine
        )

        df_recommendation = pd.read_sql(
            "SELECT * FROM recommendation",
            con=engine
        )

        df_feedback = pd.read_sql(
            "SELECT * FROM user_feedback",
            con=engine
        )

        print(f"\n[DATA] Tables loaded successfully")
        print(f"  user_profile rows    : {len(df_user_profile)}")
        print(f"  raw_session rows     : {len(df_session)}")
        print(f"  model_inference rows : {len(df_inference)}")
        print(f"  recommendation rows  : {len(df_recommendation)}")
        print(f"  user_feedback rows   : {len(df_feedback)}")

        # --------------------------------------------------------
        # 5. GET ALL USER IDs
        # --------------------------------------------------------
        user_ids = df_user_profile['user_id'].tolist()
        print(f"\n[USERS] Processing {len(user_ids)} users: {user_ids}")

        # --------------------------------------------------------
        # 6. COMPUTE AND UPSERT SUMMARY FOR EACH USER
        # --------------------------------------------------------
        for user_id in user_ids:
            summary = compute_user_summary(
                df_session,
                df_inference,
                df_feedback,
                df_recommendation,
                user_id
            )
            upsert_user_summary(db, summary)

        print(f"\n{'='*50}")
        print(f"User Summary Aggregation Complete")
        print(f"{'='*50}")

    finally:
        db.close()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Run for specific user
        user_id = sys.argv[1]
        print(f"Running for specific user: {user_id}")
        
        db = SessionLocal()
        try:
            df_session        = pd.read_sql("SELECT * FROM raw_session_data", con=engine)
            df_inference      = pd.read_sql("SELECT * FROM model_inference", con=engine)
            df_recommendation = pd.read_sql("SELECT * FROM recommendation", con=engine)
            df_feedback       = pd.read_sql("SELECT * FROM user_feedback", con=engine)

            summary = compute_user_summary(
                df_session, df_inference,
                df_feedback, df_recommendation,
                user_id
            )
            upsert_user_summary(db, summary)
        finally:
            db.close()
    else:
        # Run for all users
        main()