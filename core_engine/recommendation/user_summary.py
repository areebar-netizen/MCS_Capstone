# ============================================================
# core_engine/recommendation.py
# Bridge between Django and recommendation engine
# Now uses Django ORM instead of SQLAlchemy
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import django
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Setup Django environment ──────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
WEBAPP   = ROOT / 'webapp'
if str(WEBAPP) not in sys.path:
    sys.path.insert(0, str(WEBAPP))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'secondBrain.settings')
django.setup()

# ── Django model imports ─────────────────────────────────────
from secondBrain_App.models import (
    UserProfile, SessionSummary, Recommendation,
    UserFeedback
)
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# MAIN ENTRY POINT
# Called from tasks_realtime.py → save_session_summary()
# ============================================================
def generate_recommendation_for_session(user_email, session_id, final_summary):
    """
    Main entry point called from Django after EEG session ends.

    Args:
        user_email   : user's email
        session_id   : session identifier from EEG task
        final_summary: dict with average_focus_score,
                       relaxed/neutral/concentrating seconds

    Returns:
        recommendation_text (str) from Gemini API
    """
    try:
        print(f"[REC] Generating recommendation for {user_email}, session {session_id}")

        # ── Get user profile ──────────────────────────────────
        try:
            user_profile = UserProfile.objects.get(email=user_email)
        except UserProfile.DoesNotExist:
            print(f"[REC] UserProfile not found for {user_email}")
            return _fallback_recommendation(final_summary)

        # ── Get session ───────────────────────────────────────
        try:
            session = SessionSummary.objects.get(session_id=session_id)
        except SessionSummary.DoesNotExist:
            print(f"[REC] SessionSummary not found for {session_id}")
            return _fallback_recommendation(final_summary)

        # ── Get user summary from SessionSummary ───────────────
        total_sessions = SessionSummary.objects.filter(user=user_profile).count()
        
        # Calculate aggregates from SessionSummary
        user_summary_data = {
            'total_sessions': total_sessions,
            'average_focus_score': 0,
            'most_effective_stimulus': 'lo_fi',
            'least_effective_stimulus': 'none',
            'optimal_focus_time_of_day': 'morning',
            'average_feedback_rating': 0,
            'overall_sentiment_score': 0
        }
        
        if total_sessions > 0:
            sessions = SessionSummary.objects.filter(user=user_profile)
            avg_focus = sessions.aggregate(models.Avg('average_focus_score'))['average_focus_score__avg'] or 0
            user_summary_data['average_focus_score'] = avg_focus
        
        # Get feedback aggregates
        feedbacks = UserFeedback.objects.filter(user=user_profile)
        if feedbacks.exists():
            avg_rating = feedbacks.aggregate(models.Avg('overall_rating'))['overall_rating__avg'] or 0
            user_summary_data['average_feedback_rating'] = avg_rating

        print(f"[REC] User has {total_sessions} sessions — Phase {'1' if total_sessions <= 5 else '2'}")

        # ── Route to Phase 1 or Phase 2 ──────────────────────
        if total_sessions <= 5:
            return _phase1_llm(user_profile, session, final_summary)
        else:
            return _phase2_llm(
                user_profile, session,
                user_summary_data, user_email,
                final_summary
            )

    except Exception as e:
        print(f"[REC ERROR] {e}")
        return _fallback_recommendation(final_summary)


# ============================================================
# PHASE 1 — Sessions 1-5
# Uses: user profile + current session + EEG results
# ============================================================
def _phase1_llm(user_profile, session, final_summary):
    """Phase 1 LLM recommendation using profile and session data only."""
    client = genai.Client(api_key=API_KEY)

    avg_focus             = final_summary.get('average_focus_score', 0)
    concentrating_seconds = final_summary.get('concentrating_seconds', 0)
    neutral_seconds       = final_summary.get('neutral_seconds', 0)
    relaxed_seconds       = final_summary.get('relaxed_seconds', 0)

    # ── Map UserProfile fields to stimulus preferences ────────
    # sound_environment from onboarding maps to preferred stimulus
    preferred = user_profile.sound_environment or 'lo_fi'
    avoided   = 'loud music' if user_profile.sleep_quality == '0' else 'none'

    contents = """You are an AI-powered Study Optimization Advisor analyzing EEG brainwave data.
USER: prefers {preferred} study environment, sleep quality: {sleep}.
SESSION: duration {duration} mins, energy before session unknown.
EEG RESULTS: avg focus score {avg_focus:.2f}, concentrating {conc}s, neutral {neut}s, relaxed {relax}s.

RESPOND WITH:
1. 1-2 line fun personalized recommendation based on their EEG session results
2. Recommended Study Methods (3-4 bullet points)
3. Optimal study environment for this user
""".format(
        preferred = preferred,
        sleep     = user_profile.sleep_quality,
        duration  = round(session.total_duration_seconds / 60, 1),
        avg_focus = float(avg_focus),
        conc      = concentrating_seconds,
        neut      = neutral_seconds,
        relax     = relaxed_seconds
    )

    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = contents
    )
    return response.text


# ============================================================
# PHASE 2 — Sessions 6+
# Uses: user profile + session + summary history + feedback
# ============================================================
def _phase2_llm(user_profile, session, user_summary_data, user_email, final_summary):
    """Phase 2 LLM recommendation using full history and feedback."""
    client = genai.Client(api_key=API_KEY)

    avg_focus             = final_summary.get('average_focus_score', 0)
    concentrating_seconds = final_summary.get('concentrating_seconds', 0)
    neutral_seconds       = final_summary.get('neutral_seconds', 0)
    relaxed_seconds       = final_summary.get('relaxed_seconds', 0)

    preferred = user_profile.sound_environment or 'lo_fi'

    # ── Get last feedback ─────────────────────────────────────
    last_feedback = UserFeedback.objects.filter(
        user__email=user_email
    ).order_by('-created_at').first()

    last_overall_rating = last_feedback.overall_rating if last_feedback else 'N/A'
    last_sentiment      = last_feedback.sentiment      if last_feedback else 'N/A'

    contents = """You are an AI-powered Study Optimization Advisor. EEG session just ended.
USER: prefers {preferred}, sleep quality: {sleep}, study goals: {goals}.
EEG RESULTS: avg focus {avg_focus:.2f}, concentrating {conc}s, neutral {neut}s, relaxed {relax}s.
HISTORY ({sessions} sessions): avg focus {hist_focus}, best stimulus {best}, worst {worst}, optimal time {opt_time}.
FEEDBACK: avg rating {avg_rating}/5, sentiment {sentiment}, last rating {last_rating}/5 ({last_sent}).

RULES:
- NEVER recommend {worst}
- PRIORITIZE {best}
- If last_overall_rating <= 2, try something different
- If overall_sentiment_score < 0.3, be more creative

RESPOND WITH:
1. 1-2 line fun personalized recommendation referencing their EEG results and history
2. Recommended Study Methods (3-4 bullet points)
3. Optimal study environment for this user
4. One line on what to avoid
""".format(
        preferred  = preferred,
        sleep      = user_profile.sleep_quality,
        goals      = user_profile.main_goals,
        avg_focus  = float(avg_focus),
        conc       = concentrating_seconds,
        neut       = neutral_seconds,
        relax      = relaxed_seconds,
        sessions   = user_summary_data['total_sessions'],
        hist_focus = user_summary_data['average_focus_score'],
        best       = user_summary_data['most_effective_stimulus'],
        worst      = user_summary_data['least_effective_stimulus'],
        opt_time   = user_summary_data['optimal_focus_time_of_day'],
        avg_rating = user_summary_data['average_feedback_rating'],
        sentiment  = user_summary_data['overall_sentiment_score'],
        last_rating= last_overall_rating,
        last_sent  = last_sentiment
    )

    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = contents
    )
    return response.text


# ============================================================
# FALLBACK
# ============================================================
def _fallback_recommendation(final_summary):
    """Simple fallback if LLM or DB lookup fails."""
    avg_focus = final_summary.get('average_focus_score', 0)
    if avg_focus >= 0.7:
        return "Great focus session! Keep using the same environment and routine that worked today."
    elif avg_focus >= 0.4:
        return "Decent session! Try reducing distractions next time — a quieter space might boost your score."
    else:
        return "Tough session! Consider trying a different study location or time of day next time."