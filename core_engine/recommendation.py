# ============================================================
# core_engine/recommendation.py
# Bridge between Django and recommendation engine
# Called from tasks_realtime.py after EEG session ends
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
ROOT   = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / 'webapp'
if str(WEBAPP) not in sys.path:
    sys.path.insert(0, str(WEBAPP))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'secondBrain.settings')

# Only call django.setup() if not already configured
try:
    from django.conf import settings
    if not settings.configured:
        django.setup()
except Exception:
    django.setup()

# ── Django model imports ──────────────────────────────────────
from secondBrain_App.models import (
    UserProfile, SessionSummary, Recommendation,
    UserFeedback
)
from django.db import models
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# HELPER
# ============================================================
def get_time_of_day():
    from django.utils import timezone
    hour = timezone.localtime(timezone.now()).hour
    if hour < 12:   return 'morning'
    elif hour < 17: return 'afternoon'
    else:           return 'evening'


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
        import time
        max_retries = 3
        session = None
        
        for attempt in range(max_retries):
            try:
                session = SessionSummary.objects.get(session_id=session_id)
                break
            except SessionSummary.DoesNotExist:
                if attempt < max_retries - 1:
                    print(f"[REC] SessionSummary not found for {session_id} — retry {attempt + 1}/{max_retries}")
                    time.sleep(1)  # Wait 1 second and retry
                else:
                    print(f"[REC] SessionSummary not found for {session_id} after {max_retries} attempts — using fallback")
                    return _fallback_recommendation(final_summary)

        # ── Get user summary from SessionSummary ───────────────
        total_sessions = SessionSummary.objects.filter(user=user_profile).count()
        
        # Calculate aggregates from SessionSummary
        user_summary_data = {
            'total_sessions': total_sessions,
            'average_focus_score': 0,
            'best_focus_score': 0,
            'most_effective_stimulus': 'lo_fi',
            'least_effective_stimulus': 'none',
            'optimal_focus_time_of_day': 'morning',
            'average_feedback_rating': 0,
            'overall_sentiment_score': 0
        }
        
        if total_sessions > 0:
            sessions = SessionSummary.objects.filter(user=user_profile)
            avg_focus = sessions.aggregate(models.Avg('average_focus_score'))['average_focus_score__avg'] or 0
            best_focus = sessions.aggregate(models.Max('peak_focus_score'))['peak_focus_score__max'] or 0
            user_summary_data['average_focus_score'] = avg_focus
            user_summary_data['best_focus_score'] = best_focus
        
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
# ============================================================
def _phase1_llm(user_profile, session, final_summary):
    """Phase 1 LLM recommendation using profile and session data only."""
    client = genai.Client(api_key=API_KEY)

    avg_focus             = final_summary.get('average_focus_score', 0)
    concentrating_seconds = final_summary.get('concentrating_seconds', 0)
    neutral_seconds       = final_summary.get('neutral_seconds', 0)
    relaxed_seconds       = final_summary.get('relaxed_seconds', 0)

    # Extract brainwave inferences
    neural_state      = final_summary.get('neural_state', 'Unknown')
    signal_integrity  = final_summary.get('signal_integrity', 'Unknown')
    focus_depth       = final_summary.get('focus_depth', 'Unknown')
    beta_avg          = final_summary.get('beta_avg', 0)
    gamma_avg         = final_summary.get('gamma_avg', 0)
    alpha_avg         = final_summary.get('alpha_avg', 0)
    theta_avg         = final_summary.get('theta_avg', 0)

    # Map UserProfile fields to readable values
    sound_env     = user_profile.sound_environment or 'unknown'
    study_goals   = user_profile.main_goals        or 'study improvement'
    sleep_quality = user_profile.sleep_quality     or 'unknown'
    learning_style= user_profile.learning_style    or 'unknown'

    contents = """You are an AI-powered Study Optimization Advisor analyzing EEG brainwave data.

USER PROFILE:
- Sound preference   : {sound}
- Sleep quality      : {sleep}
- Learning style     : {style}
- Study goals        : {goals}

EEG SESSION RESULTS:
- Avg focus score    : {avg_focus:.2f} (0=no focus, 1=full focus)
- Concentrating time : {conc} seconds
- Neutral time       : {neut} seconds
- Relaxed/distracted : {relax} seconds
- Session duration   : {duration} mins
- Time of day        : {time_of_day}

BRAINWAVE ANALYSIS:
- Neural State       : {neural_state}
- Signal Integrity   : {signal_integrity}
- Focus Depth        : {focus_depth}
- Beta waves        : {beta_avg:.2f} Hz
- Gamma waves       : {gamma_avg:.2f} Hz
- Alpha waves       : {alpha_avg:.2f} Hz
- Theta waves       : {theta_avg:.2f} Hz

RESPOND WITH:
1. 1-2 line fun personalized recommendation based on their EEG session results
2. Recommended Study Methods (3-4 bullet points)
3. Optimal study environment for this user
""".format(
        sound        = sound_env,
        sleep        = sleep_quality,
        style        = learning_style,
        goals        = study_goals,
        avg_focus    = float(avg_focus),
        conc         = concentrating_seconds,
        neut         = neutral_seconds,
        relax        = relaxed_seconds,
        duration     = round(session.total_duration_seconds / 60, 1),
        time_of_day  = get_time_of_day(),
        neural_state = neural_state,
        signal_integrity = signal_integrity,
        focus_depth  = focus_depth,
        beta_avg     = beta_avg,
        gamma_avg    = gamma_avg,
        alpha_avg    = alpha_avg,
        theta_avg    = theta_avg
    )

    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = contents
    )
    return response.text


# ============================================================
# PHASE 2 — Sessions 6+
# ============================================================
def _phase2_llm(user_profile, session, user_summary_data, user_email, final_summary):
    """Phase 2 LLM recommendation using full history and feedback."""
    client = genai.Client(api_key=API_KEY)

    avg_focus             = final_summary.get('average_focus_score', 0)
    concentrating_seconds = final_summary.get('concentrating_seconds', 0)
    neutral_seconds       = final_summary.get('neutral_seconds', 0)
    relaxed_seconds       = final_summary.get('relaxed_seconds', 0)

    # Extract brainwave inferences
    neural_state      = final_summary.get('neural_state', 'Unknown')
    signal_integrity  = final_summary.get('signal_integrity', 'Unknown')
    focus_depth       = final_summary.get('focus_depth', 'Unknown')
    beta_avg          = final_summary.get('beta_avg', 0)
    gamma_avg         = final_summary.get('gamma_avg', 0)
    alpha_avg         = final_summary.get('alpha_avg', 0)
    theta_avg         = final_summary.get('theta_avg', 0)

    sound_env     = user_profile.sound_environment or 'unknown'
    study_goals   = user_profile.main_goals        or 'study improvement'
    sleep_quality = user_profile.sleep_quality     or 'unknown'
    learning_style= user_profile.learning_style    or 'unknown'

    # ── Get last feedback ─────────────────────────────────────
    try:
        last_feedback       = UserFeedback.objects.filter(
            user__email=user_email
        ).order_by('-created_at').first()
        last_overall_rating = last_feedback.overall_rating if last_feedback else 'N/A'
        last_sentiment      = last_feedback.sentiment      if last_feedback else 'N/A'
    except Exception:
        last_overall_rating = 'N/A'
        last_sentiment      = 'N/A'

    contents = """You are an AI-powered Study Optimization Advisor. EEG session just ended.

USER PROFILE:
- Sound preference   : {sound}
- Sleep quality      : {sleep}
- Learning style     : {style}
- Study goals        : {goals}

EEG SESSION RESULTS:
- Avg focus score    : {avg_focus:.2f}
- Concentrating time : {conc} seconds
- Neutral time       : {neut} seconds
- Relaxed/distracted : {relax} seconds
- Session duration   : {duration} mins
- Time of day        : {time_of_day}

BRAINWAVE ANALYSIS:
- Neural State       : {neural_state}
- Signal Integrity   : {signal_integrity}
- Focus Depth        : {focus_depth}
- Beta waves        : {beta_avg:.2f} Hz
- Gamma waves       : {gamma_avg:.2f} Hz
- Alpha waves       : {alpha_avg:.2f} Hz
- Theta waves       : {theta_avg:.2f} Hz

HISTORY ({sessions} sessions):
- Historical avg focus      : {hist_focus}
- Best focus score ever     : {best_focus}
- Most effective stimulus   : {best_stim}
- Least effective stimulus  : {worst_stim}
- Optimal focus time        : {opt_time}
- Avg feedback rating       : {avg_rating}/5
- Overall sentiment score   : {sentiment}
- Last session rating       : {last_rating}/5 ({last_sent})

RULES:
- NEVER recommend {worst_stim}
- PRIORITIZE {best_stim}
- If last rating <= 2, try something completely different
- If sentiment score < 0.3, be more creative and suggest new approaches

RESPOND WITH:
1. 1-2 line fun personalized recommendation referencing their EEG results and history
2. Recommended Study Methods (3-4 bullet points)
3. Optimal study environment for this user
4. One line on what to avoid based on their history
""".format(
        sound        = sound_env,
        sleep        = sleep_quality,
        style        = learning_style,
        goals        = study_goals,
        avg_focus    = float(avg_focus),
        conc         = concentrating_seconds,
        neut         = neutral_seconds,
        relax        = relaxed_seconds,
        duration     = round(session.total_duration_seconds / 60, 1),
        time_of_day  = get_time_of_day(),
        neural_state = neural_state,
        signal_integrity = signal_integrity,
        focus_depth  = focus_depth,
        beta_avg     = beta_avg,
        gamma_avg    = gamma_avg,
        alpha_avg    = alpha_avg,
        theta_avg    = theta_avg,
        sessions     = user_summary_data['total_sessions'],
        hist_focus   = user_summary_data['average_focus_score'],
        best_focus   = user_summary_data['best_focus_score'],
        best_stim    = user_summary_data['most_effective_stimulus'],
        worst_stim   = user_summary_data['least_effective_stimulus'],
        opt_time     = user_summary_data['optimal_focus_time_of_day'],
        avg_rating   = user_summary_data['average_feedback_rating'],
        sentiment    = user_summary_data['overall_sentiment_score'],
        last_rating  = last_overall_rating,
        last_sent    = last_sentiment
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


# ============================================================
# SAVE RECOMMENDATION
# ============================================================
def save_recommendation(user_email, session_id, inference_id,
                        category, stimulus, trigger, message):
    """
    Save recommendation to database using Django models.
    Replaces the SQLAlchemy-based save_recommendation from the subdirectory.
    """
    from django.utils import timezone
    import uuid
    
    try:
        user_profile = UserProfile.objects.get(email=user_email)
        
        recommendation = Recommendation.objects.create(
            recommendation_id=str(uuid.uuid4()),
            user=user_profile,
            session_id=session_id,
            inference_id=inference_id,
            recommendation_category=category,
            stimulus_name=stimulus,
            trigger_reason=trigger,
            action_started_at=timezone.now(),
            message=message
        )
        
        print(f"[SAVED] Recommendation: {stimulus} ({category}) for user {user_email}")
        return recommendation.recommendation_id
        
    except UserProfile.DoesNotExist:
        print(f"[ERROR] UserProfile not found for {user_email}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to save recommendation: {e}")
        return None