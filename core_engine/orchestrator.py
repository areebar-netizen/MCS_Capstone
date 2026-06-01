import json
from google.genai import types
from ai_validation import judge_response, client


JUDGES = ["Judge1", "Judge2", "Judge3"]


class Orchestrator:
    DEBATE_PROMPT = """
    You are a final consensus judge for an EEG-based study recommendation system.

    You will receive evaluations from 3 AI judges.
    Each judge reviewed 5 different sessions using the same rubric.

    Your job:
    1. Compare the judges' evaluations.
    2. Identify which recommendations were strongest and why.
    3. Identify recurring weaknesses.
    4. Extract reusable feedback rules for future recommendations.
    5. Decide what the recommendation system should do differently next time.

    Return only valid JSON with:
    {
      "best_recommendation_patterns": [],
      "weak_recommendation_patterns": [],
      "recurring_issues": [],
      "future_prompt_rules": [],
      "overall_summary": ""
    }
    """

    def __init__(self):
        self.judges = JUDGES

    def assign_judges(self, sessions):
        if len(sessions) != 15:
            raise ValueError(f"Expected 15 sessions, got {len(sessions)}")

        return {
            "Judge1": sessions[0:5],
            "Judge2": sessions[5:10],
            "Judge3": sessions[10:15],
        }

    def run_judge_batch(self, judge_name, sessions):
        results = []

        for session in sessions:
            recommendation = session["recommendation"]
            context = session["context"]

            judgment = judge_response(
                context=context,
                recommendation=recommendation
            )

            results.append({
                "judge": judge_name,
                "session_id": session.get("session_id"),
                "context": context,
                "recommendation": recommendation,
                "judgment": judgment,
            })

        return results

    def debate_judge_outputs(self, all_judge_results):
        payload = {
            "judge_results": all_judge_results
        }

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{self.DEBATE_PROMPT}\n{json.dumps(payload, indent=2)}",
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )

        return json.loads(response.text)

    def run_ai_judge_orchestration(self, evaluated_sessions):
        allocations = self.assign_judges(evaluated_sessions)

        all_judge_results = []

        for judge_name, sessions in allocations.items():
            print(f"\n[ORCH] Running {judge_name} on {len(sessions)} sessions")

            judge_batch_results = self.run_judge_batch(
                judge_name=judge_name,
                sessions=sessions
            )

            all_judge_results.extend(judge_batch_results)

        print("\n[ORCH] Running final judge debate/consensus")

        consensus = self.debate_judge_outputs(all_judge_results)

        return {
            "judge_results": all_judge_results,
            "consensus": consensus
        }