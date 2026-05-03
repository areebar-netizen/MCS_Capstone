
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import django
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
from django.db import models


API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

JUDGE_PROMPT = """
You are an AI judge for an EEG-based study recommendation system.

Evaluate whether the recommendation is appropriate for the user context.

Score each criterion from 1 to 5:
1. Context Match
2. Avoidance Safety
3. Specificity
4. Reasoning Quality
5. Non-Contradiction
6. Practicality
7. Possible edge case suggestions

Return only valid JSON.
"""

def judge_response(context, recommendation):
    judge_payload = {
        'context': context,
        'recommendation': recommendation
    }

    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = f'{JUDGE_PROMPT}\n{json.dumps(judge_payload, indent=2)}',
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    )

    return json.loads(response.text)