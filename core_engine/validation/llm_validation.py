import os

from llm_validation_framework import (
    ValidationFramework,
    LLMProvider,
    Pipe,
    ToxicityAgent,
    AccuracyAgent,
)

API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_KEY = API_KEY

# Main LLM
llm = LLMProvider(
    provider="gemini",
    model="gemini-3-flash-preview",
    key=GEMINI_KEY,
)

# Input validation
input_guardrail = Pipe(
    steps=[
        ToxicityAgent()
    ],
    verbose=False,
)

# Output validation
output_guardrail = Pipe(
    steps=[
        ToxicityAgent(),
        AccuracyAgent(
            provider="gemini",
            model="gemini-3-flash-preview",
        )
    ],
    verbose=False,
)

# Framework
vf = ValidationFramework(
    llm=llm,
    input_guardrail=input_guardrail,
    output_guardrail=output_guardrail,
)

# Run validation
result = vf.validate("""You are an AI-powered Study Optimization Advisor analyzing EEG brainwave data.

USER PROFILE:

Sound preference : lo-fi instrumental
Sleep quality : 6.5 hours, slightly irregular
Learning style : visual + hands-on
Study goals : improve exam retention
Subject studying : Biology

EEG SESSION RESULTS:

Avg focus score : 0.67 (0=no focus, 1=full focus)
Concentrating time : 1800 seconds
Neutral time : 900 seconds
Relaxed/distracted : 600 seconds
Session duration : 50 mins
Time of day : evening (7:30 PM)

BRAINWAVE ANALYSIS:

Neural State : lightly focused with mild fatigue
Signal Integrity : good
Focus Depth : moderate deep focus bursts
Beta waves : 18.42 Hz
Gamma waves : 38.15 Hz
Alpha waves : 9.87 Hz
Theta waves : 6.34 Hz

RESPOND WITH:

1-2 line fun personalized recommendation based on their EEG session results
Recommended Study Methods (3-4 bullet points)
Optimal study environment for this user
Tailor study methods specifically for Biology""")

print(result)