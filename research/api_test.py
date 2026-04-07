from google import genai
import warnings
warnings.filterwarnings("ignore")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="xxx")

response = client.models.generate_content(
    model="gemini-3-flash-preview", 
    contents="You are a focus optimization assistant. Based on the user's current focus state and preferences, recommend the most suitable stimulus to help them regain or maintain focus. The user is currently distracted. User likes lo-fi music but avoid loud music and user has poor sleep quality. What do you recommend? Generate 1-2 line specific recommendation"
)
print(response.text)