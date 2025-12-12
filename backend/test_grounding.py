import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool,
)

# Load project settings from .env (make sure your .env has PROJECT_ID and REGION)
load_dotenv()

# --- CONFIGURATION ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-default-project-id")
REGION = os.environ.get("GCP_REGION", "us-central1")
MODEL_NAME = "gemini-2.5-flash" 
SEARCH_PROMPT = "What are the latest developments in serverless LLM deployment strategies as of today?"


client = genai.Client(http_options=HttpOptions(api_version="v1"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="When is the next total solar eclipse in the United States?",
    config=GenerateContentConfig(
        tools=[
            # Use Google Search Tool
            Tool(google_search=GoogleSearch())
        ],
    ),
)

print(response.text)
# Example response:
# 'The next total solar eclipse in the United States will occur on ...'
