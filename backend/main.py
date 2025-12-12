import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv
from typing import Type

# Import necessary Vertex AI components
from google.cloud import aiplatform
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool,
)

# Load project settings from .env (for local run only)
load_dotenv() 

# --- CONFIGURATION (Ensure these match your .env and gcloud setup) ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-default-project-id") # Use your real project ID
REGION = os.environ.get("GCP_REGION", "us-central1")
MODEL_NAME = "gemini-2.5-flash" # The fastest and most cost-effective Gemini model

# Initialize Vertex AI client for the project/region
try:
    # This init uses the credentials from 'gcloud auth application-default login'
    aiplatform.init(project=PROJECT_ID, location=REGION)
    print(f"Vertex AI initialized for project {PROJECT_ID} in {REGION}")
except Exception as e:
    print(f"Error initializing Vertex AI: {e}. Ensure gcloud auth is complete.")
    # Exit or raise error if initialization fails

# --- 1. Define the Custom Tool using Gemini Grounding ---
# --- 1. Define the Custom Tool using Gemini Grounding ---

# Note: We do NOT import GoogleSearch to fix the error.
# We define the input schema, as required by CrewAI's BaseTool
class GoogleSearchInput(BaseModel):
    """Input schema for Google Search Tool."""
    query: str = Field(..., description="The specific search query to find up-to-date information.")

class GoogleSearchGroundingTool(BaseTool):
    name: str = "Google Search Tool"
    description: str = "A tool to search the live internet using Google Search for the most current, factual information."
    args_schema: Type[BaseModel] = GoogleSearchInput
    # Inside the GoogleSearchGroundingTool class definition:

    def _run(self, query: str) -> str:
        """The synchronous method that executes the search via Gemini/Vertex AI."""
        
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"Ground your answer using Google Search to find the latest information for: '{query}', and then provide a comprehensive, bulleted answer.",
            config=GenerateContentConfig(
                tools=[
                    # Use Google Search Tool
                    Tool(google_search=GoogleSearch())
                ],
            ),
        )
        
        # 4. Return the final, grounded text string. CrewAI expects a string output from _run.
        return response.text

# Initialize the custom search tool instance
google_search_tool = GoogleSearchGroundingTool()


# --- 2. Define the LLM for the Crew ---
# We use CrewAI's custom LLM class to configure the Vertex AI integration cleanly
gemini_llm = LLM(
    model=f"vertex_ai/{MODEL_NAME}",
    vertex_project=PROJECT_ID,
    vertex_location=REGION,
    temperature=0.3, # Lower temperature for factual content planning
    verbose=True
)


# --- 3. Define the Agents (Roles) ---

researcher = Agent(
    role='Expert Content Researcher',
    goal='Gather the latest, most relevant, and factual information on the given topic',
    backstory=(
        "You are a meticulous researcher specialized in finding and synthesizing "
        "key facts and trends from the live internet using your dedicated Google Search Tool."
    ),
    tools=[google_search_tool],  # Researcher is the only one who needs the tool
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

strategist = Agent(
    role='Content Strategist & Planner',
    goal='Develop a detailed, structured content plan and outline based on the research provided',
    backstory=(
        "You are a strategic planner with expertise in transforming raw research "
        "into an actionable, logical content outline, ensuring it meets the goal of the topic."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=True 
)

editor = Agent(
    role='Senior Content Editor',
    goal='Refine the content plan for clarity, tone, and professional quality',
    backstory=(
        "You are a seasoned editor. Your job is to ensure the final plan is polished, "
        "logical, and ready for a writer to execute with a compelling tone."
    ),
    llm=gemini_llm,
    verbose=True
)

# --- 4. Define the Tasks and Crew ---

research_task = Task(
    description=(
        "Conduct deep research for the latest trends, statistics, and essential facts "
        "about **{topic}**. The output must be a concise, bulleted list of 5-7 key findings."
    ),
    expected_output='A bulleted list of 5-7 key findings, including any cited sources from the search tool.',
    agent=researcher
)

planning_task = Task(
    description=(
        "Analyze the research notes provided and create a comprehensive blog post outline. "
        "The outline must include a Title, 4 main Section Headings, and 3-4 detailed "
        "bullet points under each section detailing the content to be written."
    ),
    expected_output='A clean, markdown-formatted outline with clear headings and detailed sub-points.',
    agent=strategist,
    context=[research_task] 
)

editing_task = Task(
    description=(
        "Review the complete content plan for logical flow, tone, and professional quality. "
        "Make any necessary corrections and ensure the final output is flawless and compelling."
    ),
    expected_output='The final, polished content plan, ready for a writer.',
    agent=editor,
    context=[planning_task]
)

# Create the Crew
content_crew = Crew(
    agents=[researcher, strategist, editor],
    tasks=[research_task, planning_task, editing_task],
    process=Process.sequential, 
    verbose=True # Verbose logging shows the agent's thought process (useful for learning)
)

# --- Run the Crew ---
if __name__ == "__main__":
    print("--- Starting Content Planning Crew with Vertex AI ---")
    
    # Define the topic for the agents to work on
    inputs = {'topic': 'Request Waiting List in Distributed Systems'}
    
    # Kick off the multi-agent workflow
    result = content_crew.kickoff(inputs=inputs)
    
    print("\n\n################################################")
    print("--- Final Content Plan Output ---")
    print(result)
    print("################################################")
