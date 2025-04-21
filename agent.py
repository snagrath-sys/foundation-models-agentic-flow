
from smolagents import LiteLLMModel
from dotenv import load_dotenv
import os

load_dotenv()

# Replace all calls to HfApiModel
llm_model = LiteLLMModel(
    model_id="gemini/gemini-2.0-flash", # you can see other model names here: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models. It is important to prefix the name with "gemini/"
    api_key=os.getenv("GEMINI_API_KEY"),
    max_tokens=8192
)


from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from agent_tools import symptom_disease_predict


agent = CodeAgent(tools=[DuckDuckGoSearchTool(), symptom_disease_predict], model=llm_model)

agent.run("What is disease for the following symptoms: skin rash, itching, head ache")