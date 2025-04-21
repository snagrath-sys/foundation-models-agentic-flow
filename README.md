# Foundation Models Agentic Flow

This project introduces an agent that leverages agency: health data sets, specialized clinical LLMs, and foundation models to answer clinical queries.

## Features
- Uses health datasets and clinical knowledge.
- Integrates with specialized clinical large language models (LLMs).
- Employs foundation models for advanced clinical reasoning.
- Provides tools for clinical symptom-to-disease prediction and general clinical queries.

---

## Steps to Run the Project

### 1. Clone the Repository
```bash
!git clone https://github.com/snagrath-sys/foundation-models-agentic-flow.git
```

### 2. Install Dependencies
```bash
!pip install smolagents
!pip install python-dotenv==1.0.1
!pip install google-auth smolagents[litellm]
!pip install llama-index-tools-google llama-index-llms-gemini llama-index-embeddings-gemini
```
**Note:** After installing, restart your session if running in a notebook environment.

### 3. Change Directory
```bash
%cd foundation-models-agentic-flow
```

### 4. Setting up API Keys
```python
from smolagents import LiteLLMModel
from google.colab import userdata

# Replace all calls to HfApiModel
llm_model = LiteLLMModel(
    model_id="gemini/gemini-2.0-flash",  # see other model names: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models (prefix with "gemini/")
    api_key=userdata.get('GEMINI_API_KEY'),
    max_tokens=8192
)
```

### 5. Providing Agent the Tools and Example Prompt
```python
from smolagents import CodeAgent, DuckDuckGoSearchTool
from agent_tools import symptom_disease_predict

agent = CodeAgent(tools=[DuckDuckGoSearchTool(), symptom_disease_predict], model=llm_model)

agent.run("What is disease for the following symptoms: skin rash, itching, head ache")
```

---


## Acknowledgements
- [smolagents](https://github.com/smol-ai/smol-agents)
- Google Gemini
