# Inquiro: LLM-Powered Multi-Agent Academic Research System

Inquiro is an LLM-powered multi-agent system for supporting academic research. Four specialised agents operate in a linear pipeline to decompose research questions into subtasks, retrieve relevant literature, process findings, and generate structured offline reports.

## Architecture

```
User Query               outputs/
    │                    ├── {ts}_{query}.md
    ▼                    └── {ts}_{query}.json
┌──────────┐     ┌───────────┐     ┌───────────┐     ┌────────┐
│ Planner  │ ──► │ Retriever │ ──► │ Processor │ ──► │ Writer │
│ (Mistral)│     │ (httpx)   │     │ (Mistral) │     │(Jinja2)│
└──────────┘     └───────────┘     └───────────┘     └────────┘
```

## Agents

| Agent | Backed by | Role |
|---|---|---|
| **Planner** | Mistral-7B (LLM) | Decompose research goals into concrete tasks with search keywords |
| **Retriever** | arXiv API (httpx) | Execute API queries using programmatic keyword parameters |
| **Processor** | Mistral-7B (LLM) | Filter, summarize and score results |
| **Writer** | Jinja2 + pathlib | Compile findings into markdown reports + JSON logs |

## Execution Instructions

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) with `mistral` pulled (`ollama pull mistral`)
- Docker (optional, for containerised setup)

### Local Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # edit OLLAMA_MODEL if not using mistral
```

### Command-Line Interface

```bash
python -m interfaces.cli "your research question" --session-id my-session
```

### Streamlit Web UI

```bash
streamlit run src/interfaces/streamlit_app.py
```

### Docker (app + Ollama)

```bash
docker compose up
```

Then open http://localhost:8501.

## Outputs

Reports are saved to `outputs/` as both `.md` (readable report via Jinja2 template) and `.json` (structured data with scores and metadata). Each file includes a timestamp.

## Acknowledgements

This project makes use of the following external libraries, frameworks, and services:

| Component | Purpose | Source |
|---|---|---|
| **LangGraph** | Stateful multi-agent orchestration (state graph, nodes, edges) | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) (MIT) |
| **LangChain** | LLM abstraction layer, message types | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) (MIT) |
| **Ollama** | Local LLM inference server | [ollama/ollama](https://github.com/ollama/ollama) (MIT) |
| **ChatOllama** | LangChain integration for Ollama | [langchain-ai/langchain-ollama](https://github.com/langchain-ai/langchain-ollama) (MIT) |
| **arXiv API** | Academic paper search (no API key required) | [arxiv.org](https://info.arxiv.org/help/api/index.html) |
| **httpx** | HTTP client for API calls | [encode/httpx](https://github.com/encode/httpx) (BSD) |
| **Jinja2** | Template engine for report rendering | [pallets/jinja](https://github.com/pallets/jinja) (BSD-3-Clause) |
| **Streamlit** | Web UI framework | [streamlit/streamlit](https://github.com/streamlit/streamlit) (Apache 2.0) |
| **Pydantic** | Settings management and data models | [pydantic/pydantic](https://github.com/pydantic/pydantic) (MIT) |
| **Rich** | Terminal formatting for CLI output | [Textualize/rich](https://github.com/Textualize/rich) (MIT) |
| **Mistral 7B** | Default local LLM model | [mistralai/mistral-src](https://github.com/mistralai/mistral-src) (Apache 2.0) |

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API endpoint (Ollama, llama.cpp, NIM, OpenAI) |
| `LLM_MODEL` | `mistral` | Model name to use |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
