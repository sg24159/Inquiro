# Inquiro: LLM-Powered Multi-Agent Academic Research System

Inquiro is an LLM-powered multi-agent system for supporting academic research. Four specialised agents operate in a linear pipeline to decompose research questions into subtasks, retrieve relevant literature, process findings, and generate structured offline reports.

## Architecture

```
User Query               outputs/
    │                    ├── {ts}_{query}.md
    ▼                    └── {ts}_{query}.json
┌──────────┐     ┌───────────┐     ┌───────────┐     ┌────────┐
│ Planner  │ ──► │ Retriever │ ──► │ Processor │ ──► │ Writer │
│  (LLM)   │     │  (httpx)  │     │   (LLM)   │     │(Jinja2)│
└──────────┘     └───────────┘     └───────────┘     └────────┘
```

### Agent Descriptions

| Agent | Backed by | Role |
|---|---|---|
| **Planner** | LLM | Decompose research goals into concrete tasks with search keywords |
| **Retriever** | arXiv and DDG APIs | Execute API queries using programmatic keyword parameters |
| **Processor** | LLM | Filter, summarize and score results |
| **Writer** | Jinja2 + pathlib | Compile findings into markdown reports + JSON logs |

## Setup Instructions

There are multiple running modes available, depending on what services your system provides.

### Common Setup

Copy the configuration file and adjust values as needed.

This is required for all modes of operation.

```bash
cp .env.example .env 
```

### Option 1: Self-contained Web Interface

***Prerequisites***: docker compose

In this mode, all dependencies will be handled by docker. 
The most convenient option, but some features, such as GPU acceleration, may not be available.

*If you already have the Ollama service installed and running on your computer, please use Option 2.*

```bash
docker compose up -d --build
```

Then navigate to the Streamlit UI at http://localhost:8501.

### Option 2: Web Interface with External LLM Provider

***Prerequisites***: docker compose, local LLM service or API key for remote service.

Adjust `LLM_BASE_URL` in `.env` to point to your LLM provider.
Edit `LLM_MODEL` and `API_KEY` as required.

Host networking is enabled in `docker-compose.yml` no additional port configuration is required.

Configuring your local LLM server with 16k of context is recommended, although 2k is sufficient for most queries.

```bash
docker compose up -d --build inquiro
```

### Option 3: Command Line Interface

***Prerequisites***: Python 3.11+

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
```

```bash
python -m interfaces.cli "research question"
```

### Option 4: Web Interface Without Docker

***Prerequisites***: Python 3.11+

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
```

```bash
streamlit run src/interfaces/streamlit_app.py
```

Then navigate to the Streamlit UI at http://localhost:8501.

## Results

Reports are saved to `outputs/` as both `.md` (readable report) and `.json` (report + agent logs).

## AI Usage Disclosure

All source files, test files and prompts are written in whole or in part by OpenCode 'Big Pickle' unless otherwise cited.

This README was initially drafted by OpenCode but has been substantially revised by hand.

## Acknowledgements

This project makes use of the following external libraries, frameworks, and services:

| Component | Purpose | Source |
|---|---|---|
| **LangGraph** | Stateful multi-agent orchestration (state graph, nodes, edges) | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) (MIT) |
| **LangChain** | LLM abstraction layer, message types | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) (MIT) |
| **LangChain Community** | Community integrations and utilities | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) (MIT) |
| **LangChain OpenAI** | OpenAI-compatible LLM client (used for Ollama) | [langchain-ai/langchain-openai](https://github.com/langchain-ai/langchain-openai) (MIT) |
| **Ollama** | Local LLM inference server | [ollama/ollama](https://github.com/ollama/ollama) (MIT) |
| **arXiv API** | Academic paper search (no API key required) | [arxiv.org](https://info.arxiv.org/help/api/index.html) |
| **DuckDuckGo Search** | Web search fallback for retriever | [deedy5/duckduckgo_search](https://github.com/deedy5/duckduckgo_search) (MIT) |
| **httpx** | HTTP client for API calls | [encode/httpx](https://github.com/encode/httpx) (BSD) |
| **Jinja2** | Template engine for report rendering | [pallets/jinja](https://github.com/pallets/jinja) (BSD-3-Clause) |
| **Streamlit** | Web UI framework | [streamlit/streamlit](https://github.com/streamlit/streamlit) (Apache 2.0) |
| **Pydantic** | Data models and validation | [pydantic/pydantic](https://github.com/pydantic/pydantic) (MIT) |
| **Pydantic Settings** | Environment-based configuration | [pydantic/pydantic-settings](https://github.com/pydantic/pydantic-settings) (MIT) |
| **PyYAML** | YAML config file parsing | [yaml/pyyaml](https://github.com/yaml/pyyaml) (MIT) |
| **OpenCode** | AI coding assistant used for development | [anomalyco/opencode](https://github.com/anomalyco/opencode) (MIT) |
| **Rich** | Terminal formatting for CLI output | [Textualize/rich](https://github.com/Textualize/rich) (MIT) |
| **Mistral 7B** | Baseline local LLM model | [mistralai/mistral-src](https://github.com/mistralai/mistral-src) (Apache 2.0) |
| **Qwen3.5 4B** | Default local LLM model | [QwenLM/Qwen3.5](https://github.com/QwenLM/Qwen3.6#qwen35) (Apache 2.0) |

## Configuration

All configuration is via environment variables (see `.env.example`):

### Key values

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API endpoint (Ollama, llama.cpp, NIM, OpenAI) |
| `LLM_MODEL` | `qwen3.5:4b` | Model name to use |
| `API_KEY`   | None | Optional key for external providers |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
