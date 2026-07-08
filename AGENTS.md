# AI Agent Guidelines for Inquiro

This file defines conventions and expectations for AI coding agents working on this project (e.g., opencode, Copilot). Follow these rules unless a specific task explicitly overrides them.

## Code Style & Documentation

- **Comments explain *why*, not *what***. The code itself should be self-documenting through clear naming and structure. Reserve comments for design rationale, trade-offs, and non-obvious decisions.
- Use descriptive names over inline explanations.
- No docstrings on trivial helpers (one-liners); use them for public API surfaces and complex logic.
- One blank line between top-level definitions; two blank lines between classes.

## Project Architecture Rules

- **`src/` layout**: All application code lives under `src/`. The `src/` directory is the package root — imports use `from config import ...`, `from coordinator import ...`, etc. (no `src.` prefix).
- **Domain structure**: Each agent domain (planning, retrieval, processing, writing) has exactly two files:
  - `agent.py` — the LangGraph node function (and only that agent's node logic)
  - `tools.py` — utility functions used by that agent
- **LLM access**: Always import via `from shared import llm as llm_module` and call `llm_module.get_llm()`. Never import `ChatOllama` directly outside `src/shared/llm.py`. This keeps mocking possible in tests.
- **Non-LLM agents**: The retriever and writer agents do not use an LLM. They call APIs (httpx) and render templates (Jinja2) directly. Their node functions should not use `llm_module.get_llm()`.
- **State mutations**: Node functions return dictionaries with the keys they want to update in `ResearchState`. They must not mutate `state` in place.
- **Linear pipeline**: The graph has a fixed linear flow: `planner → retriever → processor → writer → END`. The router is unused; edges are hardcoded in `graph.py`.

## Testing

- Run `pytest tests/ -v` before any final commit.
- All new agent nodes should have a unit test using fixtures from `tests/conftest.py`.
- LLM-backed agents (planner, processor): mock `shared.llm.get_llm`.
- Non-LLM agents (retriever): mock `httpx.get`.
- Functional tests that exercise the full compiled graph should use mocked LLM + httpx to stay fast and hermetic.
- Log test results and any remediations in `docs/TEST_LOG.md`.

## Commits

- One logical change per commit.
- Commit messages: short (under 72 chars) summary line, blank line, then bullet points of what and why.
- Do not commit secrets, `.env` files, `__pycache__`, `outputs/`, `*.db`.
