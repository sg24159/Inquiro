from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from coordinator.graph import build_research_graph


@pytest.fixture
def mock_llm():
    with patch("shared.llm.get_llm") as mock:
        llm_instance = MagicMock()
        llm_instance.invoke.return_value = AIMessage(
            content=(
                "TASK|Investigate machine learning basics|ML, machine learning, algorithms\n"
                "TASK|Explore neural networks|neural networks, deep learning, backpropagation\n"
            )
        )
        mock.return_value = llm_instance
        yield mock


@pytest.fixture
def mock_httpx():
    """Mock httpx.get so retrieval doesn't call real arXiv API."""
    with patch("httpx.get") as mock:
        mock.return_value.status_code = 200
        mock.return_value.text = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.56789</id>
    <title>Test Paper Title</title>
    <summary>This is a sufficiently long test abstract with enough words to pass the noise filter threshold.</summary>
    <published>2024-03-15</published>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>"""
        mock.return_value.raise_for_status = MagicMock()
        yield mock


@pytest.fixture
def mock_processor_llm():
    with patch("shared.llm.get_llm") as mock:
        llm_instance = MagicMock()
        llm_instance.invoke.return_value = AIMessage(
            content=(
                "FINDING|Machine learning is a broad field|0.92|Test Paper\n"
                "FINDING|Neural networks require large data|0.78|Test Paper\n"
            )
        )
        mock.return_value = llm_instance
        yield mock


@pytest.fixture
def mock_llm_chain():
    """Mock for planner→scorer→summarizer→synthesizer on successive invoke calls."""
    with patch("shared.llm.get_llm") as mock:
        llm_instance = MagicMock()
        llm_instance.invoke.side_effect = [
            AIMessage(
                content=(
                    "TASK|Investigate machine learning basics|ML, machine learning, algorithms\n"
                    "TASK|Explore neural networks|neural networks, deep learning, backpropagation\n"
                )
            ),
            AIMessage(content="##final score: 2"),
            AIMessage(content="FINDING|Machine learning is a broad field that enables computers to learn."),
            AIMessage(content="Machine learning enables computers to learn from data without explicit programming. Key approaches include supervised learning with labeled datasets and unsupervised learning for pattern discovery."),
        ]
        mock.return_value = llm_instance
        yield mock


@pytest.fixture
def graph():
    return build_research_graph().compile()
