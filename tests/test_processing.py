from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from processing.agent import _parse_findings, processor_node
from processing.tools import filter_noise, jaccard_similarity
from coordinator.state import ResearchState
from shared.models import RawResult


def test_jaccard_similarity():
    sim = jaccard_similarity("machine learning", "machine learning")
    assert abs(sim - 1.0) < 0.01
    sim = jaccard_similarity("machine learning", "deep learning")
    assert 0 < sim < 1.0


def test_filter_noise_short_snippets():
    results = [
        RawResult(source="a", title="Short", snippet="Too short."),
        RawResult(source="b", title="Long enough", snippet="This is a sufficiently long snippet that has enough words for testing purposes and should be kept."),
    ]
    filtered = filter_noise(results)
    assert len(filtered) == 1
    assert filtered[0].title == "Long enough"


def test_filter_noise_deduplicates():
    results = [
        RawResult(source="a", title="Same Duplicate Title", snippet="This is a research abstract about machine learning algorithms and their applications to real world problems."),
        RawResult(source="b", title="Same Duplicate Title", snippet="This is another research abstract about machine learning algorithms and their real world applications as well."),
    ]
    filtered = filter_noise(results)
    assert len(filtered) == 1


def test_parse_findings():
    text = (
        "FINDING|Key insight found|0.92|Source A\n"
        "FINDING|Another key point|0.85|Source B\n"
    )
    result = _parse_findings(text)
    assert len(result) == 2
    assert abs(result[0].relevance_score - 0.92) < 0.01
    assert result[0].source == "Source A"


def test_parse_findings_empty():
    assert _parse_findings("") == []


def test_parse_findings_with_markdown_noise():
    """Should strip markdown list prefixes before matching FINDING|."""
    text = (
        "- FINDING|Key insight|0.9|Source A\n"
        "* FINDING|Another point|0.8|Source B"
    )
    result = _parse_findings(text)
    assert len(result) == 2
    assert result[0].summary == "Key insight"
    assert result[1].summary == "Another point"


def test_parse_findings_with_preamble():
    """Non-matching lines before/after FINDING| should be skipped."""
    text = (
        "Here are the summarized results:\n"
        "FINDING|First finding|0.9|Source A\n"
        "FINDING|Second finding|0.8|Source B\n"
        "--- end ---"
    )
    result = _parse_findings(text)
    assert len(result) == 2
    assert result[0].summary == "First finding"
    assert result[1].summary == "Second finding"


def test_processor_node_skips_llm_on_empty():
    """Empty raw_results should skip LLM call, return empty, log warning."""
    with patch("shared.llm.get_llm") as mock:
        state = ResearchState(query="test", messages=[], raw_results=[])
        result = processor_node(state, {"configurable": {"thread_id": "t"}})
        assert result["processed_findings"] == []
        mock.assert_not_called()
        assert any("[WARN]" in log for log in result["logs"])


def test_processor_node_all_results_filtered():
    """When filter_noise removes everything, processor logs warning, skips LLM."""
    results = [
        RawResult(source="s", title="t", snippet="Short."),
    ]
    with patch("shared.llm.get_llm") as mock:
        state = ResearchState(query="test", messages=[], raw_results=results)
        result = processor_node(state, {"configurable": {"thread_id": "t"}})
        assert result["processed_findings"] == []
        mock.assert_not_called()
        assert any("No raw results" in log for log in result["logs"])


def test_parse_findings_source_from_last_segment():
    """Source should be parsed from the LAST pipe-delimited segment.

    If the summary contains extra pipes, the source is still found
    by taking the final segment rather than a positional index.
    """
    text = "FINDING|Some summary with extra | pipe|0.85|Final Source"
    result = _parse_findings(text)
    assert len(result) == 1
    assert result[0].summary == "Some summary with extra | pipe"
    assert abs(result[0].relevance_score - 0.85) < 0.01
    assert result[0].source == "Final Source"


def test_processor_node_with_results():
    """Valid raw_results → calls LLM, parses findings."""
    results = [
        RawResult(
            source="s1",
            title="Paper A",
            snippet="This is a sufficiently long abstract about machine learning that passes the noise filter.",
        ),
    ]
    with patch("shared.llm.get_llm") as mock:
        llm_instance = MagicMock()
        llm_instance.invoke.return_value = AIMessage(content=(
            "FINDING|Machine learning is key|0.91|Paper A\n"
            "FINDING|AI requires data|0.85|Paper A\n"
        ))
        mock.return_value = llm_instance
        state = ResearchState(query="test", messages=[], raw_results=results)
        result = processor_node(state, {"configurable": {"thread_id": "t"}})
        assert len(result["processed_findings"]) == 2
        assert result["processed_findings"][0].summary == "Machine learning is key"
        assert abs(result["processed_findings"][0].relevance_score - 0.91) < 0.01
        assert result["processed_findings"][1].summary == "AI requires data"
        assert abs(result["processed_findings"][1].relevance_score - 0.85) < 0.01
