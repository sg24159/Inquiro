from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from processing.agent import _parse_score, _parse_summary, processor_node
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


def test_parse_score():
    text = "##final score: 3"
    result = _parse_score(text)
    assert result == 3


def test_parse_score_empty():
    assert _parse_score("") == -1


def test_parse_score_with_markdown_noise():
    text = "- ##final score: 2"
    result = _parse_score(text)
    assert result == 2


def test_parse_score_with_preamble():
    text = (
        "Here is my analysis:\n"
        "##final score: 3\n"
        "--- end ---"
    )
    result = _parse_score(text)
    assert result == 3


def test_parse_score_case_insensitive():
    text = "##FINAL SCORE: 1"
    result = _parse_score(text)
    assert result == 1


def test_parse_summary():
    text = "FINDING|Key insight about machine learning"
    result = _parse_summary(text)
    assert result == "Key insight about machine learning"


def test_parse_summary_empty():
    assert _parse_summary("") is None


def test_parse_summary_with_markdown_noise():
    text = "- FINDING|Key insight\n"
    result = _parse_summary(text)
    assert result == "Key insight"


def test_parse_summary_with_preamble():
    text = (
        "Here is the summary:\n"
        "FINDING|Key insight\n"
        "--- end ---"
    )
    result = _parse_summary(text)
    assert result == "Key insight"


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


def test_processor_node_below_threshold():
    """Papers scoring below threshold should be dropped."""
    results = [
        RawResult(
            source="s1",
            title="Paper A",
            snippet="This is a sufficiently long abstract about machine learning that passes the noise filter.",
        ),
    ]
    with patch("shared.llm.get_llm") as mock:
        scorer = MagicMock()
        scorer.invoke.return_value = AIMessage(content="##final score: 1")
        summarizer = MagicMock()
        mock.side_effect = [scorer, summarizer]
        state = ResearchState(query="test", messages=[], raw_results=results)
        result = processor_node(state, {"configurable": {"thread_id": "t"}})
        assert result["processed_findings"] == []
        assert any("skipped" in log for log in result["logs"])


def test_processor_node_with_results():
    """Valid raw_results → scores then summarizes, returns findings above threshold."""
    results = [
        RawResult(
            source="s1",
            title="Paper A",
            snippet="This is a sufficiently long abstract about machine learning that passes the noise filter.",
        ),
    ]
    with patch("shared.llm.get_llm") as mock:
        scorer = MagicMock()
        scorer.invoke.return_value = AIMessage(content="##final score: 3")
        summarizer = MagicMock()
        summarizer.invoke.return_value = AIMessage(content="FINDING|Machine learning is key")
        mock.side_effect = [scorer, summarizer]
        state = ResearchState(query="test", messages=[], raw_results=results)
        result = processor_node(state, {"configurable": {"thread_id": "t"}})
        assert len(result["processed_findings"]) == 1
        assert result["processed_findings"][0].summary == "Machine learning is key"
        assert result["processed_findings"][0].relevance_score == 3
        assert result["processed_findings"][0].source_url == "s1"


def test_parse_score_fractional():
    """Fractional score should fail int() conversion, return -1."""
    assert _parse_score("final score: 2.5") == -1


def test_parse_score_empty_after_colon():
    """Empty value after colon should fail int() conversion, return -1."""
    assert _parse_score("final score: ") == -1


def test_parse_score_negative():
    """Negative score parses as int but is treated as failure by threshold."""
    assert _parse_score("final score: -1") == -1


def test_parse_summary_returns_first():
    """Multiple FINDING lines should return the first one."""
    text = "FINDING|First finding\nFINDING|Second finding"
    assert _parse_summary(text) == "First finding"


def test_filter_noise_empty_list():
    """Empty input should return empty list."""
    assert filter_noise([]) == []


def test_filter_noise_single_item():
    """Single long-enough item should be kept."""
    results = [
        RawResult(source="a", title="A", snippet="This is a sufficiently long snippet that passes the filter."),
    ]
    filtered = filter_noise(results)
    assert len(filtered) == 1


def test_processor_node_llm_failure():
    """LLM exception during scoring should propagate (not silently swallowed)."""
    results = [
        RawResult(
            source="s1",
            title="Paper A",
            snippet="This is a sufficiently long abstract about machine learning that passes the noise filter.",
        ),
    ]
    with patch("shared.llm.get_llm") as mock:
        scorer = MagicMock()
        scorer.invoke.side_effect = RuntimeError("LLM connection lost")
        mock.return_value = scorer
        state = ResearchState(query="test", messages=[], raw_results=results)
        try:
            processor_node(state, {"configurable": {"thread_id": "t"}})
            raise AssertionError("Expected RuntimeError")
        except RuntimeError as e:
            assert "LLM connection lost" in str(e)


def test_processor_node_mixed_threshold_results():
    """Papers above and below threshold should be split correctly."""
    results = [
        RawResult(
            source="s1",
            title="High Score Paper",
            snippet="This is a sufficiently long abstract about machine learning that passes the noise filter.",
        ),
        RawResult(
            source="s2",
            title="Low Score Paper",
            snippet="Another sufficiently long abstract about deep neural networks for image classification tasks.",
        ),
    ]
    with patch("shared.llm.get_llm") as mock:
        scorer = MagicMock()
        scorer.invoke.side_effect = [
            AIMessage(content="##final score: 3"),
            AIMessage(content="##final score: 1"),
        ]
        summarizer = MagicMock()
        summarizer.invoke.return_value = AIMessage(content="FINDING|Key insight")
        mock.side_effect = [scorer, summarizer]
        state = ResearchState(query="test", messages=[], raw_results=results)
        result = processor_node(state, {"configurable": {"thread_id": "t"}})
        assert len(result["processed_findings"]) == 1
        assert result["processed_findings"][0].source == "High Score Paper"
        assert any("skipped" in log for log in result["logs"])
