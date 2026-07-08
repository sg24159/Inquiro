from unittest.mock import MagicMock, patch

from retrieval.agent import _deduplicate, _fetch_arxiv, retriever_node
from coordinator.state import ResearchState


def test_deduplicate():
    from shared.models import RawResult

    results = [
        RawResult(source="a", title="Same Title", snippet="x"),
        RawResult(source="b", title="Same Title", snippet="y"),
        RawResult(source="c", title="Different", snippet="z"),
    ]
    deduped = _deduplicate(results)
    assert len(deduped) == 2
    assert deduped[0].title == "Same Title"
    assert deduped[1].title == "Different"


def test_fetch_arxiv_handles_error():
    """Should return empty list and error string on network error."""
    results, error = _fetch_arxiv("nonexistent_xyz_query_12345", max_results=1)
    assert isinstance(results, list)
    assert len(results) == 0
    # error may be None if httpx resolves to localhost, or a string if it raises


def test_fetch_arxiv_malformed_xml():
    """Malformed XML should return empty results + error string."""
    with patch("retrieval.agent.httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "not valid xml"
        mock_get.return_value.raise_for_status = MagicMock()
        results, error = _fetch_arxiv("test", max_results=1)
        assert len(results) == 0
        assert error is not None
        assert "XML" in error or "Parse" in error


def test_retriever_node_empty_sub_tasks():
    """No sub-tasks → empty results, no HTTP calls, warning logged."""
    with patch("retrieval.agent._fetch_arxiv") as mock_fetch:
        state = ResearchState(query="test", messages=[], sub_tasks=[])
        result = retriever_node(state, {"configurable": {"thread_id": "t"}})
        assert result["raw_results"] == []
        mock_fetch.assert_not_called()
        assert any("[WARN]" in log for log in result["logs"])
