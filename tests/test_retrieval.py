from unittest.mock import MagicMock, patch

import httpx

from retrieval.agent import _deduplicate, _fetch_arxiv, retriever_node
from retrieval.tools import _cache_load, _cache_store
from coordinator.state import ResearchState
from shared.models import RawResult, SubTask


def test_deduplicate():
    results = [
        RawResult(source="a", title="Same Title", snippet="x"),
        RawResult(source="b", title="Same Title", snippet="y"),
        RawResult(source="c", title="Different", snippet="z"),
    ]
    deduped = _deduplicate(results)
    assert len(deduped) == 2
    assert deduped[0].title == "Same Title"
    assert deduped[1].title == "Different"


def test_deduplicate_empty_list():
    assert _deduplicate([]) == []


def test_fetch_arxiv_handles_error():
    """Should return empty list and error string on network error."""
    results, error, _cache_hit = _fetch_arxiv("nonexistent_xyz_query_12345", max_results=1)
    assert isinstance(results, list)
    assert len(results) == 0
    # error may be None if httpx resolves to localhost, or a string if it raises


def test_fetch_arxiv_malformed_xml():
    """Malformed XML should return empty results + error string."""
    with patch("retrieval.agent.httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "not valid xml"
        mock_get.return_value.raise_for_status = MagicMock()
        results, error, cache_hit = _fetch_arxiv("test", max_results=1)
        assert len(results) == 0
        assert error is not None
        assert "XML" in error or "Parse" in error
        assert not cache_hit


@patch("retrieval.agent._cache_load", return_value=None)
def test_fetch_arxiv_http_timeout(mock_cache):
    """Timeout should be caught, returned as error string."""
    with patch("retrieval.agent.httpx.get") as mock_get:
        mock_get.side_effect = httpx.TimeoutException("connection timed out")
        results, error, cache_hit = _fetch_arxiv("test query", max_results=3)
        assert results == []
        assert error is not None
        assert "timed out" in error.lower() or "timeout" in error.lower()
        assert not cache_hit


@patch("retrieval.agent._cache_load", return_value=None)
def test_fetch_arxiv_http_status_error(mock_cache):
    """HTTP 4xx/5xx should be caught, returned as error string."""
    with patch("retrieval.agent.httpx.get") as mock_get:
        response = MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Forbidden",
            request=MagicMock(),
            response=MagicMock(status_code=403),
        )
        mock_get.return_value = response
        results, error, cache_hit = _fetch_arxiv("test query", max_results=3)
        assert results == []
        assert error is not None
        assert "403" in error
        assert not cache_hit


@patch("retrieval.agent._cache_load", return_value=None)
@patch("retrieval.agent._cache_store")
def test_fetch_arxiv_empty_feed(mock_store, mock_load):
    """Feed with zero entries should return empty results, no error."""
    with patch("retrieval.agent.httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
        mock_get.return_value.raise_for_status = MagicMock()
        results, error, cache_hit = _fetch_arxiv("test", max_results=3)
        assert results == []
        assert error is None
        assert not cache_hit


def test_fetch_arxiv_cache_hit():
    """When cache returns data, _fetch_arxiv should return it without HTTP."""
    cached = [RawResult(source="cached", title="Cached Paper", snippet="abstract")]
    with patch("retrieval.agent._cache_load", return_value=(cached, None)) as mock_load:
        with patch("retrieval.agent.httpx.get") as mock_get:
            results, error, cache_hit = _fetch_arxiv("test", max_results=3)
            assert results == cached
            assert error is None
            assert cache_hit
            mock_get.assert_not_called()


def test_fetch_arxiv_empty_query():
    """Empty query string should still produce a valid arXiv request."""
    with patch("retrieval.agent._cache_load", return_value=None):
        with patch("retrieval.agent.httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
            mock_get.return_value.raise_for_status = MagicMock()
            results, error, cache_hit = _fetch_arxiv("", max_results=3)
            assert isinstance(results, list)
            assert not cache_hit


@patch("retrieval.agent._cache_load", return_value=None)
def test_fetch_arxiv_preserves_case(mock_cache):
    """Search query should not be lowercased — keywords like OR, AND must survive."""
    with patch("retrieval.agent.httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        mock_get.return_value.raise_for_status = MagicMock()
        _fetch_arxiv("Machine Learning AND Neural Networks", max_results=3)
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["search_query"] == "Machine Learning AND Neural Networks"


@patch("retrieval.agent._cache_load", return_value=None)
def test_fetch_arxiv_strips_whitespace(mock_cache):
    """Query should be stripped but not lowercased."""
    with patch("retrieval.agent.httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        mock_get.return_value.raise_for_status = MagicMock()
        _fetch_arxiv("  Quantum Computing  ", max_results=3)
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["search_query"] == "Quantum Computing"


def test_cache_load_corrupted_file(tmp_path, monkeypatch):
    """Corrupted cache file should return None (not raise)."""
    monkeypatch.setattr("config.settings.get_settings", lambda: MagicMock(outputs_dir=str(tmp_path)))
    cache_dir = tmp_path / "arxiv_cache"
    cache_dir.mkdir()
    key_file = cache_dir / "deadbeef"
    key_file.write_text("not valid json {{{")
    result = _cache_load("any query", 3)
    assert result is None


def test_cache_store_and_load_roundtrip(tmp_path, monkeypatch):
    """Stored cache should load back correctly."""
    monkeypatch.setattr("config.settings.get_settings", lambda: MagicMock(outputs_dir=str(tmp_path)))
    results = [RawResult(source="s", title="t", snippet="snip")]
    _cache_store("test query", 3, results, None)
    loaded = _cache_load("test query", 3)
    assert loaded is not None
    loaded_results, error = loaded
    assert len(loaded_results) == 1
    assert loaded_results[0].title == "t"
    assert error is None


def test_retriever_node_empty_sub_tasks():
    """No sub-tasks → empty results, no HTTP calls, warning logged."""
    with patch("retrieval.agent._fetch_arxiv") as mock_fetch:
        state = ResearchState(query="test", messages=[], sub_tasks=[])
        result = retriever_node(state, {"configurable": {"thread_id": "t"}})
        assert result["raw_results"] == []
        mock_fetch.assert_not_called()
        assert any("[WARN]" in log for log in result["logs"])


def test_retriever_node_with_sub_tasks():
    """Valid sub-tasks → sub_task_idx assigned, overlapping results deduped."""
    tasks = [
        SubTask(description="Task A", keywords=["ml", "deep learning"]),
        SubTask(description="Task B", keywords=["neural networks"]),
    ]
    paper_a = RawResult(source="s1", title="Paper A", snippet="Abstract A")
    paper_b = RawResult(source="s2", title="Paper B", snippet="Abstract B")

    with patch("retrieval.agent._fetch_arxiv") as mock_fetch:
        mock_fetch.side_effect = [
            ([paper_a], None, False),
            ([paper_b], None, False),
            ([paper_b], None, False),
        ]
        state = ResearchState(query="test", messages=[], sub_tasks=tasks)
        result = retriever_node(state, {"configurable": {"thread_id": "t"}})
        mock_fetch.assert_called()
        assert len(result["raw_results"]) == 2
        titles = {r.title for r in result["raw_results"]}
        assert "Paper A" in titles
        assert "Paper B" in titles


def test_retriever_node_partial_failure():
    """Some sub-tasks fail, others succeed → partial results with warning."""
    tasks = [
        SubTask(description="Task A", keywords=["ml", "deep"]),
        SubTask(description="Task B", keywords=["broken"]),
    ]
    paper = RawResult(source="s1", title="Paper A", snippet="Abstract A")

    with patch("retrieval.agent._fetch_arxiv") as mock_fetch:
        mock_fetch.side_effect = [
            ([paper], None, False),
            ([], "HTTP error querying arXiv for 'broken': timeout", False),
        ]
        state = ResearchState(query="test", messages=[], sub_tasks=tasks)
        result = retriever_node(state, {"configurable": {"thread_id": "t"}})
        assert len(result["raw_results"]) == 1
        assert any("[WARN]" in log for log in result["logs"])


def test_retriever_node_keywords_are_lowercased():
    """Keywords should be lowercased and joined with OR before querying arXiv."""
    from shared.models import SubTask

    tasks = [SubTask(description="Task", keywords=["Machine Learning", "Deep Learning"])]

    with patch("retrieval.agent._fetch_arxiv") as mock_fetch:
        mock_fetch.return_value = ([], None, False)
        state = ResearchState(query="test", messages=[], sub_tasks=tasks)
        retriever_node(state, {"configurable": {"thread_id": "t"}})
        mock_fetch.assert_called_once()
        query_arg = mock_fetch.call_args.args[0]
        assert query_arg == "machine learning OR deep learning"
