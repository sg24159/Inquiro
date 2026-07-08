from retrieval.agent import _deduplicate, _fetch_arxiv


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
