from processing.agent import _parse_findings
from processing.tools import filter_noise, jaccard_similarity
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
