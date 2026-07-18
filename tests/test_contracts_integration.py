import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from coordinator.state import ResearchState
from planning.agent import planner_node
from retrieval.agent import retriever_node
from processing.agent import processor_node
from writing.agent import writer_node
from shared.models import RawResult, ReportAssets, SubTask


STATE_FIELDS = ResearchState.__annotations__

EXPECTED_TYPES = {
    "query": str,
    "sub_tasks": list,
    "raw_results": list,
    "processed_findings": list,
    "iteration_count": int,
    "logs": list,
    "messages": list,
}

# Fields that are Optional or union types need custom checks
OPTIONAL_FIELDS = {"report"}


def _type_matches(value, field_name):
    if field_name == "report":
        return value is None or isinstance(value, ReportAssets)
    expected = EXPECTED_TYPES.get(field_name)
    if expected is None:
        return True
    return isinstance(value, expected)


@pytest.fixture
def base_state():
    return ResearchState(
        query="test query",
        messages=[],
        sub_tasks=[],
        raw_results=[],
        processed_findings=[],
    )


@pytest.fixture
def mock_llm_for_planner():
    with patch("shared.llm.get_llm") as mock:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(
            content="TASK|Test task|keyword1, keyword2"
        )
        mock.return_value = llm
        yield mock


@pytest.fixture
def mock_httpx():
    with patch("retrieval.agent._fetch_arxiv") as mock:
        mock.return_value = (
            [RawResult(source="s", title="Paper", snippet="Abstract text here")],
            None,
            False,
        )
        yield mock


@pytest.fixture
def mock_llm_for_processor():
    with patch("shared.llm.get_llm") as mock:
        scorer = MagicMock()
        scorer.invoke.return_value = AIMessage(content="##final score: 3")
        summarizer = MagicMock()
        summarizer.invoke.return_value = AIMessage(content="FINDING|A finding")
        mock.side_effect = [scorer, summarizer]
        yield mock


def _validate_state_keys(returned_keys):
    for key in returned_keys:
        assert key in STATE_FIELDS, (
            f"Node returned unexpected key '{key}'. "
            f"Valid keys: {list(STATE_FIELDS.keys())}"
        )


def _validate_state_types(result):
    for key, value in result.items():
        if key in EXPECTED_TYPES:
            assert _type_matches(value, key), (
                f"Key '{key}' has type {type(value).__name__}, "
                f"expected {EXPECTED_TYPES[key]}"
            )


class TestPlannerNodeContracts:
    def test_return_keys_are_valid(self, base_state, mock_llm_for_planner):
        result = planner_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_keys(result.keys())

    def test_return_types_are_correct(self, base_state, mock_llm_for_planner):
        result = planner_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_types(result)


class TestRetrieverNodeContracts:
    def test_return_keys_are_valid(self, base_state):
        result = retriever_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_keys(result.keys())

    def test_return_types_are_correct(self, base_state):
        result = retriever_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_types(result)

    def test_with_sub_tasks(self, base_state, mock_httpx):
        base_state["sub_tasks"] = [
            SubTask(description="Task", keywords=["ml"]),
        ]
        result = retriever_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_keys(result.keys())
        _validate_state_types(result)


class TestProcessorNodeContracts:
    def test_return_keys_are_valid(self, base_state):
        result = processor_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_keys(result.keys())

    def test_return_types_are_correct(self, base_state):
        result = processor_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_types(result)

    def test_with_raw_results(self, base_state, mock_llm_for_processor):
        base_state["raw_results"] = [
            RawResult(
                source="s1",
                title="Paper A",
                snippet="This is a sufficiently long abstract about machine learning that passes the noise filter.",
            ),
        ]
        result = processor_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_keys(result.keys())
        _validate_state_types(result)


class TestWriterNodeContracts:
    def test_return_keys_are_valid(self, base_state, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = writer_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_keys(result.keys())

    def test_return_types_are_correct(self, base_state, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = writer_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_types(result)

    def test_with_findings(self, base_state, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from shared.models import ProcessedFinding
        base_state["processed_findings"] = [
            ProcessedFinding(summary="A finding", relevance_score=2, source="src"),
        ]
        base_state["sub_tasks"] = [
            SubTask(description="Task", keywords=["ml"]),
        ]
        result = writer_node(base_state, {"configurable": {"thread_id": "t"}})
        _validate_state_keys(result.keys())
        _validate_state_types(result)
