from shared.contracts import (
    PlannerInput,
    PlannerOutput,
    ProcessorInput,
    ProcessorOutput,
    RetrieverInput,
    RetrieverOutput,
    WriterInput,
    WriterOutput,
    validate_contract,
)
from shared.models import ProcessedFinding, RawResult, ReportAssets, SubTask


def test_planner_input_valid():
    assert validate_contract({"query": "test"}, PlannerInput) == []


def test_planner_input_missing():
    warnings = validate_contract({}, PlannerInput)
    assert len(warnings) >= 1
    assert all("Contract" in w for w in warnings)


def test_planner_output_valid():
    tasks = [SubTask(description="Test", keywords=["a", "b"])]
    assert validate_contract({"sub_tasks": tasks}, PlannerOutput) == []


def test_planner_output_wrong_type():
    warnings = validate_contract({"sub_tasks": "not a list"}, PlannerOutput)
    assert len(warnings) >= 1


def test_retriever_input_empty():
    assert validate_contract({"sub_tasks": []}, RetrieverInput) == []


def test_retriever_output_valid():
    results = [RawResult(source="s", title="t", snippet="snip")]
    assert validate_contract({"raw_results": results}, RetrieverOutput) == []


def test_processor_input_valid():
    results = [RawResult(source="s", title="t", snippet="snip")]
    assert validate_contract({"raw_results": results}, ProcessorInput) == []


def test_processor_output_valid():
    findings = [ProcessedFinding(summary="s", relevance_score=2, source="src")]
    assert validate_contract({"processed_findings": findings}, ProcessorOutput) == []


def test_processor_output_score_out_of_range():
    """Pydantic ge/le on relevance_score should catch bad values."""
    from pydantic import ValidationError

    try:
        ProcessedFinding(summary="s", relevance_score=4, source="src")
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected ValidationError for score > 3")


def test_writer_input_valid():
    tasks = [SubTask(description="Test", keywords=["a"])]
    findings = [ProcessedFinding(summary="s", relevance_score=2, source="src")]
    data = {"query": "test", "sub_tasks": tasks, "processed_findings": findings}
    assert validate_contract(data, WriterInput) == []


def test_writer_output_valid():
    from pathlib import Path

    assets = ReportAssets(
        markdown_path="/tmp/test.md", json_path="/tmp/test.json", title="test"
    )
    assert validate_contract({"report": assets}, WriterOutput) == []
