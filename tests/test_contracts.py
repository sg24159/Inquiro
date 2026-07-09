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
    assets = ReportAssets(
        markdown_path="/tmp/test.md", json_path="/tmp/test.json", title="test"
    )
    assert validate_contract({"report": assets}, WriterOutput) == []


def test_writer_output_none_report():
    """WriterOutput allows report=None."""
    assert validate_contract({"report": None}, WriterOutput) == []


def test_writer_input_missing_query():
    warnings = validate_contract({"sub_tasks": [], "processed_findings": []}, WriterInput)
    assert len(warnings) >= 1


def test_writer_input_missing_findings():
    warnings = validate_contract({"query": "test", "sub_tasks": []}, WriterInput)
    assert len(warnings) >= 1


def test_relevance_score_lower_boundary():
    """Score of 0 is valid (ge=0)."""
    finding = ProcessedFinding(summary="s", relevance_score=0, source="src")
    assert finding.relevance_score == 0


def test_relevance_score_upper_boundary():
    """Score of 3 is valid (le=3)."""
    finding = ProcessedFinding(summary="s", relevance_score=3, source="src")
    assert finding.relevance_score == 3


def test_relevance_score_above_range():
    from pydantic import ValidationError

    try:
        ProcessedFinding(summary="s", relevance_score=4, source="src")
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected ValidationError for score > 3")


def test_relevance_score_below_range():
    from pydantic import ValidationError

    try:
        ProcessedFinding(summary="s", relevance_score=-1, source="src")
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected ValidationError for score < 0")


def test_validate_contract_rejects_non_dict():
    """validate_contract with non-dict should raise TypeError."""
    try:
        validate_contract("not a dict", PlannerInput)
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass
