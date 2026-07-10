import json
import tempfile
from pathlib import Path

from coordinator.state import ResearchState
from shared.models import ProcessedFinding, SubTask
from writing.agent import _save_assets


def test_writer_node(tmp_path, monkeypatch):
    """writer_node renders template, saves markdown + JSON with findings."""
    monkeypatch.chdir(tmp_path)
    from writing.agent import writer_node

    tasks = [SubTask(description="Test Task", keywords=["ml"])]
    findings = [
        ProcessedFinding(summary="A finding", relevance_score=2, source="src"),
    ]
    state = ResearchState(
        query="test",
        messages=[],
        sub_tasks=tasks,
        processed_findings=findings,
    )
    result = writer_node(state, {"configurable": {"thread_id": "t"}})
    assert result["report"] is not None
    assert Path(result["report"].markdown_path).exists()
    body = Path(result["report"].markdown_path).read_text()
    assert "Test Task" in body
    assert "A finding" in body
    assert "Synthesized Answer" in body
    assert "was not produced" in body
    assert result["logs"][0].startswith("[Writer]")


def test_writer_node_with_synthesized_answer(tmp_path, monkeypatch):
    """When synthesized_answer is set, it appears in the report."""
    monkeypatch.chdir(tmp_path)
    from writing.agent import writer_node

    tasks = [SubTask(description="Test Task", keywords=["ml"])]
    findings = [
        ProcessedFinding(summary="A finding", relevance_score=2, source="src"),
    ]
    state = ResearchState(
        query="test",
        messages=[],
        sub_tasks=tasks,
        processed_findings=findings,
        synthesized_answer="Machine learning is a key field.",
    )
    result = writer_node(state, {"configurable": {"thread_id": "t"}})
    body = Path(result["report"].markdown_path).read_text()
    assert "Machine learning is a key field." in body
    assert "was not produced" not in body


def test_save_assets_empty_findings(tmp_path, monkeypatch):
    """Empty findings should still produce valid files."""
    monkeypatch.chdir(tmp_path)
    assets = _save_assets(title="Empty Report", md_body="# Report", findings=[])
    assert Path(assets.markdown_path).exists()
    assert Path(assets.json_path).exists()
    with open(assets.json_path) as f:
        data = json.load(f)
    assert data["findings"] == []


def test_save_assets_creates_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    findings = [
        ProcessedFinding(summary="First finding", relevance_score=3, source="src_a"),
    ]
    sub_tasks = [
        SubTask(description="Test task", keywords=["ml", "ai"]),
    ]
    assets = _save_assets(
        title="Test Query",
        md_body="# Report Body",
        findings=findings,
        query="test query",
        sub_tasks=sub_tasks,
        logs=["[Planner] done", "[Retriever] done"],
    )
    assert Path(assets.markdown_path).exists()
    assert Path(assets.json_path).exists()
    with open(assets.json_path) as f:
        data = json.load(f)
    assert data["title"] == "Test Query"
    assert data["query"] == "test query"
    assert len(data["findings"]) == 1
    assert len(data["sub_tasks"]) == 1
    assert data["sub_tasks"][0]["description"] == "Test task"
    assert data["sub_tasks"][0]["keywords"] == ["ml", "ai"]
    assert data["logs"] == ["[Planner] done", "[Retriever] done"]


def test_save_assets_special_characters_in_title(tmp_path, monkeypatch):
    """Special characters in title should be sanitized in filename."""
    monkeypatch.chdir(tmp_path)
    assets = _save_assets(
        title="What is 2+2? (and why?)",
        md_body="# Report",
        findings=[],
    )
    assert Path(assets.markdown_path).exists()
    filename = Path(assets.markdown_path).name
    assert not any(c in filename for c in "?()")


def test_writer_node_missing_template(monkeypatch, tmp_path):
    """Writer node should propagate Jinja2 errors when template is broken."""
    from unittest.mock import patch as mock_patch
    from writing.agent import writer_node
    from coordinator.state import ResearchState

    monkeypatch.chdir(tmp_path)
    state = ResearchState(
        query="test",
        messages=[],
        sub_tasks=[],
        processed_findings=[],
    )
    with mock_patch("writing.agent.Environment") as mock_env_cls:
        mock_env = mock_env_cls.return_value
        mock_env.get_template.side_effect = Exception("template render failed")
        try:
            writer_node(state, {"configurable": {"thread_id": "t"}})
            raise AssertionError("Expected Exception")
        except Exception as e:
            assert "template render failed" in str(e)
