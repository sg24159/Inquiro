import json
import tempfile
from pathlib import Path

from shared.models import ProcessedFinding, SubTask
from writing.agent import _save_assets


def test_save_assets_creates_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    findings = [
        ProcessedFinding(summary="First finding", relevance_score=0.9, source="src_a"),
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
