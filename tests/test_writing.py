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
    assets = _save_assets("Test Query", "# Report Body", findings)
    assert Path(assets.markdown_path).exists()
    assert Path(assets.json_path).exists()
    with open(assets.json_path) as f:
        data = json.load(f)
    assert data["title"] == "Test Query"
    assert len(data["findings"]) == 1
