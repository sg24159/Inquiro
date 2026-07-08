"""Jinja2 template rendering and file I/O for report generation."""

import json
from datetime import datetime
from pathlib import Path


def _save_report(md_body: str, json_data: dict, title: str) -> tuple[str, str]:
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:50]
    stem = f"{ts}_{safe}"
    md_path = out / f"{stem}.md"
    md_path.write_text(md_body)
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    return str(md_path), str(json_path)
