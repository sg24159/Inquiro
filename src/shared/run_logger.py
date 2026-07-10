import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config.settings import get_settings


def _commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def log_run(run_data: dict) -> None:
    settings = get_settings()
    out = Path(settings.outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "eval_runs.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _commit_hash(),
        "config": {
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "arxiv_max_results": settings.arxiv_max_results,
            "relevance_threshold": settings.relevance_threshold,
            "chat_template_kwargs": settings.chat_template_kwargs,
        },
    }
    record.update(run_data)

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
