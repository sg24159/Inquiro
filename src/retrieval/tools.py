"""HTTP-based retrieval utilities for academic APIs."""
import json
import os
from hashlib import sha256
from pathlib import Path

from shared.models import RawResult


def _cache_dir() -> Path:
    from config.settings import get_settings
    return Path(get_settings().outputs_dir) / "arxiv_cache"


def _cache_key(query: str, max_results: int) -> str:
    raw = f"{query.strip().lower()}|{max_results}".encode()
    return sha256(raw).hexdigest()


def _cache_load(query: str, max_results: int) -> tuple[list[RawResult], str | None] | None:
    path = _cache_dir() / _cache_key(query, max_results)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        results = [RawResult.model_validate(r) for r in data["results"]]
        return results, data.get("error")
    except Exception:
        return None


def _cache_store(query: str, max_results: int, results: list[RawResult], error: str | None) -> None:
    cache_dir = _cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_dir / _cache_key(query, max_results)
    data = {
        "results": [r.model_dump() for r in results],
        "error": error,
    }
    path.write_text(json.dumps(data, indent=2, default=str))
