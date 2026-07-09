"""HTTP-based retrieval utilities for academic APIs."""
import json
import os
from hashlib import sha256
from pathlib import Path

from shared.models import RawResult

CACHE_DIR = Path("outputs") / "arxiv_cache"


def _cache_key(query: str, max_results: int) -> str:
    raw = f"{query}|{max_results}".encode()
    return sha256(raw).hexdigest()


def _cache_load(query: str, max_results: int) -> tuple[list[RawResult], str | None] | None:
    path = CACHE_DIR / _cache_key(query, max_results)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        results = [RawResult.model_validate(r) for r in data["results"]]
        return results, data.get("error")
    except Exception:
        return None


def _cache_store(query: str, max_results: int, results: list[RawResult], error: str | None) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = CACHE_DIR / _cache_key(query, max_results)
    data = {
        "results": [r.model_dump() for r in results],
        "error": error,
    }
    path.write_text(json.dumps(data, indent=2, default=str))
