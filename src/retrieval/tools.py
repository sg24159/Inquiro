"""HTTP-based retrieval utilities for academic and web APIs."""
import json
import os
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

from shared.models import RawResult


def _format_title_query(keywords: list[str]) -> str:
    """Build an arXiv title-only OR query from keywords.

    Multi-word keywords are wrapped in quotes for exact-phrase matching.
    """
    parts = []
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        if " " in kw:
            parts.append(f'ti:"{kw}"')
        else:
            parts.append(f"ti:{kw}")
    return " OR ".join(parts)


def _arxiv_cache_dir() -> Path:
    from config.settings import get_settings
    return Path(get_settings().outputs_dir) / "arxiv_cache"


def _web_cache_dir() -> Path:
    from config.settings import get_settings
    return Path(get_settings().outputs_dir) / "web_cache"


def _cache_key(query: str, max_results: int) -> str:
    raw = f"{query.strip().lower()}|{max_results}".encode()
    return sha256(raw).hexdigest()


def _cache_load(cache_dir: Path, query: str, max_results: int) -> tuple[list[RawResult], str | None] | None:
    path = cache_dir / _cache_key(query, max_results)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        results = [RawResult.model_validate(r) for r in data["results"]]
        return results, data.get("error")
    except Exception:
        return None


def _cache_store(cache_dir: Path, query: str, max_results: int, results: list[RawResult], error: str | None) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_dir / _cache_key(query, max_results)
    data = {
        "results": [r.model_dump() for r in results],
        "error": error,
    }
    path.write_text(json.dumps(data, indent=2, default=str))


def _infer_site_name(url: str) -> str:
    """Extract a human-readable site name from a URL for citation purposes."""
    try:
        domain = urlparse(url).netloc.lower()
        for prefix in ("www.", "en.", "de.", "fr.", "es.", "pt.", "ru.", "ja.", "zh.", "ko."):
            domain = domain.removeprefix(prefix)
        name = domain.split(".")[0]
        return name.capitalize() if name else "Web"
    except Exception:
        return "Web"


def _fetch_ddg(description: str, max_results: int = 5) -> tuple[list[RawResult], str | None, bool]:
    import time
    from duckduckgo_search import DDGS

    cache_dir = _web_cache_dir()
    cached = _cache_load(cache_dir, description, max_results)
    if cached is not None:
        return cached[0], cached[1], True

    results: list[RawResult] = []
    error: str | None = None
    for attempt in range(3):
        results.clear()
        error = None
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(description, max_results=max_results, backend="html"):
                    title = (r.get("title") or "").strip()
                    link = (r.get("link") or r.get("href") or "").strip()
                    snippet = (r.get("body") or r.get("snippet") or r.get("description") or "").strip()
                    if not title and not snippet:
                        continue
                    results.append(
                        RawResult(
                            source=link,
                            title=title,
                            snippet=snippet[:2000],
                            source_type="web",
                            authors=[_infer_site_name(link)],
                        )
                    )
            if results:
                _cache_store(cache_dir, description, max_results, results, error)
                break
            if attempt < 4:
                time.sleep(15)
        except Exception as e:
            error = f"DDG error querying '{description[:80]}' (attempt {attempt + 1}): {e}"
            if attempt < 4:
                time.sleep(15)
    return results, error, False
