import httpx

from coordinator.state import ResearchState
from retrieval.tools import _cache_load, _cache_store
from shared.contracts import RetrieverInput, RetrieverOutput, validate_contract
from shared.models import RawResult

ARXIV_URL = "https://export.arxiv.org/api/query"


def retriever_node(state: ResearchState, config) -> dict:
    sub_tasks = state.get("sub_tasks", [])
    warnings = validate_contract({"sub_tasks": sub_tasks}, RetrieverInput)
    logs: list[str] = list(warnings)

    if not sub_tasks:
        logs.append("[Retriever] [WARN] No sub-tasks to query — returning empty.")
        logs.extend(validate_contract({"raw_results": []}, RetrieverOutput))
        return {"raw_results": [], "logs": logs}

    total_keywords = sum(len(t.keywords) for t in sub_tasks)
    all_results: list[RawResult] = []
    for idx, task in enumerate(sub_tasks):
        for keyword in task.keywords:
            results, error = _fetch_arxiv(keyword, max_results=3)
            if error:
                logs.append(f"  [WARN] {error}")
            kw_log = f"  sub_task[{idx}] keyword '{keyword}': {len(results)} results"
            logs.append(kw_log)
            for r in results:
                r.sub_task_idx = idx
            all_results.extend(results)
    deduplicated = _deduplicate(all_results)
    logs.insert(
        0,
        f"[Retriever] Queried {total_keywords} keyword sets across "
        f"{len(sub_tasks)} sub-tasks, found {len(deduplicated)} unique results.",
    )
    logs.extend(validate_contract({"raw_results": deduplicated}, RetrieverOutput))
    return {
        "raw_results": deduplicated,
        "logs": logs,
    }


def _fetch_arxiv(query: str, max_results: int = 3) -> tuple[list[RawResult], str | None]:
    cached = _cache_load(query, max_results)
    if cached is not None:
        return cached

    params = {
        "search_query": f"abs:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
    }
    results: list[RawResult] = []
    error: str | None = None
    try:
        resp = httpx.get(ARXIV_URL, params=params, timeout=15.0)
        resp.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = entry.findtext("a:title", "", ns).replace("\n", " ").strip()
            summary = entry.findtext("a:summary", "", ns).replace("\n", " ").strip()
            link_el = entry.find("a:id", ns)
            source = link_el.text.strip() if link_el is not None else ""
            results.append(RawResult(source=source, title=title, snippet=summary[:500]))
        _cache_store(query, max_results, results, error)
    except httpx.HTTPError as e:
        error = f"HTTP error querying arXiv for '{query}': {e}"
    except ET.ParseError as e:
        error = f"XML parse error for arXiv response on '{query}': {e}"
    return results, error


def _deduplicate(results: list[RawResult]) -> list[RawResult]:
    seen_titles: set[str] = set()
    deduped: list[RawResult] = []
    for r in results:
        key = r.title.lower().strip()[:80]
        if key and key not in seen_titles:
            seen_titles.add(key)
            deduped.append(r)
    return deduped
