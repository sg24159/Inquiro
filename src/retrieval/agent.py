import httpx

from coordinator.state import ResearchState
from shared.models import RawResult

ARXIV_URL = "http://export.arxiv.org/api/query"


def retriever_node(state: ResearchState, config) -> dict:
    sub_tasks = state.get("sub_tasks", [])
    all_results: list[RawResult] = []
    for idx, task in enumerate(sub_tasks):
        for keyword in task.keywords:
            results = _fetch_arxiv(keyword, max_results=3)
            for r in results:
                r.sub_task_idx = idx
            all_results.extend(results)
    deduplicated = _deduplicate(all_results)
    return {
        "raw_results": deduplicated,
        "logs": [
            f"[Retriever] Queried {sum(len(t.keywords) for t in sub_tasks)} "
            f"keyword sets, found {len(deduplicated)} unique results."
        ],
    }


def _fetch_arxiv(query: str, max_results: int = 3) -> list[RawResult]:
    params = {
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
    }
    results: list[RawResult] = []
    try:
        resp = httpx.get(ARXIV_URL, params=params, timeout=15.0)
        resp.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = entry.findtext("a:title", "").replace("\n", " ").strip()
            summary = entry.findtext("a:summary", "").replace("\n", " ").strip()
            link_el = entry.find("a:id", ns)
            source = link_el.text.strip() if link_el is not None else ""
            results.append(RawResult(source=source, title=title, snippet=summary[:500]))
    except httpx.HTTPError:
        pass
    return results


def _deduplicate(results: list[RawResult]) -> list[RawResult]:
    seen_titles: set[str] = set()
    deduped: list[RawResult] = []
    for r in results:
        key = r.title.lower().strip()[:80]
        if key and key not in seen_titles:
            seen_titles.add(key)
            deduped.append(r)
    return deduped
