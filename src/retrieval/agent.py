import httpx

from config.settings import get_settings
from coordinator.state import ResearchState
from retrieval.tools import _cache_load, _cache_store, _format_title_query
from shared.contracts import RetrieverInput, RetrieverOutput, validate_contract
from shared.models import RawResult

ARXIV_URL = "https://export.arxiv.org/api/query"


def retriever_node(state: ResearchState, config) -> dict:
    settings = get_settings()
    sub_tasks = state.get("sub_tasks", [])
    warnings = validate_contract({"sub_tasks": sub_tasks}, RetrieverInput)
    logs: list[str] = list(warnings)

    if not sub_tasks:
        logs.append("[Retriever] [WARN] No sub-tasks to query — returning empty.")
        logs.extend(validate_contract({"raw_results": []}, RetrieverOutput))
        return {"raw_results": [], "logs": logs}

    all_results: list[RawResult] = []
    cache_hits = 0
    total_queries = 0
    for idx, task in enumerate(sub_tasks):
        keywords = task.keywords
        if not keywords:
            continue
        title_query = _format_title_query(keywords)
        abstract_query = " OR ".join(kw.strip().lower() for kw in keywords)
        title_results, title_error, title_hit = _fetch_arxiv(
            title_query, max_results=settings.arxiv_max_results
        )
        abstract_results, abstract_error, abstract_hit = _fetch_arxiv(
            abstract_query, max_results=settings.arxiv_max_results
        )
        total_queries += 2
        cache_hits += sum([title_hit, abstract_hit])
        if title_error:
            logs.append(f"  [WARN] {title_error}")
        if abstract_error:
            logs.append(f"  [WARN] {abstract_error}")
        merged = title_results + abstract_results
        kw_log = (
            f"  sub_task[{idx}]: {len(merged)} results"
            f" ({len(title_results)} title + {len(abstract_results)} abstract)"
        )
        logs.append(kw_log)
        for r in merged:
            r.sub_task_idx = idx
        all_results.extend(merged)
    deduplicated = _deduplicate(all_results)
    cache_msg = f", {cache_hits}/{total_queries} from cache" if total_queries else ""
    logs.insert(
        0,
        f"[Retriever] Queried {len(sub_tasks)} sub-tasks, "
        f"found {len(deduplicated)} unique results{cache_msg}.",
    )
    logs.extend(validate_contract({"raw_results": deduplicated}, RetrieverOutput))
    return {
        "raw_results": deduplicated,
        "logs": logs,
    }


def _fetch_arxiv(query: str, max_results: int = 3) -> tuple[list[RawResult], str | None, bool]:
    cached = _cache_load(query, max_results)
    if cached is not None:
        return cached[0], cached[1], True

    params = {
        "search_query": query.strip(),
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
            authors = [
                author.findtext("a:name", "", ns).strip()
                for author in entry.findall("a:author", ns)
                if author.findtext("a:name", "", ns).strip()
            ]
            published = entry.findtext("a:published", "", ns).strip()
            results.append(
                RawResult(
                    source=source,
                    title=title,
                    snippet=summary[:2000],
                    authors=authors,
                    published=published,
                )
            )
        _cache_store(query, max_results, results, error)
    except httpx.HTTPError as e:
        error = f"HTTP error querying arXiv for '{query}': {e}"
    except ET.ParseError as e:
        error = f"XML parse error for arXiv response on '{query}': {e}"
    return results, error, False


def _deduplicate(results: list[RawResult]) -> list[RawResult]:
    seen_titles: set[str] = set()
    deduped: list[RawResult] = []
    for r in results:
        key = r.title.lower().strip()[:80]
        if key and key not in seen_titles:
            seen_titles.add(key)
            deduped.append(r)
    return deduped
