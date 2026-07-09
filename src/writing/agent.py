from pathlib import Path

from jinja2 import Environment, PackageLoader

from coordinator.state import ResearchState
from shared.contracts import WriterInput, WriterOutput, validate_contract
from shared.models import ReportAssets, SubTask


def writer_node(state: ResearchState, config) -> dict:
    query = state.get("query", "Research Report")
    sub_tasks = state.get("sub_tasks", [])
    findings = state.get("processed_findings", [])
    all_logs = state.get("logs", [])
    warnings = validate_contract(
        {"query": query, "sub_tasks": sub_tasks, "processed_findings": findings},
        WriterInput,
    )
    env = Environment(loader=PackageLoader("writing", "templates"))
    md_body = env.get_template("report.md.j2").render(
        title=query,
        sub_tasks=sub_tasks,
        findings=findings,
    )
    assets = _save_assets(
        title=query,
        md_body=md_body,
        findings=findings,
        query=query,
        sub_tasks=sub_tasks,
        logs=all_logs,
    )
    logs = [f"[Writer] Saved report to {assets.markdown_path}"]
    logs.extend(warnings)
    logs.extend(validate_contract({"report": assets}, WriterOutput))
    return {
        "report": assets,
        "logs": logs,
    }


def _save_assets(
    title: str,
    md_body: str,
    findings: list,
    query: str = "",
    sub_tasks: list[SubTask] | None = None,
    logs: list[str] | None = None,
) -> ReportAssets:
    from datetime import datetime
    import json

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:50].strip("-_")
    stem = f"{ts}_{safe}"

    md_path = out / f"{stem}.md"
    md_path.write_text(md_body)

    json_path = out / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "title": title,
                "query": query,
                "generated": datetime.now().isoformat(),
                "sub_tasks": [
                    {"description": st.description, "keywords": st.keywords}
                    for st in (sub_tasks or [])
                ],
                "findings": [
                    {
                        "summary": f.summary,
                        "relevance_score": f.relevance_score,
                        "source": f.source,
                        "source_url": f.source_url,
                    }
                    for f in findings
                ],
                "logs": logs or [],
            },
            indent=2,
        )
    )
    return ReportAssets(
        markdown_path=str(md_path),
        json_path=str(json_path),
        title=title,
    )
