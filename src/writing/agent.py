from pathlib import Path

from jinja2 import Environment, PackageLoader

from coordinator.state import ResearchState
from shared.models import ReportAssets


def writer_node(state: ResearchState, config) -> dict:
    query = state.get("query", "Research Report")
    sub_tasks = state.get("sub_tasks", [])
    findings = state.get("processed_findings", [])
    env = Environment(loader=PackageLoader("writing", "templates"))
    md_body = env.get_template("report.md.j2").render(
        title=query,
        sub_tasks=sub_tasks,
        findings=findings,
    )
    assets = _save_assets(query, md_body, findings)
    return {
        "report": assets,
        "logs": [f"[Writer] Saved report to {assets.markdown_path}"],
    }


def _save_assets(
    title: str, md_body: str, findings: list
) -> ReportAssets:
    from datetime import datetime
    import json

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:50]
    stem = f"{ts}_{safe}"

    md_path = out / f"{stem}.md"
    md_path.write_text(md_body)

    json_path = out / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "title": title,
                "generated": datetime.now().isoformat(),
                "findings": [
                    {
                        "summary": f.summary,
                        "relevance_score": f.relevance_score,
                        "source": f.source,
                    }
                    for f in findings
                ],
            },
            indent=2,
        )
    )
    return ReportAssets(
        markdown_path=str(md_path),
        json_path=str(json_path),
        title=title,
    )
