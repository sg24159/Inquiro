from langchain_core.messages import HumanMessage, SystemMessage

from config import agents_config
from coordinator.state import ResearchState
from processing.tools import filter_noise
from shared import llm as llm_module
from shared.models import ProcessedFinding


def processor_node(state: ResearchState, config) -> dict:
    raw = state.get("raw_results", [])
    filtered = filter_noise(raw)
    context_lines = "\n".join(
        f"[{i}] {r.title}\n    {r.snippet[:200]}" for i, r in enumerate(filtered)
    )
    llm = llm_module.get_llm(temperature=0.2)
    system_prompt = agents_config["processor"]["system_prompt"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            "Summarise and score the following research results "
            "(relevance 0-1). Output one line per result:\n\n"
            f"{context_lines}\n\n"
            "Output format:\n"
            "FINDING|summary|relevance_score|source_title"
        )),
    ]
    response = llm.invoke(messages)
    findings = _parse_findings(response.content)
    return {
        "processed_findings": findings,
        "messages": [response],
        "logs": [
            f"[Processor] Filtered {len(raw)} → {len(filtered)} after noise removal, "
            f"produced {len(findings)} scored findings."
        ],
    }


def _parse_findings(text: str) -> list[ProcessedFinding]:
    findings = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line.startswith("FINDING|"):
            continue
        parts = line.split("|")
        if len(parts) >= 4:
            try:
                score = float(parts[2])
            except ValueError:
                score = 0.5
            findings.append(
                ProcessedFinding(
                    summary=parts[1].strip(),
                    relevance_score=score,
                    source=parts[3].strip(),
                )
            )
    return findings
