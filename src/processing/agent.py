from langchain_core.messages import HumanMessage, SystemMessage

from config import agents_config
from coordinator.state import ResearchState
from processing.tools import filter_noise
from shared import llm as llm_module
from shared.contracts import ProcessorInput, ProcessorOutput, validate_contract
from shared.models import ProcessedFinding
from shared.utils import strip_line_noise


def processor_node(state: ResearchState, config) -> dict:
    raw = state.get("raw_results", [])
    warnings = validate_contract({"raw_results": raw}, ProcessorInput)
    filtered = filter_noise(raw)
    logs = [
        f"[Processor] Filtered {len(raw)} → {len(filtered)} after noise removal."
    ]
    logs.extend(warnings)

    if not filtered:
        logs.append("  [WARN] No raw results to process — skipping LLM call.")
        logs.extend(validate_contract({"processed_findings": []}, ProcessorOutput))
        return {"processed_findings": [], "logs": logs}

    llm = llm_module.get_llm(temperature=0.2)
    system_prompt = agents_config["processor"]["system_prompt"]
    findings = []

    for r in filtered:
        context = f"Title: {r.title}\nURL: {r.source}\nAbstract: {r.snippet[:200]}"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                "Summarise and score the following research result "
                "(relevance 0-1).\n\n"
                f"{context}\n\n"
                "Output format:\n"
                "FINDING|summary|relevance_score|source_title"
            )),
        ]
        response = llm.invoke(messages)
        parsed = _parse_findings(response.content)
        for f in parsed:
            f.source_url = r.source
            findings.append(f)
        logs.append(f"  Raw LLM response:\n{response.content}")

    logs.append(f"  Produced {len(findings)} scored findings across {len(filtered)} papers.")
    if not findings:
        logs.append("  [WARN] No FINDING| lines parsed from any LLM response.")
    logs.extend(validate_contract({"processed_findings": findings}, ProcessorOutput))
    return {
        "processed_findings": findings,
        "logs": logs,
    }


def _parse_findings(text: str) -> list[ProcessedFinding]:
    findings = []
    for line in text.strip().split("\n"):
        cleaned = strip_line_noise(line)
        if not cleaned.startswith("FINDING|"):
            continue
        parts = cleaned.split("|")
        if len(parts) >= 4:
            summary = "|".join(parts[1:-2]).strip()
            try:
                score = float(parts[-2])
            except ValueError:
                score = 0.5
            source = parts[-1].strip()
            findings.append(
                ProcessedFinding(
                    summary=summary,
                    relevance_score=score,
                    source=source,
                )
            )
    return findings
