from langchain_core.messages import HumanMessage, SystemMessage

from config import agents_config
from config.settings import get_settings
from coordinator.state import ResearchState
from processing.tools import filter_noise
from shared import llm as llm_module
from shared.contracts import ProcessorInput, ProcessorOutput, validate_contract
from shared.models import ProcessedFinding
from shared.utils import strip_line_noise


def processor_node(state: ResearchState, config) -> dict:
    raw = state.get("raw_results", [])
    query = state.get("query", "")
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

    scorer_prompt = agents_config["processor"]["scorer_prompt"]
    summarizer_prompt = agents_config["processor"]["summarizer_prompt"]
    scorer_llm = llm_module.get_llm(temperature=0.0)
    summarizer_llm = llm_module.get_llm(temperature=0.2)
    settings = get_settings()
    threshold = settings.relevance_threshold
    findings = []

    for r in filtered:
        score = _score_paper(scorer_llm, scorer_prompt, query, r)
        if score < threshold:
            logs.append(
                f"  Paper '{r.title[:60]}': score={score} < threshold={threshold} — skipped."
            )
            continue

        summary = _summarize_paper(summarizer_llm, summarizer_prompt, r)
        if summary is None:
            logs.append(
                f"  Paper '{r.title[:60]}': score={score} but failed to parse summary — skipped."
            )
            continue

        findings.append(
            ProcessedFinding(
                summary=summary,
                relevance_score=score,
                source=r.title,
                source_url=r.source,
            )
        )
        logs.append(f"  Paper '{r.title[:60]}': score={score} — included.")

    logs.append(
        f"  Produced {len(findings)} findings across {len(filtered)} papers "
        f"(threshold={threshold})."
    )
    if not findings:
        logs.append("  [WARN] No papers passed the relevance threshold.")
    logs.extend(validate_contract({"processed_findings": findings}, ProcessorOutput))
    return {
        "processed_findings": findings,
        "logs": logs,
    }


def _score_paper(llm, prompt: str, query: str, paper) -> int:
    context = f"Title: {paper.title}\nAbstract: {paper.snippet[:2000]}"
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Query: {query}\nPassage: {context}"),
    ]
    response = llm.invoke(messages)
    return _parse_score(response.content)


def _parse_score(text: str) -> int:
    for line in text.strip().split("\n"):
        cleaned = strip_line_noise(line)
        if "final score" in cleaned.lower():
            parts = cleaned.split(":")
            if len(parts) >= 2:
                try:
                    return int(parts[-1].strip())
                except ValueError:
                    pass
    return -1


def _summarize_paper(llm, prompt: str, paper) -> str | None:
    context = f"Title: {paper.title}\nAbstract: {paper.snippet[:2000]}"
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=context),
    ]
    response = llm.invoke(messages)
    return _parse_summary(response.content)


def _parse_summary(text: str) -> str | None:
    for line in text.strip().split("\n"):
        cleaned = strip_line_noise(line)
        if cleaned.startswith("FINDING|"):
            parts = cleaned.split("|", 1)
            if len(parts) >= 2:
                return parts[1].strip()
    return None
