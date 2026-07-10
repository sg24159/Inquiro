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
    synthesizer_prompt = agents_config["processor"].get("synthesizer_prompt", "")
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

        summary = _summarize_paper(summarizer_llm, summarizer_prompt, query, r)
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
                year=r.published[:4] if r.published else "",
            )
        )
        logs.append(f"  Paper '{r.title[:60]}': score={score} — included.")

    findings.sort(
        key=lambda f: (f.year or "0", f.relevance_score),
        reverse=True,
    )
    logs.append(
        f"  Produced {len(findings)} findings across {len(filtered)} papers "
        f"(threshold={threshold})."
    )
    synthesized_answer = ""
    if findings:
        synthesized_answer = _synthesize_answer(
            summarizer_llm, synthesizer_prompt, query, findings
        )
        logs.append(
            f"  Generated synthesized answer ({len(synthesized_answer)} chars)."
        )
    else:
        logs.append("  [WARN] No papers passed the relevance threshold.")
    logs.extend(
        validate_contract(
            {"processed_findings": findings, "synthesized_answer": synthesized_answer},
            ProcessorOutput,
        )
    )
    return {
        "processed_findings": findings,
        "synthesized_answer": synthesized_answer,
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


def _summarize_paper(llm, prompt: str, query: str, paper) -> str | None:
    formatted_prompt = prompt.format(query=query)
    authors = ", ".join(paper.authors) if paper.authors else "N/A"
    published = paper.published or "N/A"
    context = (
        f"Title: {paper.title}\n"
        f"Authors: {authors}\n"
        f"Published: {published}\n"
        f"Abstract: {paper.snippet[:2000]}"
    )
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=context),
    ]
    response = llm.invoke(messages)
    return _parse_summary(response.content)


def _synthesize_answer(llm, prompt: str, query: str, findings: list) -> str:
    findings_text = "\n\n".join(
        f"Paper: {f.source}\n"
        f"Year: {f.year}\n"
        f"Relevance: {f.relevance_score}/3\n"
        f"Summary: {f.summary}"
        for f in findings
    )
    formatted_prompt = prompt.format(query=query, findings=findings_text)
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content="Write the synthesized answer."),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _parse_summary(text: str) -> str | None:
    for line in text.strip().split("\n"):
        cleaned = strip_line_noise(line)
        if cleaned.startswith("FINDING|"):
            parts = cleaned.split("|", 1)
            if len(parts) >= 2:
                return parts[1].strip()
    return None
