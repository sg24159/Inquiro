import time

from langchain_core.messages import HumanMessage, SystemMessage

from config import agents_config
from config.settings import get_settings
from coordinator.state import ResearchState
from processing.tools import filter_noise
from shared import llm as llm_module
from shared.contracts import ProcessorInput, ProcessorOutput, validate_contract
from shared.models import ProcessedFinding, _format_citation_author
from shared.utils import strip_line_noise


def processor_node(state: ResearchState, config) -> dict:
    _t0 = time.time()
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
    score_dist: dict[int, int] = {}
    scorer_tok_in = 0
    scorer_tok_out = 0
    summarizer_tok_in = 0
    summarizer_tok_out = 0

    for r in filtered:
        score = _score_paper(scorer_llm, scorer_prompt, query, r)
        score_dist[score] = score_dist.get(score, 0) + 1
        su = getattr(_score_paper, "_last_usage", None) or {}
        if isinstance(su, dict):
            scorer_tok_in += su.get("input_tokens", 0)
            scorer_tok_out += su.get("output_tokens", 0)

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
        stu = getattr(_summarize_paper, "_last_usage", None) or {}
        if isinstance(stu, dict):
            summarizer_tok_in += stu.get("input_tokens", 0)
            summarizer_tok_out += stu.get("output_tokens", 0)

        findings.append(
            ProcessedFinding(
                summary=summary,
                relevance_score=score,
                source=r.title,
                source_url=r.source,
                year=r.published[:4] if r.published else "",
                citation_author=_format_citation_author(r.authors),
            )
        )
        logs.append(f"  '{r.title[:60]}': score={score} — included.")

    findings.sort(
        key=lambda f: (f.year or "0", f.relevance_score),
        reverse=True,
    )
    logs.append(
        f"  Produced {len(findings)} findings across {len(filtered)} sources "
        f"(threshold={threshold})."
    )
    synthesized_answer = ""
    synthesis_tok_in = 0
    synthesis_tok_out = 0
    if findings:
        synthesized_answer = _synthesize_answer(
            summarizer_llm, synthesizer_prompt, query, findings
        )
        metadata = getattr(_synthesize_answer, "_last_metadata", {})
        actual_model = metadata.get("model_name", summarizer_llm.model_name)
        resolved_model = llm_module.resolve_model_info(
            summarizer_llm.openai_api_base, actual_model
        )
        su = getattr(_synthesize_answer, "_last_usage", {})
        if isinstance(su, dict):
            synthesis_tok_in = su.get("input_tokens", 0)
            synthesis_tok_out = su.get("output_tokens", 0)
        tokens = ""
        if su:
            tokens = (
                f" | in={(su or {}).get('input_tokens', '?')} "
                f"out={(su or {}).get('output_tokens', '?')} "
                f"total={(su or {}).get('total_tokens', '?')}"
            )
        logs.append(
            f"  Generated synthesized answer ({len(synthesized_answer)} chars)"
            f"{tokens} via {resolved_model} ({summarizer_llm.openai_api_base})."
        )
    else:
        resolved_model = llm_module.resolve_model_info(
            summarizer_llm.openai_api_base, summarizer_llm.model_name
        )
        logs.append("  [WARN] No sources passed the relevance threshold.")
    elapsed = time.time() - _t0
    logs.append(f"  Completed in {elapsed:.1f}s")
    logs.extend(
        validate_contract(
            {"processed_findings": findings, "synthesized_answer": synthesized_answer},
            ProcessorOutput,
        )
    )
    return {
        "processed_findings": findings,
        "synthesized_answer": synthesized_answer,
        "resolved_model": resolved_model,
        "logs": logs,
        "processor_stats": {
            "elapsed_s": round(elapsed, 1),
            "papers_in": len(filtered),
            "findings_out": len(findings),
            "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
            "scorer_tokens_in": scorer_tok_in,
            "scorer_tokens_out": scorer_tok_out,
            "summarizer_tokens_in": summarizer_tok_in,
            "summarizer_tokens_out": summarizer_tok_out,
            "synthesis_tokens_in": synthesis_tok_in,
            "synthesis_tokens_out": synthesis_tok_out,
        },
    }


def _score_paper(llm, prompt: str, query: str, paper) -> int:
    context = f"Title: {paper.title}\nAbstract: {paper.snippet[:2000]}"
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Query: {query}\nPassage: {context}"),
    ]
    response = llm.invoke(messages)
    _score_paper._last_usage = (
        response.usage_metadata if hasattr(response, "usage_metadata") else None
    )
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
    _summarize_paper._last_usage = (
        response.usage_metadata if hasattr(response, "usage_metadata") else None
    )
    return _parse_summary(response.content)


def _synthesize_answer(llm, prompt: str, query: str, findings: list) -> str:
    findings_text = "\n\n".join(
        f"Source: {f.source}\n"
        f"Author: {f.citation_author}\n"
        f"Year: {f.year}\n"
        f"URL: {f.source_url}\n"
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
    _synthesize_answer._last_metadata = response.response_metadata
    _synthesize_answer._last_usage = response.usage_metadata
    return response.content.strip()


def _parse_summary(text: str) -> str | None:
    for line in text.strip().split("\n"):
        cleaned = strip_line_noise(line)
        if cleaned.startswith("FINDING|"):
            parts = cleaned.split("|", 1)
            if len(parts) >= 2:
                return parts[1].strip()
    return None
