from langchain_core.messages import HumanMessage, SystemMessage

from config import agents_config
from coordinator.state import ResearchState
from shared import llm as llm_module
from shared.contracts import PlannerInput, PlannerOutput, validate_contract
from shared.models import SubTask
from shared.utils import strip_line_noise


def planner_node(state: ResearchState, config) -> dict:
    warnings = validate_contract({"query": state.get("query", "")}, PlannerInput)
    llm = llm_module.get_llm(temperature=0.3)
    system_prompt = agents_config["planner"]["system_prompt"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Research query: {state.get('query', '')}\n\n"
            "Break this query into 3-5 focused sub-tasks. "
            "For each sub-task, provide:\n"
            "- A description of what to investigate\n"
            "- 3-5 search keywords suitable for querying arxiv\n\n"
            "Output format:\n"
            "TASK|description|keyword1, keyword2, keyword3"
        )),
    ]
    response = llm.invoke(messages)
    sub_tasks = _parse_sub_tasks(response.content)
    logs = [f"[Planner] Decomposed into {len(sub_tasks)} sub-tasks."]
    if sub_tasks:
        for st in sub_tasks:
            logs.append(f"  TASK: {st.description} ({', '.join(st.keywords)})")
    else:
        logs.append(
            f"  [WARN] No TASK| lines found in LLM response"
        )
    logs.append(
        f"  Raw LLM response:\n{response.content}"
    )
    warnings.extend(
        validate_contract({"sub_tasks": sub_tasks}, PlannerOutput)
    )
    logs.extend(warnings)
    return {
        "sub_tasks": sub_tasks,
        "messages": [response],
        "logs": logs,
    }


def _parse_sub_tasks(text: str) -> list[SubTask]:
    tasks = []
    for line in text.strip().split("\n"):
        cleaned = strip_line_noise(line)
        if not cleaned.startswith("TASK|"):
            continue
        parts = cleaned.split("|")
        if len(parts) >= 3:
            desc = "|".join(parts[1:-1]).strip()
            keywords = [k.strip() for k in parts[-1].split(",") if k.strip()]
            tasks.append(SubTask(description=desc, keywords=keywords))
    return tasks
