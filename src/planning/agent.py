from langchain_core.messages import HumanMessage, SystemMessage

from config import agents_config
from coordinator.state import ResearchState
from shared import llm as llm_module
from shared.models import SubTask


def planner_node(state: ResearchState, config) -> dict:
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
    return {
        "sub_tasks": sub_tasks,
        "messages": [response],
        "logs": [f"[Planner] Decomposed into {len(sub_tasks)} sub-tasks."],
    }


def _parse_sub_tasks(text: str) -> list[SubTask]:
    tasks = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line.startswith("TASK|"):
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            desc = parts[1].strip()
            keywords = [k.strip() for k in parts[2].split(",") if k.strip()]
            tasks.append(SubTask(description=desc, keywords=keywords))
    return tasks
