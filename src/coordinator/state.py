import operator
from typing import Annotated

from langgraph.graph import MessagesState

from shared.models import ProcessedFinding, RawResult, ReportAssets, SubTask


class ResearchState(MessagesState):
    query: str
    sub_tasks: list[SubTask] = []
    raw_results: list[RawResult] = []
    processed_findings: list[ProcessedFinding] = []
    report: ReportAssets | None = None
    iteration_count: int = 0
    synthesized_answer: str = ""
    resolved_model: str = ""
    pipeline_start_time: float = 0.0
    planner_stats: dict = {}
    retriever_stats: dict = {}
    processor_stats: dict = {}
    writer_stats: dict = {}
    logs: Annotated[list[str], operator.add] = []
