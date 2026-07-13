from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.constants import END
from langgraph.graph import StateGraph

from coordinator.state import ResearchState
from planning.agent import planner_node
from processing.agent import processor_node
from retrieval.agent import retriever_node
from writing.agent import writer_node


def build_research_graph() -> StateGraph:
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("processor", processor_node)
    builder.add_node("writer", writer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "processor")
    builder.add_edge("processor", "writer")
    builder.add_edge("writer", END)

    return builder


def _make_serde():
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("shared.models", "SubTask"),
            ("shared.models", "RawResult"),
            ("shared.models", "ProcessedFinding"),
            ("shared.models", "ReportAssets"),
        ]
    )


def compile_graph(checkpointer=None):
    builder = build_research_graph()
    if checkpointer is None:
        checkpointer = MemorySaver(serde=_make_serde())
    return builder.compile(checkpointer=checkpointer)
