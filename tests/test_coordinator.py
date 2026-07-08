def test_graph_builds():
    from coordinator.graph import build_research_graph

    graph = build_research_graph().compile()
    assert graph is not None
    nodes = [n for n in graph.get_graph().nodes]
    assert "planner" in nodes
    assert "retriever" in nodes
    assert "processor" in nodes
    assert "writer" in nodes


def test_graph_execution(mock_llm, mock_httpx, graph):
    """End-to-end: all nodes run with mocks."""
    from coordinator.graph import build_research_graph
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_research_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test"}}
    for event in graph.stream(
        {"query": "test", "messages": []}, config, stream_mode="updates"
    ):
        pass
    final = graph.get_state(config)
    assert isinstance(final.values.get("sub_tasks"), list)
    assert len(final.values["sub_tasks"]) >= 1
    assert isinstance(final.values.get("raw_results"), list)
    assert isinstance(final.values.get("processed_findings"), list)
    assert final.values.get("processed_findings") is not None
    assert final.values.get("report") is not None
    assert isinstance(final.values.get("logs"), list)
