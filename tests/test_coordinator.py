def test_graph_builds():
    from coordinator.graph import build_research_graph

    graph = build_research_graph().compile()
    assert graph is not None
    nodes = [n for n in graph.get_graph().nodes]
    assert "planner" in nodes
    assert "retriever" in nodes
    assert "processor" in nodes
    assert "writer" in nodes


def test_graph_execution(mock_llm, mock_httpx, tmp_path, monkeypatch):
    """End-to-end: all nodes run with mocks."""
    monkeypatch.chdir(tmp_path)
    from coordinator.graph import build_research_graph
    from langgraph.checkpoint.memory import MemorySaver

    graph_compiled = build_research_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test"}}
    for event in graph_compiled.stream(
        {"query": "test", "messages": []}, config, stream_mode="updates"
    ):
        pass
    final = graph_compiled.get_state(config)
    assert isinstance(final.values.get("sub_tasks"), list)
    assert len(final.values["sub_tasks"]) >= 1
    assert isinstance(final.values.get("raw_results"), list)
    assert isinstance(final.values.get("processed_findings"), list)
    assert final.values.get("processed_findings") is not None
    assert final.values.get("report") is not None
    assert isinstance(final.values.get("logs"), list)


def test_graph_execution_full_chain(mock_llm_chain, mock_httpx, tmp_path, monkeypatch):
    """E2E: planner produces TASK, processor produces FINDING, report has findings."""
    monkeypatch.chdir(tmp_path)
    from coordinator.graph import build_research_graph
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_research_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "full-test"}}
    for event in graph.stream(
        {"query": "machine learning", "messages": []}, config, stream_mode="updates"
    ):
        pass
    final = graph.get_state(config)
    assert len(final.values["sub_tasks"]) >= 1
    assert len(final.values["raw_results"]) >= 1
    assert len(final.values["processed_findings"]) >= 1
    assert final.values["report"] is not None
    assert any("[WARN]" not in log for log in final.values["logs"])


def test_graph_execution_empty_sub_tasks(mock_httpx, tmp_path, monkeypatch):
    """E2E: when planner returns 0 sub-tasks, graph completes gracefully."""
    monkeypatch.chdir(tmp_path)
    from coordinator.graph import build_research_graph
    from langgraph.checkpoint.memory import MemorySaver
    from unittest.mock import MagicMock, patch
    from langchain_core.messages import AIMessage

    # Patch LLM to return empty (no TASK| lines)
    with patch("shared.llm.get_llm") as mock:
        llm_instance = MagicMock()
        llm_instance.invoke.return_value = AIMessage(content="No sub-tasks found.")
        mock.return_value = llm_instance
        graph = build_research_graph().compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "empty-test"}}
        for event in graph.stream(
            {"query": "test", "messages": []}, config, stream_mode="updates"
        ):
            pass
        final = graph.get_state(config)
        assert len(final.values["sub_tasks"]) == 0
        assert len(final.values["raw_results"]) == 0
        assert len(final.values["processed_findings"]) == 0
        assert final.values["report"] is not None
        assert any("[WARN]" in log for log in final.values["logs"])
