from planning.agent import _parse_sub_tasks


def test_parse_sub_tasks():
    text = (
        "TASK|Investigate AI regulation|AI, regulation, policy\n"
        "TASK|Compare global approaches|comparison, governance, international\n"
    )
    result = _parse_sub_tasks(text)
    assert len(result) == 2
    assert result[0].description == "Investigate AI regulation"
    assert result[0].keywords == ["AI", "regulation", "policy"]


def test_parse_sub_tasks_empty():
    assert _parse_sub_tasks("") == []


def test_parse_sub_tasks_malformed():
    text = "TASK|only description"
    result = _parse_sub_tasks(text)
    assert len(result) == 0


def test_planner_node(mock_llm, graph):
    """Integration check: planner node emits sub_tasks."""
    from planning.agent import planner_node
    from coordinator.state import ResearchState

    state = ResearchState(query="test", messages=[])
    result = planner_node(state, {"configurable": {"thread_id": "t"}})
    assert "sub_tasks" in result
    assert len(result["sub_tasks"]) >= 1
