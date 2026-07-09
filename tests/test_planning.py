from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

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


def test_parse_sub_tasks_with_markdown_noise():
    """Should strip markdown list prefixes before parsing."""
    text = (
        "- TASK|Investigate AI|AI, regulation\n"
        "* TASK|Explore ethics|ethics, bias\n"
        "1. TASK|Test topic|a, b"
    )
    result = _parse_sub_tasks(text)
    assert len(result) == 3
    assert result[0].description == "Investigate AI"
    assert result[1].description == "Explore ethics"
    assert result[2].description == "Test topic"


def test_planner_node_malformed_llm():
    """When LLM returns no TASK| lines, planner returns empty + warning."""
    from planning.agent import planner_node
    from coordinator.state import ResearchState

    with patch("shared.llm.get_llm") as mock:
        llm_instance = MagicMock()
        llm_instance.invoke.return_value = AIMessage(
            content="I did not format this as requested."
        )
        mock.return_value = llm_instance
        state = ResearchState(query="test", messages=[])
        result = planner_node(state, {"configurable": {"thread_id": "t"}})
        assert len(result["sub_tasks"]) == 0
        assert any("[WARN]" in log for log in result["logs"])


def test_parse_sub_tasks_comma_in_description():
    """Description with commas must not consume the keyword segment.

    LLMs naturally write prose descriptions like
    'Analyze social media, focusing on depression'.
    The keyword segment is the LAST pipe-delimited fragment.
    """
    text = (
        "TASK|Analyze social media, focusing on depression|"
        "social media, depression, teenagers"
    )
    result = _parse_sub_tasks(text)
    assert len(result) == 1
    assert result[0].description == "Analyze social media, focusing on depression"
    assert result[0].keywords == ["social media", "depression", "teenagers"]
