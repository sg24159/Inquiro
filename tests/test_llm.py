from unittest.mock import MagicMock, patch

from config.settings import settings
from shared.llm import get_llm


@patch("shared.llm.ChatOpenAI")
def test_get_llm_passes_settings(mock_chat):
    mock_chat.return_value = MagicMock()
    get_llm()
    mock_chat.assert_called_once_with(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=0.0,
        api_key=settings.api_key or "no-key-required",
    )


@patch("shared.llm.ChatOpenAI")
def test_get_llm_custom_model_and_temperature(mock_chat):
    mock_chat.return_value = MagicMock()
    get_llm(model="gpt-4", temperature=0.7)
    mock_chat.assert_called_once_with(
        model="gpt-4",
        base_url=settings.llm_base_url,
        temperature=0.7,
        api_key=settings.api_key or "no-key-required",
    )
