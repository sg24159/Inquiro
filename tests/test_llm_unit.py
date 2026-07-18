from unittest.mock import MagicMock, patch

from config.settings import get_settings


@patch("shared.llm.ChatOpenAI")
@patch("shared.llm.get_settings")
def test_get_llm_passes_settings(mock_get_settings, mock_chat):
    settings = get_settings()
    settings.chat_template_kwargs = ""
    mock_get_settings.return_value = settings
    mock_chat.return_value = MagicMock()

    from shared.llm import get_llm
    get_llm()

    mock_chat.assert_called_once_with(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=0.0,
        api_key=settings.api_key or "no-key-required",
    )


@patch("shared.llm.ChatOpenAI")
@patch("shared.llm.get_settings")
def test_get_llm_custom_model_and_temperature(mock_get_settings, mock_chat):
    settings = get_settings()
    settings.chat_template_kwargs = ""
    mock_get_settings.return_value = settings
    mock_chat.return_value = MagicMock()

    from shared.llm import get_llm
    get_llm(model="gpt-4", temperature=0.7)

    mock_chat.assert_called_once_with(
        model="gpt-4",
        base_url=settings.llm_base_url,
        temperature=0.7,
        api_key=settings.api_key or "no-key-required",
    )
