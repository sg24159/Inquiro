from langchain_openai import ChatOpenAI

from config.settings import get_settings


def get_llm(model: str | None = None, temperature: float = 0.0):
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=temperature,
        api_key=settings.api_key or "no-key-required",
    )
