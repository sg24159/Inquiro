from langchain_ollama import ChatOllama

from config.settings import settings


def get_llm(model: str | None = None, temperature: float = 0.0):
    return ChatOllama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )
