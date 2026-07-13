import json
from pathlib import Path

import httpx

from langchain_openai import ChatOpenAI

from config.settings import get_settings


def get_llm(model: str | None = None, temperature: float = 0.0):
    settings = get_settings()
    kwargs: dict = {}
    if settings.chat_template_kwargs:
        try:
            parsed = json.loads(settings.chat_template_kwargs)
            if isinstance(parsed, dict) and parsed:
                kwargs["extra_body"] = {"chat_template_kwargs": parsed}
        except json.JSONDecodeError:
            pass
    return ChatOpenAI(
        model=model or settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=temperature,
        api_key=settings.api_key or "no-key-required",
        **kwargs,
    )


def resolve_model_info(base_url: str, model_alias: str) -> str:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=2)
        resp.raise_for_status()
        for entry in resp.json().get("data", []):
            if entry.get("id") != model_alias:
                continue
            args = entry.get("status", {}).get("args", [])
            model_path = ""
            for i, arg in enumerate(args):
                if arg == "--model" and i + 1 < len(args):
                    model_path = args[i + 1]
            if not model_path:
                return model_alias
            label = Path(model_path).stem
            meta = entry.get("meta", {})
            details = ", ".join(
                str(meta[k]) for k in ("n_params", "ftype") if meta.get(k)
            )
            if details:
                label = f"{label} ({details})"
            return label
    except Exception:
        pass
    return str(model_alias)
