"""OpenRouter LLM client via LangChain ChatOpenAI."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at OpenRouter."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature,
        max_tokens=2048,
    )


def get_llm_cheap(temperature: float = 0.1) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at OpenRouter."""
    settings = get_settings()
    return ChatOpenAI(
        model="stepfun/step-3.5-flash:free",
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature,
        max_tokens=2048,
        extra_body={"provider": {"only": ["stepfun/fp8"]}},
    )
