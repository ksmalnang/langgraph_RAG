"""Classify query node — decides whether retrieval is needed."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.nodes.generate import _format_history
from app.agent.prompts import CLASSIFY_SYSTEM_PROMPT
from app.agent.state import ClassificationInput, ClassificationUpdate
from app.services.llm import get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def classify_query(state: ClassificationInput) -> ClassificationUpdate:
    """Read query/history and return only routing fields."""
    query = state["query"]
    history = state.get("chat_history", [])
    logger.info("Classifying query: %s", query[:80])

    history_str = _format_history(history)
    system_prompt = CLASSIFY_SYSTEM_PROMPT.format(chat_history=history_str)

    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    response = await llm.ainvoke(messages)
    content = response.content.strip()

    route = "fallback"
    reason = "Parsing failed"
    try:
        # Safely parse codeblocks if LLM wraps JSON response
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()

        data = json.loads(content)
        route = data.get("route", "fallback")
        reason = data.get("reason", "No reason provided")

        if route not in (
            "fallback",
            "retrieval_only",
            "student_only",
            "both",
            "nilai_semester",
        ):
            logger.warning("Invalid route returned by LLM: %s", route)
            route = "fallback"

    except Exception as e:
        logger.warning("JSON parse error for classification: %s", e)

    need_retrieval = route in ("retrieval_only", "both")
    logger.debug(
        "Classification result: route=%s, need_retrieval=%s, reason=%s",
        route,
        need_retrieval,
        reason,
    )

    return {"route": route, "need_retrieval": need_retrieval}
