"""Classify query node — decides whether retrieval is needed."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import CLASSIFY_SYSTEM_PROMPT
from app.agent.state import AgentState
from app.services.llm import get_llm
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def classify_query(state: AgentState) -> dict:
    """Classify the user query and set ``need_retrieval``."""
    query = state["query"]
    logger.info("Classifying query: %s", query[:80])

    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    response = await llm.ainvoke(messages)
    classification = response.content.strip().lower()

    need_retrieval = "retrieval" in classification
    logger.info(
        "Classification: %s → need_retrieval=%s", classification, need_retrieval
    )

    return {"need_retrieval": need_retrieval}
