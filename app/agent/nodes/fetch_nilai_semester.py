import logging
from typing import Any

import httpx
from langchain_core.messages import HumanMessage

from app.agent.nodes.fetch import _scrape_nilaimhs
from app.agent.state import AgentState
from app.config import get_settings
from app.services.llm import get_llm_cheap
from app.services.siakad_session import get_siakad_cookies

logger = logging.getLogger(__name__)


def _map_semester_to_periode(angkatan: int, semester_number: int) -> str:
    if not (1 <= semester_number <= 14):
        raise ValueError("semester_number must be between 1 and 14")
    if not (2000 <= angkatan <= 2100):
        raise ValueError("angkatan must be between 2000 and 2100")

    year = angkatan + (semester_number - 1) // 2
    term = 1 if semester_number % 2 != 0 else 2
    return f"{year}{term}"

SEMESTER_EXTRACTOR_PROMPT = """\
You are a precise assistant. Your only job is to extract the semester number from
the user's query.

User query: {query}

Rules:
- Reply with ONLY a single integer representing the semester number (e.g. "4")
- The integer must be between 1 and 14 inclusive
- If no specific semester number is mentioned or it is ambiguous, reply with exactly: NONE
- Do not explain, do not add punctuation, do not add any other text

Examples:
  "nilai semester 4 saya" → 4
  "IP saya semester 7 berapa?" → 7
  "transkrip semester ini" → NONE
  "nilai saya" → NONE\
"""

async def fetch_nilai_semester(state: AgentState) -> dict[str, Any]:
    student_data = state.get("student_data")
    if not student_data:
        logger.warning("student_data not available")
        return {"nilai_semester_detail": None}

    angkatan = student_data["mahasiswa"].get("angkatan")
    if not angkatan or not str(angkatan).isdigit():
        logger.warning("angkatan not available or invalid")
        return {"nilai_semester_detail": None}

    prompt = SEMESTER_EXTRACTOR_PROMPT.format(query=state["query"])
    llm = get_llm_cheap(temperature=0.0)
    response = await llm.ainvoke([HumanMessage(content=prompt)]) # Ensure it's called using ainvoke and awaited if the llm expects async invocation, assuming async invocation

    res_text = response.content.strip()

    if res_text == "NONE" or not res_text:
        return {"nilai_semester_detail": None}

    if res_text.isdigit() and 1 <= int(res_text) <= 14:
        semester_number = int(res_text)
    else:
        logger.warning(f"LLM returned unexpected value: {res_text!r}")
        return {"nilai_semester_detail": None}

    try:
        computed_periode = _map_semester_to_periode(int(angkatan), semester_number)
    except ValueError as e:
        logger.warning(f"ValueError computing periode: {e}")
        return {"nilai_semester_detail": None}

    periode_options = student_data["nilai_semester"]["periode_options"]
    available = [p["value"] for p in periode_options]
    if computed_periode not in available:
        logger.warning(f"Computed periode {computed_periode} not in available options: {available}")
        return {"nilai_semester_detail": None}

    logger.info(f"Fetching nilai for semester {semester_number} → periode {computed_periode}")

    try:
        session_id = state.get("session_id")
        cookies_dict = await get_siakad_cookies(session_id)
        if not cookies_dict:
            logger.warning("cookies not found")
            return {"nilai_semester_detail": None}

        settings = get_settings()
        async with httpx.AsyncClient(
            cookies=cookies_dict,
            timeout=settings.siakad_timeout_seconds,
        ) as client:
            result = await _scrape_nilaimhs(client, periode=computed_periode)
            nilai_semester_detail = result["nilai_semester"]
            logger.info(f"Successfully fetched nilai for periode {computed_periode}, total MK: {nilai_semester_detail['total_mata_kuliah']}")
            return {"nilai_semester_detail": nilai_semester_detail}

    except ConnectionError as e:
        logger.warning(
            "Dependency failure dependency=siakad operation=fetch_nilai_semester flow=student mode=auth session_id=%s detail=%s",
            state.get("session_id"),
            e,
        )
        return {"nilai_semester_detail": None}
    except httpx.TimeoutException as e:
        logger.warning(
            "Dependency failure dependency=siakad operation=fetch_nilai_semester flow=student mode=timeout session_id=%s detail=%s",
            state.get("session_id"),
            e,
        )
        return {"nilai_semester_detail": None}
    except httpx.TransportError as e:
        logger.warning(
            "Dependency failure dependency=siakad operation=fetch_nilai_semester flow=student mode=transport session_id=%s detail=%s",
            state.get("session_id"),
            e,
        )
        return {"nilai_semester_detail": None}
    except Exception as e:
        logger.error(
            "Dependency failure dependency=siakad operation=fetch_nilai_semester flow=student mode=unexpected session_id=%s detail=%s",
            state.get("session_id"),
            e,
            exc_info=True,
        )
        return {"nilai_semester_detail": None}
