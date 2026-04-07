"""Generate answer nodes — with and without document context."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import (
    FALLBACK_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    STUDENT_CONTEXT_TEMPLATE,
)
from app.agent.state import AgentState
from app.services.llm import get_llm, get_llm_cheap
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _format_history(chat_history: list[dict]) -> str:
    """Format chat history into a readable string."""
    if not chat_history:
        return ""

    lines = []
    # Keep last 10 turns for context window
    for turn in chat_history[-10:]:
        role = turn.get("role")
        content = turn.get("content")

        # Skip entry if it's missing role or content
        if not role or not content:
            continue

        role_title = role.capitalize()
        lines.append(f"{role_title}: {content}")

    return "\n".join(lines)


def _format_context(documents: list[dict]) -> str:
    """Format reranked documents into a context block."""
    if not documents:
        return "(no documents)"
    parts = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        score = doc.get("relevance_score", 0.0)
        headings = doc.get("headings", [])
        heading_str = " > ".join(headings) if headings else "N/A"
        parts.append(f"[{i}] (relevance: {score:.2f}, section: {heading_str})\n{text}")
    return "\n\n".join(parts)


def _format_jadwal_summary(jadwal: list[dict]) -> str:
    """Format jadwal list into a readable summary."""
    if not jadwal:
        return "-"
    lines = []
    for j in jadwal:
        lines.append(
            f"- {j.get('hari')} {j.get('tanggal')} | "
            f"{j.get('mulai')}-{j.get('selesai')} | "
            f"{j.get('kelas_mata_kuliah')} | "
            f"Ruang: {j.get('ruang') or 'TBD'}"
        )
    return "\n".join(lines)


def _format_nilai_summary(nilai: list[dict]) -> str:
    """Format nilai list into a readable summary."""
    if not nilai:
        return "-"
    lines = []
    for n in nilai:
        lines.append(
            f"- [{n.get('kode')}] {n.get('nama_mata_kuliah')} "
            f"(Kelas {n.get('nama_kelas')}) → {n.get('nilai_akhir')}"
        )
    return "\n".join(lines)


def _format_berita_summary(berita: list[dict]) -> str:
    """Format berita list into a readable summary (max 3 items)."""
    if not berita:
        return "-"
    lines = []
    for b in berita[:3]:
        lines.append(f"- [{b.get('tanggal')}] {b.get('judul')}")
    return "\n".join(lines)


def _format_nilai_semester_detail(nilai_semester_detail: dict) -> str:
    if not nilai_semester_detail:
        return ""

    periode = nilai_semester_detail.get("periode_dipilih", "-")
    nilai_list = nilai_semester_detail.get("nilai", [])

    if not nilai_list:
        return f"=== Nilai Semester {periode} ===\n(tidak ada data nilai)"

    lines = [f"=== Nilai Semester {periode} ==="]
    for n in nilai_list:
        line = (
            f"- [{n.get('kode')}] {n.get('nama_mata_kuliah')} "
            f"(Kelas {n.get('nama_kelas')}) → Nilai Akhir: {n.get('nilai_akhir')}"
        )
        komponen = n.get("komponen_nilai", [])
        if komponen:
            komponen_str = ", ".join(
                f"{k['komponen']} {k['bobot_persen']}%={k['nilai']}"
                for k in komponen
            )
            line += f"\n  Komponen: {komponen_str}"
        lines.append(line)

    return "\n".join(lines)


def _format_student_context(student_data: dict) -> str:
    """Format nested student_data into a readable string block."""
    if not student_data:
        return ""

    mhs = student_data.get("mahasiswa", {})
    transkrip_data = student_data.get("transkrip", {})
    nilai_data = student_data.get("nilai_semester", {})
    jadwal_data = student_data.get("jadwal", {})
    berita_data = student_data.get("berita", {})

    # Basic info
    nama = mhs.get("nama", "-")
    nim = mhs.get("nim", "-")
    prodi = mhs.get("program_studi", "-")
    semester = mhs.get("semester", "-")
    angkatan = mhs.get("angkatan", "-")
    status = mhs.get("status", "-")
    pembimbing = mhs.get("pembimbing_akademik", "-")

    # Parse "138 / 3.66" → total_sks="138", ipk="3.66"
    sks_ipk_raw = mhs.get("total_sks_ipk", "-/-").split("/")
    total_sks = sks_ipk_raw[0].strip() if len(sks_ipk_raw) > 0 else "-"
    ipk = sks_ipk_raw[1].strip() if len(sks_ipk_raw) > 1 else "-"

    sks_lulus_raw = mhs.get("sks_lulus_ipk_lulus", "-/-").split("/")
    sks_lulus = sks_lulus_raw[0].strip() if len(sks_lulus_raw) > 0 else "-"

    # Summaries
    periode_aktif = nilai_data.get("periode_dipilih", "-")
    nilai_list = nilai_data.get("nilai", [])
    transkrip_list = transkrip_data.get("transkrip", [])
    jadwal_list = jadwal_data.get("jadwal", [])
    berita_list = berita_data.get("berita", [])

    return STUDENT_CONTEXT_TEMPLATE.format(
        nama=nama,
        nim=nim,
        prodi=prodi,
        semester=semester,
        angkatan=angkatan,
        status=status,
        pembimbing=pembimbing,
        total_sks=total_sks,
        sks_lulus=sks_lulus,
        ipk=ipk,
        periode_aktif=periode_aktif,
        total_mk_semester=len(nilai_list),
        nilai_summary=_format_nilai_summary(nilai_list),
        total_mk_transkrip=len(transkrip_list),
        total_jadwal=len(jadwal_list),
        jadwal_summary=_format_jadwal_summary(jadwal_list),
        berita_summary=_format_berita_summary(berita_list),
    )


async def generate_answer(state: AgentState) -> dict:
    """Generate an answer with document context (RAG path)."""
    query = state["query"]
    history = state.get("chat_history", [])
    documents = state.get("reranked_documents", [])

    context_str = _format_context(documents)

    student_data = state.get("student_data")
    if student_data is not None:
        student_str = _format_student_context(student_data)
        if student_str:
            context_str = f"{student_str}\n\n=== Document Context ===\n{context_str}"

    nilai_detail = state.get("nilai_semester_detail")
    if nilai_detail is not None:
        nilai_detail_str = _format_nilai_semester_detail(nilai_detail)
        if nilai_detail_str:
            context_str = f"{nilai_detail_str}\n\n{context_str}"

    history_str = _format_history(history)

    system_prompt = RAG_SYSTEM_PROMPT.format(
        context=context_str, chat_history=history_str
    )

    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    logger.info("Generating RAG answer for: %s", query[:80])
    response = await llm.ainvoke(messages)
    answer = response.content.strip()

    logger.debug("Generated answer length: %d chars", len(answer))
    return {"answer": answer}


async def generate_answer_fallback(state: AgentState) -> dict:
    """Generate an answer without document context (fallback path)."""
    query = state["query"]
    history = state.get("chat_history", [])

    history_str = _format_history(history)
    system_prompt = FALLBACK_SYSTEM_PROMPT.format(chat_history=history_str)

    llm = get_llm(temperature=0.5)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    logger.info("Generating fallback answer for: %s", query[:80])
    response = await llm.ainvoke(messages)
    answer = response.content.strip()

    return {"answer": answer, "sources": []}


async def rewrite_question(state: AgentState) -> dict:
    """Rewrite the query for better retrieval results."""
    query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)

    llm = get_llm_cheap(temperature=0.1)
    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    logger.info("Rewriting query (attempt %d): %s", rewrite_count + 1, query[:80])
    response = await llm.ainvoke(messages)
    rewritten = response.content.strip()

    logger.info("Rewritten query: %s", rewritten[:80])
    return {"query": rewritten, "rewrite_count": rewrite_count + 1}
