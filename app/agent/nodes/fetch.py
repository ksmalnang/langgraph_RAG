from __future__ import annotations

import asyncio
import logging
from typing import Any

from bs4 import BeautifulSoup
import httpx

from app.agent.state import FetchStudentInput, FetchStudentUpdate
from app.services.siakad_session import (
    cache_student_data,
    get_cached_student_data,
    get_siakad_cookies,
)

logger = logging.getLogger(__name__)

# ============================================================
# URLs
# ============================================================
BASE_URL = "https://situ2.unpas.ac.id"
TRANSKRIP_URL = f"{BASE_URL}/siakad/list_transkrip"
NILAI_MHS_URL = f"{BASE_URL}/siakad/list_nilaimhs"
JADWAL_URL = f"{BASE_URL}/siakad/list_jadwalkuliahsmt"
BERITA_URL = f"{BASE_URL}/siakad/list_berita"
BERITA_DETAIL_URL = f"{BASE_URL}/siakad/data_berita/detail"

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ============================================================
# Helpers
# ============================================================


def _check_redirect(resp: httpx.Response) -> None:
    """Raise kalau session expired (redirect ke login page)."""
    if "gate/login" in str(resp.url):
        raise ConnectionError("Session expired, redirect ke login page.")


def _extract_periode_options(soup: BeautifulSoup) -> tuple[list[dict], str | None]:
    """
    Extract opsi periode dari dropdown #periode.
    Return (periode_options, selected_value).
    """
    options: list[dict] = []
    selected: str | None = None

    select_el = soup.find("select", id="periode")
    if select_el:
        for opt in select_el.find_all("option"):
            is_selected = opt.has_attr("selected")
            options.append(
                {
                    "value": opt["value"],
                    "label": opt.text.strip(),
                    "selected": is_selected,
                }
            )
            if is_selected:
                selected = opt["value"]

    return options, selected


# ============================================================
# Transkrip
# ============================================================


async def _scrape_transkrip(client: httpx.AsyncClient) -> dict[str, Any]:
    """Scrape halaman transkrip → return IPK + daftar MK."""
    resp = await client.get(
        TRANSKRIP_URL,
        headers={**HEADERS_BASE, "Referer": NILAI_MHS_URL},
    )
    _check_redirect(resp)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # IPK
    ipk_value = None
    label_th = soup.find("th", string=lambda t: t and "Indeks Prestasi Kumulatif" in t)
    if label_th:
        ipk_th = label_th.find_next_sibling("th")
        if ipk_th:
            try:
                ipk_value = float(ipk_th.text.strip().replace(",", "."))
            except ValueError:
                ipk_value = ipk_th.text.strip()

    # Tabel transkrip
    transkrip = []
    tabel = soup.find("table", class_="dataTable")
    if tabel:
        tbody = tabel.find("tbody")
        if tbody:
            for baris in tbody.find_all("tr"):
                kolom = baris.find_all("td")
                if len(kolom) == 8:
                    transkrip.append(
                        {
                            "no": kolom[0].text.strip(),
                            "kode": kolom[1].text.strip(),
                            "nama_mata_kuliah": kolom[2].text.strip(),
                            "semester": kolom[3].text.strip(),
                            "sks": kolom[4].text.strip(),
                            "grade": kolom[5].text.strip(),
                            "nilai_mutu": kolom[6].text.strip(),
                            "bobot": kolom[7].text.strip(),
                        }
                    )

    return {
        "ipk": ipk_value,
        "total_mata_kuliah": len(transkrip),
        "transkrip": transkrip,
    }


# ============================================================
# Nilai MHS — data mahasiswa + nilai semester digabung
# karena keduanya ada di halaman yang sama (list_nilaimhs)
# ============================================================


async def _scrape_nilaimhs(
    client: httpx.AsyncClient,
    periode: str | None = None,
) -> dict[str, Any]:
    """
    Scrape halaman list_nilaimhs dalam satu request.
    Return data mahasiswa + nilai semester sekaligus.

    Args:
        periode: e.g. "20242" (2024 Genap). None = pakai selected default.
    """
    # GET pertama — ambil halaman default + periode options
    resp = await client.get(
        NILAI_MHS_URL,
        headers={**HEADERS_BASE, "Referer": "https://situ2.unpas.ac.id/"},
    )
    _check_redirect(resp)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    periode_options, selected_periode = _extract_periode_options(soup)
    target_periode = periode or selected_periode

    # POST kalau periode yang diminta beda dari selected
    if periode and periode != selected_periode:
        resp = await client.post(
            NILAI_MHS_URL,
            data={"periode": target_periode},
            headers={
                **HEADERS_BASE,
                "Referer": NILAI_MHS_URL,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        _check_redirect(resp)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

    # --- Data Mahasiswa ---
    # Struktur: div.callout-info > div.row > label + div.col-md-3 (pasangan)
    mahasiswa: dict[str, Any] = {}
    callout = soup.find("div", class_="callout-info")
    if callout:
        label_map = {
            "NIM": "nim",
            "Tahun Kurikulum": "tahun_kurikulum",
            "Nama Mahasiswa": "nama",
            "Semester": "semester",
            "Program Studi": "program_studi",
            "Pembimbing Akademik": "pembimbing_akademik",
            "Status Mahasiswa": "status",
            "SKS Lulus / IPK Lulus": "sks_lulus_ipk_lulus",
            "Angkatan": "angkatan",
            "Total SKS / IPK": "total_sks_ipk",
        }
        for row in callout.find_all("div", class_="row"):
            labels = row.find_all("label")
            values = row.find_all("div", class_=lambda c: c and "col-md-3" in c)
            for label_el, value_el in zip(labels, values, strict=False):
                key = label_map.get(label_el.text.strip())
                if key:
                    mahasiswa[key] = value_el.text.strip()

    # --- Nilai per Semester ---
    # Kolom: 0=Kurikulum, 1=Kode MK, 2=Nama MK, 3=Nama Kelas,
    #        4=nested table komponen nilai, 7=Nilai Akhir
    nilai_list = []
    tabel = soup.find("table", class_="dataTable")
    if tabel:
        tbody = tabel.find("tbody")
        if tbody:
            for baris in tbody.find_all("tr", recursive=False):
                # Skip separator row (background-color: #34495E)
                if baris.get("style") and "background-color" in baris["style"]:
                    continue

                kolom = baris.find_all("td", recursive=False)
                if len(kolom) < 5:
                    continue

                # Nilai akhir di kolom terakhir, bisa float atau string (e.g. "T")
                nilai_akhir_raw = kolom[-1].text.strip()
                try:
                    nilai_akhir: Any = float(nilai_akhir_raw.replace(",", "."))
                except ValueError:
                    nilai_akhir = nilai_akhir_raw

                # Komponen nilai dari nested table di kolom ke-4
                komponen: list[dict] = []
                nested_table = kolom[4].find("table")
                if nested_table:
                    for nested_row in nested_table.find_all("tr"):
                        nested_cols = nested_row.find_all("td")
                        if len(nested_cols) == 3:
                            komponen.append(
                                {
                                    "komponen": nested_cols[0].text.strip(),
                                    "bobot_persen": nested_cols[1].text.strip(),
                                    "nilai": nested_cols[2].text.strip(),
                                }
                            )

                nilai_list.append(
                    {
                        "kurikulum": kolom[0].text.strip(),
                        "kode": kolom[1].text.strip(),
                        "nama_mata_kuliah": kolom[2].text.strip(),
                        "nama_kelas": kolom[3].text.strip(),
                        "komponen_nilai": komponen,
                        "nilai_akhir": nilai_akhir,
                    }
                )

    return {
        "mahasiswa": mahasiswa,
        "nilai_semester": {
            "periode_dipilih": target_periode,
            "periode_options": periode_options,
            "total_mata_kuliah": len(nilai_list),
            "nilai": nilai_list,
        },
    }


# ============================================================
# Jadwal Kuliah
# ============================================================


async def _scrape_jadwal_kuliah(
    client: httpx.AsyncClient,
    periode: str | None = None,
    tglkuliah: str = "",
) -> dict[str, Any]:
    """
    Scrape jadwal kuliah per semester.

    Args:
        periode:   e.g. "20242" (2024 Genap). None = pakai selected default.
        tglkuliah: Filter tanggal spesifik, format "YYYY-MM-DD". Kosong = semua.
    """
    # GET pertama — ambil halaman default + periode options
    resp = await client.get(
        JADWAL_URL,
        headers={**HEADERS_BASE, "Referer": "https://situ2.unpas.ac.id/"},
    )
    _check_redirect(resp)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    periode_options, selected_periode = _extract_periode_options(soup)
    target_periode = periode or selected_periode

    # POST untuk load data (jadwal selalu butuh POST untuk render tabel)
    if target_periode:
        resp = await client.post(
            JADWAL_URL,
            data={"periode": target_periode, "tglkuliah": tglkuliah},
            headers={
                **HEADERS_BASE,
                "Referer": JADWAL_URL,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        _check_redirect(resp)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

    # Kolom: 0=No, 1=Hari, 2=Tanggal, 3=Mulai, 4=Selesai,
    #        5=Jenis, 6=Kelas MK, 7=Materi, 8=Ruang
    jadwal_list = []
    tabel = soup.find("table", class_="dataTable")
    if tabel:
        tbody = tabel.find("tbody")
        if tbody:
            for baris in tbody.find_all("tr"):
                kolom = baris.find_all("td")
                if len(kolom) < 9:
                    continue

                jadwal_list.append(
                    {
                        "no": kolom[0].text.strip(),
                        "hari": kolom[1].text.strip(),
                        "tanggal": kolom[2].text.strip(),
                        "mulai": kolom[3].text.strip(),
                        "selesai": kolom[4].text.strip(),
                        "jenis": kolom[5].text.strip(),
                        "kelas_mata_kuliah": kolom[6].text.strip(),
                        "materi": " ".join(
                            kolom[7].text.split()
                        ),  # normalize whitespace
                        "ruang": kolom[8].text.strip() or None,
                    }
                )

    return {
        "periode_dipilih": target_periode,
        "periode_options": periode_options,
        "total_jadwal": len(jadwal_list),
        "jadwal": jadwal_list,
    }


# ============================================================
# Berita / Pengumuman
# ============================================================


async def _scrape_detail_berita(
    client: httpx.AsyncClient,
    berita_id: str,
) -> dict[str, Any]:
    """
    Fetch detail satu berita via /siakad/data_berita/detail/{id}.

    Field yang di-scrape:
        - judulberita  → div#block-judulberita > div.input-judulberita
        - isiberita    → div#block-isiberita > div.input-isiberita
        - fileberita   → div#block-fileberita > a[data-type="download"] (nama file)
    """
    resp = await client.get(
        f"{BERITA_DETAIL_URL}/{berita_id}",
        headers={**HEADERS_BASE, "Referer": BERITA_URL},
    )
    _check_redirect(resp)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    def _get_field(block_id: str) -> str | None:
        block = soup.find("div", id=f"block-{block_id}")
        if block:
            content = block.find("div", class_=f"input-{block_id}")
            if content:
                return content.get_text(separator="\n").strip()
        return None

    # Nama file lampiran
    file_name = None
    block_file = soup.find("div", id="block-fileberita")
    if block_file:
        link_el = block_file.find("a", attrs={"data-type": "download"})
        if link_el:
            file_name = link_el.text.strip()

    return {
        "id": berita_id,
        "judul": _get_field("judulberita"),
        "isi": _get_field("isiberita"),
        "file": file_name,
    }


async def _scrape_berita(
    client: httpx.AsyncClient,
    fetch_detail: bool = False,
) -> dict[str, Any]:
    """
    Scrape list berita/pengumuman dari list_berita.

    Kolom: 0=Tanggal, 1=Penulis, 2=Judul, 3=Aksi (button dengan data-id)

    Args:
        fetch_detail: Kalau True, fetch konten detail tiap berita.
                      Default False biar ga spam request.
    """
    resp = await client.get(
        BERITA_URL,
        headers={**HEADERS_BASE, "Referer": "https://situ2.unpas.ac.id/"},
    )
    _check_redirect(resp)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    berita_list = []
    tabel = soup.find("table", class_="dataTable")
    if tabel:
        tbody = tabel.find("tbody")
        if tbody:
            for baris in tbody.find_all("tr"):
                kolom = baris.find_all("td")
                if len(kolom) < 4:
                    continue

                # ID berita dari data-id button detail
                btn = baris.find("button", attrs={"data-id": True})
                berita_id = btn["data-id"] if btn else None

                entry: dict[str, Any] = {
                    "id": berita_id,
                    "tanggal": kolom[0].text.strip(),
                    "penulis": kolom[1].text.strip(),
                    "judul": kolom[2].text.strip(),
                    "detail": None,
                }

                if fetch_detail and berita_id:
                    try:
                        entry["detail"] = await _scrape_detail_berita(client, berita_id)
                    except Exception as e:
                        logger.warning(
                            "[berita] Gagal fetch detail id=%s: %s", berita_id, e
                        )

                berita_list.append(entry)

    return {
        "total": len(berita_list),
        "berita": berita_list,
    }


# ============================================================
# LangGraph Node
# ============================================================


async def fetch_student_data(state: FetchStudentInput) -> FetchStudentUpdate:
    """
    Node LangGraph: scrape semua data akademik mahasiswa dari SIAKAD.

    Expect di state:
        - session_id : untuk lookup cookie di Redis

    Return ke state:
        - student_data        : hasil scraping kalau success
        - student_fetch_error : True kalau gagal
        - need_retrieval      : di-preserve dari classifier

    Struktur student_data:
        {
            "mahasiswa":      { nim, nama, program_studi, ... },
            "nilai_semester": { periode_dipilih, periode_options, nilai: [...] },
            "transkrip":      { ipk, total_mata_kuliah, transkrip: [...] },
            "jadwal":         { periode_dipilih, periode_options, jadwal: [...] },
            "berita":         { total, berita: [...] },
        }
    """
    session_id: str | None = state.get("session_id")
    if not session_id:
        logger.warning("[fetch] session_id tidak ada di state.")
        return {
            "student_data": None,
            "student_fetch_error": True,
        }

    # 1. Cek cache Redis
    cached_data = await get_cached_student_data(session_id)
    if cached_data:
        logger.debug("[fetch] Cache hit untuk session_id=%s", session_id)
        return {
            "student_data": cached_data,
            "student_fetch_error": False,
        }

    # 2. Load cookies dari Redis
    cookies_dict = await get_siakad_cookies(session_id)
    if not cookies_dict:
        logger.warning("[fetch] Cookie tidak ditemukan untuk session_id=%s", session_id)
        return {
            "student_data": None,
            "student_fetch_error": True,
        }

    cookies = httpx.Cookies(cookies_dict)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=30.0, cookies=cookies
        ) as client:
            # Jalankan semua scraper secara concurrent
            # _scrape_nilaimhs di-await terpisah karena GET dulu baru POST
            nilaimhs_result, transkrip_result, jadwal_result, berita_result = (
                await asyncio.gather(
                    _scrape_nilaimhs(client),
                    _scrape_transkrip(client),
                    _scrape_jadwal_kuliah(client),
                    _scrape_berita(client, fetch_detail=False),
                )
            )

        student_data: dict[str, Any] = {
            "mahasiswa": nilaimhs_result["mahasiswa"],
            "nilai_semester": nilaimhs_result["nilai_semester"],
            "transkrip": transkrip_result,
            "jadwal": jadwal_result,
            "berita": berita_result,
        }

        logger.info(
            "[fetch] Berhasil fetch data SIAKAD, session_id=%s, IPK=%s",
            session_id,
            student_data["transkrip"].get("ipk"),
        )

        # 3. Cache ke Redis
        await cache_student_data(session_id, student_data)

        return {
            "student_data": student_data,
            "student_fetch_error": False,
        }

    except ConnectionError as e:
        logger.warning("[fetch] Session expired: %s", e)
        return {
            "student_data": None,
            "student_fetch_error": True,
        }
    except Exception as e:
        logger.error("[fetch] Error tidak terduga: %s", e)
        return {
            "student_data": None,
            "student_fetch_error": True,
        }
