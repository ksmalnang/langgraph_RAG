"""SIAKAD Session Management with Redis."""

import json
import logging

from bs4 import BeautifulSoup
import httpx

from app.services.memory import get_redis

logger = logging.getLogger(__name__)

# ============================================================
# URLs
# ============================================================
LOGIN_URL = "https://situ2.unpas.ac.id/gate/login"
SIAKAD_LOGIN_URL = "https://situ2.unpas.ac.id/siakad/login"

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


async def _login(client: httpx.AsyncClient, email: str, password: str) -> None:
    """Fetch CSRF token lalu POST credentials."""
    resp = await client.get(LOGIN_URL, headers=HEADERS_BASE)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    try:
        token_val = soup.find("input", {"name": "__token"})["value"]
        client_id_val = soup.find("input", {"name": "client_id"})["value"]
        redirect_uri_val = soup.find("input", {"name": "redirect_uri"})["value"]
    except (TypeError, KeyError):
        raise ValueError("Gagal ekstrak CSRF token dari halaman login.")

    payload = {
        "email": email,
        "password": password,
        "__token": token_val,
        "_token": "",
        "client_id": client_id_val,
        "redirect_uri": redirect_uri_val,
    }

    resp_post = await client.post(
        LOGIN_URL,
        data=payload,
        headers={**HEADERS_BASE, "Content-Type": "application/x-www-form-urlencoded"},
    )

    if "Email atau Password salah" in resp_post.text or str(resp_post.url) == LOGIN_URL:
        raise ValueError("Kredensial tidak valid.")


async def _activate_siakad(client: httpx.AsyncClient) -> None:
    """Aktivasi modul SIAKAD setelah login."""
    payload = {
        "oldpass": "",
        "newpass": "",
        "renewpass": "",
        "act": "",
        "sessdata": "",
        "kodemodul": "siakad",
        "koderole": "mhs",
        "kodeunit": "55201",
    }
    resp = await client.post(
        SIAKAD_LOGIN_URL,
        data=payload,
        headers={
            **HEADERS_BASE,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://situ2.unpas.ac.id/gate/menu",
        },
    )
    resp.raise_for_status()


async def init_siakad_session(session_id: str, email: str, password: str) -> bool:
    """Login ke SIAKAD dan simpan cookies ke Redis dengan TTL 1 jam."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            await _login(client, email, password)
            await _activate_siakad(client)

            # Ambil cookies dan simpan ke Redis
            cookies_dict = dict(client.cookies)
            cookie_str = json.dumps(cookies_dict)

            r = await get_redis()
            redis_key = f"siakad_session:{session_id}"
            await r.setex(redis_key, 3600, cookie_str)
            logger.info("Session %s berhasil disimpan di Redis.", session_id)
            return True
    except ValueError as e:
        logger.warning("Kredensial tidak valid untuk %s: %s", session_id, e)
        return False
    except Exception as e:
        logger.error("Gagal inisiasi SIAKAD session untuk %s: %s", session_id, e)
        return False


async def get_siakad_cookies(session_id: str) -> dict | None:
    """Ambil cookies dari Redis berdasarkan session_id."""
    try:
        r = await get_redis()
        redis_key = f"siakad_session:{session_id}"
        cookie_str = await r.get(redis_key)
        if cookie_str:
            return json.loads(cookie_str)
        return None
    except Exception as e:
        logger.error(
            "Redis unavailable saat mengambil cookies untuk %s: %s", session_id, e
        )
        return None


async def cache_student_data(session_id: str, student_data: dict) -> bool:
    """Simpan student_data sebagai JSON string ke Redis dengan TTL 1 jam."""
    try:
        r = await get_redis()
        redis_key = f"student_data:{session_id}"
        await r.setex(redis_key, 3600, json.dumps(student_data))
        logger.debug("Berhasil cache student_data untuk session_id=%s", session_id)
        return True
    except Exception as e:
        logger.warning(
            "Redis unavailable saat set cache student_data untuk %s: %s", session_id, e
        )
        return False


async def get_cached_student_data(session_id: str) -> dict | None:
    """Ambil cached student_data dari Redis."""
    try:
        r = await get_redis()
        redis_key = f"student_data:{session_id}"
        cached_str = await r.get(redis_key)
        if cached_str:
            return json.loads(cached_str)
        return None
    except Exception as e:
        logger.warning(
            "Redis unavailable saat get cache student_data untuk %s: %s", session_id, e
        )
        return None
