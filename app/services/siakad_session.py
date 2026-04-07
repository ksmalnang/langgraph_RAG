"""SIAKAD Session Management with Redis."""

import json
import logging

from bs4 import BeautifulSoup
import httpx

from app.config import get_settings
from app.services.resilience import retry_async
from app.services.memory import get_redis
from app.utils.exceptions import MemoryStoreError, SiakadAuthError

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
    try:
        resp = await retry_async(
            operation="login_page_get",
            dependency="siakad",
            fn=lambda: client.get(LOGIN_URL, headers=HEADERS_BASE),
            retry_on=(httpx.TimeoutException, httpx.TransportError),
        )
        resp.raise_for_status()
    except httpx.TimeoutException as exc:
        raise SiakadAuthError("SIAKAD login page request timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise SiakadAuthError(
            f"SIAKAD login page returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.TransportError as exc:
        raise SiakadAuthError("SIAKAD login page request failed due to network error") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    try:
        token_val = soup.find("input", {"name": "__token"})["value"]
        client_id_val = soup.find("input", {"name": "client_id"})["value"]
        redirect_uri_val = soup.find("input", {"name": "redirect_uri"})["value"]
    except (TypeError, KeyError):
        raise SiakadAuthError("Gagal ekstrak CSRF token dari halaman login.")

    payload = {
        "email": email,
        "password": password,
        "__token": token_val,
        "_token": "",
        "client_id": client_id_val,
        "redirect_uri": redirect_uri_val,
    }

    try:
        resp_post = await client.post(
            LOGIN_URL,
            data=payload,
            headers={**HEADERS_BASE, "Content-Type": "application/x-www-form-urlencoded"},
        )
    except httpx.TimeoutException as exc:
        raise SiakadAuthError("SIAKAD credential submission timed out") from exc
    except httpx.TransportError as exc:
        raise SiakadAuthError(
            "SIAKAD credential submission failed due to network error"
        ) from exc

    if "Email atau Password salah" in resp_post.text or str(resp_post.url) == LOGIN_URL:
        raise SiakadAuthError("Kredensial tidak valid.")


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
    try:
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
    except httpx.TimeoutException as exc:
        raise SiakadAuthError("SIAKAD activation timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise SiakadAuthError(
            f"SIAKAD activation returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.TransportError as exc:
        raise SiakadAuthError("SIAKAD activation failed due to network error") from exc


async def init_siakad_session(session_id: str, email: str, password: str) -> bool:
    """Login ke SIAKAD dan simpan cookies ke Redis dengan TTL 1 jam."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=settings.siakad_timeout_seconds,
        ) as client:
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
    except SiakadAuthError as exc:
        logger.warning(
            "Dependency failure dependency=siakad operation=init_session flow=student mode=auth session_id=%s detail=%s",
            session_id,
            exc.message,
        )
        return False
    except MemoryStoreError as exc:
        logger.error(
            "Dependency failure dependency=redis operation=setex_siakad_session flow=student mode=cache_write session_id=%s detail=%s",
            session_id,
            exc.message,
        )
        return False
    except Exception as exc:
        logger.error(
            "Dependency failure dependency=siakad operation=init_session flow=student mode=unexpected session_id=%s error=%s",
            session_id,
            exc,
            exc_info=True,
        )
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
    except Exception as exc:
        logger.warning(
            "Dependency failure dependency=redis operation=get_siakad_cookies flow=student mode=read session_id=%s error=%s",
            session_id,
            exc,
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
