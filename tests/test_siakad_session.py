from unittest.mock import AsyncMock, MagicMock, patch

from httpx import Response
import pytest

from app.services.siakad_session import (
    cache_student_data,
    get_cached_student_data,
    get_siakad_cookies,
    init_siakad_session,
)


@pytest.fixture
def mock_redis():
    with patch("app.services.siakad_session.get_redis") as mock_get_redis:
        mock_r = AsyncMock()
        mock_get_redis.return_value = mock_r
        yield mock_r


# --------------------------------------------------------------------------------
# test init_siakad_session
# --------------------------------------------------------------------------------

@patch("app.services.siakad_session.httpx.AsyncClient", new_callable=MagicMock)
@pytest.mark.asyncio
async def test_init_siakad_session__happy_path(mock_client_class, mock_redis):
    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    # Mock _login (GET and POST)
    mock_get_resp = MagicMock(spec=Response)
    mock_get_resp.text = '<input name="__token" value="abc"><input name="client_id" value="123"><input name="redirect_uri" value="url">'
    mock_get_resp.raise_for_status = MagicMock()

    mock_post_resp = MagicMock(spec=Response)
    mock_post_resp.text = "success"
    mock_post_resp.url = "https://situ2.unpas.ac.id/gate/menu"
    mock_post_resp.raise_for_status = MagicMock()

    # Mock _activate_siakad (POST)
    mock_activate_resp = MagicMock(spec=Response)
    mock_activate_resp.raise_for_status = MagicMock()

    mock_client.get.return_value = mock_get_resp
    mock_client.post.side_effect = [mock_post_resp, mock_activate_resp]

    mock_client.cookies = {"cookie_key": "cookie_val"}

    res = await init_siakad_session("sess-1", "user", "pass")
    assert res is True

    # assert redis was called
    mock_redis.setex.assert_called_once()
    called_args = mock_redis.setex.call_args[0]
    assert called_args[0] == "siakad_session:sess-1"
    assert called_args[1] == 3600
    assert "cookie_key" in called_args[2]


@patch("app.services.siakad_session.httpx.AsyncClient", new_callable=MagicMock)
@pytest.mark.asyncio
async def test_init_siakad_session__invalid_credentials(mock_client_class, mock_redis):
    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    mock_get_resp = MagicMock()
    mock_get_resp.text = '<input name="__token" value="abc"><input name="client_id" value="123"><input name="redirect_uri" value="url">'

    mock_post_resp = MagicMock()
    mock_post_resp.text = "Email atau Password salah"
    mock_post_resp.url = "https://situ2.unpas.ac.id/gate/login"

    mock_client.get.return_value = mock_get_resp
    mock_client.post.return_value = mock_post_resp

    res = await init_siakad_session("sess-2", "user", "wrong_pass")
    assert res is False
    mock_redis.setex.assert_not_called()


@patch("app.services.siakad_session.httpx.AsyncClient", new_callable=MagicMock)
@pytest.mark.asyncio
async def test_init_siakad_session__missing_token(mock_client_class, mock_redis):
    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    mock_get_resp = MagicMock()
    mock_get_resp.text = "<html>no tokens here</html>"
    mock_client.get.return_value = mock_get_resp

    res = await init_siakad_session("sess-3", "user", "pass")
    assert res is False


@patch("app.services.siakad_session.httpx.AsyncClient", new_callable=MagicMock)
@pytest.mark.asyncio
async def test_init_siakad_session__network_error(mock_client_class, mock_redis):
    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client
    mock_client.get.side_effect = Exception("network down")

    res = await init_siakad_session("sess-4", "user", "pass")
    assert res is False


# --------------------------------------------------------------------------------
# test get_siakad_cookies
# --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_siakad_cookies__cache_hit(mock_redis):
    mock_redis.get.return_value = '{"my_cookie": "test"}'
    res = await get_siakad_cookies("sess-5")
    assert res == {"my_cookie": "test"}
    mock_redis.get.assert_called_with("siakad_session:sess-5")


@pytest.mark.asyncio
async def test_get_siakad_cookies__cache_miss(mock_redis):
    mock_redis.get.return_value = None
    res = await get_siakad_cookies("sess-6")
    assert res is None


@pytest.mark.asyncio
async def test_get_siakad_cookies__redis_error(mock_redis):
    mock_redis.get.side_effect = Exception("Redis connection lost")
    res = await get_siakad_cookies("sess-7")
    assert res is None


# --------------------------------------------------------------------------------
# test cache_student_data
# --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_student_data__happy_path(mock_redis):
    data = {"nim": "123"}
    res = await cache_student_data("sess-8", data)
    assert res is True
    mock_redis.setex.assert_called_once()
    assert mock_redis.setex.call_args[0][0] == "student_data:sess-8"


@pytest.mark.asyncio
async def test_cache_student_data__redis_error(mock_redis):
    mock_redis.setex.side_effect = Exception("error")
    res = await cache_student_data("sess-9", {"nim": "123"})
    assert res is False


# --------------------------------------------------------------------------------
# test get_cached_student_data
# --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_cached_student_data__cache_hit(mock_redis):
    mock_redis.get.return_value = '{"nim": "999"}'
    res = await get_cached_student_data("sess-10")
    assert res == {"nim": "999"}


@pytest.mark.asyncio
async def test_get_cached_student_data__cache_miss(mock_redis):
    mock_redis.get.return_value = None
    res = await get_cached_student_data("sess-11")
    assert res is None


@pytest.mark.asyncio
async def test_get_cached_student_data__redis_error(mock_redis):
    mock_redis.get.side_effect = Exception("redis out of memory")
    res = await get_cached_student_data("sess-12")
    assert res is None
