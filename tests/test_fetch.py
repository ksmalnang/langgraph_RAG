from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.nodes.fetch import (
    _scrape_berita,
    _scrape_detail_berita,
    _scrape_jadwal_kuliah,
    _scrape_nilaimhs,
    _scrape_transkrip,
    fetch_student_data,
)

# ============================================================
# HTML Fixtures
# ============================================================

PERIODE_SELECT_HTML = """
<select id="periode" name="periode" class="form-control">
  <option value="20252">2025 Genap</option>
  <option value="20251">2025 Ganjil</option>
  <option value="20242" selected>2024 Genap</option>
</select>
"""

CALLOUT_INFO_HTML = """
<div class="callout callout-info">
  <div class="row">
    <label class="col-md-3">NIM</label>
    <div class="col-md-3">213040001</div>
    <label class="col-md-3">Tahun Kurikulum</label>
    <div class="col-md-3">221</div>
  </div>
  <div class="row">
    <label class="col-md-3">Nama Mahasiswa</label>
    <div class="col-md-3">BUDI SANTOSO</div>
    <label class="col-md-3">Semester</label>
    <div class="col-md-3">10</div>
  </div>
  <div class="row">
    <label class="col-md-3">Program Studi</label>
    <div class="col-md-3">Teknik Informatika</div>
    <label class="col-md-3">Pembimbing Akademik</label>
    <div class="col-md-3">Dr. RIRIN DWI AGUSTIN, ST. MT</div>
  </div>
  <div class="row">
    <label class="col-md-3">Status Mahasiswa</label>
    <div class="col-md-3">Aktif</div>
    <label class="col-md-3">SKS Lulus / IPK Lulus</label>
    <div class="col-md-3">138 / 3.66</div>
  </div>
  <div class="row">
    <label class="col-md-3">Angkatan</label>
    <div class="col-md-3">2021</div>
    <label class="col-md-3">Total SKS / IPK</label>
    <div class="col-md-3">138 / 3.66</div>
  </div>
</div>
"""

NILAI_TABLE_HTML = """
<table class="table table-bordered dataTable">
  <thead>
    <tr>
      <th>Kurikulum</th><th>Kode MK</th><th>Nama MK</th><th>Nama Kelas</th>
      <th colspan="3">Nilai Komponen</th><th>Nilai Akhir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="text-center">221</td>
      <td>IF21W0801</td>
      <td>Islam Dasar Ilmu</td>
      <td class="text-center">A</td>
      <td colspan="3" style="padding:0px">
        <table class="table table-condensed table-bordered">
          <tbody>
            <tr>
              <td style="width:185px">KEHADIRAN</td>
              <td style="width:50px" align="center">20.00</td>
              <td align="center">81.25</td>
            </tr>
          </tbody>
        </table>
      </td>
      <td>80.51</td>
    </tr>
    <tr style="background-color: #34495E"><td colspan="8"></td></tr>
    <tr>
      <td class="text-center">221</td>
      <td>IF21W0805</td>
      <td>Tugas Akhir</td>
      <td class="text-center">A</td>
      <td colspan="3" style="padding:0px">
        <table class="table table-condensed table-bordered"></table>
      </td>
      <td>T</td>
    </tr>
  </tbody>
</table>
"""

E_NILAI_TABLE_HTML = """
<table class="table table-bordered dataTable">
  <thead>
    <tr>
      <th>Kurikulum</th><th>Kode MK</th><th>Nama MK</th><th>Nama Kelas</th>
      <th colspan="3">Nilai Komponen</th><th>Nilai Akhir</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
"""

JADWAL_TABLE_HTML = """
<table class="table table-bordered table-striped dataTable">
  <thead>
    <tr>
      <th>No</th><th>Hari</th><th>Tanggal</th><th>Mulai</th><th>Selesai</th>
      <th>Jenis</th><th>Kelas Mata Kuliah</th><th>Materi</th><th>Ruang</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td><td>Selasa</td><td>4 Feb 2025</td>
      <td class="text-center">16:30</td><td class="text-center">18:10</td>
      <td>Kuliah</td><td>Ilmu Budaya Sunda - B</td>
      <td class="text-break">Pengantar Budaya Sunda</td>
      <td>RUANG KULIAH SB108</td>
    </tr>
    <tr>
      <td>2</td><td>Jumat</td><td>7 Feb 2025</td>
      <td class="text-center">08:50</td><td class="text-center">10:30</td>
      <td>Kuliah</td><td>Keamanan Informasi - A</td>
      <td class="text-break">Pendahuluan Keamanan Informasi</td>
      <td></td>
    </tr>
  </tbody>
</table>
"""

E_JADWAL_TABLE_HTML = """
<table class="table table-bordered table-striped dataTable">
  <thead>
    <tr>
      <th>No</th><th>Hari</th><th>Tanggal</th><th>Mulai</th><th>Selesai</th>
      <th>Jenis</th><th>Kelas Mata Kuliah</th><th>Materi</th><th>Ruang</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
"""

BERITA_TABLE_HTML = """
<table class="table table-bordered table-striped dataTable">
  <thead>
    <tr><th>Tanggal</th><th>Penulis</th><th>Judul</th><th>Aksi</th></tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">12 Feb 2026, 10:15:13</td>
      <td>Ghulam Guntara</td>
      <td>Pengumuman Perubahan Jam Perkuliahan Semester Genap 2025-2026 Fakultas Teknik</td>
      <td nowrap="">
        <button class="btn btn-info btn-xs btn-flat" type="button"
          data-id="371" data-type="edit"><i class="fa fa-eye"></i></button>
      </td>
    </tr>
  </tbody>
</table>
"""

E_BERITA_TABLE_HTML = """
<table class="table table-bordered table-striped dataTable">
  <thead>
    <tr><th>Tanggal</th><th>Penulis</th><th>Judul</th><th>Aksi</th></tr>
  </thead>
  <tbody>
  </tbody>
</table>
"""

BERITA_DETAIL_HTML = """
<div id="block-judulberita" class="row bord-bottom">
  <label class="col-md-3">Judul</label>
  <div class="col-md-9 input-judulberita">
    Pengumuman Perubahan Jam Perkuliahan Semester Genap 2025-2026 Fakultas Teknik
  </div>
</div>
<div id="block-isiberita" class="row bord-bottom">
  <label class="col-md-3">Pengumuman</label>
  <div class="col-md-9 input-isiberita">
    Salam Teknik<br><br>Silahkan unduh surat edaran terlampir.
  </div>
</div>
<div id="block-fileberita" class="row bord-bottom">
  <label class="col-md-3">File</label>
  <div class="col-md-9 input-fileberita">
    <a href="javascript:void(0)" data-type="download" data-name="fileberita">
      4435_075_Unpas.FT.D_Q_II_2026.pdf
    </a>
  </div>
</div>
"""

TRANSKRIP_TABLE_HTML = """
<table class="dataTable">
  <thead>
    <tr>
      <th>No</th><th>Kode</th><th>Nama MK</th><th>Semester</th>
      <th>SKS</th><th>Grade</th><th>Nilai Mutu</th><th>Bobot</th>
    </tr>
    <tr>
      <th colspan="2">Indeks Prestasi Kumulatif</th>
      <th>3.66</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td><td>IF21W0801</td><td>Islam Dasar Ilmu</td>
      <td>1</td><td>2</td><td>A</td><td>4.00</td><td>8.00</td>
    </tr>
  </tbody>
</table>
"""

E_TRANSKRIP_TABLE_HTML = """
<table class="dataTable">
  <thead>
    <tr>
      <th>No</th><th>Kode</th><th>Nama MK</th><th>Semester</th>
      <th>SKS</th><th>Grade</th><th>Nilai Mutu</th><th>Bobot</th>
    </tr>
    <tr>
      <th colspan="2">Indeks Prestasi Kumulatif</th>
      <th>3.66</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
"""


# ============================================================
# Helper Mocks
# ============================================================

def make_mock_response(html: str, url: str = "https://situ2.unpas.ac.id/siakad/list_nilaimhs"):
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.url = url
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ============================================================
# Tests: _scrape_transkrip
# ============================================================

@pytest.mark.asyncio
async def test_scrape_transkrip__happy_path():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(TRANSKRIP_TABLE_HTML)
    result = await _scrape_transkrip(mock_client)

    assert "ipk" in result
    assert result["ipk"] == 3.66
    assert result["total_mata_kuliah"] == 1
    assert result["transkrip"][0]["kode"] == "IF21W0801"
    assert result["transkrip"][0]["grade"] == "A"


@pytest.mark.asyncio
async def test_scrape_transkrip__empty_table():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(E_TRANSKRIP_TABLE_HTML)
    result = await _scrape_transkrip(mock_client)

    assert result["ipk"] == 3.66
    assert result["total_mata_kuliah"] == 0
    assert result["transkrip"] == []


@pytest.mark.asyncio
async def test_scrape_transkrip__session_expired():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(TRANSKRIP_TABLE_HTML, url="https://situ2.unpas.ac.id/gate/login")
    with pytest.raises(ConnectionError, match="Session expired"):
        await _scrape_transkrip(mock_client)


# ============================================================
# Tests: _scrape_nilaimhs
# ============================================================

@pytest.mark.asyncio
async def test_scrape_nilaimhs__happy_path():
    mock_client = AsyncMock()
    html = PERIODE_SELECT_HTML + CALLOUT_INFO_HTML + NILAI_TABLE_HTML
    mock_client.get.return_value = make_mock_response(html)

    result = await _scrape_nilaimhs(mock_client)

    # Assert Mahasiswa
    assert "mahasiswa" in result
    assert result["mahasiswa"]["nim"] == "213040001"
    assert result["mahasiswa"]["nama"] == "BUDI SANTOSO"
    assert result["mahasiswa"]["angkatan"] == "2021"

    # Assert Nilai
    assert "nilai_semester" in result
    assert result["nilai_semester"]["periode_dipilih"] == "20242"
    assert result["nilai_semester"]["total_mata_kuliah"] == 2

    # Check that T is string, 80.51 is float
    assert result["nilai_semester"]["nilai"][0]["nilai_akhir"] == 80.51
    assert result["nilai_semester"]["nilai"][1]["nilai_akhir"] == "T"


@pytest.mark.asyncio
async def test_scrape_nilaimhs__empty_table():
    mock_client = AsyncMock()
    html = PERIODE_SELECT_HTML + CALLOUT_INFO_HTML + E_NILAI_TABLE_HTML
    mock_client.get.return_value = make_mock_response(html)

    result = await _scrape_nilaimhs(mock_client)

    # NIM still correctly parsed
    assert result["mahasiswa"]["nim"] == "213040001"

    # Empty table
    assert result["nilai_semester"]["nilai"] == []
    assert result["nilai_semester"]["total_mata_kuliah"] == 0


@pytest.mark.asyncio
async def test_scrape_nilaimhs__with_post():
    mock_client = AsyncMock()
    html_get = PERIODE_SELECT_HTML + CALLOUT_INFO_HTML + ""
    html_post = PERIODE_SELECT_HTML + CALLOUT_INFO_HTML + NILAI_TABLE_HTML

    mock_client.get.return_value = make_mock_response(html_get)
    mock_client.post.return_value = make_mock_response(html_post)

    # Passing periode "20251" which is different from "20242"
    result = await _scrape_nilaimhs(mock_client, periode="20251")

    assert mock_client.post.called
    assert result["nilai_semester"]["periode_dipilih"] == "20251"
    assert result["mahasiswa"]["nim"] == "213040001"


@pytest.mark.asyncio
async def test_scrape_nilaimhs__session_expired():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(PERIODE_SELECT_HTML, url="https://situ2.unpas.ac.id/gate/login")
    with pytest.raises(ConnectionError, match="Session expired"):
        await _scrape_nilaimhs(mock_client)


# ============================================================
# Tests: _scrape_jadwal_kuliah
# ============================================================

@pytest.mark.asyncio
async def test_scrape_jadwal_kuliah__happy_path():
    mock_client = AsyncMock()
    html_get = PERIODE_SELECT_HTML
    html_post = PERIODE_SELECT_HTML + JADWAL_TABLE_HTML

    mock_client.get.return_value = make_mock_response(html_get)
    mock_client.post.return_value = make_mock_response(html_post)

    result = await _scrape_jadwal_kuliah(mock_client)

    assert result["periode_dipilih"] == "20242"
    assert result["total_jadwal"] == 2
    assert result["jadwal"][0]["ruang"] == "RUANG KULIAH SB108"
    assert result["jadwal"][1]["ruang"] is None


@pytest.mark.asyncio
async def test_scrape_jadwal_kuliah__empty_table():
    mock_client = AsyncMock()
    html_post = PERIODE_SELECT_HTML + E_JADWAL_TABLE_HTML
    mock_client.get.return_value = make_mock_response(PERIODE_SELECT_HTML)
    mock_client.post.return_value = make_mock_response(html_post)

    result = await _scrape_jadwal_kuliah(mock_client)
    assert result["total_jadwal"] == 0
    assert result["jadwal"] == []


@pytest.mark.asyncio
async def test_scrape_jadwal_kuliah__session_expired():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(PERIODE_SELECT_HTML, url="https://situ2.unpas.ac.id/gate/login")
    with pytest.raises(ConnectionError, match="Session expired"):
        await _scrape_jadwal_kuliah(mock_client)


# ============================================================
# Tests: _scrape_berita
# ============================================================

@pytest.mark.asyncio
async def test_scrape_berita__happy_path():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(BERITA_TABLE_HTML)

    result = await _scrape_berita(mock_client, fetch_detail=False)
    assert result["total"] == 1
    assert result["berita"][0]["id"] == "371"
    assert result["berita"][0]["penulis"] == "Ghulam Guntara"


@pytest.mark.asyncio
async def test_scrape_berita__empty_table():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(E_BERITA_TABLE_HTML)

    result = await _scrape_berita(mock_client, fetch_detail=False)
    assert result["total"] == 0
    assert result["berita"] == []


@pytest.mark.asyncio
async def test_scrape_berita__session_expired():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(BERITA_TABLE_HTML, url="https://situ2.unpas.ac.id/gate/login")
    with pytest.raises(ConnectionError, match="Session expired"):
        await _scrape_berita(mock_client)


@pytest.mark.asyncio
async def test_scrape_berita__with_detail():
    mock_client = AsyncMock()
    mock_client.get.side_effect = [
        make_mock_response(BERITA_TABLE_HTML),        # list response
        make_mock_response(BERITA_DETAIL_HTML),       # detail response
    ]

    result = await _scrape_berita(mock_client, fetch_detail=True)
    assert result["total"] == 1
    assert result["berita"][0]["id"] == "371"
    assert result["berita"][0]["detail"] is not None
    assert result["berita"][0]["detail"]["file"] == "4435_075_Unpas.FT.D_Q_II_2026.pdf"


# ============================================================
# Tests: _scrape_detail_berita
# ============================================================

@pytest.mark.asyncio
async def test_scrape_detail_berita__happy_path():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(BERITA_DETAIL_HTML)

    result = await _scrape_detail_berita(mock_client, "371")
    assert result["id"] == "371"
    assert result["file"] == "4435_075_Unpas.FT.D_Q_II_2026.pdf"
    assert "Salam Teknik" in result["isi"]


@pytest.mark.asyncio
async def test_scrape_detail_berita__session_expired():
    mock_client = AsyncMock()
    mock_client.get.return_value = make_mock_response(BERITA_DETAIL_HTML, url="https://situ2.unpas.ac.id/gate/login")
    with pytest.raises(ConnectionError, match="Session expired"):
        await _scrape_detail_berita(mock_client, "371")


# ============================================================
# Tests: fetch_student_data
# ============================================================

@patch("app.agent.nodes.fetch.get_cached_student_data", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch.get_siakad_cookies", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch.cache_student_data", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch._scrape_nilaimhs", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch._scrape_transkrip", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch._scrape_jadwal_kuliah", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch._scrape_berita", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_fetch_student_data__happy_path(
    mock_berita, mock_jadwal, mock_transkrip, mock_nilaimhs,
    mock_cache_set, mock_get_cookies, mock_cache_get
):
    mock_cache_get.return_value = None
    mock_get_cookies.return_value = {"SIAKAD_CLOUD_ACCESS": "unpas-test"}
    mock_nilaimhs.return_value = {"mahasiswa": {"nim": "213"}, "nilai_semester": {}}
    mock_transkrip.return_value = {"ipk": 3.66, "total_mata_kuliah": 1, "transkrip": []}
    mock_jadwal.return_value = {"periode_dipilih": "20242", "total_jadwal": 1, "jadwal": []}
    mock_berita.return_value = {"total": 1, "berita": []}

    state = {"session_id": "test-session-123", "need_retrieval": False}
    result = await fetch_student_data(state)

    assert set(result) == {"student_data", "student_fetch_error"}
    assert result["student_fetch_error"] is False
    assert result["student_data"] is not None
    assert set(result["student_data"].keys()) == {"mahasiswa", "nilai_semester", "transkrip", "jadwal", "berita"}

    mock_cache_set.assert_called_once()
    assert mock_cache_set.call_args[0][0] == "test-session-123"
    assert result["student_data"] == mock_cache_set.call_args[0][1]


@patch("app.agent.nodes.fetch.get_cached_student_data", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_fetch_student_data__cache_hit(mock_cache_get):
    dummy_student_data = {
        "mahasiswa": {},
        "nilai_semester": {},
        "transkrip": {},
        "jadwal": {},
        "berita": {},
    }
    mock_cache_get.return_value = dummy_student_data

    # get_siakad_cookies will NOT be called
    with patch("app.agent.nodes.fetch.get_siakad_cookies") as mock_get_cookies:
        state = {"session_id": "test-session-123", "need_retrieval": False}
        result = await fetch_student_data(state)

        mock_get_cookies.assert_not_called()
        assert set(result) == {"student_data", "student_fetch_error"}
        assert result["student_data"] == dummy_student_data
        assert result["student_fetch_error"] is False


@patch("app.agent.nodes.fetch.get_cached_student_data", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch.get_siakad_cookies", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch._scrape_nilaimhs", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_fetch_student_data__session_expired(mock_nilaimhs, mock_get_cookies, mock_cache_get):
    mock_cache_get.return_value = None
    mock_get_cookies.return_value = {"SIAKAD_CLOUD_ACCESS": "unpas-test"}
    mock_nilaimhs.side_effect = ConnectionError("Session expired, redirect ke login page.")

    # We mock others as valid, just one is failing
    with patch("app.agent.nodes.fetch._scrape_transkrip", new_callable=AsyncMock), \
         patch("app.agent.nodes.fetch._scrape_jadwal_kuliah", new_callable=AsyncMock), \
         patch("app.agent.nodes.fetch._scrape_berita", new_callable=AsyncMock):

        state = {"session_id": "test-session-123", "need_retrieval": False}
        result = await fetch_student_data(state)

        assert set(result) == {"student_data", "student_fetch_error"}
        assert result["student_fetch_error"] is True
        assert result["student_data"] is None


@pytest.mark.asyncio
async def test_fetch_student_data__missing_session_id():
    state = {"need_retrieval": False}
    result = await fetch_student_data(state)
    assert set(result) == {"student_data", "student_fetch_error"}
    assert result["student_fetch_error"] is True
    assert result["student_data"] is None


@patch("app.agent.nodes.fetch.get_cached_student_data", new_callable=AsyncMock)
@patch("app.agent.nodes.fetch.get_siakad_cookies", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_fetch_student_data__no_cookies(mock_get_cookies, mock_cache_get):
    mock_cache_get.return_value = None
    mock_get_cookies.return_value = None

    state = {"session_id": "test-session-123", "need_retrieval": False}
    result = await fetch_student_data(state)

    assert set(result) == {"student_data", "student_fetch_error"}
    assert result["student_fetch_error"] is True
    assert result["student_data"] is None
