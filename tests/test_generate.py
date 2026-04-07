from app.agent.nodes.generate import (
    _format_berita_summary,
    _format_context,
    _format_history,
    _format_jadwal_summary,
    _format_nilai_summary,
    _format_student_context,
)

MOCK_STUDENT_DATA = {
    "mahasiswa": {
        "nim": "213040001",
        "nama": "BUDI SANTOSO",
        "program_studi": "Teknik Informatika",
        "semester": "10",
        "angkatan": "2021",
        "status": "Aktif",
        "pembimbing_akademik": "Dr. RIRIN DWI AGUSTIN, ST. MT",
        "sks_lulus_ipk_lulus": "138 / 3.66",
        "total_sks_ipk": "138 / 3.66",
        "tahun_kurikulum": "221",
    },
    "nilai_semester": {
        "periode_dipilih": "20242",
        "periode_options": [
            {"value": "20242", "label": "2024 Genap", "selected": True},
        ],
        "total_mata_kuliah": 2,
        "nilai": [
            {
                "kurikulum": "221",
                "kode": "IF21W0801",
                "nama_mata_kuliah": "Islam Dasar Ilmu",
                "nama_kelas": "A",
                "komponen_nilai": [
                    {"komponen": "KEHADIRAN", "bobot_persen": "20.00", "nilai": "81.25"}
                ],
                "nilai_akhir": 80.51,
            },
            {
                "kurikulum": "221",
                "kode": "IF21W0805",
                "nama_mata_kuliah": "Tugas Akhir",
                "nama_kelas": "A",
                "komponen_nilai": [],
                "nilai_akhir": "T",
            },
        ],
    },
    "transkrip": {
        "ipk": 3.66,
        "total_mata_kuliah": 1,
        "transkrip": [
            {
                "no": "1",
                "kode": "IF21W0801",
                "nama_mata_kuliah": "Islam Dasar Ilmu",
                "semester": "1",
                "sks": "2",
                "grade": "A",
                "nilai_mutu": "4.00",
                "bobot": "8.00",
            }
        ],
    },
    "jadwal": {
        "periode_dipilih": "20242",
        "periode_options": [],
        "total_jadwal": 1,
        "jadwal": [
            {
                "no": "1",
                "hari": "Selasa",
                "tanggal": "4 Feb 2025",
                "mulai": "16:30",
                "selesai": "18:10",
                "jenis": "Kuliah",
                "kelas_mata_kuliah": "Ilmu Budaya Sunda - B",
                "materi": "Pengantar Budaya Sunda",
                "ruang": "RUANG KULIAH SB108",
            }
        ],
    },
    "berita": {
        "total": 1,
        "berita": [
            {
                "id": "371",
                "tanggal": "12 Feb 2026, 10:15:13",
                "penulis": "Ghulam Guntara",
                "judul": "Pengumuman Perubahan Jam Perkuliahan Semester Genap 2025-2026",
                "detail": None,
            }
        ],
    },
}


def test_format_history__happy_path():
    history = [
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I am fine."},
    ]
    formatted = _format_history(history)
    assert "User: How are you?" in formatted
    assert "Assistant: I am fine." in formatted


def test_format_history__empty():
    assert _format_history([]) == ""
    assert _format_history([{"role": "user"}]) == ""  # missing content


def test_format_history__limit_10():
    history = [{"role": f"user{i}", "content": f"msg{i}"} for i in range(12)]
    formatted = _format_history(history)
    assert "User0: msg0" not in formatted
    assert "User1: msg1" not in formatted
    assert "User2: msg2" in formatted
    assert "User11: msg11" in formatted


def test_format_context__happy_path():
    documents = [
        {"text": "doc one text", "relevance_score": 0.95, "headings": ["H1", "H2"]},
        {"text": "doc two text", "relevance_score": 0.80},
    ]
    formatted = _format_context(documents)
    assert "[1] (relevance: 0.95, section: H1 > H2)" in formatted
    assert "doc one text" in formatted
    assert "[2] (relevance: 0.80, section: N/A)" in formatted
    assert "doc two text" in formatted


def test_format_context__empty():
    assert _format_context([]) == "(no documents)"


def test_format_student_context__happy_path():
    result = _format_student_context(MOCK_STUDENT_DATA)

    # Mahasiswa fields
    assert "BUDI SANTOSO" in result
    assert "213040001" in result
    assert "Teknik Informatika" in result
    assert "3.66" in result

    # Nilai semester section
    assert "Islam Dasar Ilmu" in result
    assert "80.51" in result
    assert "Tugas Akhir" in result
    assert "T" in result

    # Jadwal section
    assert "Ilmu Budaya Sunda - B" in result
    assert "16:30" in result
    assert "RUANG KULIAH SB108" in result

    # Berita section
    assert "Pengumuman Perubahan Jam Perkuliahan" in result
    assert "12 Feb 2026" in result


def test_format_student_context__empty():
    assert _format_student_context({}) == ""


def test_format_jadwal_summary__happy_path():
    jadwal = [
        {
            "hari": "Selasa",
            "tanggal": "4 Feb 2025",
            "mulai": "16:30",
            "selesai": "18:10",
            "kelas_mata_kuliah": "Ilmu Budaya Sunda - B",
            "ruang": "RUANG KULIAH SB108",
        }
    ]
    result = _format_jadwal_summary(jadwal)
    assert "Selasa" in result
    assert "16:30" in result
    assert "Ilmu Budaya Sunda - B" in result
    assert "RUANG KULIAH SB108" in result


def test_format_jadwal_summary__none_ruang():
    jadwal = [
        {
            "hari": "Jumat",
            "tanggal": "7 Feb 2025",
            "mulai": "08:50",
            "selesai": "10:30",
            "kelas_mata_kuliah": "Keamanan Informasi - A",
            "ruang": None,
        }
    ]
    result = _format_jadwal_summary(jadwal)
    assert "Jumat" in result
    assert "TBD" in result  # None ruang must render as "TBD"


def test_format_nilai_summary__happy_path():
    nilai = [
        {
            "kode": "IF21W0801",
            "nama_mata_kuliah": "Islam Dasar Ilmu",
            "nama_kelas": "A",
            "nilai_akhir": 80.51,
        }
    ]
    result = _format_nilai_summary(nilai)
    assert "IF21W0801" in result
    assert "Islam Dasar Ilmu" in result
    assert "80.51" in result


def test_format_nilai_summary__string_nilai_akhir():
    # nilai_akhir can be "T" for Tugas Akhir — must not crash
    nilai = [
        {
            "kode": "IF21W0805",
            "nama_mata_kuliah": "Tugas Akhir",
            "nama_kelas": "A",
            "nilai_akhir": "T",
        }
    ]
    result = _format_nilai_summary(nilai)
    assert "Tugas Akhir" in result
    assert "T" in result


def test_format_berita_summary__happy_path():
    berita = [
        {
            "tanggal": "12 Feb 2026, 10:15:13",
            "judul": "Pengumuman Perubahan Jam Perkuliahan",
        }
    ]
    result = _format_berita_summary(berita)
    assert "12 Feb 2026" in result
    assert "Pengumuman Perubahan Jam Perkuliahan" in result


def test_format_jadwal_summary__empty():
    assert _format_jadwal_summary([]) == "-"


def test_format_nilai_summary__empty():
    assert _format_nilai_summary([]) == "-"


def test_format_berita_summary__empty():
    assert _format_berita_summary([]) == "-"
