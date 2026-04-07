import pytest

from app.agent.nodes.fetch_nilai_semester import _map_semester_to_periode


def test_map_semester_to_periode__sem1():
    assert _map_semester_to_periode(2021, 1) == "20211"

def test_map_semester_to_periode__sem2():
    assert _map_semester_to_periode(2021, 2) == "20212"

def test_map_semester_to_periode__sem3():
    assert _map_semester_to_periode(2021, 3) == "20221"

def test_map_semester_to_periode__sem4():
    assert _map_semester_to_periode(2021, 4) == "20222"

def test_map_semester_to_periode__sem5():
    assert _map_semester_to_periode(2021, 5) == "20231"

def test_map_semester_to_periode__sem6():
    assert _map_semester_to_periode(2021, 6) == "20232"

def test_map_semester_to_periode__sem7():
    assert _map_semester_to_periode(2021, 7) == "20241"

def test_map_semester_to_periode__sem8():
    assert _map_semester_to_periode(2021, 8) == "20242"

def test_map_semester_to_periode__sem9():
    assert _map_semester_to_periode(2021, 9) == "20251"

def test_map_semester_to_periode__sem10():
    assert _map_semester_to_periode(2021, 10) == "20252"

def test_map_semester_to_periode__angkatan_2022_sem1():
    assert _map_semester_to_periode(2022, 1) == "20221"

def test_map_semester_to_periode__angkatan_2022_sem2():
    assert _map_semester_to_periode(2022, 2) == "20222"

def test_map_semester_to_periode__angkatan_2019_sem6():
    assert _map_semester_to_periode(2019, 6) == "20212"

def test_map_semester_to_periode__sem14():
    assert _map_semester_to_periode(2021, 14) == "20272"

def test_map_semester_to_periode__raises_on_semester_0():
    with pytest.raises(ValueError):
        _map_semester_to_periode(2021, 0)

def test_map_semester_to_periode__raises_on_semester_15():
    with pytest.raises(ValueError):
        _map_semester_to_periode(2021, 15)

def test_map_semester_to_periode__raises_on_angkatan_1999():
    with pytest.raises(ValueError):
        _map_semester_to_periode(1999, 1)

def test_map_semester_to_periode__raises_on_angkatan_2101():
    with pytest.raises(ValueError):
        _map_semester_to_periode(2101, 1)
