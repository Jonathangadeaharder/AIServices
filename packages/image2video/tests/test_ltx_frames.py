import pytest

from image2video.ltx_frames import normalize_ltx_frame_count


def test_normalize_ltx_frame_count_eight_seconds_at_24fps():
    assert normalize_ltx_frame_count(8, 24) == 193


def test_normalize_ltx_frame_count_four_seconds_at_24fps():
    assert normalize_ltx_frame_count(4, 24) == 97


def test_normalize_ltx_frame_count_one_second_at_24fps():
    assert normalize_ltx_frame_count(1, 24) == 25


def test_normalize_ltx_frame_count_rejects_invalid():
    with pytest.raises(ValueError):
        normalize_ltx_frame_count(0, 24)
    with pytest.raises(ValueError):
        normalize_ltx_frame_count(4, 0)
