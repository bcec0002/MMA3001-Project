import pytest
from src.labels import read_labels


def test_reads_valid_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("1 0.5 0.5 0.2 0.1\n3 0.25 0.75 0.4 0.3\n")
    assert read_labels(f) == [(1, 0.5, 0.5, 0.2, 0.1), (3, 0.25, 0.75, 0.4, 0.3)]


def test_empty_file_gives_empty_list(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert read_labels(f) == []


def test_blank_and_short_lines_are_skipped(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("\n1 0.5 0.5\n2 0.1 0.2 0.3 0.4\n")
    assert read_labels(f) == [(2, 0.1, 0.2, 0.3, 0.4)]


def test_non_numeric_value_raises(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("1 0.5 abc 0.2 0.1\n")
    with pytest.raises(ValueError):
        read_labels(f)