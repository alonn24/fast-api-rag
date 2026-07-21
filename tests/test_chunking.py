import pytest
from app.services.chunking import chunk_text


def test_chunk_text_short_text_returns_single_chunk():
    assert chunk_text("one two three", chunk_size=500, overlap=50) == ["one two three"]


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("", chunk_size=500, overlap=50) == []


def test_chunk_text_splits_with_overlap():
    words = [f"w{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "w0"
    assert chunks[0].split()[-1] == "w499"
    assert chunks[1].split()[0] == "w450"
    assert chunks[1].split()[-1] == "w949"
    assert chunks[2].split()[0] == "w900"
    assert chunks[2].split()[-1] == "w1199"


def test_chunk_text_invalid_overlap_raises_value_error():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)


def test_chunk_text_invalid_chunk_size_raises_value_error():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=0, overlap=0)
