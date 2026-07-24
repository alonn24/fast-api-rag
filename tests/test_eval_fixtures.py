import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "app" / "eval" / "fixtures"


def test_documents_fixture_has_expected_shape():
    documents = json.loads((FIXTURES_DIR / "documents.json").read_text())

    assert len(documents) >= 3
    for doc in documents:
        assert isinstance(doc["title"], str) and doc["title"]
        assert isinstance(doc["source"], str) and doc["source"]
        assert isinstance(doc["content"], str) and len(doc["content"]) > 0
        assert isinstance(doc["metadata"], dict)


def test_cases_fixture_has_expected_shape():
    cases = json.loads((FIXTURES_DIR / "cases.json").read_text())

    assert len(cases) >= 3
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"

    for case in cases:
        assert isinstance(case["question"], str) and case["question"]
        assert isinstance(case["expected_answer"], str) and case["expected_answer"]
        assert isinstance(case["expected_source_titles"], list) and case["expected_source_titles"]


def test_case_expected_source_titles_reference_real_documents():
    documents = json.loads((FIXTURES_DIR / "documents.json").read_text())
    cases = json.loads((FIXTURES_DIR / "cases.json").read_text())

    document_titles = {doc["title"] for doc in documents}
    for case in cases:
        for title in case["expected_source_titles"]:
            assert title in document_titles, f"case {case['id']} references unknown document {title!r}"
