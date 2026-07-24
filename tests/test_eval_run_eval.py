import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select

from app.db.models import Chunk, Document
from app.eval.run_eval import ingest_documents, load_cases, load_documents

FIXTURES_DIR = Path(__file__).parent.parent / "app" / "eval" / "fixtures"


def test_load_documents_reads_json_array():
    documents = load_documents(FIXTURES_DIR / "documents.json")
    assert isinstance(documents, list)
    assert documents[0]["title"] == "Vacation Policy"


def test_load_cases_reads_json_array():
    cases = load_cases(FIXTURES_DIR / "cases.json")
    assert isinstance(cases, list)
    assert cases[0]["id"] == "vacation-accrual"


def test_load_documents_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing.json")


def _fake_embeddings():
    fake = Mock()
    fake.embed_documents.side_effect = lambda texts: [[0.01] * 1024 for _ in texts]
    return fake


@pytest.mark.asyncio
async def test_ingest_documents_creates_documents_and_chunks(db_session):
    documents = [
        {"title": "Doc A", "source": "test", "content": " ".join(["word"] * 50), "metadata": {}},
        {"title": "Doc B", "source": "test", "content": " ".join(["word"] * 50), "metadata": {"lang": "en"}},
    ]

    await ingest_documents(db_session, _fake_embeddings(), documents)

    doc_count = (await db_session.execute(select(func.count()).select_from(Document))).scalar_one()
    chunk_count = (await db_session.execute(select(func.count()).select_from(Chunk))).scalar_one()
    assert doc_count == 2
    assert chunk_count == 2


from types import SimpleNamespace

from app.eval.judge import JudgeError, JudgmentResult, Verdict
from app.eval.report import case_passed
from app.eval.run_eval import evaluate_case, orchestrate, parse_args
from app.services.search import SearchResult


ALL_PASS = JudgmentResult(
    correctness=Verdict(passed=True, rationale="ok"),
    faithfulness=Verdict(passed=True, rationale="ok"),
    retrieval_relevance=Verdict(passed=True, rationale="ok"),
)


def _agent_answer(text="15 vacation days.", sources=None):
    import uuid

    if sources is None:
        sources = [
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                document_title="Vacation Policy",
                content="15 days of paid vacation.",
                distance=0.05,
            )
        ]
    return SimpleNamespace(answer=text, sources=sources)


@pytest.mark.asyncio
async def test_evaluate_case_returns_case_result_on_success(monkeypatch):
    import app.eval.run_eval as run_eval_module

    async def fake_run_agent_query(question, *, client, session, embeddings, model, max_iterations=5):
        return _agent_answer()

    monkeypatch.setattr(run_eval_module, "run_agent_query", fake_run_agent_query)
    monkeypatch.setattr(run_eval_module, "judge_case", lambda **kwargs: ALL_PASS)

    case = {
        "id": "vacation-accrual",
        "question": "How many vacation days?",
        "expected_answer": "15 days.",
        "expected_source_titles": ["Vacation Policy"],
    }

    result = await evaluate_case(
        case,
        agent_client=Mock(),
        judge_client=Mock(),
        session=Mock(),
        embeddings=Mock(),
        agent_model="claude-opus-4-8",
        judge_model="claude-sonnet-5",
    )

    assert result.case_id == "vacation-accrual"
    assert result.agent_answer == "15 vacation days."
    assert result.error is None
    assert case_passed(result) is True
    assert result.retrieved_sources == [{"document_title": "Vacation Policy", "content": "15 days of paid vacation."}]


@pytest.mark.asyncio
async def test_evaluate_case_records_error_when_agent_raises(monkeypatch):
    import app.eval.run_eval as run_eval_module

    async def failing_run_agent_query(*args, **kwargs):
        raise RuntimeError("anthropic call failed")

    monkeypatch.setattr(run_eval_module, "run_agent_query", failing_run_agent_query)

    case = {
        "id": "vacation-accrual",
        "question": "How many vacation days?",
        "expected_answer": "15 days.",
        "expected_source_titles": ["Vacation Policy"],
    }

    result = await evaluate_case(
        case,
        agent_client=Mock(),
        judge_client=Mock(),
        session=Mock(),
        embeddings=Mock(),
        agent_model="claude-opus-4-8",
        judge_model="claude-sonnet-5",
    )

    assert result.error is not None
    assert "anthropic call failed" in result.error
    assert result.judgment is None
    assert case_passed(result) is False


@pytest.mark.asyncio
async def test_evaluate_case_records_error_when_judge_raises(monkeypatch):
    import app.eval.run_eval as run_eval_module

    async def fake_run_agent_query(question, *, client, session, embeddings, model, max_iterations=5):
        return _agent_answer()

    def failing_judge_case(**kwargs):
        raise JudgeError("judge API call failed: timeout")

    monkeypatch.setattr(run_eval_module, "run_agent_query", fake_run_agent_query)
    monkeypatch.setattr(run_eval_module, "judge_case", failing_judge_case)

    case = {
        "id": "vacation-accrual",
        "question": "How many vacation days?",
        "expected_answer": "15 days.",
        "expected_source_titles": ["Vacation Policy"],
    }

    result = await evaluate_case(
        case,
        agent_client=Mock(),
        judge_client=Mock(),
        session=Mock(),
        embeddings=Mock(),
        agent_model="claude-opus-4-8",
        judge_model="claude-sonnet-5",
    )

    assert result.error is not None
    assert "judge API call failed" in result.error
    assert result.agent_answer == "15 vacation days."  # agent ran fine; judge failed
    assert case_passed(result) is False


@pytest.mark.asyncio
async def test_orchestrate_returns_zero_when_all_cases_pass(monkeypatch, tmp_path):
    import app.eval.run_eval as run_eval_module

    async def fake_ingest_documents(session, embeddings, documents):
        return None

    async def fake_evaluate_case(case, **kwargs):
        return run_eval_module.CaseResult(
            case_id=case["id"],
            question=case["question"],
            expected_answer=case["expected_answer"],
            agent_answer="answer",
            retrieved_sources=[],
            expected_source_titles=case["expected_source_titles"],
            judgment=ALL_PASS,
            error=None,
        )

    monkeypatch.setattr(run_eval_module, "ingest_documents", fake_ingest_documents)
    monkeypatch.setattr(run_eval_module, "evaluate_case", fake_evaluate_case)

    output_path = tmp_path / "report.html"
    cases = [{"id": "c1", "question": "q1?", "expected_answer": "a1", "expected_source_titles": ["Doc"]}]

    exit_code = await orchestrate(
        documents=[],
        cases=cases,
        session=Mock(),
        embeddings=Mock(),
        agent_client=Mock(),
        judge_client=Mock(),
        agent_model="claude-opus-4-8",
        judge_model="claude-sonnet-5",
        output_path=output_path,
    )

    assert exit_code == 0
    assert output_path.exists()


@pytest.mark.asyncio
async def test_orchestrate_returns_one_when_any_case_fails(monkeypatch, tmp_path):
    import app.eval.run_eval as run_eval_module

    async def fake_ingest_documents(session, embeddings, documents):
        return None

    ONE_FAIL = JudgmentResult(
        correctness=Verdict(passed=False, rationale="wrong"),
        faithfulness=Verdict(passed=True, rationale="ok"),
        retrieval_relevance=Verdict(passed=True, rationale="ok"),
    )

    async def fake_evaluate_case(case, **kwargs):
        return run_eval_module.CaseResult(
            case_id=case["id"],
            question=case["question"],
            expected_answer=case["expected_answer"],
            agent_answer="answer",
            retrieved_sources=[],
            expected_source_titles=case["expected_source_titles"],
            judgment=ONE_FAIL,
            error=None,
        )

    monkeypatch.setattr(run_eval_module, "ingest_documents", fake_ingest_documents)
    monkeypatch.setattr(run_eval_module, "evaluate_case", fake_evaluate_case)

    output_path = tmp_path / "report.html"
    cases = [{"id": "c1", "question": "q1?", "expected_answer": "a1", "expected_source_titles": ["Doc"]}]

    exit_code = await orchestrate(
        documents=[],
        cases=cases,
        session=Mock(),
        embeddings=Mock(),
        agent_client=Mock(),
        judge_client=Mock(),
        agent_model="claude-opus-4-8",
        judge_model="claude-sonnet-5",
        output_path=output_path,
    )

    assert exit_code == 1


def test_parse_args_defaults():
    args = parse_args([])
    assert args.judge_model == "claude-sonnet-5"
    assert str(args.output) == "eval_report.html"
    assert args.documents.name == "documents.json"
    assert args.cases.name == "cases.json"


def test_parse_args_overrides():
    args = parse_args(
        ["--judge-model", "claude-opus-4-8", "--output", "custom.html", "--documents", "d.json", "--cases", "c.json"]
    )
    assert args.judge_model == "claude-opus-4-8"
    assert str(args.output) == "custom.html"
    assert str(args.documents) == "d.json"
    assert str(args.cases) == "c.json"
