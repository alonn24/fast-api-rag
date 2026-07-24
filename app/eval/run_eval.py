import os

os.environ.setdefault(
    "DATABASE_URL",
    os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5433/rag_test"),
)

import argparse
import asyncio
import json
import sys
from pathlib import Path

import anthropic
import voyageai
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tool_loop import run_agent_query
from app.api.documents import create_document
from app.config import get_settings
from app.db.schema import apply_schema
from app.db.session import engine, session_factory
from app.eval.judge import JudgeError, judge_case
from app.eval.report import CaseResult, case_passed, write_report
from app.schemas import DocumentCreateRequest
from app.services.embeddings import EmbeddingService

DEFAULT_DOCUMENTS_PATH = Path(__file__).parent / "fixtures" / "documents.json"
DEFAULT_CASES_PATH = Path(__file__).parent / "fixtures" / "cases.json"
DEFAULT_OUTPUT_PATH = Path("eval_report.html")
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"


def load_documents(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text())


async def truncate_tables() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE chunks, documents RESTART IDENTITY CASCADE"))


async def ingest_documents(session: AsyncSession, embeddings: EmbeddingService, documents: list[dict]) -> None:
    for doc in documents:
        body = DocumentCreateRequest(
            title=doc["title"],
            source=doc["source"],
            content=doc["content"],
            metadata=doc.get("metadata", {}),
        )
        await create_document(body, session=session, embeddings=embeddings)


async def evaluate_case(
    case: dict,
    *,
    agent_client: anthropic.Anthropic,
    judge_client: anthropic.Anthropic,
    session: AsyncSession,
    embeddings: EmbeddingService,
    agent_model: str,
    judge_model: str,
) -> CaseResult:
    try:
        agent_answer = await run_agent_query(
            case["question"],
            client=agent_client,
            session=session,
            embeddings=embeddings,
            model=agent_model,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case["id"],
            question=case["question"],
            expected_answer=case["expected_answer"],
            agent_answer=None,
            retrieved_sources=[],
            expected_source_titles=case["expected_source_titles"],
            judgment=None,
            error=f"agent error: {exc}",
        )

    retrieved_sources = [
        {"document_title": s.document_title, "content": s.content} for s in agent_answer.sources
    ]

    try:
        judgment = judge_case(
            client=judge_client,
            model=judge_model,
            question=case["question"],
            expected_answer=case["expected_answer"],
            expected_source_titles=case["expected_source_titles"],
            agent_answer=agent_answer.answer,
            retrieved_sources=retrieved_sources,
        )
    except JudgeError as exc:
        return CaseResult(
            case_id=case["id"],
            question=case["question"],
            expected_answer=case["expected_answer"],
            agent_answer=agent_answer.answer,
            retrieved_sources=retrieved_sources,
            expected_source_titles=case["expected_source_titles"],
            judgment=None,
            error=str(exc),
        )

    return CaseResult(
        case_id=case["id"],
        question=case["question"],
        expected_answer=case["expected_answer"],
        agent_answer=agent_answer.answer,
        retrieved_sources=retrieved_sources,
        expected_source_titles=case["expected_source_titles"],
        judgment=judgment,
        error=None,
    )


async def orchestrate(
    *,
    documents: list[dict],
    cases: list[dict],
    session: AsyncSession,
    embeddings: EmbeddingService,
    agent_client: anthropic.Anthropic,
    judge_client: anthropic.Anthropic,
    agent_model: str,
    judge_model: str,
    output_path: Path,
) -> int:
    await ingest_documents(session, embeddings, documents)

    results: list[CaseResult] = []
    for case in cases:
        results.append(
            await evaluate_case(
                case,
                agent_client=agent_client,
                judge_client=judge_client,
                session=session,
                embeddings=embeddings,
                agent_model=agent_model,
                judge_model=judge_model,
            )
        )

    write_report(results, judge_model=judge_model, output_path=output_path)

    passed = sum(1 for r in results if case_passed(r))
    total = len(results)
    print(f"{passed}/{total} cases passed. Report written to {output_path}")

    return 0 if passed == total else 1


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    documents = load_documents(args.documents)
    cases = load_cases(args.cases)

    await apply_schema(engine)
    await truncate_tables()

    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
    embeddings = EmbeddingService(voyage_client, model=settings.embedding_model)

    async with session_factory() as session:
        exit_code = await orchestrate(
            documents=documents,
            cases=cases,
            session=session,
            embeddings=embeddings,
            agent_client=anthropic_client,
            judge_client=anthropic_client,
            agent_model=settings.claude_model,
            judge_model=args.judge_model,
            output_path=args.output,
        )

    await truncate_tables()
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM-as-judge agent evaluation.")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
