# Agent Evaluation via LLM-as-Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, re-runnable evaluation harness for `run_agent_query` (`app/agent/tool_loop.py`) that scores agent answers on correctness, faithfulness, and retrieval_relevance using an independent LLM judge, and renders a static HTML report.

**Architecture:** A new `app/eval/` package with hand-written JSON fixtures (documents + cases), a `judge.py` module that makes one forced tool-use Anthropic call per case to get structured pass/fail verdicts, a `report.py` module that turns verdicts into a single static HTML file, and a `run_eval.py` CLI that wires ingestion → agent run → judging → report together. It reuses the existing `create_document` route function and `run_agent_query` directly rather than duplicating ingestion/agent logic. Design source: `docs/superpowers/specs/2026-07-24-agent-eval-llm-judge-design.md`.

**Tech Stack:** Python 3.11, FastAPI/SQLAlchemy async, `anthropic` SDK (tool-use with forced `tool_choice`), `voyageai`, pytest/pytest-asyncio, stdlib `json`/`html`/`argparse` (no new dependencies).

## Global Constraints

- Judge scores exactly three independent binary (pass/fail) dimensions per case: `correctness`, `faithfulness`, `retrieval_relevance` — each with a rationale string. No numeric scale.
- Judge runs on a separate, configurable model (`--judge-model`, default `claude-sonnet-5`), independent of `settings.claude_model`.
- Judge output must come from a forced Anthropic tool-use call (a `submit_judgment` tool) so it's reliably parseable — no free-text parsing.
- Fixtures (`app/eval/fixtures/documents.json`, `app/eval/fixtures/cases.json`) are hand-written, checked into git, and overridable via `--documents` / `--cases` CLI flags.
- The eval ingests its own fixture documents (via the existing `POST /documents` logic, called as a function) rather than assuming a pre-populated DB, against the same dockerized Postgres/pgvector instance `tests/conftest.py` uses. Tables are truncated before and after the run.
- No retries or silent fallbacks on judge API failure — a failing judge call marks that case as errored (not skipped, not defaulted to pass), and the run still exits nonzero.
- Exit code: `0` only if every case passes every dimension with no errors; `1` otherwise. No pass-rate threshold.
- The report is a single self-contained static HTML file (no server, no external JS/CSS/font/image assets) written to disk, default path `eval_report.html`, overridable via `--output`.
- Report contents: header with overall pass rate, pass rate per dimension, judge model, timestamp; one row per case with pass/fail badges per dimension and an expandable detail (expected vs. actual answer, retrieved sources, judge rationale per verdict); failing cases sorted to the top.
- This harness is intentionally excluded from the default `pytest` run (it costs real Anthropic + Voyage API money and is non-deterministic) — invoked manually via `uv run python -m app.eval.run_eval`. Only the pure/mockable pieces (judge parsing, report rendering, fixture loading, DB ingestion with mocked embeddings, mocked orchestration) get automated pytest coverage; a full real-API run is manual/documented, not asserted by tests.
- CI wiring is out of scope for this plan.

---

### Task 1: Package scaffold and fixtures

**Files:**
- Create: `app/eval/__init__.py`
- Create: `app/eval/fixtures/documents.json`
- Create: `app/eval/fixtures/cases.json`
- Test: `tests/test_eval_fixtures.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `app/eval/fixtures/documents.json` — JSON array of `{title: str, source: str, content: str, metadata: dict}`. `app/eval/fixtures/cases.json` — JSON array of `{id: str, question: str, expected_answer: str, expected_source_titles: list[str]}`. Later tasks (`run_eval.py`) load these two files by path. `app.eval` is an empty, importable package.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_fixtures.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_fixtures.py -v`
Expected: FAIL — `FileNotFoundError` (no `app/eval/fixtures/documents.json` yet).

- [ ] **Step 3: Create the package init and fixture files**

```python
# app/eval/__init__.py
```

```json
// app/eval/fixtures/documents.json
[
  {
    "title": "Vacation Policy",
    "source": "hr-handbook",
    "content": "Full-time employees accrue 15 days of paid vacation per year, accrued monthly at 1.25 days per month. Unused vacation days roll over up to a maximum of 5 days into the next calendar year; any excess beyond 5 days is forfeited at year end. Employees must submit vacation requests through the HR portal at least two weeks in advance for any request longer than 3 consecutive days.",
    "metadata": {"category": "hr"}
  },
  {
    "title": "Remote Work Policy",
    "source": "hr-handbook",
    "content": "Employees may work remotely up to 3 days per week with their manager's approval. Fully remote arrangements (more than 3 days per week) require VP-level sign-off and are reviewed annually. Remote employees must be reachable during core hours of 10am-4pm in their local timezone and must attend the quarterly in-person team offsite.",
    "metadata": {"category": "hr"}
  },
  {
    "title": "Expense Reimbursement Policy",
    "source": "hr-handbook",
    "content": "Employees can be reimbursed for business-related expenses including travel, client meals, and approved software subscriptions. Expense reports must be submitted within 30 days of the expense via the Expensify system, with an itemized receipt required for any single expense over $25. Reimbursements are processed within 10 business days of manager approval.",
    "metadata": {"category": "hr"}
  }
]
```

```json
// app/eval/fixtures/cases.json
[
  {
    "id": "vacation-accrual",
    "question": "How many vacation days do full-time employees accrue per year, and how does rollover work?",
    "expected_answer": "Full-time employees accrue 15 vacation days per year (1.25 days per month). Up to 5 unused days roll over into the next calendar year; anything beyond that is forfeited.",
    "expected_source_titles": ["Vacation Policy"]
  },
  {
    "id": "remote-days-per-week",
    "question": "How many days per week can an employee work remotely without needing VP-level approval?",
    "expected_answer": "Up to 3 days per week with manager approval. Going fully remote (more than 3 days per week) requires VP-level sign-off.",
    "expected_source_titles": ["Remote Work Policy"]
  },
  {
    "id": "expense-receipt-threshold",
    "question": "Above what dollar amount do I need an itemized receipt for an expense, and how long do I have to submit an expense report?",
    "expected_answer": "Itemized receipts are required for any single expense over $25, and expense reports must be submitted within 30 days of the expense.",
    "expected_source_titles": ["Expense Reimbursement Policy"]
  }
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_fixtures.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/__init__.py app/eval/fixtures/documents.json app/eval/fixtures/cases.json tests/test_eval_fixtures.py
git commit -m "feat: add app/eval package scaffold and eval fixtures"
```

---

### Task 2: Judge module (`judge.py`)

**Files:**
- Create: `app/eval/judge.py`
- Test: `tests/test_eval_judge.py`

**Interfaces:**
- Consumes: `anthropic.Anthropic` client (same type used in `app/agent/tool_loop.py`). No dependency on Task 1's fixtures.
- Produces:
  - `class JudgeError(Exception)`
  - `@dataclass class Verdict: passed: bool; rationale: str`
  - `@dataclass class JudgmentResult: correctness: Verdict; faithfulness: Verdict; retrieval_relevance: Verdict`
  - `def judge_case(*, client: anthropic.Anthropic, model: str, question: str, expected_answer: str, expected_source_titles: list[str], agent_answer: str, retrieved_sources: list[dict]) -> JudgmentResult` — raises `JudgeError` on any API failure or malformed/missing tool output. `retrieved_sources` items are `{"document_title": str, "content": str}`.
  - These are consumed by Task 5's `evaluate_case`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_judge.py
from types import SimpleNamespace
from unittest.mock import Mock

import anthropic
import pytest

from app.eval.judge import JudgeError, judge_case


def _tool_use_response(input_data: dict):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="toolu_1", name="submit_judgment", input=input_data)],
        stop_reason="tool_use",
    )


VALID_INPUT = {
    "correctness": {"verdict": "pass", "rationale": "Matches the reference answer."},
    "faithfulness": {"verdict": "pass", "rationale": "All claims are grounded in retrieved chunks."},
    "retrieval_relevance": {"verdict": "fail", "rationale": "Retrieved chunks are from the wrong document."},
}


def test_judge_case_parses_tool_output_into_judgment_result():
    fake_client = Mock()
    fake_client.messages.create = Mock(return_value=_tool_use_response(VALID_INPUT))

    result = judge_case(
        client=fake_client,
        model="claude-sonnet-5",
        question="How many vacation days per year?",
        expected_answer="15 days.",
        expected_source_titles=["Vacation Policy"],
        agent_answer="Employees get 15 vacation days per year.",
        retrieved_sources=[{"document_title": "Vacation Policy", "content": "15 days of paid vacation."}],
    )

    assert result.correctness.passed is True
    assert result.correctness.rationale == "Matches the reference answer."
    assert result.faithfulness.passed is True
    assert result.retrieval_relevance.passed is False
    assert result.retrieval_relevance.rationale == "Retrieved chunks are from the wrong document."

    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "submit_judgment"}
    assert call_kwargs["tools"][0]["name"] == "submit_judgment"


def test_judge_case_raises_judge_error_on_api_failure():
    fake_client = Mock()
    fake_client.messages.create = Mock(
        side_effect=anthropic.APIConnectionError(request=Mock())
    )

    with pytest.raises(JudgeError):
        judge_case(
            client=fake_client,
            model="claude-sonnet-5",
            question="q",
            expected_answer="a",
            expected_source_titles=["Doc"],
            agent_answer="a",
            retrieved_sources=[],
        )


def test_judge_case_raises_judge_error_when_no_tool_use_block():
    fake_client = Mock()
    fake_client.messages.create = Mock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="I refuse to use the tool.")],
            stop_reason="end_turn",
        )
    )

    with pytest.raises(JudgeError):
        judge_case(
            client=fake_client,
            model="claude-sonnet-5",
            question="q",
            expected_answer="a",
            expected_source_titles=["Doc"],
            agent_answer="a",
            retrieved_sources=[],
        )


def test_judge_case_raises_judge_error_on_malformed_tool_input():
    fake_client = Mock()
    fake_client.messages.create = Mock(return_value=_tool_use_response({"correctness": {"verdict": "pass"}}))

    with pytest.raises(JudgeError):
        judge_case(
            client=fake_client,
            model="claude-sonnet-5",
            question="q",
            expected_answer="a",
            expected_source_titles=["Doc"],
            agent_answer="a",
            retrieved_sources=[],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.judge'`

- [ ] **Step 3: Write the implementation**

```python
# app/eval/judge.py
import json
from dataclasses import dataclass

import anthropic

SUBMIT_JUDGMENT_TOOL = {
    "name": "submit_judgment",
    "description": (
        "Submit pass/fail verdicts with a short rationale for each of the three "
        "evaluation dimensions: correctness, faithfulness, retrieval_relevance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            dimension: {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["pass", "fail"]},
                    "rationale": {"type": "string"},
                },
                "required": ["verdict", "rationale"],
            }
            for dimension in ("correctness", "faithfulness", "retrieval_relevance")
        },
        "required": ["correctness", "faithfulness", "retrieval_relevance"],
    },
}

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating the output of a retrieval-augmented Q&A agent. "
    "You will be given a question, a reference answer, the agent's actual answer, the source "
    "chunks the agent's retrieval tool returned, and the document titles the answer is expected "
    "to draw from. Score three independent binary dimensions using the submit_judgment tool:\n\n"
    "- correctness: does the agent's answer match the substance of the reference answer?\n"
    "- faithfulness: is every claim in the agent's answer grounded in the retrieved chunks, with "
    "no unsupported claims or hallucination beyond what was retrieved?\n"
    "- retrieval_relevance: did retrieval surface chunks from the expected source document(s) at "
    "all, regardless of what the final answer did with them?\n\n"
    "Each dimension is pass or fail, never anything in between. Always call submit_judgment "
    "exactly once with all three dimensions filled in."
)


class JudgeError(Exception):
    pass


@dataclass
class Verdict:
    passed: bool
    rationale: str


@dataclass
class JudgmentResult:
    correctness: Verdict
    faithfulness: Verdict
    retrieval_relevance: Verdict


def _build_user_content(
    *,
    question: str,
    expected_answer: str,
    expected_source_titles: list[str],
    agent_answer: str,
    retrieved_sources: list[dict],
) -> str:
    return json.dumps(
        {
            "question": question,
            "reference_answer": expected_answer,
            "expected_source_titles": expected_source_titles,
            "agent_answer": agent_answer,
            "retrieved_sources": retrieved_sources,
        },
        indent=2,
    )


def _verdict_from(dimension_data: dict) -> Verdict:
    return Verdict(passed=dimension_data["verdict"] == "pass", rationale=dimension_data["rationale"])


def judge_case(
    *,
    client: anthropic.Anthropic,
    model: str,
    question: str,
    expected_answer: str,
    expected_source_titles: list[str],
    agent_answer: str,
    retrieved_sources: list[dict],
) -> JudgmentResult:
    user_content = _build_user_content(
        question=question,
        expected_answer=expected_answer,
        expected_source_titles=expected_source_titles,
        agent_answer=agent_answer,
        retrieved_sources=retrieved_sources,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=JUDGE_SYSTEM_PROMPT,
            tools=[SUBMIT_JUDGMENT_TOOL],
            tool_choice={"type": "tool", "name": "submit_judgment"},
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APIError as exc:
        raise JudgeError(f"judge API call failed: {exc}") from exc

    tool_block = next((block for block in response.content if block.type == "tool_use"), None)
    if tool_block is None:
        raise JudgeError("judge response did not include a submit_judgment tool_use block")

    data = tool_block.input
    try:
        return JudgmentResult(
            correctness=_verdict_from(data["correctness"]),
            faithfulness=_verdict_from(data["faithfulness"]),
            retrieval_relevance=_verdict_from(data["retrieval_relevance"]),
        )
    except (KeyError, TypeError) as exc:
        raise JudgeError(f"malformed submit_judgment input: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_judge.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/judge.py tests/test_eval_judge.py
git commit -m "feat: add LLM-as-judge scoring module"
```

---

### Task 3: Report module (`report.py`)

**Files:**
- Create: `app/eval/report.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: `Verdict`, `JudgmentResult` from `app.eval.judge` (Task 2).
- Produces:
  - `@dataclass class CaseResult: case_id: str; question: str; expected_answer: str; agent_answer: str | None; retrieved_sources: list[dict]; expected_source_titles: list[str]; judgment: JudgmentResult | None; error: str | None`
  - `def case_passed(result: CaseResult) -> bool`
  - `def render_report(results: list[CaseResult], *, judge_model: str, generated_at: datetime) -> str` — returns the full HTML document as a string.
  - `def write_report(results: list[CaseResult], *, judge_model: str, output_path: Path) -> None` — writes `render_report(...)` to `output_path` using the current UTC time.
  - These are consumed by Task 5's `orchestrate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_report.py
from datetime import datetime, timezone
from pathlib import Path

from app.eval.judge import JudgmentResult, Verdict
from app.eval.report import CaseResult, case_passed, render_report, write_report

ALL_PASS = JudgmentResult(
    correctness=Verdict(passed=True, rationale="Matches."),
    faithfulness=Verdict(passed=True, rationale="Grounded."),
    retrieval_relevance=Verdict(passed=True, rationale="Right doc."),
)

ONE_FAIL = JudgmentResult(
    correctness=Verdict(passed=True, rationale="Matches."),
    faithfulness=Verdict(passed=False, rationale="Unsupported claim about pricing."),
    retrieval_relevance=Verdict(passed=True, rationale="Right doc."),
)


def _result(case_id, judgment=ALL_PASS, error=None):
    return CaseResult(
        case_id=case_id,
        question=f"Question for {case_id}?",
        expected_answer="Expected answer text.",
        agent_answer="Agent answer text." if error is None else None,
        retrieved_sources=[{"document_title": "Vacation Policy", "content": "15 days of vacation."}],
        expected_source_titles=["Vacation Policy"],
        judgment=judgment,
        error=error,
    )


def test_case_passed_true_when_all_dimensions_pass():
    assert case_passed(_result("vacation-accrual")) is True


def test_case_passed_false_when_any_dimension_fails():
    assert case_passed(_result("remote-days-per-week", judgment=ONE_FAIL)) is False


def test_case_passed_false_when_case_errored():
    assert case_passed(_result("expense-receipt-threshold", judgment=None, error="judge API call failed: timeout")) is False


def test_render_report_includes_header_and_case_details():
    results = [_result("vacation-accrual"), _result("remote-days-per-week", judgment=ONE_FAIL)]
    generated_at = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)

    html = render_report(results, judge_model="claude-sonnet-5", generated_at=generated_at)

    assert "1/2" in html  # overall pass rate: 1 of 2 cases fully passed
    assert "claude-sonnet-5" in html
    assert "2026-07-24" in html
    assert "Question for vacation-accrual?" in html
    assert "Question for remote-days-per-week?" in html
    assert "Unsupported claim about pricing." in html
    assert "Vacation Policy" in html


def test_render_report_sorts_failing_cases_to_top():
    results = [_result("passing-case"), _result("failing-case", judgment=ONE_FAIL)]
    generated_at = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)

    html = render_report(results, judge_model="claude-sonnet-5", generated_at=generated_at)

    assert html.index("Question for failing-case?") < html.index("Question for passing-case?")


def test_render_report_escapes_html_in_case_content():
    results = [
        CaseResult(
            case_id="xss-case",
            question="<script>alert(1)</script>?",
            expected_answer="expected",
            agent_answer="actual",
            retrieved_sources=[],
            expected_source_titles=["Doc"],
            judgment=ALL_PASS,
            error=None,
        )
    ]
    generated_at = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)

    html = render_report(results, judge_model="claude-sonnet-5", generated_at=generated_at)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_report_writes_file_to_disk(tmp_path):
    output_path = tmp_path / "report.html"
    results = [_result("vacation-accrual")]

    write_report(results, judge_model="claude-sonnet-5", output_path=output_path)

    assert output_path.exists()
    assert "claude-sonnet-5" in output_path.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.report'`

- [ ] **Step 3: Write the implementation**

```python
# app/eval/report.py
import html as html_lib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.eval.judge import JudgmentResult

DIMENSIONS = ("correctness", "faithfulness", "retrieval_relevance")


@dataclass
class CaseResult:
    case_id: str
    question: str
    expected_answer: str
    agent_answer: str | None
    retrieved_sources: list[dict]
    expected_source_titles: list[str]
    judgment: JudgmentResult | None
    error: str | None


def case_passed(result: CaseResult) -> bool:
    if result.error is not None or result.judgment is None:
        return False
    return all(getattr(result.judgment, dimension).passed for dimension in DIMENSIONS)


def _badge(passed: bool) -> str:
    label, css_class = ("PASS", "pass") if passed else ("FAIL", "fail")
    return f'<span class="badge {css_class}">{label}</span>'


def _dimension_rows(result: CaseResult) -> str:
    if result.judgment is None:
        return f'<p class="error">Error: {html_lib.escape(result.error or "unknown error")}</p>'

    rows = []
    for dimension in DIMENSIONS:
        verdict = getattr(result.judgment, dimension)
        rows.append(
            f"<tr><td>{html_lib.escape(dimension)}</td>"
            f"<td>{_badge(verdict.passed)}</td>"
            f"<td>{html_lib.escape(verdict.rationale)}</td></tr>"
        )
    return f"<table class='dimensions'><thead><tr><th>Dimension</th><th>Verdict</th><th>Rationale</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _sources_list(sources: list[dict]) -> str:
    if not sources:
        return "<p><em>No sources retrieved.</em></p>"
    items = "".join(
        f"<li><strong>{html_lib.escape(s['document_title'])}</strong>: {html_lib.escape(s['content'])}</li>"
        for s in sources
    )
    return f"<ul class='sources'>{items}</ul>"

def _case_section(result: CaseResult) -> str:
    passed = case_passed(result)
    overall_badge = _badge(passed)
    summary_badges = "".join(
        _badge(getattr(result.judgment, d).passed) if result.judgment else _badge(False) for d in DIMENSIONS
    )

    return f"""
    <details class="case {'passed' if passed else 'failed'}">
      <summary>{overall_badge} {summary_badges} {html_lib.escape(result.question)}</summary>
      <div class="case-body">
        <h4>Expected answer</h4>
        <p>{html_lib.escape(result.expected_answer)}</p>
        <h4>Agent answer</h4>
        <p>{html_lib.escape(result.agent_answer) if result.agent_answer else '<em>none (case errored)</em>'}</p>
        <h4>Expected source titles</h4>
        <p>{html_lib.escape(', '.join(result.expected_source_titles))}</p>
        <h4>Retrieved sources</h4>
        {_sources_list(result.retrieved_sources)}
        <h4>Judge verdicts</h4>
        {_dimension_rows(result)}
      </div>
    </details>
    """


STYLE = """
body { font-family: -apple-system, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 0.25rem; font-weight: bold; font-size: 0.85rem; }
.badge.pass { background: #d4edda; color: #155724; }
.badge.fail { background: #f8d7da; color: #721c24; }
.case { border: 1px solid #ddd; border-radius: 0.25rem; margin-bottom: 0.5rem; padding: 0.5rem 0.75rem; }
.case.failed { border-color: #dc3545; }
.case summary { cursor: pointer; font-weight: 600; }
table.dimensions { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
table.dimensions th, table.dimensions td { text-align: left; border-bottom: 1px solid #eee; padding: 0.25rem 0.5rem; }
.error { color: #721c24; font-weight: bold; }
header.summary { margin-bottom: 1.5rem; }
"""


def render_report(results: list[CaseResult], *, judge_model: str, generated_at: datetime) -> str:
    total = len(results)
    passed_count = sum(1 for r in results if case_passed(r))

    dimension_pass_rates = []
    for dimension in DIMENSIONS:
        scored = [r for r in results if r.judgment is not None]
        dim_passed = sum(1 for r in scored if getattr(r.judgment, dimension).passed)
        dimension_pass_rates.append((dimension, dim_passed, len(scored)))

    ordered_results = sorted(results, key=lambda r: case_passed(r))  # failing (False) sorts first

    dimension_summary = "".join(
        f"<li>{html_lib.escape(dim)}: {dim_passed}/{dim_total}</li>" for dim, dim_passed, dim_total in dimension_pass_rates
    )

    cases_html = "".join(_case_section(r) for r in ordered_results)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Agent Eval Report</title>
<style>{STYLE}</style>
</head>
<body>
<header class="summary">
  <h1>Agent Evaluation Report</h1>
  <p>Overall: <strong>{passed_count}/{total}</strong> cases passed</p>
  <ul>{dimension_summary}</ul>
  <p>Judge model: <code>{html_lib.escape(judge_model)}</code></p>
  <p>Generated at: {generated_at.isoformat()}</p>
</header>
<main>
{cases_html}
</main>
</body>
</html>
"""


def write_report(results: list[CaseResult], *, judge_model: str, output_path: Path) -> None:
    html_doc = render_report(results, judge_model=judge_model, generated_at=datetime.now(timezone.utc))
    output_path.write_text(html_doc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_report.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/report.py tests/test_eval_report.py
git commit -m "feat: add static HTML report renderer for eval results"
```

---

### Task 4: Fixture loading and DB ingestion (`run_eval.py`, part 1)

**Files:**
- Create: `app/eval/run_eval.py`
- Test: `tests/test_eval_run_eval.py`

**Interfaces:**
- Consumes: `app.api.documents.create_document(body: DocumentCreateRequest, session: AsyncSession, embeddings: EmbeddingService) -> DocumentCreateResponse` (existing, `app/api/documents.py:19`). `app.schemas.DocumentCreateRequest` (existing). `app.services.embeddings.EmbeddingService` (existing).
- Produces:
  - `def load_documents(path: Path) -> list[dict]`
  - `def load_cases(path: Path) -> list[dict]`
  - `async def truncate_tables() -> None` — truncates `chunks, documents` on the module-level `engine` from `app.db.session`.
  - `async def ingest_documents(session: AsyncSession, embeddings: EmbeddingService, documents: list[dict]) -> None`
  - These are consumed by Task 5's `orchestrate` and `main_async`.

This task's tests exercise `ingest_documents` against the real dockerized Postgres (same DB the rest of the test suite already uses via `tests/conftest.py`), with a mocked `EmbeddingService` so no real Voyage API calls happen — consistent with `tests/test_documents_api.py`'s existing pattern.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_run_eval.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_run_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.run_eval'`

- [ ] **Step 3: Write the implementation**

```python
# app/eval/run_eval.py
import json
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import create_document
from app.db.session import engine
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose up -d && uv run pytest tests/test_eval_run_eval.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/run_eval.py tests/test_eval_run_eval.py
git commit -m "feat: add eval fixture loading and DB ingestion"
```

---

### Task 5: Case evaluation, orchestration, and CLI (`run_eval.py`, part 2)

**Files:**
- Modify: `app/eval/run_eval.py`
- Modify: `tests/test_eval_run_eval.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `run_agent_query(question, *, client, session, embeddings, model, max_iterations=5) -> AgentAnswer` (`app/agent/tool_loop.py:41`, existing). `AgentAnswer.sources: list[SearchResult]` where `SearchResult` has `.document_title: str` and `.content: str` (`app/services/search.py`, existing). `judge_case(...) -> JudgmentResult` and `JudgeError` (Task 2, `app/eval/judge.py`). `CaseResult`, `case_passed`, `write_report` (Task 3, `app/eval/report.py`). `load_documents`, `load_cases`, `truncate_tables`, `ingest_documents` (Task 4, `app/eval/run_eval.py`). `app.config.get_settings() -> Settings` with `.anthropic_api_key`, `.voyage_api_key`, `.embedding_model`, `.claude_model` (existing, `app/config.py`). `app.db.session.session_factory` (existing, `app/db/session.py:9`).
- Produces:
  - `async def evaluate_case(case: dict, *, agent_client, judge_client, session, embeddings, agent_model: str, judge_model: str) -> CaseResult`
  - `async def orchestrate(*, documents: list[dict], cases: list[dict], session, embeddings, agent_client, judge_client, agent_model: str, judge_model: str, output_path: Path) -> int` — returns the process exit code.
  - `async def main_async(args: "argparse.Namespace") -> int`
  - `def parse_args(argv: list[str] | None = None) -> "argparse.Namespace"`
  - `def main(argv: list[str] | None = None) -> int`
  - `if __name__ == "__main__": sys.exit(main())`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_run_eval.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_run_eval.py -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_case' from 'app.eval.run_eval'`

- [ ] **Step 3: Extend the implementation**

Append to `app/eval/run_eval.py` (add these imports to the existing import block at the top, then add the new functions below the existing ones):

```python
import argparse
import asyncio
import sys

import anthropic
import voyageai
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tool_loop import run_agent_query
from app.config import get_settings
from app.db.session import session_factory
from app.eval.judge import JudgeError, judge_case
from app.eval.report import CaseResult, case_passed, write_report
```

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose up -d && uv run pytest tests/test_eval_run_eval.py -v`
Expected: PASS (11 tests)

Then run the full suite to confirm nothing else broke:

Run: `uv run pytest -v`
Expected: PASS (all tests, including pre-existing ones)

- [ ] **Step 5: Document the manual run command in the README**

In `README.md`, after the existing `## Run tests` section, add:

```markdown
## Run the agent evaluation

`app/eval/` is an LLM-as-judge evaluation harness for the `/query` agent. It ingests its
own fixture documents (`app/eval/fixtures/`), runs each fixture question through the real
agent, and scores the answer with an independent judge model. It costs real Anthropic +
Voyage API calls and is intentionally excluded from the default `pytest` run.

```bash
docker compose up -d
uv run python -m app.eval.run_eval
```

Writes `eval_report.html` (override with `--output`) and exits nonzero if any case fails
any of the three judged dimensions (correctness, faithfulness, retrieval_relevance). Use
`--judge-model`, `--documents`, `--cases` to override defaults.
```

- [ ] **Step 6: Commit**

```bash
git add app/eval/run_eval.py tests/test_eval_run_eval.py README.md
git commit -m "feat: add eval orchestration, CLI entrypoint, and README docs"
```

---

## Self-Review Notes

- **Spec coverage:** three judged dimensions (Task 2) · fixture layout and hand-written JSON (Task 1) · runner flow steps 1-8 including truncate-before/after, ingestion via `create_document`, real `run_agent_query`, judge call, report render, exit code (Tasks 4-5) · CLI flags `--judge-model`/`--output`/`--cases`/`--documents` (Task 5) · judge uses forced tool-use `submit_judgment` on a separate configurable model (Task 2) · report is a single static self-contained HTML file with header, per-case rows, expandable detail, failing-first sort (Task 3) · no retries/fallback on judge failure, case marked errored (Task 2 raises `JudgeError`, Task 5 catches and records) · exit nonzero on any failure, no threshold (Task 5 `orchestrate`) · excluded from default pytest run / manual invocation documented (README update in Task 5, and only pure/mocked pieces are asserted by tests) · CI wiring out of scope (not addressed, as specified).
- **Placeholder scan:** no TBD/TODO markers; every step has runnable code and exact commands.
- **Type consistency:** `CaseResult`, `JudgmentResult`, `Verdict` field names and types are identical across Task 3's definition and Task 5's construction sites. `evaluate_case` and `orchestrate` keyword names match between their Task 5 definitions and the test calls. `ingest_documents(session, embeddings, documents)` signature (Task 4) matches its Task 5 call sites (positional) and Task 4/5 test calls.
