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
