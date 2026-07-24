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
