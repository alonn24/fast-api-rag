# Agent Evaluation via LLM-as-Judge — Design

## Goal

Build a repeatable evaluation harness for the `POST /query` agentic retrieval
endpoint (`run_agent_query` in `app/agent/tool_loop.py`), so changes to the
system prompt, retrieval, or model choice can be checked for quality
regressions before they ship. This is a regression safety net, not a one-off
report — it's meant to be re-run as the agent evolves and eventually gated in
CI.

## What gets judged

For each test case, an LLM judge scores three independent binary
(pass/fail) dimensions, each with a short rationale:

- **correctness** — does the agent's final answer match the substance of a
  known reference answer?
- **faithfulness** — is every claim in the agent's answer grounded in the
  chunks `search_documents` actually retrieved (no unsupported claims /
  hallucination beyond what was retrieved)?
- **retrieval_relevance** — did `search_documents` surface chunks from the
  expected source document(s) at all, independent of what the final answer
  did with them?

Binary pass/fail (not a 1–5 scale) was chosen so results threshold cleanly
for CI gating without picking an arbitrary numeric cutoff.

## Layout

```
app/eval/
  __init__.py
  fixtures/
    documents.json   # seed docs: [{title, source, content, metadata}]
    cases.json        # [{id, question, expected_answer, expected_source_titles}]
  judge.py            # LLM-as-judge scoring
  report.py           # HTML report rendering
  run_eval.py          # CLI entrypoint / orchestration
```

`documents.json` and `cases.json` are hand-written and checked into git,
grown over time as new cases / failure modes are identified. The eval is
self-contained: it ingests its own fixture documents rather than assuming
the DB is pre-populated, so it's reproducible in any environment including
CI.

## Runner flow (`run_eval.py`)

1. Load `fixtures/documents.json` and `fixtures/cases.json` (paths
   overridable via `--documents` / `--cases` for future test-set variants).
2. Connect to the dockerized Postgres/pgvector DB — the same instance
   `tests/conftest.py` uses — and truncate the relevant tables.
3. Ingest the fixture documents through the existing ingestion logic behind
   `POST /documents` (called directly as a function, not over HTTP).
4. For each case, call `run_agent_query(...)` — using the real
   `EmbeddingService` and `anthropic.Anthropic` client, hitting real Voyage
   and Anthropic APIs — to get an `AgentAnswer` (answer text + retrieved
   sources).
5. Pass `{question, expected_answer, expected_source_titles, agent_answer,
   retrieved_sources}` to the judge for scoring.
6. Collect per-case, per-dimension verdicts and rationales.
7. Truncate tables again (cleanup).
8. Render the HTML report to disk, print a one-line summary to stdout, and
   exit `1` if any case failed any dimension (or errored), else `0`.

CLI flags:
- `--judge-model` (default `claude-sonnet-5`)
- `--output` (report path, default `eval_report.html`)
- `--cases`, `--documents` (override fixture paths)

## Judge (`judge.py`)

One judge call per case, using Anthropic tool-use with a `submit_judgment`
tool so the output is reliably parseable rather than free text. The judge
receives the question, reference answer, agent's actual answer, retrieved
chunks (title + content), and expected source titles, and returns the three
pass/fail verdicts with rationales.

The judge runs on a separate model (`--judge-model`, default
`claude-sonnet-5`) independent of `settings.claude_model` (the agent's own
model), to avoid self-grading bias by default. The judge model is
configurable via the CLI flag for future experimentation.

## Report (`report.py`)

A single self-contained static HTML file (no server, no external assets):

- Header: overall pass rate, pass rate per dimension, judge model used,
  timestamp.
- One row per case: question, pass/fail badge per dimension, expandable
  detail showing expected vs. actual answer, retrieved sources, and the
  judge's rationale per verdict.
- Failing cases sorted to the top.

The report is a plain file written to disk (e.g. suitable as a CI
artifact) — not published or served anywhere.

## Error handling & scope boundaries

- No retries or silent fallbacks for judge API failures. If a judge call
  errors, that case is reported as an error (not skipped, not defaulted to
  pass), and the run still exits nonzero.
- This eval costs real API money per run (Anthropic + Voyage calls) and is
  non-deterministic by nature of using real models — intentionally kept
  out of the default `pytest` run. It's invoked manually via
  `uv run python -m app.eval.run_eval`, and later via a separate CI job.
- Exit criteria: nonzero if *any* case fails *any* dimension. No
  pass-rate threshold — every case in the fixture set is expected to pass;
  cases that can't reliably pass should be fixed or removed rather than
  tolerated via a threshold.
- CI workflow wiring (GitHub Actions file, secrets configuration) is out of
  scope for this design. This design produces the runnable script and
  fixtures; hooking it into CI is a followup once the fixture set has a
  meaningful number of real cases.

## Out of scope (for now)

- Generating Q&A cases automatically from ingested documents.
- Tracking eval results over time / trend dashboards.
- Judging tool-call efficiency or agent process behavior beyond the three
  dimensions above.
