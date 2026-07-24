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
