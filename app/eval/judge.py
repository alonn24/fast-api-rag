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
