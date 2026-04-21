from src.catalog.answer_format_prompts import (
    AIME_PROMPT,
    DROP_PROMPT,
    GSM8K_PROMPT,
    HOTPOTQA_PROMPT,
    HUMANEVAL_PROMPT,
    LIVECODE_PROMPT,
    MATH_PROMPT,
    MBPP_PROMPT,
    MUSIQUEANS_PROMPT,
)


ANSWER_FORMAT_REQUIREMENTS = {
    "MATH": MATH_PROMPT,
    "MATHDEMO": MATH_PROMPT,
    "HumanEval": HUMANEVAL_PROMPT,
    "DROP": DROP_PROMPT,
    "GSM8K": GSM8K_PROMPT,
    "HotpotQA": HOTPOTQA_PROMPT,
    "MBPP": MBPP_PROMPT,
    "AIME": AIME_PROMPT,
    "LiveCode": LIVECODE_PROMPT,
    "MusiqueAns": MUSIQUEANS_PROMPT,
}


def get_answer_format_requirements(dataset: str | None) -> str:
    if not dataset:
        return ""
    return ANSWER_FORMAT_REQUIREMENTS.get(dataset, "")
