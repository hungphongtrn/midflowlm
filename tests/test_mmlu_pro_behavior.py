import pytest

from src.eval.mmlu_pro_behavior import (
    build_behavior_record,
    create_mmlu_pro_prompt,
    extract_first_valid_answer,
    summarize_behavior_records,
)


@pytest.fixture
def mock_tokenizer():
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.bos_token = "<|begin_of_text|>"
    tokenizer.assistant_token = "<|im_start|>assistant"
    # apply_chat_template should return a simple prompt string for testing
    tokenizer.apply_chat_template.return_value = "<|im_start|>user\nAnswer the following multiple choice question. Respond with only the letter of the correct answer (A, B, C, etc.).\n\nQuestion: What is 2+2?\n\nOptions:\nA. 1\nB. 2\nC. 3\nD. 4\n\nAnswer:<|im_start|>assistant\n"
    return tokenizer


@pytest.fixture
def sample_question():
    return "What is 2+2?", ["A: 1", "B: 2", "C: 3", "D: 4"]


def test_extract_first_valid_answer_finds_letter_after_reasoning_text():
    valid_options = ["A", "B", "C", "D"]

    assert (
        extract_first_valid_answer("The answer is C because...", valid_options) == "C"
    )
    assert extract_first_valid_answer("I think (B) fits best.", valid_options) == "B"
    assert extract_first_valid_answer("No clue", valid_options) is None


def test_build_behavior_record_captures_first_token_and_answer_hit():
    question = {
        "question": "What is 2+2?",
        "options": ["1", "2", "3", "4"],
        "correct_answer": "D",
        "category": "math",
    }

    record = build_behavior_record(
        sample_index=7,
        question=question,
        prompt_text="prompt",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[10, 11],
        generated_text="The answer is D.",
        first_generated_text="The",
        model_name="trained_midblock",
        checkpoint_path="outputs/checkpoints/best.ckpt",
        num_steps=4,
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
        stopped_on_eos=False,
    )

    assert record["sample_index"] == 7
    assert record["first_generated_text"] == "The"
    assert record["first_answer_letter"] == "D"
    assert record["found_valid_answer"] is True
    assert record["generated_text"] == "The answer is D."


def test_summarize_behavior_records_counts_hits_and_common_outputs():
    records = [
        {
            "model_name": "trained_midblock",
            "num_steps": 1,
            "first_generated_text": "The",
            "first_answer_letter": None,
            "found_valid_answer": False,
            "generated_text": "The model keeps talking.",
        },
        {
            "model_name": "trained_midblock",
            "num_steps": 1,
            "first_generated_text": "Answer",
            "first_answer_letter": "C",
            "found_valid_answer": True,
            "generated_text": "Answer: C",
        },
        {
            "model_name": "trained_midblock",
            "num_steps": 1,
            "first_generated_text": "The",
            "first_answer_letter": "C",
            "found_valid_answer": True,
            "generated_text": "The answer is C.",
        },
    ]

    summary = summarize_behavior_records(records, example_count=1)
    group = summary[0]

    assert group["model_name"] == "trained_midblock"
    assert group["num_steps"] == 1
    assert group["sample_count"] == 3
    assert group["answer_hit_rate"] == 2 / 3
    assert group["top_first_generated_texts"][0] == ["The", 2]
    assert group["top_answer_letters"][0] == ["C", 2]
    assert len(group["example_completions"]) == 1
def test_create_mmlu_pro_prompt_default_behavior():
    """Test that default mode (no argument) produces thinking tags."""
    prompt = create_mmlu_pro_prompt(
        "What is 2+2?",
        ["A: 1", "B: 2", "C: 3", "D: 4"],
        mock_tokenizer,
    )
    # Default should include thinking tags (same as closed_think)
    assert "<think>" in prompt

def test_create_mmlu_pro_prompt_supports_closed_think_prefill():
    """Test that closed_think mode produces assistant prefill with thinking tags."""
    from unittest.mock import MagicMock
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "Question: What is 2+2?\n\nAnswer:"
    prompt = create_mmlu_pro_prompt(
        "What is 2+2?",
        ["A: 1", "B: 2", "C: 3", "D: 4"],
        mock_tok,
        prompt_behavior="closed_think",
    )
    expected = "<|im_start|>assistant\n<think>\n"
    assert expected in prompt, f"Expected {repr(expected)} in prompt"


def test_create_mmlu_pro_prompt_supports_stripped_prefill():
    """Test that stripped mode produces prompt without thinking tags."""
    from unittest.mock import MagicMock
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "Question: What is 2+2?\n\nAnswer:"
    prompt = create_mmlu_pro_prompt(
        "What is 2+2?",
        ["A: 1", "B: 2", "C: 3", "D: 4"],
        mock_tok,
        prompt_behavior="stripped",
    )
    assert "<think>" not in prompt


def test_create_mmlu_pro_prompt_default_behavior():
    """Test that default mode (no argument) produces thinking tags."""
    from unittest.mock import MagicMock
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "Question: What is 2+2?\n\nAnswer:"
    prompt = create_mmlu_pro_prompt(
        "What is 2+2?",
        ["A: 1", "B: 2", "C: 3", "D: 4"],
        mock_tok,
    )
    # Default should include thinking tags (same as closed_think)
    assert "<think>" in prompt
