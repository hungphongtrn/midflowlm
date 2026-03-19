import yaml
import pytest
from unittest.mock import MagicMock


def test_mixed_corpus_config_uses_new_loader():
    cfg = yaml.safe_load(open("configs/v0_mixed_corpus.yaml"))
    assert cfg["data"]["loader"] == "mixture"
    assert len(cfg["data"]["mixture_components"]) >= 5


def test_mixed_corpus_config_keeps_hidden_state_only_loss():
    cfg = yaml.safe_load(open("configs/v0_mixed_corpus.yaml"))
    assert cfg["loss"]["kl_weight"] == 0.0
    assert cfg["loss"]["ce_weight"] == 0.0


def test_plain_text_component_returns_raw_text():
    from src.data.mixed_corpus import format_example_text

    text = format_example_text(
        {"text": "hello world"}, {"format_type": "plain_text", "text_field": "text"}
    )
    assert text == "hello world"


def test_chat_messages_component_uses_chat_template():
    from src.data.mixed_corpus import format_example_text

    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = (
        "{% for message in messages %}{{ message.content }}{% endfor %}"
    )
    mock_tokenizer.apply_chat_template.return_value = "Hi Hello"

    rendered = format_example_text(
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        },
        {"format_type": "chat_messages", "messages_field": "messages"},
        tokenizer=mock_tokenizer,
    )
    assert "Hi" in rendered
    mock_tokenizer.apply_chat_template.assert_called_once()


def test_chat_messages_fallback_without_chat_template():
    from src.data.mixed_corpus import format_example_text

    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None

    rendered = format_example_text(
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        },
        {"format_type": "chat_messages", "messages_field": "messages"},
        tokenizer=mock_tokenizer,
    )
    assert "user: Hi" in rendered
    assert "assistant: Hello" in rendered


def test_mcq_component_renders_answer_letter():
    from src.data.mixed_corpus import format_example_text

    example = {
        "question": "What is the capital of France?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["London", "Paris", "Berlin", "Madrid"],
        },
        "answerKey": "B",
    }
    component_cfg = {
        "format_type": "mcq_choices",
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answerKey",
    }

    rendered = format_example_text(example, component_cfg)
    assert "Question: What is the capital of France?" in rendered
    assert "A. London" in rendered
    assert "B. Paris" in rendered
    assert "Answer: B" in rendered


def test_mcq_component_renders_all_options():
    from src.data.mixed_corpus import format_example_text

    example = {
        "question": "Which planet is closest to the Sun?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["Venus", "Mercury", "Mars", "Jupiter"],
        },
        "answerKey": "B",
    }
    component_cfg = {
        "format_type": "mcq_choices",
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answerKey",
    }

    rendered = format_example_text(example, component_cfg)
    for label, text in zip(
        ["A", "B", "C", "D"], ["Venus", "Mercury", "Mars", "Jupiter"]
    ):
        assert f"{label}. {text}" in rendered
