from datasets import Dataset
from unittest.mock import patch


class DummyTokenizer:
    def __init__(self):
        self.name_or_path = "qwen-dummy"
        self.chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\\n{content}<|im_end|>\\n"
        return text

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=False):
        ids = [ord(ch) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return type("Tokenized", (), {"input_ids": ids})


def _mock_reasoning_dataset():
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": "What is 2+2?"},
                    {"from": "gpt", "value": "It's 4."},
                ],
                "meta": {"input_tokens": 50, "output_tokens": 10},
            },
            {
                "conversations": [
                    {"from": "human", "value": "Explain recursion."},
                    {"from": "assistant", "value": "Recursion calls itself."},
                ],
                "meta": {"input_tokens": 9000, "output_tokens": 2000},
            },
            {
                "conversations": [
                    {"from": "human", "value": "No meta sample"},
                    {"from": "gpt", "value": "Should be kept."},
                ],
            },
        ]
    )


def test_filter_by_token_budget_filters_and_keeps_missing_meta():
    from src.data.reasoning_sft import filter_by_token_budget

    ds = _mock_reasoning_dataset()
    filtered = filter_by_token_budget(ds, max_total_tokens=100)

    assert len(filtered) == 2
    remaining_total_tokens = []
    for sample in filtered:
        meta = sample.get("meta")
        if isinstance(meta, dict) and meta.get("input_tokens") is not None and meta.get("output_tokens") is not None:
            remaining_total_tokens.append(int(meta["input_tokens"]) + int(meta["output_tokens"]))
    assert remaining_total_tokens == [60]


def test_filter_by_token_budget_keeps_malformed_token_counts():
    from src.data.reasoning_sft import filter_by_token_budget

    ds = Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "meta": {"input_tokens": "N/A", "output_tokens": 5},
            }
        ]
    )

    filtered = filter_by_token_budget(ds, max_total_tokens=10)
    assert len(filtered) == 1


def _find_subsequence(seq, subseq):
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i : i + len(subseq)] == subseq:
            return i
    return -1


def test_tokenize_reasoning_dataset_outputs_ids_and_masked_labels():
    from src.data.reasoning_sft import tokenize_reasoning_dataset

    ds = Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": "What is 2+2?"},
                    {"from": "gpt", "value": "It's 4."},
                ],
                "meta": {"input_tokens": 10, "output_tokens": 10},
            }
        ]
    )
    tok = DummyTokenizer()

    out = tokenize_reasoning_dataset(ds, tok, max_length=128, num_proc=1)
    ex = out[0]

    assert "input_ids" in out.column_names
    assert "labels" in out.column_names
    assert len(ex["input_ids"]) <= 128
    assert len(ex["labels"]) == len(ex["input_ids"])
    assert any(lbl == -100 for lbl in ex["labels"])
    assert any(lbl != -100 for lbl in ex["labels"])


def test_tokenize_reasoning_dataset_applies_glm_role_mapping_for_labels():
    from src.data.reasoning_sft import tokenize_reasoning_dataset

    user_text = "USER_ONLY_MARKER"
    assistant_text = "ASSIST_ONLY_MARKER"
    ds = Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": user_text},
                    {"from": "gpt", "value": assistant_text},
                ]
            }
        ]
    )

    tok = DummyTokenizer()
    out = tokenize_reasoning_dataset(ds, tok, max_length=512, num_proc=1)
    ex = out[0]

    user_ids = [ord(ch) for ch in user_text]
    assistant_ids = [ord(ch) for ch in assistant_text]

    user_start = _find_subsequence(ex["input_ids"], user_ids)
    assistant_start = _find_subsequence(ex["input_ids"], assistant_ids)

    assert user_start >= 0
    assert assistant_start >= 0
    assert all(lbl == -100 for lbl in ex["labels"][user_start : user_start + len(user_ids)])
    assert all(
        lbl != -100
        for lbl in ex["labels"][assistant_start : assistant_start + len(assistant_ids)]
    )


def test_create_reasoning_sft_datasets_returns_expected_columns():
    from src.data.reasoning_sft import create_reasoning_sft_datasets

    ds = Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ],
                "meta": {"input_tokens": 5, "output_tokens": 5},
            },
            {
                "conversations": [
                    {"from": "human", "value": "Q2"},
                    {"from": "assistant", "value": "A2"},
                ],
                "meta": {"input_tokens": 5, "output_tokens": 5},
            },
            {
                "conversations": [
                    {"from": "human", "value": "Q3"},
                    {"from": "gpt", "value": "A3"},
                ],
                "meta": {"input_tokens": 5, "output_tokens": 5},
            },
        ]
    )

    with patch("src.data.reasoning_sft.pack_tokenized_dataset", lambda d, **_: d):
        train_ds, eval_ds = create_reasoning_sft_datasets(
            ds,
            DummyTokenizer(),
            max_length=64,
            num_proc=1,
            val_split=0.34,
            seed=7,
        )

    assert len(train_ds) > 0
    assert len(eval_ds) > 0
    assert set(train_ds.column_names) == {"input_ids", "attention_mask", "labels"}
    assert set(eval_ds.column_names) == {"input_ids", "attention_mask", "labels"}


def test_train_val_are_disjoint_by_content():
    from src.data.reasoning_sft import create_reasoning_sft_datasets

    ds = Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": f"Q{i}"},
                    {"from": "gpt", "value": f"A{i}"},
                ],
                "meta": {"input_tokens": 5, "output_tokens": 5},
            }
            for i in range(20)
        ]
    )

    with patch("src.data.reasoning_sft.pack_tokenized_dataset", lambda d, **_: d):
        train_ds, val_ds = create_reasoning_sft_datasets(
            ds,
            DummyTokenizer(),
            max_length=128,
            num_proc=1,
            val_split=0.25,
            seed=7,
        )

    assert len(train_ds) + len(val_ds) == len(ds)

    train_rows = {tuple(ex["input_ids"]) for ex in train_ds}
    val_rows = {tuple(ex["input_ids"]) for ex in val_ds}
    assert train_rows.isdisjoint(val_rows)


def test_create_reasoning_sft_datasets_validates_val_split():
    from src.data.reasoning_sft import create_reasoning_sft_datasets

    ds = _mock_reasoning_dataset()
    with patch("src.data.reasoning_sft.pack_tokenized_dataset", lambda d, **_: d):
        try:
            create_reasoning_sft_datasets(
                ds,
                DummyTokenizer(),
                max_length=64,
                num_proc=1,
                val_split=0.0,
                seed=7,
            )
            assert False, "Expected ValueError for invalid val_split"
        except ValueError as exc:
            assert "val_split" in str(exc)


def test_create_reasoning_sft_datasets_raises_when_empty_after_filtering():
    from src.data.reasoning_sft import create_reasoning_sft_datasets

    ds = Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "meta": {"input_tokens": 1000, "output_tokens": 1000},
            }
        ]
    )

    with patch("src.data.reasoning_sft.pack_tokenized_dataset", lambda d, **_: d):
        try:
            create_reasoning_sft_datasets(
                ds,
                DummyTokenizer(),
                max_length=10,
                num_proc=1,
                val_split=0.5,
                seed=7,
            )
            assert False, "Expected ValueError for empty dataset after filtering"
        except ValueError as exc:
            assert "after token-budget filtering" in str(exc)


def test_create_reasoning_sft_datasets_raises_when_empty_after_subsampling():
    from src.data.reasoning_sft import create_reasoning_sft_datasets

    ds = Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ],
                "meta": {"input_tokens": 5, "output_tokens": 5},
            },
            {
                "conversations": [
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ],
                "meta": {"input_tokens": 5, "output_tokens": 5},
            },
        ]
    )

    with patch("src.data.reasoning_sft.pack_tokenized_dataset", lambda d, **_: d):
        try:
            create_reasoning_sft_datasets(
                ds,
                DummyTokenizer(),
                max_length=64,
                num_proc=1,
                val_split=0.5,
                max_eval_samples=0,
                seed=7,
            )
            assert False, "Expected ValueError for empty split after subsampling"
        except ValueError as exc:
            assert "after subsampling" in str(exc)
