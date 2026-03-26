import pytest
from collm.data.prompts import get_prompt, get_answer


def test_prompt_with_signal_contains_placeholders():
    prompt = get_prompt(
        history_titles=["Movie A", "Movie B"],
        target_title="Movie C",
        use_collaborative_signal=True,
    )
    assert "[USER_TOKEN]" in prompt
    assert "[ITEM_TOKEN]" in prompt


def test_prompt_without_signal_no_placeholders():
    prompt = get_prompt(
        history_titles=["Movie A", "Movie B"],
        target_title="Movie C",
        use_collaborative_signal=False,
    )
    assert "[USER_TOKEN]" not in prompt
    assert "[ITEM_TOKEN]" not in prompt


def test_prompt_contains_history():
    prompt = get_prompt(
        history_titles=["Movie A", "Movie B"],
        target_title="Movie C",
        use_collaborative_signal=True,
    )
    assert "Movie A" in prompt
    assert "Movie B" in prompt
    assert "Movie C" in prompt


def test_answer_yes_no():
    assert "Yes" in get_answer(label=1)
    assert "No" in get_answer(label=0)


import pandas as pd
import torch
from collm.data.dataset import RecDataset


def test_dataset_len(sample_pickle):
    ds = RecDataset(sample_pickle, dataset_type="movielens")
    assert len(ds) == 3


def test_dataset_getitem(sample_pickle):
    ds = RecDataset(sample_pickle, dataset_type="movielens")
    item = ds[0]
    assert "uid" in item
    assert "iid" in item
    assert "title" in item
    assert "history_titles" in item
    assert "label" in item
    assert isinstance(item["label"], int)


def test_dataset_label_is_int(sample_pickle):
    ds = RecDataset(sample_pickle, dataset_type="movielens")
    for i in range(len(ds)):
        item = ds[i]
        assert item["label"] in (0, 1)


import torch
from transformers import AutoTokenizer
from collm.data.collator import RecDataCollator


@pytest.fixture
def tiny_tokenizer(tmp_path):
    """使用 GPT-2 tokenizer 並注冊占位符（不下載 LLM 本體）"""
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["[USER_TOKEN]", "[ITEM_TOKEN]"]})
    return tok


@pytest.fixture
def sample_batch():
    return [
        {"uid": 0, "iid": 5, "title": "Movie A",
         "history_iid": [1, 2], "history_titles": ["X", "Y"], "label": 1},
        {"uid": 1, "iid": 6, "title": "Movie B",
         "history_iid": [3], "history_titles": ["Z"], "label": 0},
    ]


def test_collator_returns_required_keys(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=True,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=False,
    )
    batch = collator(sample_batch)
    for key in ["input_ids", "attention_mask", "labels",
                "user_ids", "target_item_ids",
                "user_placeholder_pos", "item_placeholder_pos"]:
        assert key in batch, f"Missing key: {key}"


def test_collator_with_signal_finds_placeholder_positions(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=True,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=False,
    )
    batch = collator(sample_batch)
    assert batch["user_placeholder_pos"].max().item() != -1
    assert batch["item_placeholder_pos"].max().item() != -1


def test_collator_without_signal_placeholder_is_neg1(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=False,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=False,
    )
    batch = collator(sample_batch)
    assert (batch["user_placeholder_pos"] == -1).all()
    assert (batch["item_placeholder_pos"] == -1).all()


def test_collator_sequential_includes_seq_history(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=True,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=True,
    )
    batch = collator(sample_batch)
    assert "seq_history" in batch
    assert batch["seq_history"].shape == (2, 5)


def test_collator_non_sequential_no_seq_history(tiny_tokenizer, sample_batch):
    collator = RecDataCollator(
        tokenizer=tiny_tokenizer,
        use_collaborative_signal=True,
        max_seq_len=64,
        max_history_len=5,
        is_sequential=False,
    )
    batch = collator(sample_batch)
    assert "seq_history" not in batch
