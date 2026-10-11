import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from diffusiongemma_e4b.clean_teacher import clean_teacher


def test_in_place_cleaning_partitions_real_token_counts_and_keeps_original_bytes(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    rows = [
        # Legacy equal context/prompt was wrapped, unlike the current policy.
        {"prompt_text": "one two", "context_text": "one two",
         "metadata": {"teacher_input_tokens": 2}, "estimated_tokens": 10},
        {"prompt_text": "one two", "context_text": "one two",
         "metadata": {"teacher_input_tokens": 2, "teacher_input_policy": "full_stored_prompt",
                      "teacher_finish_reason": "stop"}, "estimated_tokens": 20},
        {"prompt_text": "one two", "metadata": {"teacher_input_tokens": 2,
         "teacher_finish_reason": "length"}, "estimated_tokens": 30},
        {"prompt_text": "one two", "metadata": {}, "estimated_tokens": 40},
    ]
    original_lines = [(json.dumps(row, indent=None) + "\n").encode() for row in rows]
    path = tmp_path / "teacher.jsonl"
    path.write_bytes(b"".join(original_lines))
    report = clean_teacher(path, tokenizer)
    assert path.read_bytes() == b"".join(original_lines[1:])
    assert report["counts"]["output_tokens_removed"] == 10
    assert report["counts"]["output_tokens_kept"] == 90
    assert report["counts"]["missing_input_accounting"] == 1
    assert report["removal_reasons"] == {"input_truncated": 1}
    report = clean_teacher(path, tokenizer, drop_length_limited=True)
    assert path.read_bytes() == original_lines[1] + original_lines[3]
    assert report["removal_reasons"] == {"length_limited": 1}
    assert list(tmp_path.iterdir()) == [path]
