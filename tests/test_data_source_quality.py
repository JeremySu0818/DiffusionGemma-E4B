from __future__ import annotations

from pathlib import Path

from diffusiongemma_e4b.data_sources import _prompt_record


def test_streaming_parquet_does_not_retain_entire_decoded_row_group(tmp_path):
    import gc
    import pyarrow as pa
    import pyarrow.parquet as pq
    from diffusiongemma_e4b.data_sources import _open_hf_dataset

    path = tmp_path / "rows.parquet"
    table = pa.table({"prompt": [str(i) + "x" * 16384 for i in range(1024)]})
    pq.write_table(
        table, path, row_group_size=1024, data_page_size=65536,
        write_batch_size=32, use_dictionary=False,
    )
    del table
    gc.collect()
    baseline = pa.total_allocated_bytes()
    dataset = _open_hf_dataset({"id": str(tmp_path)})
    rows = iter(dataset)
    first = next(rows)
    checkpoint = dataset.state_dict()
    try:
        # A two-row queue cannot fix a scanner holding a 16 MiB row group.
        assert pa.total_allocated_bytes() - baseline < 4 * 1024 * 1024
        assert first["prompt"].startswith("0x")
        assert sum(1 for _ in rows) == 1023
    finally:
        rows.close()
    resumed = _open_hf_dataset({"id": str(tmp_path)})
    resumed.load_state_dict(checkpoint)
    resumed_rows = iter(resumed)
    try:
        assert next(resumed_rows)["prompt"].startswith("1x")
    finally:
        resumed_rows.close()


def test_prefetch_prepares_prompt_without_retaining_raw_payload(tmp_path):
    import weakref
    from diffusiongemma_e4b.data_sources import _prepare_source_rows

    refs = []

    class RawRow(dict):
        pass

    class Rows:
        def __iter__(self):
            return self

        def __next__(self):
            row = RawRow(prompt="p" * 1000, unused_payload=bytearray(1024 * 1024))
            refs.append(weakref.ref(row))
            return row

    prepared = _prepare_source_rows(Rows(), {"id": "unit/text"}, 100, tmp_path)
    try:
        item = next(prepared)
        assert len(item.record["prompt_text"]) == 100
        assert "unused_payload" not in item.record
        assert refs[0]() is None
    finally:
        prepared.close()


def test_image_only_placeholder_gets_meaningful_teacher_prompt(tmp_path: Path) -> None:
    source = {
        "id": "unit/images",
        "name": "unit_images",
        "bucket": "teacher_supervised_image",
        "role": "image_text_context",
        "modality": "image",
    }
    row = {
        "id": "image-1",
        "image": "source-image.png",
        "conversations": [
            {"from": "human", "value": "<image>"},
            {"from": "gpt", "value": "This third-party target must not be copied."},
        ],
    }

    record = _prompt_record(source, row, max_chars=1000, media_dir=tmp_path)

    assert record is not None
    assert record["prompt_text"] == (
        "Describe and analyze the image carefully, including the important visual details."
    )
    assert "third-party" not in record["prompt_text"]
    assert record["media"]["images"] == ["source-image.png"]


def test_prompt_and_context_share_one_teacher_request_budget() -> None:
    source = {
        "id": "unit/context",
        "name": "unit_context",
        "bucket": "teacher_supervised_text_reasoning_code",
        "role": "prompt_context_bank",
        "modality": "text",
    }
    row = {
        "id": "context-1",
        "question": "Q" * 400,
        "context": "C" * 900,
    }
    record = _prompt_record(source, row, max_chars=1_000)
    assert record is not None
    assert (
        len(record["prompt_text"]) + len(record["context_text"]) + 32
        <= 1_000
    )


def test_image_source_requires_real_image_and_keeps_choices_not_answer(
    tmp_path: Path,
) -> None:
    source = {
        "id": "unit/science",
        "name": "unit_science",
        "bucket": "teacher_supervised_image",
        "role": "science_diagram_context",
        "modality": "image",
    }
    without_image = _prompt_record(
        source,
        {
            "question": "Which option is correct?",
            "choices": ["first", "second"],
            "answer": 1,
            "solution": "Do not copy this target.",
            "image": None,
        },
        max_chars=1_000,
        media_dir=tmp_path,
    )
    assert without_image is None

    with_image = _prompt_record(
        source,
        {
            "question": "Which option is correct?",
            "choices": ["first", "second"],
            "answer": 1,
            "solution": "Do not copy this target.",
            "image": "diagram.png",
        },
        max_chars=1_000,
        media_dir=tmp_path,
    )
    assert with_image is not None
    assert "A. first" in with_image["prompt_text"]
    assert "B. second" in with_image["prompt_text"]
    assert "Do not copy" not in with_image["prompt_text"]
