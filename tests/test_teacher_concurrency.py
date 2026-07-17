from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest

from diffusiongemma_e4b import teacher
from diffusiongemma_e4b.teacher import (
    TeacherConfig,
    _append_record_durable,
    generate_records,
    generation_fingerprint,
    validate_generated_mix,
)


def _cfg(concurrency: int) -> TeacherConfig:
    return TeacherConfig(
        runtime="openai-compatible",
        model="teacher",
        base_url="http://teacher/v1",
        max_tokens=64,
        temperature=0.2,
        top_p=0.95,
        max_retries=0,
        min_estimated_tokens=1,
        tokenizer_name_or_path="student",
        concurrency=concurrency,
    )


def _prompts(count: int):
    for index in range(count):
        yield {
            "id": f"p{index}",
            "source": "unit/source",
            "prompt_text": f"prompt {index}",
            "modality": "text",
        }


def test_concurrent_generation_reads_prompts_on_main_thread_and_yields_submission_order(tmp_path, monkeypatch):
    main_thread = threading.get_ident()
    generator_read_threads: list[int] = []
    clients = []
    lock = threading.Lock()
    first_call_barrier = threading.Barrier(3)

    def prompt_stream():
        for row in _prompts(4):
            generator_read_threads.append(threading.get_ident())
            yield row

    class Client:
        def __init__(self):
            self.owner = threading.get_ident()
            self.calls = 0

        def generate(self, prompt, media=None):
            assert threading.get_ident() == self.owner
            self.calls += 1
            if self.calls == 1:
                first_call_barrier.wait(timeout=2)
            index = int(prompt.rsplit(" ", 1)[1])
            time.sleep((3 - index) * 0.01)
            return f"unique answer for {prompt}"

    def make_client(_cfg):
        client = Client()
        with lock:
            clients.append(client)
        return client

    monkeypatch.setattr(teacher, "make_client", make_client)
    records = list(
        generate_records(
            _cfg(3),
            prompt_stream(),
            0,
            tmp_path / "progress.json",
            data_fingerprint="unit",
            token_counter=lambda text: len(text.split()),
        )
    )

    assert [record.metadata["prompt_record_id"] for record in records] == ["p0", "p1", "p2", "p3"]
    assert generator_read_threads and set(generator_read_threads) == {main_thread}
    assert len({client.owner for client in clients}) == 3
    assert all(client.owner != main_thread for client in clients)


def test_concurrency_one_preserves_synchronous_execution(tmp_path, monkeypatch):
    main_thread = threading.get_ident()
    client_threads = []

    class Client:
        def generate(self, prompt, media=None):
            client_threads.append(threading.get_ident())
            return f"answer for {prompt}"

    def make_client(_cfg):
        client_threads.append(threading.get_ident())
        return Client()

    monkeypatch.setattr(teacher, "make_client", make_client)
    records = list(
        generate_records(
            _cfg(1),
            _prompts(2),
            0,
            tmp_path / "progress.json",
            data_fingerprint="unit",
            token_counter=lambda _text: 4,
        )
    )

    assert len(records) == 2
    assert set(client_threads) == {main_thread}


def test_durable_record_keeps_prompt_and_context_separate(tmp_path, monkeypatch):
    sent_prompts: list[str] = []

    class Client:
        def generate(self, prompt, media=None):
            sent_prompts.append(prompt)
            return "a sufficiently long unique teacher answer"

    monkeypatch.setattr(teacher, "make_client", lambda _cfg: Client())
    prompt = {
        "id": "context-prompt",
        "source": "unit/source",
        "prompt_text": "What follows from the context?",
        "context_text": "A premise used by the question.",
        "modality": "text",
    }
    records = list(
        generate_records(
            _cfg(1),
            [prompt],
            0,
            tmp_path / "progress.json",
            data_fingerprint="unit",
            token_counter=lambda text: len(text.split()),
        )
    )
    assert len(records) == 1
    assert records[0].prompt_text == prompt["prompt_text"]
    assert records[0].context_text == prompt["context_text"]
    assert sent_prompts == [
        "Context:\nA premise used by the question.\n\nRequest:\nWhat follows from the context?"
    ]


def test_target_discards_pending_and_resume_uses_stable_prompt_ids(tmp_path, monkeypatch):
    calls: list[str] = []
    lock = threading.Lock()

    class Client:
        def generate(self, prompt, media=None):
            with lock:
                calls.append(prompt)
            return f"unique answer for {prompt}"

    monkeypatch.setattr(teacher, "make_client", lambda _cfg: Client())
    cfg = _cfg(3)
    output = tmp_path / "teacher.jsonl"
    progress = tmp_path / "progress.json"

    first = []
    for record in generate_records(
        cfg,
        _prompts(5),
        5,
        progress,
        output,
        data_fingerprint="unit",
        token_counter=lambda _text: 5,
    ):
        _append_record_durable(output, record)
        first.append(record)
    assert [record.metadata["prompt_record_id"] for record in first] == ["p0"]

    second = []
    for record in generate_records(
        cfg,
        _prompts(5),
        10,
        progress,
        output,
        data_fingerprint="unit",
        token_counter=lambda _text: 5,
    ):
        _append_record_durable(output, record)
        second.append(record)

    assert [record.metadata["prompt_record_id"] for record in second] == ["p1"]
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["prompt_record_id"] for row in rows] == ["p0", "p1"]


def test_concurrency_does_not_change_generation_fingerprint():
    one = _cfg(1)
    many = replace(one, concurrency=8)

    assert generation_fingerprint(one, "data") == generation_fingerprint(many, "data")
    assert TeacherConfig("openai-compatible", "m", "http://x", 1, 0.2, 0.95).concurrency == 8


def test_generated_mix_rejects_silently_lost_required_bucket():
    config = {
        "recommended_mix": [
            {"bucket": "text", "share": 0.9, "required": True},
            {"bucket": "image", "share": 0.1, "required": True},
        ]
    }
    state = {
        "records_by_bucket": {"text": 100},
        "estimated_tokens_by_bucket": {"text": 10_000},
    }
    with pytest.raises(RuntimeError, match="image"):
        validate_generated_mix(state, config)
