from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import replace

import pytest

from diffusiongemma_e4b import teacher
from diffusiongemma_e4b.teacher import (
    _DiskPromptQueue,
    _ThroughputMeter,
    _throughput_eta,
    TeacherConfig,
    _append_record_durable,
    generate_records,
    generation_fingerprint,
    validate_generated_mix,
)


def test_throughput_eta_uses_remaining_target_tokens_over_total_tps():
    assert _throughput_eta(550, 50, 100.0) == "00:05"
    assert _throughput_eta(550, 550, 100.0) == "00:00"
    assert _throughput_eta(550, 50, None) == "?"


def test_streaming_throughput_sums_ten_workers_without_a_fixed_multiplier():
    meter = _ThroughputMeter(window_seconds=5, ewma_alpha=1.0)
    for _ in range(10):
        request_id = meter.begin_request()
        meter.record_tokens(request_id, 101.5, 22)
        meter.finish_request(request_id, 22)


    assert meter.total_tps(now=102.1) == pytest.approx(220.0)


def test_streaming_throughput_reconciles_buffered_chunks_with_api_usage():
    meter = _ThroughputMeter(window_seconds=5, ewma_alpha=1.0)
    request_id = meter.begin_request()
    meter.record_tokens(request_id, 101.5, 1)
    meter.finish_request(request_id, 2)

    assert meter.total_tps(now=102.1) == pytest.approx(2.0)


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


def test_concurrent_generation_prefetches_prompts_and_yields_completion_order(tmp_path, monkeypatch):
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

    record_ids = [record.metadata["prompt_record_id"] for record in records]
    assert record_ids[0] == "p2"
    assert set(record_ids) == {"p0", "p1", "p2", "p3"}
    assert generator_read_threads and main_thread not in set(generator_read_threads)
    assert len(set(generator_read_threads)) == 1
    assert len({client.owner for client in clients}) == 3
    assert all(client.owner != main_thread for client in clients)


def test_slow_prompt_prefetch_does_not_block_completed_requests(tmp_path, monkeypatch):
    source_block = threading.Event()

    def prompt_stream():
        yield from _prompts(2)
        source_block.wait()

    class Client:
        def generate(self, prompt, media=None):
            return f"unique answer for {prompt}"

    monkeypatch.setattr(teacher, "make_client", lambda _cfg: Client())
    records = generate_records(
        _cfg(2),
        prompt_stream(),
        0,
        tmp_path / "progress.json",
        data_fingerprint="unit",
        token_counter=lambda _text: 4,
    )
    started = time.monotonic()
    first = next(records)
    second = next(records)
    elapsed = time.monotonic() - started
    records.close()

    assert [first.metadata["prompt_record_id"], second.metadata["prompt_record_id"]] == ["p0", "p1"]
    assert elapsed < 1


def test_single_worker_still_prefetches_prompts_to_disk(tmp_path, monkeypatch):
    source_prefetched = threading.Event()

    def prompt_stream():
        yield from _prompts(3)
        source_prefetched.set()

    class Client:
        def generate(self, prompt, media=None):
            assert source_prefetched.wait(timeout=1)
            return f"unique answer for {prompt}"

    monkeypatch.setattr(teacher, "make_client", lambda _cfg: Client())
    cfg = replace(_cfg(1), prefetch_dir=tmp_path / "spool")
    records = list(
        generate_records(
            cfg,
            prompt_stream(),
            0,
            tmp_path / "progress.json",
            data_fingerprint="single-worker-spool",
            token_counter=lambda _text: 4,
        )
    )

    assert [record.metadata["prompt_record_id"] for record in records] == ["p0", "p1", "p2"]
    assert list((tmp_path / "spool").glob("prompts-*.sqlite3"))


def test_newly_spooled_prompts_fill_idle_slots_before_first_request_finishes(
    tmp_path, monkeypatch
):
    first_request_release = threading.Event()
    all_slots_started = threading.Event()
    calls: list[str] = []
    lock = threading.Lock()

    class Client:
        def generate(self, prompt, media=None):
            with lock:
                calls.append(prompt)
                if len(calls) == 3:
                    all_slots_started.set()
            if prompt == "prompt 0":
                first_request_release.wait(timeout=2)
            return f"unique answer for {prompt}"

    def delayed_prompt_stream():
        yield next(_prompts(1))
        time.sleep(0.08)
        yield from list(_prompts(4))[1:]

    monkeypatch.setattr(teacher, "make_client", lambda _cfg: Client())
    records = generate_records(
        _cfg(3),
        delayed_prompt_stream(),
        0,
        tmp_path / "progress.json",
        data_fingerprint="unit",
        token_counter=lambda _text: 4,
    )

    consumer = threading.Thread(target=lambda: list(records))
    consumer.start()
    try:
        assert all_slots_started.wait(timeout=1)
        assert len(calls) >= 3
    finally:
        first_request_release.set()
        consumer.join(timeout=2)
    assert not consumer.is_alive()


def test_resume_activity_resets_prefetch_stall_clock(tmp_path, monkeypatch):
    def prompt_stream():
        yield next(_prompts(1))
        for _ in range(2):
            time.sleep(0.03)
            yield next(_prompts(1))
        time.sleep(0.03)
        yield next(iter(list(_prompts(2))[1:]))

    class Client:
        def generate(self, prompt, media=None):
            return f"unique answer for {prompt}"

    monkeypatch.setattr(teacher, "make_client", lambda _cfg: Client())
    monkeypatch.setattr(teacher, "_PROMPT_PREFETCH_STALL_TIMEOUT_S", 0.05)
    records = list(
        generate_records(
            _cfg(2),
            prompt_stream(),
            0,
            tmp_path / "progress.json",
            data_fingerprint="unit",
            token_counter=lambda _text: 4,
        )
    )

    assert [record.metadata["prompt_record_id"] for record in records] == ["p0", "p1"]


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
    assert TeacherConfig("openai-compatible", "m", "http://x", 1, 0.2, 0.95).concurrency == 10
    assert TeacherConfig("openai-compatible", "m", "http://x", 1, 0.2, 0.95).prefetch_records == 16384


def test_prefetch_reservoir_cannot_be_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher, "make_client", lambda _cfg: object())
    with pytest.raises(ValueError, match="prefetch_records"):
        list(
            generate_records(
                replace(_cfg(2), prefetch_records=0),
                _prompts(1),
                0,
                tmp_path / "progress.json",
            )
        )


def test_prompt_prefetch_payloads_persist_and_resume_from_disk(tmp_path):
    spool_root = tmp_path / "spool"
    stop = threading.Event()
    queue = _DiskPromptQueue(spool_root, max_records=2, fingerprint="unit")
    database_path = queue.path

    assert queue.put(7, {"id": "p7", "prompt_text": "stored on disk"}, stop)
    assert database_path.is_file()
    assert queue.get() == (7, {"id": "p7", "prompt_text": "stored on disk"})
    queue.close()

    resumed = _DiskPromptQueue(spool_root, max_records=2, fingerprint="unit")
    assert resumed.get() == (7, {"id": "p7", "prompt_text": "stored on disk"})
    resumed.acknowledge("p7")
    resumed.finish()
    resumed.close()
    assert database_path.exists()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 0


def test_prompt_spool_removes_legacy_consumed_rows_on_open(tmp_path):
    spool_root = tmp_path / "spool"
    stop = threading.Event()
    queue = _DiskPromptQueue(spool_root, max_records=2, fingerprint="legacy")
    database_path = queue.path
    assert queue.put(1, {"id": "old"}, stop)
    queue._writer.execute("UPDATE prompts SET status = 'consumed' WHERE source_index = 1")
    queue._writer.commit()
    queue.close()

    reopened = _DiskPromptQueue(spool_root, max_records=2, fingerprint="legacy")
    reopened.close()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 0


def test_prompt_spool_keys_resume_payloads_by_stable_prompt_id(tmp_path):
    stop = threading.Event()
    queue = _DiskPromptQueue(tmp_path / "spool", max_records=4, fingerprint="stable-ids")

    assert queue.put(3, {"id": "p1", "prompt_text": "first ordering"}, stop)
    assert queue.put(3, {"id": "p2", "prompt_text": "changed ordering"}, stop)
    first = queue.get()
    assert first is not None
    queue.acknowledge(first[1]["id"])
    second = queue.get()
    assert second is not None
    queue.acknowledge(second[1]["id"])
    queue.close()

    assert {first[1]["id"], second[1]["id"]} == {"p1", "p2"}


def test_prompt_spool_migrates_position_keyed_database(tmp_path):
    spool_root = tmp_path / "spool"
    spool_root.mkdir()
    database_path = spool_root / "prompts-old-schema.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE prompts (source_index INTEGER PRIMARY KEY, payload TEXT NOT NULL, "
            "status TEXT NOT NULL CHECK(status IN ('queued', 'inflight', 'consumed')))"
        )
        connection.execute(
            "INSERT INTO prompts(source_index, payload, status) VALUES (?, ?, 'queued')",
            (11, json.dumps({"id": "old-prompt", "prompt_text": "resume me"})),
        )

    queue = _DiskPromptQueue(spool_root, max_records=4, fingerprint="old-schema")
    assert queue.get() == (11, {"id": "old-prompt", "prompt_text": "resume me"})
    queue.acknowledge("old-prompt")
    queue.close()

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prompts)").fetchall()
        }
        assert "prompt_key" in columns


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


def test_generate_with_retry_applies_jitter_and_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    class FlakyClient:
        def __init__(self):
            self.attempts = 0

        def generate(self, prompt, media=None):
            self.attempts += 1
            if self.attempts < 3:
                raise ValueError("temporary glitch")
            return "valid answer after retries"

    cfg = TeacherConfig(
        runtime="openai-compatible",
        model="m",
        base_url="http://x",
        max_tokens=64,
        temperature=0.2,
        top_p=0.95,
        max_retries=3,
        retry_base_s=2.0,
        min_estimated_tokens=1,
    )
    result = teacher._generate_with_retry(
        FlakyClient(), cfg, "prompt", {}, token_counter=lambda t: len(t.split())
    )
    assert result.text == "valid answer after retries"
    assert len(sleeps) == 2
    # Attempt 0: base=2.0, jitter range [1.0, 3.0]
    assert 1.0 <= sleeps[0] <= 3.0
    # Attempt 1: base=4.0, jitter range [2.0, 6.0]
    assert 2.0 <= sleeps[1] <= 6.0


def test_short_nonempty_teacher_output_is_accepted_without_retry(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    class ShortClient:
        attempts = 0

        def generate(self, prompt, media=None):
            self.attempts += 1
            return "OK"

    client = ShortClient()
    result = teacher._generate_with_retry(
        client,
        replace(_cfg(1), max_retries=5, min_estimated_tokens=8),
        "prompt",
        {},
        token_counter=lambda _text: 1,
    )

    assert result.text == "OK"
    assert client.attempts == 1
    assert sleeps == []


def test_empty_teacher_output_is_retried_and_contained(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    class EmptyThenValidClient:
        attempts = 0

        def generate(self, prompt, media=None):
            self.attempts += 1
            return "   " if self.attempts == 1 else "OK"

    client = EmptyThenValidClient()
    result = teacher._generate_with_retry(
        client,
        replace(_cfg(1), max_retries=2, retry_base_s=0, min_estimated_tokens=8),
        "prompt",
        {},
        token_counter=lambda _text: 1,
    )

    assert result.text == "OK"
    assert client.attempts == 2
    assert sleeps == [0.0]


def test_consecutive_failure_triggers_cooldown(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    class AlwaysFailingClient:
        def generate(self, prompt, media=None):
            raise ValueError("server is down")

    monkeypatch.setattr(teacher, "make_client", lambda _cfg: AlwaysFailingClient())
    cfg = TeacherConfig(
        runtime="openai-compatible",
        model="m",
        base_url="http://x",
        max_tokens=64,
        temperature=0.2,
        top_p=0.95,
        max_retries=0,
        max_consecutive_failures=3,
        retry_base_s=1.0,
        concurrency=2,
        min_estimated_tokens=1,
    )
    progress_file = tmp_path / "progress.json"
    with pytest.raises(RuntimeError, match="failed 3 consecutive prompts"):
        list(
            generate_records(
                cfg,
                _prompts(10),
                target_estimated_tokens=0,
                progress_path=progress_file,
                token_counter=lambda t: len(t.split()),
            )
        )
    # Consecutive failures should have triggered cooldown sleeps on the scheduler
    assert any(0.5 <= s <= 15.0 for s in sleeps)
