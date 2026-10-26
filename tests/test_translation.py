from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core_pipeline.errors import TranslationAuthError, TranslationFormatError
from core_pipeline.translation import (
    DeepSeekTranslator,
    EchoTranslator,
    extract_json_payload,
    normalize_translation,
    parse_translation_payload,
)
from core_pipeline.types import PipelineOptions, TextRegion

OPTIONS = PipelineOptions(source_lang="ja", target_lang="en", mode="manga")


def regions(count: int = 3) -> list[TextRegion]:
    return [
        TextRegion(
            id=index,
            text=f"原文 {index}",
            polygon=[(10, index * 40), (200, index * 40), (200, index * 40 + 30), (10, index * 40 + 30)],
        )
        for index in range(count)
    ]


def test_extract_json_payload_handles_fences_and_prose():
    raw = 'Here you go:\n```json\n{"items":[{"id":0,"translation":"Hi"}]}\n```'
    payload = extract_json_payload(raw)
    assert json.loads(payload)["items"][0]["translation"] == "Hi"


def test_parse_payload_falls_back_for_missing_ids():
    parsed, warnings = parse_translation_payload(
        '{"items":[{"id":0,"translation":"Hello"}]}',
        [0, 1],
        {0: "こんにちは", 1: "さようなら"},
    )
    assert parsed[0] == "Hello"
    assert parsed[1] == "さようなら"
    assert warnings


def test_parse_payload_accepts_id_mapping():
    parsed, _ = parse_translation_payload(
        '{"0":"Hello","1":"Bye"}', [0, 1], {0: "a", 1: "b"}
    )
    assert parsed == {0: "Hello", 1: "Bye"}


def test_parse_payload_rejects_truncated_json():
    with pytest.raises(TranslationFormatError):
        parse_translation_payload('{"items":[{"id":0,"translation":"Hi"', [0], {0: "a"})


def test_echo_translator_keeps_source():
    output = asyncio.run(EchoTranslator().translate(regions(2), OPTIONS))
    assert output.translations == {0: "原文 0", 1: "原文 1"}
    assert output.degraded is True


def test_normalize_translation_uses_simplified_chinese_glyphs():
    assert normalize_translation("随心所欲蹂躙弱者", "zh") == "随心所欲蹂躏弱者"
    assert normalize_translation("伪 圣 女", "zh") == "伪圣女"
    assert normalize_translation("Keep spaced words", "en") == "Keep spaced words"


def _response(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        },
    )


def test_deepseek_translator_posts_ordered_batch():
    captured: list[dict] = []
    content = json.dumps(
        {
            "items": [
                {"id": 0, "translation": "Hello"},
                {"id": 1, "translation": "Goodbye"},
            ]
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        captured.append(json.loads(request.content))
        return _response(content)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    translator = DeepSeekTranslator(api_key="secret", client=client, batch_size=40)
    output = asyncio.run(translator.translate(regions(2), OPTIONS))
    assert output.translations == {0: "Hello", 1: "Goodbye"}
    assert output.usage["total_tokens"] == 17
    assert len(captured) == 1
    payload = json.loads(captured[0]["messages"][1]["content"].split("\n\n", 1)[1])
    assert [item["id"] for item in payload["items"]] == [0, 1]
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["model"] == "deepseek-chat"


def test_deepseek_batches_large_pages(monkeypatch):
    calls = {"count": 0}
    payloads: list[dict] = []

    async def no_backoff(attempt: int) -> None:
        return None

    monkeypatch.setattr(DeepSeekTranslator, "_backoff", staticmethod(no_backoff))

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"].split("\n\n", 1)[1])
        payloads.append(payload)
        items = [
            {"id": item["id"], "translation": f"T{item['id']}"}
            for item in payload["items"]
        ]
        return _response(json.dumps({"items": items}))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    translator = DeepSeekTranslator(api_key="secret", client=client, batch_size=2)
    output = asyncio.run(translator.translate(regions(5), OPTIONS))
    assert calls["count"] == 3
    assert output.translations == {index: f"T{index}" for index in range(5)}
    # Every chunk still sees the whole page, not just its own two lines.
    assert all(payload["page_text"] == "\n".join(f"原文 {i}" for i in range(5)) for payload in payloads)


def test_deepseek_chunk_carries_adjacent_line_context(monkeypatch):
    captured: dict[int, dict] = {}
    limits: dict[int, int] = {}

    async def no_backoff(attempt: int) -> None:
        return None

    monkeypatch.setattr(DeepSeekTranslator, "_backoff", staticmethod(no_backoff))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"].split("\n\n", 1)[1])
        index = int(payload["chunk"].split("/")[0])
        captured[index] = payload
        limits[index] = body["max_tokens"]
        item = payload["items"][0]
        return _response(
            json.dumps({"items": [{"id": item["id"], "translation": "T"}]})
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    translator = DeepSeekTranslator(api_key="secret", client=client, batch_size=1)
    asyncio.run(translator.translate(regions(3), OPTIONS))

    first, middle, last = captured[1], captured[2], captured[3]
    assert "context_before" not in first["items"][0]
    assert first["items"][0]["context_after"] == "原文 1"
    assert middle["items"][0]["context_before"] == "原文 0"
    assert middle["items"][0]["context_after"] == "原文 2"
    assert last["items"][0]["context_before"] == "原文 1"
    assert "context_after" not in last["items"][0]
    # The completion budget follows the source length, not just the item count,
    # so a merged multi-line paragraph is not truncated.
    assert limits[1] > 512 + 96


def test_deepseek_retries_invalid_json(monkeypatch):
    calls = {"count": 0}

    async def no_backoff(attempt: int) -> None:
        return None

    monkeypatch.setattr(DeepSeekTranslator, "_backoff", staticmethod(no_backoff))

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _response("this is not json")
        return _response('{"items":[{"id":0,"translation":"ok"}]}')

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    translator = DeepSeekTranslator(api_key="secret", client=client, max_retries=2)
    output = asyncio.run(translator.translate(regions(1), OPTIONS))
    assert output.translations[0] == "ok"
    assert calls["count"] == 2


def test_deepseek_auth_error_is_not_retried():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    translator = DeepSeekTranslator(api_key="bad", client=client)
    with pytest.raises(TranslationAuthError):
        asyncio.run(translator.translate(regions(1), OPTIONS))
    assert calls["count"] == 1