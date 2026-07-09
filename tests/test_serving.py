"""Tests for the OpenAI-compatible serving client -- no network required.

We stub :func:`urllib.request.urlopen` so the request construction, SSE parsing,
and latency bookkeeping are exercised without hitting NVIDIA NIM.
"""

import io
import json

import pytest

from quantune.serving import (
    DEFAULT_BASE_URL,
    GenerationResult,
    OpenAICompatClient,
    ServingError,
    _estimate_tokens,
    _iter_sse,
)


class _FakeResponse(io.BytesIO):
    """A urlopen() result usable as a context manager and an iterator of lines."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _install_urlopen(monkeypatch, handler):
    """Route urllib.request.urlopen to ``handler(req) -> _FakeResponse``."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return handler(req)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


def test_missing_key_on_default_endpoint_raises(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatClient()  # default = NVIDIA hosted, needs a key
    with pytest.raises(ServingError):
        client.generate("hi")


def test_local_endpoint_needs_no_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    body = json.dumps(
        {"model": "local", "choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1}}
    ).encode()
    _install_urlopen(monkeypatch, lambda req: _FakeResponse(body))
    client = OpenAICompatClient(base_url="http://localhost:8000/v1")
    result = client.generate("hi")  # must not raise about a missing key
    assert result.text == "ok"


def test_blocking_generate_builds_request_and_parses(monkeypatch):
    body = json.dumps(
        {
            "model": "meta/llama-3.1-8b-instruct",
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    ).encode()
    captured = _install_urlopen(monkeypatch, lambda req: _FakeResponse(body))
    client = OpenAICompatClient(api_key="nvapi-test")
    result = client.generate("hi there", system="be terse", max_tokens=10, temperature=0.0)

    req = captured["req"]
    assert req.full_url == f"{DEFAULT_BASE_URL}/chat/completions"
    assert req.get_method() == "POST"
    assert req.headers["Authorization"] == "Bearer nvapi-test"
    sent = json.loads(req.data.decode())
    assert sent["stream"] is False
    assert sent["messages"][0] == {"role": "system", "content": "be terse"}
    assert sent["messages"][1] == {"role": "user", "content": "hi there"}

    assert isinstance(result, GenerationResult)
    assert result.text == "hello world"
    assert result.completion_tokens == 2
    assert result.time_to_first_token_s is None  # not measurable when not streaming


def test_streaming_generate_parses_sse_and_measures_ttft(monkeypatch):
    lines = [
        b'data: {"model":"m","choices":[{"delta":{"content":"He"}}]}\n',
        b"\n",
        b'data: {"model":"m","choices":[{"delta":{"content":"llo"}}]}\n',
        b'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n',
        b"data: [DONE]\n",
    ]
    captured = _install_urlopen(monkeypatch, lambda req: _FakeResponse(b"".join(lines)))
    client = OpenAICompatClient(api_key="nvapi-test")
    seen = []
    result = client.generate("hi", stream=True, on_token=seen.append)

    sent = json.loads(captured["req"].data.decode())
    assert sent["stream"] is True
    assert result.text == "Hello"
    assert seen == ["He", "llo"]
    assert result.completion_tokens == 2
    assert result.time_to_first_token_s is not None
    assert result.tokens_per_s >= 0


def test_list_models(monkeypatch):
    body = json.dumps({"data": [{"id": "a"}, {"id": "b"}]}).encode()
    captured = _install_urlopen(monkeypatch, lambda req: _FakeResponse(body))
    client = OpenAICompatClient(api_key="nvapi-test")
    assert client.list_models() == ["a", "b"]
    assert captured["req"].get_method() == "GET"
    assert captured["req"].full_url.endswith("/models")


def test_iter_sse_skips_blank_and_comment_lines():
    payload = b": comment\n\ndata: {\"x\":1}\n\ndata: [DONE]\n"
    events = list(_iter_sse(io.BytesIO(payload)))
    assert events == ['{"x":1}', "[DONE]"]


def test_estimate_tokens_fallback():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("abcd") == 1
