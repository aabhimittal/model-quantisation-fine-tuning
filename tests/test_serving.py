"""Tests for the OpenAI-compatible serving client -- no network required.

We stub :func:`urllib.request.urlopen` so the request construction, SSE parsing,
and latency bookkeeping are exercised without hitting NVIDIA NIM.
"""

import io
import json

import pytest

from quantune.serving import (
    DEFAULT_BASE_URL,
    GROUNDING_SYSTEM_PROMPT,
    GenerationResult,
    OpenAICompatClient,
    ServingError,
    _estimate_tokens,
    _is_abstention,
    _iter_sse,
    _jaccard,
    _vote,
    groundedness,
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


# -- grounding ------------------------------------------------------------- #

def _chat_response(content, usage=None):
    return json.dumps(
        {"model": "m", "choices": [{"message": {"content": content}}], "usage": usage or {}}
    ).encode()


def test_groundedness_fully_supported_is_one():
    context = ["QLoRA stores frozen base weights in 4-bit NF4 and trains a LoRA adapter."]
    answer = "QLoRA stores frozen weights in NF4 and trains a LoRA adapter."
    assert groundedness(answer, context) == 1.0


def test_groundedness_penalizes_novel_words():
    context = ["QLoRA stores frozen base weights in 4-bit NF4 and trains a LoRA adapter."]
    answer = "QLoRA is a quantum computing technique using nuclear spin resonance."
    assert groundedness(answer, context) < 0.5


def test_groundedness_empty_answer_is_one():
    assert groundedness("", ["anything"]) == 1.0


def test_is_abstention():
    assert _is_abstention("I don't know.")
    assert _is_abstention("Sorry, I do not know that.")
    assert not _is_abstention("The answer is NF4.")


def test_grounded_generate_injects_system_prompt_and_context(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda req: _FakeResponse(_chat_response("NF4 [1]", {"completion_tokens": 3}))
    )
    client = OpenAICompatClient(api_key="nvapi-test")
    result = client.generate("What dtype?", context=["Weights are stored in NF4."])

    sent = json.loads(captured["req"].data.decode())
    assert sent["messages"][0] == {"role": "system", "content": GROUNDING_SYSTEM_PROMPT}
    user = sent["messages"][1]["content"]
    assert "Context:" in user and "[1] Weights are stored in NF4." in user
    assert "Question: What dtype?" in user
    # NF4 appears in the context -> high groundedness, not an abstention.
    assert result.grounded_fraction == 1.0
    assert result.abstained is False


def test_grounded_generate_flags_abstention(monkeypatch):
    _install_urlopen(monkeypatch, lambda req: _FakeResponse(_chat_response("I don't know.")))
    client = OpenAICompatClient(api_key="nvapi-test")
    result = client.generate("unanswerable?", context=["irrelevant source text"])
    assert result.abstained is True


def test_ungrounded_generate_leaves_grounding_unset(monkeypatch):
    _install_urlopen(monkeypatch, lambda req: _FakeResponse(_chat_response("hi", {"completion_tokens": 1})))
    client = OpenAICompatClient(api_key="nvapi-test")
    result = client.generate("hi")
    assert result.grounded_fraction is None
    assert result.abstained is False


def test_grounded_flag_without_context_still_sets_system_prompt(monkeypatch):
    captured = _install_urlopen(monkeypatch, lambda req: _FakeResponse(_chat_response("ok")))
    client = OpenAICompatClient(api_key="nvapi-test")
    result = client.generate("hi", grounded=True)
    sent = json.loads(captured["req"].data.decode())
    assert sent["messages"][0]["content"] == GROUNDING_SYSTEM_PROMPT
    # No context supplied -> nothing to score against.
    assert result.grounded_fraction is None


def test_summary_reports_groundedness():
    r = GenerationResult("x", "m", 1, 1, 0.1, None, 10.0, grounded_fraction=0.5)
    assert "grounded=50%" in r.summary()
    r2 = GenerationResult("x", "m", 1, 1, 0.1, None, 10.0, grounded_fraction=0.9, abstained=True)
    assert "grounded=abstained" in r2.summary()


# -- self-consistency voting ----------------------------------------------- #

def test_jaccard_bounds():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0
    assert _jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)


def test_vote_majority_wins_over_outliers():
    answers = [
        "QLoRA stores frozen weights in 4-bit NF4 and trains a LoRA adapter.",
        "The frozen base weights are stored in NF4, a 4-bit datatype, with a LoRA adapter.",
        "It keeps frozen weights in 4-bit NF4 and trains a small LoRA adapter.",
        "QLoRA uses quantum entanglement to compress the model.",
        "QLoRA is a reinforcement learning algorithm for robots.",
    ]
    winner, votes = _vote(answers, threshold=0.4)
    assert votes == 3
    assert "nf4" in winner.lower()


def test_vote_all_distinct_returns_first_singleton():
    answers = ["alpha one", "beta two", "gamma three"]
    winner, votes = _vote(answers, threshold=0.9)
    assert votes == 1
    assert winner == "alpha one"  # ties break to the earliest


def test_vote_empty():
    assert _vote([], 0.6) == ("", 0)


def _install_sequenced_urlopen(monkeypatch, bodies):
    """Return successive response bodies on successive urlopen() calls."""
    calls = {"n": 0}
    seq = list(bodies)

    def fake_urlopen(req, timeout=None):
        body = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def test_self_consistency_votes_across_samples(monkeypatch):
    bodies = [
        _chat_response("The answer is NF4, a 4-bit datatype."),
        _chat_response("NF4 -- a 4-bit datatype -- is the answer."),
        _chat_response("The answer is NF4, a 4-bit datatype."),
        _chat_response("The answer is a quantum hologram."),
    ]
    calls = _install_sequenced_urlopen(monkeypatch, bodies)
    client = OpenAICompatClient(api_key="nvapi-test")
    result = client.self_consistency("What dtype?", n=4, threshold=0.5)

    assert calls["n"] == 4                 # made exactly n calls
    assert result.n_samples == 4
    assert result.votes == 3               # three NF4-ish answers agree
    assert result.agreement == 0.75
    assert "nf4" in result.text.lower()
    assert "self-consistency=3/4" in result.summary()


def test_self_consistency_rejects_bad_n(monkeypatch):
    client = OpenAICompatClient(api_key="nvapi-test")
    with pytest.raises(ValueError):
        client.self_consistency("hi", n=0)


def test_self_consistency_scores_groundedness_when_context(monkeypatch):
    bodies = [_chat_response("Weights are stored in NF4.")] * 3
    _install_sequenced_urlopen(monkeypatch, bodies)
    client = OpenAICompatClient(api_key="nvapi-test")
    result = client.self_consistency("dtype?", n=3, context=["Weights are stored in NF4."])
    assert result.grounded_fraction == 1.0
