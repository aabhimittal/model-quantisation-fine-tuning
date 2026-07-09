"""Run *real* text generation on GPU cloud -- without owning a GPU.

The rest of ``quantune`` runs on a laptop CPU. This module is the one place that
reaches out to an actual accelerator, and it does so by borrowing someone else's:
it speaks the **OpenAI chat-completions wire format** to any compatible endpoint.

The important observation is that the three "how do I serve an LLM fast?" answers
in the README -- **NVIDIA NIM**, **vLLM**, and **Hugging Face TGI** -- all expose
the *same* HTTP API. So a single ~200-line client, plus a swappable ``base_url``,
covers all of them:

* ``https://integrate.api.nvidia.com/v1`` -- NVIDIA's hosted NIM catalog. Free
  ``nvapi-`` key, no credit card, no local GPU: the GPU lives in NVIDIA's cloud.
* ``http://localhost:8000/v1`` -- a self-hosted vLLM or NIM container on your own
  GPU (same payloads, ``api_key`` unused).
* ``http://localhost:8080/v1`` -- Hugging Face TGI's OpenAI-compatible route.

To keep the whole package installable with nothing but NumPy, this talks HTTP with
the standard library (:mod:`urllib`) rather than the ``openai`` SDK -- the wire
format is just JSON. And because "fast, low-latency" is a *measurable* claim, every
call reports **time-to-first-token** and **tokens/sec**, not just the text.

Example
-------
::

    from quantune import OpenAICompatClient

    client = OpenAICompatClient()                  # reads NVIDIA_API_KEY from env
    result = client.generate("Explain NF4 in one sentence.", stream=True)
    print(result.text)
    print(f"{result.time_to_first_token_s:.2f}s to first token, "
          f"{result.tokens_per_s:.1f} tok/s")
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"


@dataclass
class GenerationResult:
    """The text plus the latency numbers that justify calling it "low-latency"."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float                       # wall-clock for the whole call
    time_to_first_token_s: Optional[float]  # only measurable when streaming
    tokens_per_s: float

    def summary(self) -> str:
        ttft = (
            f"{self.time_to_first_token_s * 1000:.0f} ms"
            if self.time_to_first_token_s is not None
            else "n/a (non-streaming)"
        )
        return (
            f"model={self.model}  "
            f"tokens={self.completion_tokens} (+{self.prompt_tokens} prompt)  "
            f"ttft={ttft}  "
            f"speed={self.tokens_per_s:.1f} tok/s  "
            f"total={self.latency_s:.2f}s"
        )


class ServingError(RuntimeError):
    """Raised for missing credentials or a non-2xx response from the endpoint."""


class OpenAICompatClient:
    """A minimal client for any OpenAI-compatible ``/v1`` server.

    Parameters
    ----------
    base_url:
        The ``/v1`` root. Defaults to NVIDIA's hosted NIM catalog. Point it at
        ``http://localhost:8000/v1`` to hit a self-hosted vLLM/NIM instead.
    api_key:
        Bearer token. If omitted, falls back to the ``NVIDIA_API_KEY`` (or
        ``OPENAI_API_KEY``) environment variable. Left unset for local servers.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    # -- low-level HTTP ---------------------------------------------------- #
    def _headers(self, *, stream: bool = False) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, path: str, *, method: str = "GET", payload: Optional[dict] = None, stream: bool = False):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers(stream=stream))
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310 (trusted, user-supplied base_url)
        except urllib.error.HTTPError as exc:  # surface the server's error body, not a bare 400
            body = exc.read().decode("utf-8", "replace")
            if exc.code in (401, 403):
                raise ServingError(
                    f"{exc.code} from {url}: the API key was rejected. For NVIDIA NIM, generate a "
                    f"free key at build.nvidia.com and export NVIDIA_API_KEY. Server said: {body}"
                ) from exc
            raise ServingError(f"{exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ServingError(f"could not reach {url}: {exc.reason}") from exc

    def _require_key(self) -> None:
        if not self.api_key and self.base_url == DEFAULT_BASE_URL:
            raise ServingError(
                "no API key found. Generate a free NVIDIA NIM key at build.nvidia.com "
                "(Get API Key -> nvapi-...), then `export NVIDIA_API_KEY=nvapi-...`."
            )

    # -- public API -------------------------------------------------------- #
    def list_models(self) -> List[str]:
        """Return the model ids the endpoint exposes -- also a quick auth check."""
        self._require_key()
        with self._request("/models") as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return [m["id"] for m in body.get("data", [])]

    def generate(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        system: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        """Generate a completion for ``prompt`` and measure how fast it came back.

        When ``stream=True`` the response is consumed token-by-token (so
        ``time_to_first_token_s`` is meaningful and ``on_token`` fires as text
        arrives); otherwise the full JSON reply is awaited in one shot.
        """
        self._require_key()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            return self._generate_streaming(payload, model, on_token)
        return self._generate_blocking(payload, model)

    def _generate_blocking(self, payload: dict, model: str) -> GenerationResult:
        start = time.perf_counter()
        with self._request("/chat/completions", method="POST", payload=payload) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency = time.perf_counter() - start
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0) or 0
        return GenerationResult(
            text=text,
            model=body.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=completion_tokens,
            latency_s=latency,
            time_to_first_token_s=None,
            tokens_per_s=(completion_tokens / latency) if latency > 0 and completion_tokens else 0.0,
        )

    def _generate_streaming(
        self, payload: dict, model: str, on_token: Optional[Callable[[str], None]]
    ) -> GenerationResult:
        start = time.perf_counter()
        first_token_at: Optional[float] = None
        chunks: List[str] = []
        usage = {}
        served_model = model
        with self._request("/chat/completions", method="POST", payload=payload, stream=True) as resp:
            for event in _iter_sse(resp):
                if event == "[DONE]":
                    break
                obj = json.loads(event)
                served_model = obj.get("model", served_model)
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                piece = delta.get("content")
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - start
                    chunks.append(piece)
                    if on_token:
                        on_token(piece)
        latency = time.perf_counter() - start
        text = "".join(chunks)
        # NIM/vLLM usually send usage in the final chunk; fall back to a word-ish
        # estimate so tokens/sec is still populated for servers that omit it.
        completion_tokens = usage.get("completion_tokens") or _estimate_tokens(text)
        return GenerationResult(
            text=text,
            model=served_model,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=completion_tokens,
            latency_s=latency,
            time_to_first_token_s=first_token_at,
            tokens_per_s=(completion_tokens / latency) if latency > 0 and completion_tokens else 0.0,
        )


def _iter_sse(resp) -> Iterator[str]:
    """Yield the ``data:`` payloads from a Server-Sent-Events stream.

    The OpenAI streaming format sends lines like ``data: {json}`` separated by
    blank lines, terminated by ``data: [DONE]``. We ignore comments/other fields.
    """
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[len("data:"):].strip()


def _estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) for servers that omit a usage block."""
    return max(1, round(len(text) / 4)) if text else 0
