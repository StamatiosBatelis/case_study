"""
Local-only LLM and embedding client for Ollama.

Calls Ollama's OpenAI-compatible REST endpoints over localhost using httpx.
No cloud dependencies — safe for air-gapped deployments.
Swap base_url to point at a local vLLM server with no other code changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT  = 180.0   # seconds — local inference can be slow for large prompts


# ---------------------------------------------------------------------------
# Response shims — match the attribute paths used across the codebase so
# callers don't need to touch raw dicts.
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    function: "_FunctionCall"

@dataclass
class _FunctionCall:
    name: str
    arguments: str   # JSON string, as returned by the model


@dataclass
class Message:
    role: str
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to the format expected by the chat completions API."""
        d: dict = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return d


@dataclass
class Choice:
    message: Message


@dataclass
class ChatResponse:
    choices: list[Choice]


@dataclass
class EmbeddingData:
    embedding: list[float]


@dataclass
class EmbeddingResponse:
    data: list[EmbeddingData]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Minimal synchronous HTTP client for the Ollama OpenAI-compatible API.
    All requests go to localhost — no external network calls, ever.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=DEFAULT_TIMEOUT)

    # ------------------------------------------------------------------
    # Chat completions (with optional tool-calling)
    # ------------------------------------------------------------------

    def chat_completions_create(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        response_format: Optional[dict] = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model":    model,
            "messages": messages,
            "stream":   False,
        }
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format

        resp = self._http.post(f"{self._base}/chat/completions", json=payload)
        resp.raise_for_status()
        raw = resp.json()

        choices = []
        for c in raw.get("choices", []):
            msg_raw = c.get("message", {})
            tool_calls = []
            for tc in msg_raw.get("tool_calls") or []:
                fn = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    function=_FunctionCall(
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", "{}"),
                    ),
                ))
            choices.append(Choice(message=Message(
                role=msg_raw.get("role", "assistant"),
                content=msg_raw.get("content"),
                tool_calls=tool_calls,
            )))
        return ChatResponse(choices=choices)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embeddings_create(self, model: str, input: list[str]) -> EmbeddingResponse:
        payload = {"model": model, "input": input}
        resp = self._http.post(f"{self._base}/embeddings", json=payload)
        resp.raise_for_status()
        raw = resp.json()
        data = [EmbeddingData(embedding=d["embedding"]) for d in raw.get("data", [])]
        return EmbeddingResponse(data=data)

    def __del__(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass
