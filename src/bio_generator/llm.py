from __future__ import annotations

"""LLM interaction layer using Anthropic API directly."""

from typing import Any, Dict

import anthropic

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator


class TokenUsage:
    """Track token usage across steps."""

    def __init__(self) -> None:
        self.by_step: Dict[str, Dict[str, int]] = {}

    def record(self, step: str, input_tokens: int, output_tokens: int) -> None:
        if step in self.by_step:
            self.by_step[step]["input_tokens"] += input_tokens
            self.by_step[step]["output_tokens"] += output_tokens
        else:
            self.by_step[step] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

    @property
    def total_tokens(self) -> int:
        return sum(
            s["input_tokens"] + s["output_tokens"] for s in self.by_step.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": sum(s["input_tokens"] for s in self.by_step.values()),
            "output_tokens": sum(s["output_tokens"] for s in self.by_step.values()),
            "total_tokens": self.total_tokens,
            "by_step": dict(self.by_step),
        }


class LLMClient:
    """Wrapper around Anthropic API."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.7) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.usage = TokenUsage()

    @traceable(name="llm_generate")
    def generate(self, system: str, user: str, step: str) -> str:
        """Generate a response and track tokens."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        self.usage.record(step, response.usage.input_tokens, response.usage.output_tokens)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text.strip()
