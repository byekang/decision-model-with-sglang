"""Thin Sonnet 4.5 wrapper for reasoning nodes (specialist summary, final respond)."""

import time
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    latency_ms: float
    usage: dict
    cost_usd: float


class BedrockLLM:
    def __init__(self, client, model_id: str, price: dict, name: str = "sonnet-4-5"):
        self.client = client
        self.model_id = model_id
        self.price = price
        self.name = name

    def generate(self, *, system: str, user: str, max_tokens: int = 400) -> LLMResult:
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model_id, max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
        )
        ms = (time.perf_counter() - t0) * 1000.0
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}
        cost = (usage["input_tokens"] * self.price["input_per_mtok"]
                + usage["output_tokens"] * self.price["output_per_mtok"]) / 1e6
        return LLMResult(text=text, latency_ms=round(ms, 2), usage=usage,
                         cost_usd=round(cost, 8))
