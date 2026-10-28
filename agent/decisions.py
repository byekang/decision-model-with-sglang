"""Decision backends: Haiku 4.5 via Bedrock (Arm A) and SGLang /v1/systemone (Arm B).

Both receive the same question text and label set. Invalid outputs fall back to a
per-decision default so the graph always proceeds; invalid rate is a reported metric.
"""

import json
import re
import time
import urllib.request
from dataclasses import dataclass

from .prompts import HAIKU_DECISION_SYSTEM

FALLBACKS = {"d1": "in_scope", "d2": "general_faq", "d4": "clarify"}  # d3 -> first tool


@dataclass
class DecisionResult:
    label: str
    confidence: float | None
    latency_ms: float
    valid: bool
    usage: dict
    cost_usd: float
    raw: str


def _fallback(decision_id: str, choices: list[str]) -> str:
    return FALLBACKS.get(decision_id, choices[0])


class HaikuDecision:
    name = "haiku-4-5"

    def __init__(self, client, model_id: str, price: dict):
        self.client = client
        self.model_id = model_id
        self.price = price

    def decide(self, *, state_text: str, question: str, choices: list[str],
               decision_id: str) -> DecisionResult:
        labels = "\n".join(f"- {c}" for c in choices)
        user = f"{question}\n\nInput:\n{state_text}\n\nLabels:\n{labels}"
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model_id, max_tokens=64,
            system=HAIKU_DECISION_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        ms = (time.perf_counter() - t0) * 1000.0
        text = resp.content[0].text if resp.content else ""
        label, valid = None, False
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            label = json.loads(m.group(0))["label"]
            valid = label in choices
        except Exception:
            pass
        if not valid:
            label = _fallback(decision_id, choices)
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}
        cost = (usage["input_tokens"] * self.price["input_per_mtok"]
                + usage["output_tokens"] * self.price["output_per_mtok"]) / 1e6
        return DecisionResult(label=label, confidence=None, latency_ms=round(ms, 2),
                              valid=valid, usage=usage, cost_usd=round(cost, 8),
                              raw=text[:200])


class SGLangDecision:
    name = "sglang-qwen3-vl-8b"

    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct", port: int = 8001,
                 host: str | None = None):
        # host 우선순위: 인자 > (8002는 STRANDS_HOST 우선) > SGLANG_HOST > localhost
        # (workshop에서는 SGLang과 strands-decider가 각각 다른 EC2에서 서빙될 수 있음)
        import os
        self.model = model
        if host is None and port == 8002:
            host = os.environ.get("STRANDS_HOST")
        host = host or os.environ.get("SGLANG_HOST", "localhost")
        self.url = f"http://{host}:{port}/v1/systemone"

    def decide(self, *, state_text: str, question: str, choices: list[str],
               decision_id: str) -> DecisionResult:
        body = json.dumps({
            "state": state_text,
            "model": self.model,
            "questions": {decision_id: {
                "type": "choice",
                "instructions": question,
                "criteria": {c: None for c in choices},
            }},
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        resp = json.load(urllib.request.urlopen(req, timeout=120))
        ms = (time.perf_counter() - t0) * 1000.0
        ans = resp["answers"][decision_id]
        label, conf = ans["choice"], ans["confidence"]
        valid = label in choices
        if not valid:
            label = _fallback(decision_id, choices)
        return DecisionResult(label=label, confidence=conf, latency_ms=round(ms, 2),
                              valid=valid, usage=resp.get("usage", {}), cost_usd=0.0,
                              raw=json.dumps(ans)[:200])
