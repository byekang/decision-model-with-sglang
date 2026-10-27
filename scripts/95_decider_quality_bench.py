"""Decision backend quality bench on the Korean v2 dataset (user hypothesis check).

Hypothesis: strands-decider(2B, 영어 중심) < Laya(multilingual) < SGLang Qwen3-VL-8B
on Korean banking decisions. Measures D1(3-way guardrail, n=160) and
D2(12-way routing, n=140, strict/lenient) accuracy + warm latency per backend,
over data/agent_convs_v2ko.jsonl text_open. All local — no Bedrock cost.

Backends: sglang(:8001 systemone), strands(:8002 systemone),
laya(in-process Router, model="multilingual" 고정, max_len=8192).
Output: results/summary/decider_quality_ko.json
Usage: HF_HOME=/data/hf-cache /data/venvs/agent/bin/python scripts/95_decider_quality_bench.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_DIR, SUMMARY_DIR, log, read_jsonl
from agent.decisions import SGLangDecision
from agent.prompts import D1_QUESTION, D2_QUESTION
from agent.state import CATEGORIES, D1_CHOICES

PHASE = "decider_quality"


class LayaBackend:
    name = "laya-multilingual"

    def __init__(self):
        from laya import Router
        self.router = Router()

    def decide(self, state_text, question, choices):
        qs = {"q": {"type": "choice", "instructions": question,
                    "criteria": {c: None for c in choices}}}
        t0 = time.perf_counter()
        r = self.router.predict(state_text, qs, model="multilingual", max_len=8192)
        ms = (time.perf_counter() - t0) * 1000
        a = r["answers"]["q"]
        return a["choice"], a.get("confidence"), ms


class SystemOneBackend:
    def __init__(self, name, port):
        self.name = name
        self.client = SGLangDecision(model=name, port=port)

    def decide(self, state_text, question, choices):
        r = self.client.decide(state_text=state_text, question=question,
                               choices=choices, decision_id="q")
        return r.label, r.confidence, r.latency_ms


def main():
    convs = read_jsonl(DATA_DIR / "agent_convs_v2ko.jsonl")
    backends = [
        SystemOneBackend("sglang-qwen3vl-8b", 8001),
        SystemOneBackend("strands-decider-2b", 8002),
        LayaBackend(),
    ]
    out = {}
    for bk in backends:
        # warm-up
        for _ in range(3):
            bk.decide(convs[0]["text_open"], D1_QUESTION, D1_CHOICES)
        d1_ok = d2_ok = d2_len_ok = 0
        d1_n = d2_n = 0
        lats = []
        t_start = time.perf_counter()
        for c in convs:
            label, conf, ms = bk.decide(c["text_open"], D1_QUESTION, D1_CHOICES)
            lats.append(ms)
            d1_n += 1
            d1_ok += label == c["gold"]["d1"]
            if c["gold"]["d2"]:
                label2, _, ms2 = bk.decide(c["text_open"], D2_QUESTION, CATEGORIES)
                lats.append(ms2)
                d2_n += 1
                hit = label2 == c["gold"]["d2"]
                d2_ok += hit
                d2_len_ok += hit or label2 in c["gold"].get("d2_accept_also", [])
        res = {
            "d1_acc": round(d1_ok / d1_n, 4), "d1_n": d1_n,
            "d2_strict": round(d2_ok / d2_n, 4),
            "d2_lenient": round(d2_len_ok / d2_n, 4), "d2_n": d2_n,
            "latency_ms_p50": round(statistics.median(lats), 1),
            "latency_ms_mean": round(statistics.mean(lats), 1),
            "wall_s": round(time.perf_counter() - t_start, 1),
        }
        out[bk.name] = res
        log(PHASE, f"{bk.name}: d1={res['d1_acc']} d2_strict={res['d2_strict']} "
                   f"d2_lenient={res['d2_lenient']} p50={res['latency_ms_p50']}ms")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    p = SUMMARY_DIR / "decider_quality_ko.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log(PHASE, f"wrote {p}")


if __name__ == "__main__":
    main()
