"""CPU-only decision backend bench on the Korean v2 dataset (GPU 불가 시나리오).

GPU quota가 0인 계정 대비: 판단 backend를 CPU만으로 돌렸을 때의 품질/latency 실측.
scripts/95와 같은 데이터(agent_convs_v2ko.jsonl)·지표(D1 3지 n=160, D2 12지 n=140).

Backends:
- laya-multilingual (322M, in-process, CPU 강제)
- strands-decider-2b-cpu (:8003, --device cpu 로 별도 기동)
- HF prefill-only scorer (transformers, CPU, fp32): Qwen3-0.6B / 1.7B / 4B
- qwen3-vl-8b-cpu (text-only, subsample) — GPU 모델의 CPU 하한 참고치

Thread cap: THREADS env (기본 8 = m5/c7i.2xlarge급 가정).
Usage:
  CUDA_VISIBLE_DEVICES= THREADS=8 HF_HOME=/data/hf-cache \
    /data/venvs/agent/bin/python scripts/96_cpu_decider_bench.py
Output: results/summary/decider_quality_cpu.json
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
THREADS = int(os.environ.get("THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_DIR, SUMMARY_DIR, log, read_jsonl  # noqa: E402
from agent.decisions import SGLangDecision  # noqa: E402
from agent.prompts import D1_QUESTION, D2_QUESTION  # noqa: E402
from agent.state import CATEGORIES, D1_CHOICES  # noqa: E402

PHASE = "cpu_decider"
LETTERS = "ABCDEFGHIJKL"


class LayaBackend:
    name = "laya-322m-cpu"

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


class HFPrefillBackend:
    """transformers로 prefill 1회 -> 선택지 letter token 확률만 읽는 decision 호출.

    systemone과 같은 원리(decode 없음). prompt wording은 서버 구현과 다르므로
    품질 수치는 '동일 방식의 근사치'로 해석할 것.
    """

    def __init__(self, model_id, short, dtype="float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(THREADS)
        self.name = short
        self.tok = AutoTokenizer.from_pretrained(model_id)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=getattr(torch, dtype)).eval()
        except Exception:  # VL 모델(text-only 사용)
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=getattr(torch, dtype)).eval()
        self.letter_ids = [self.tok.encode(l, add_special_tokens=False)[0]
                           for l in LETTERS]

    def close(self):
        import gc
        del self.model
        gc.collect()

    def decide(self, state_text, question, choices):
        torch = self.torch
        lines = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
        user = (f"{question}\n\nInput:\n{state_text}\n\nChoices:\n{lines}\n\n"
                f"Answer with the single letter of the best choice.")
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(text, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
        ms = (time.perf_counter() - t0) * 1000
        sub = logits[self.letter_ids[:len(choices)]]
        probs = torch.softmax(sub, dim=-1)
        k = int(probs.argmax())
        return choices[k], float(probs[k]), ms


def run(bk, convs, lats_extra=None):
    for _ in range(2):
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
    return {
        "d1_acc": round(d1_ok / d1_n, 4), "d1_n": d1_n,
        "d2_strict": round(d2_ok / d2_n, 4) if d2_n else None,
        "d2_lenient": round(d2_len_ok / d2_n, 4) if d2_n else None, "d2_n": d2_n,
        "latency_ms_p50": round(statistics.median(lats), 1),
        "latency_ms_mean": round(statistics.mean(lats), 1),
        "wall_s": round(time.perf_counter() - t_start, 1),
        "threads": THREADS,
    }


def main():
    convs = read_jsonl(DATA_DIR / "agent_convs_v2ko.jsonl")
    out_path = SUMMARY_DIR / "decider_quality_cpu.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}

    plan = []
    only = os.environ.get("ONLY", "")
    def want(name):
        return (not only) or (name in only)

    if want("laya"):
        plan.append(("laya-322m-cpu", lambda: LayaBackend(), convs))
    if want("strands"):
        plan.append(("strands-decider-2b-cpu",
                     lambda: SystemOneBackend("strands-decider-2b-cpu", 8003), convs))
    if want("qwen06"):
        plan.append(("qwen3-0.6b-cpu-prefill",
                     lambda: HFPrefillBackend("Qwen/Qwen3-0.6B", "qwen3-0.6b-cpu-prefill"), convs))
    if want("qwen17"):
        plan.append(("qwen3-1.7b-cpu-prefill",
                     lambda: HFPrefillBackend("Qwen/Qwen3-1.7B", "qwen3-1.7b-cpu-prefill"), convs))
    if want("qwen4b"):
        plan.append(("qwen3-4b-cpu-prefill",
                     lambda: HFPrefillBackend("Qwen/Qwen3-4B", "qwen3-4b-cpu-prefill"), convs))
    if want("vl8b"):
        plan.append(("qwen3-vl-8b-cpu-prefill-subsample",
                     lambda: HFPrefillBackend("Qwen/Qwen3-VL-8B-Instruct",
                                              "qwen3-vl-8b-cpu-prefill-subsample"),
                     convs[:40]))

    for name, mk, data in plan:
        log(PHASE, f"== {name} (threads={THREADS}, n={len(data)}) ==")
        try:
            bk = mk()
            res = run(bk, data)
            if hasattr(bk, "close"):
                bk.close()
        except Exception as exc:
            res = {"error": str(exc)[:300]}
        out[name] = res
        log(PHASE, f"{name}: {json.dumps(res, ensure_ascii=False)}")
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log(PHASE, f"wrote {out_path}")


if __name__ == "__main__":
    main()
