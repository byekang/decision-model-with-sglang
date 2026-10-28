"""Workshop demo: three realistic Korean banking-support cases, measuring how fast
the SGLang decision layer answers each decision vs Bedrock Haiku 4.5.

Case 1 (text)  : lost wallet -> card freeze. 4 decisions (guardrail D1, routing D2,
                 tool D3, escalation D4) via /v1/systemone (prefill-only) vs Haiku.
Case 2 (image) : customer disputes an overseas card-approval push (receipt screenshot).
                 2 VLM decisions (evidence type, incident class) via /v1/chat/completions
                 (Qwen3-VL reads the Korean screenshot) vs Haiku with image blocks.
Case 3 (image) : "is this bank SMS real?" — phishing vs legitimate screenshot, 1 VLM
                 decision on each image.

Each decision runs WARMUP un-timed + --reps timed calls per backend; p50/mean/p95,
label distribution and (systemone) confidence are reported.
Output: results/summary/workshop_latency.json
Usage: /data/venvs/agent/bin/python scripts/93_workshop_demo.py [--reps 10]
"""

import argparse
import os
import base64
import json
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (BEDROCK_MODEL_IDS, DATA_DIR, PRICING, RAW_DIR, SUMMARY_DIR,
                    log, set_status)
from agent.decisions import HaikuDecision, SGLangDecision
from agent.prompts import D1_QUESTION, D2_QUESTION, D3_QUESTION, D4_QUESTION
from agent.state import CATEGORIES, D1_CHOICES, D4_CHOICES
from agent.tools import CATEGORY_TOOLS

PHASE = "workshop_demo"
WARMUP = 2
IMG_DIR = DATA_DIR / "workshop"
SPEND_FILE = RAW_DIR.parent / "bedrock_spend.json"
SGLANG_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHAT_URL = f"http://{os.environ.get('SGLANG_HOST', 'localhost')}:8001/v1/chat/completions"

VLM_PROMPT = ('위 이미지를 보고 다음 질문에 답하라. {question}\n'
              '아래 Labels 중 정확히 하나를 골라 {{"label": "<label>"}} JSON 객체로만 '
              '응답하라. 다른 텍스트는 출력하지 마라.\n\nLabels:\n{labels}')

# ---------------------------------------------------------------- scenarios --
CASE1_MSG = "지갑을 도둑맞았어요. 카드 정지시켜 주세요."
CASE1_TOOL_CTX = (
    f"Customer message:\n{CASE1_MSG}\n\n"
    'Tool result:\n{"tool": "freeze_card", "status": "ok", '
    '"result": "The requested operation completed successfully in the banking system.", '
    '"reference_id": "REF-ws-case1"}\n\n'
    "Specialist facts:\n- 고객 카드(끝자리 7712) 즉시 정지 완료\n"
    "- 재발급 신청 가능 상태\n- 정지 시점 이후 승인 내역 없음")

CASE2_MSG = "방금 이 승인 알림을 받았는데 제가 결제한 게 아니에요. 어떻게 해야 하나요?"
CASE2_Q_DOCTYPE = "이 이미지는 어떤 유형의 화면인가?"
CASE2_L_DOCTYPE = ["card_approval_notice", "bank_statement", "sms_screenshot",
                   "id_document", "other"]
CASE2_Q_INCIDENT = ("고객이 이 화면을 첨부하며 본인이 결제한 것이 아니라고 말한다. "
                    "화면 내용(승인 유형, 가맹점, 금액)을 근거로 이 사건을 분류하면?")
CASE2_L_INCIDENT = ["unauthorized_overseas_payment", "duplicate_charge",
                    "wrong_amount", "normal_payment"]

CASE3_MSG = "은행에서 이런 문자가 왔는데 링크를 눌러야 하나요?"
CASE3_Q = ("이 문자 메시지 화면이 은행의 정상 안내인지, 피싱(스미싱) 의심인지 판별하라. "
           "발신번호, 링크 주소, 요구 내용(보안카드 번호 등)을 근거로 판단하라.")
CASE3_L = ["phishing_suspected", "legitimate_notice"]


def total_spend():
    return json.loads(SPEND_FILE.read_text())["usd"] if SPEND_FILE.exists() else 0.0


def add_spend(usd):
    SPEND_FILE.write_text(json.dumps({"usd": total_spend() + usd}))


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def parse_label(text, valid):
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        label = json.loads(m.group(0))["label"]
        return label, label in valid
    except Exception:
        return text.strip()[:60], False


def sglang_vlm_decide(image: Path, question: str, labels: list[str]):
    body = json.dumps({
        "model": SGLANG_MODEL, "temperature": 0, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(image)}},
            {"type": "text", "text": VLM_PROMPT.format(
                question=question, labels="\n".join(f"- {l}" for l in labels))},
        ]}],
    }, ensure_ascii=False).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    ms = (time.perf_counter() - t0) * 1000.0
    label, valid = parse_label(resp["choices"][0]["message"]["content"] or "", set(labels))
    return label, None, ms, valid, 0.0


def make_haiku_vlm(client):
    price = PRICING["haiku-4-5"]

    def decide(image: Path, question: str, labels: list[str]):
        img_block = {"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(image.read_bytes()).decode()}}
        txt = VLM_PROMPT.format(question=question,
                                labels="\n".join(f"- {l}" for l in labels))
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=BEDROCK_MODEL_IDS["haiku-4-5"], max_tokens=64,
            messages=[{"role": "user", "content": [img_block,
                                                   {"type": "text", "text": txt}]}])
        ms = (time.perf_counter() - t0) * 1000.0
        text = resp.content[0].text if resp.content else ""
        label, valid = parse_label(text, set(labels))
        cost = (resp.usage.input_tokens * price["input_per_mtok"]
                + resp.usage.output_tokens * price["output_per_mtok"]) / 1e6
        return label, None, ms, valid, cost
    return decide


def stats(samples):
    lat = sorted(s[2] for s in samples)
    k = lambda p: lat[max(0, min(len(lat) - 1, round(p / 100 * (len(lat) - 1))))]
    confs = [s[1] for s in samples if s[1] is not None]
    cost = sum(s[4] for s in samples)
    return {"n": len(samples),
            "labels": dict(Counter(s[0] for s in samples)),
            "valid_rate": round(sum(s[3] for s in samples) / len(samples), 3),
            "latency_ms": {"p50": round(k(50), 1), "mean": round(
                statistics.mean(lat), 1), "p95": round(k(95), 1)},
            "confidence_mean": (round(statistics.mean(confs), 4) if confs else None),
            "bedrock_cost_usd": round(cost, 6)}


def run_decision(name, backends, call_args, reps):
    """backends: {bk_name: callable(*call_args) -> (label, conf, ms, valid, cost)}"""
    out = {}
    for bk, fn in backends.items():
        for _ in range(WARMUP):
            fn(*call_args)
        samples = []
        for _ in range(reps):
            s = fn(*call_args)
            samples.append(s)
            add_spend(s[4])
        out[bk] = stats(samples)
        log(PHASE, f"{name} [{bk}] p50={out[bk]['latency_ms']['p50']}ms "
                   f"labels={out[bk]['labels']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region="us-west-2", max_retries=8)
    sg = SGLangDecision(model=SGLANG_MODEL)
    hk = HaikuDecision(client, BEDROCK_MODEL_IDS["haiku-4-5"], PRICING["haiku-4-5"])
    hk_vlm = make_haiku_vlm(client)

    def sg_txt(state, question, choices, did):
        r = sg.decide(state_text=state, question=question, choices=choices,
                      decision_id=did)
        return r.label, r.confidence, r.latency_ms, r.valid, 0.0

    def hk_txt(state, question, choices, did):
        r = hk.decide(state_text=state, question=question, choices=choices,
                      decision_id=did)
        return r.label, r.confidence, r.latency_ms, r.valid, r.cost_usd

    results = {"reps": args.reps, "cases": {}}

    # Case 1: lost card (text, 4 decisions)
    case1 = {}
    for did, q, choices, state in (
            ("d1_guardrail", D1_QUESTION, D1_CHOICES, CASE1_MSG),
            ("d2_routing", D2_QUESTION, CATEGORIES, CASE1_MSG),
            ("d3_tool", D3_QUESTION.format(category="card_security"),
             CATEGORY_TOOLS["card_security"], CASE1_MSG),
            ("d4_escalation", D4_QUESTION, D4_CHOICES, CASE1_TOOL_CTX)):
        case1[did] = run_decision(
            f"case1/{did}",
            {"sglang_systemone": sg_txt, "bedrock_haiku": hk_txt},
            (state, q, choices, did.split("_")[0]), args.reps)
    results["cases"]["case1_lost_card"] = {
        "customer_message": CASE1_MSG, "decisions": case1}

    # Case 2: disputed overseas approval (image, 2 VLM decisions)
    receipt = IMG_DIR / "receipt_dispute.png"
    case2 = {
        "v1_evidence_type": run_decision(
            "case2/v1_evidence_type",
            {"sglang_qwen3vl_chat": sglang_vlm_decide, "bedrock_haiku": hk_vlm},
            (receipt, CASE2_Q_DOCTYPE, CASE2_L_DOCTYPE), args.reps),
        "v2_incident_class": run_decision(
            "case2/v2_incident_class",
            {"sglang_qwen3vl_chat": sglang_vlm_decide, "bedrock_haiku": hk_vlm},
            (receipt, CASE2_Q_INCIDENT, CASE2_L_INCIDENT), args.reps),
    }
    results["cases"]["case2_dispute_receipt"] = {
        "customer_message": CASE2_MSG, "image": str(receipt), "decisions": case2}

    # Case 3: phishing SMS triage (image, 1 VLM decision x 2 images)
    case3 = {}
    for key, img, gold in (("phishing_sms", IMG_DIR / "phishing_sms.png",
                            "phishing_suspected"),
                           ("legit_sms", IMG_DIR / "legit_sms.png",
                            "legitimate_notice")):
        case3[key] = run_decision(
            f"case3/{key}",
            {"sglang_qwen3vl_chat": sglang_vlm_decide, "bedrock_haiku": hk_vlm},
            (img, CASE3_Q, CASE3_L), args.reps)
        case3[key]["gold"] = gold
    results["cases"]["case3_sms_triage"] = {
        "customer_message": CASE3_MSG, "decisions": case3}

    results["session_spend_usd"] = round(total_spend(), 4)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = SUMMARY_DIR / "workshop_latency.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log(PHASE, f"wrote {out}; session spend=${total_spend():.3f}")
    set_status(PHASE, "demo", "done", note=f"spend ${total_spend():.3f}")


if __name__ == "__main__":
    main()
