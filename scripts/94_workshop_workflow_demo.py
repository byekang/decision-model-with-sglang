"""Workshop Scenario A: three Korean customer questions through the full v2
multi-stage recursive workflow (agent/graph_v2.py), run on both decision backends.

Each question is scripted to exercise one signature path of the complex graph:
  ws-a1 : vague question  -> clarification loop -> tool success -> resolve
  ws-a2 : transfer delay  -> tool_error -> retry loop (alternative tool) -> resolve
  ws-a3 : unknown overseas approval -> success + fraud/AML flag -> KYC sub-flow

Prints a per-node timeline (decision vs LLM latency) per arm and saves
results/summary/workshop_workflow.json for the guide.
Usage: /data/venvs/agent/bin/python scripts/94_workshop_workflow_demo.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (BEDROCK_MODEL_IDS, PRICING, RAW_DIR, SUMMARY_DIR, log,
                    set_status)
from agent.bedrock_llm import BedrockLLM
from agent.decisions import HaikuDecision, SGLangDecision
from agent.graph_v2 import build_graph_v2

PHASE = "workshop_workflow"
SPEND_FILE = RAW_DIR.parent / "bedrock_spend.json"
SGLANG_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

SCENARIOS = [
    {
        "id": "ws-a1-vague-clarify",
        "title": "모호한 문의 -> clarification loop",
        "text_open": "카드가 안 긁히는데 어떻게 해야 하나요?",
        "script": {
            "vague": True,
            "clarify_reply": ("어제부터 오프라인 매장 단말기에서만 승인이 거절돼요. "
                              "관련 정보: Visa 체크카드(끝자리 7712), 마지막 시도 "
                              "2026-10-27 TechMart 강남점, 금액 45,000원입니다."),
            "no_info_reply": "지금 알려드릴 수 있는 정보는 그게 전부예요.",
            "tool_outcomes": ["success"], "risk_flag": False, "kyc_result": None,
        },
        "expected_path": "D1 -> D2 -> D5(need_clarification) -> 질문/답변 -> D5 -> D3 -> D6(goal_met) -> 요약 -> D7(no_risk) -> D4",
    },
    {
        "id": "ws-a2-retry-loop",
        "title": "시스템 오류 -> tool retry loop",
        "text_open": ("어제 오후에 보낸 해외송금이 아직 수취인에게 도착하지 않았어요. "
                      "송금 번호는 TR-20261026-8841, 금액은 USD 2,000입니다. 확인해 주세요."),
        "script": {
            "vague": False, "clarify_reply": None,
            "no_info_reply": "지금 알려드릴 수 있는 정보는 그게 전부예요.",
            "tool_outcomes": ["tool_error", "success"],
            "risk_flag": False, "kyc_result": None,
        },
        "expected_path": "D1 -> D2 -> D5 -> D3 -> tool_error -> D6(try_alternative) -> D3(대체 tool) -> D6(goal_met) -> 요약 -> D7 -> D4",
    },
    {
        "id": "ws-a3-risk-kyc",
        "title": "이상거래 -> risk verification sub-flow (KYC)",
        # 주의: "제가 쓰지 않은/모르는 해외 결제" 류 피해 신고 표현은 Qwen3-VL-8B가
        # D1에서 conf 1.0으로 sensitive_compliance 과차단함(probe 3/4 표현 차단).
        # KYC 경로 시연을 위해 중립 표현 사용 — guardrail 한계는 가이드에 명시.
        "text_open": ("카드 끝자리 7712로 해외 결제(USD 1,284, LUXE GADGETS) 승인이 "
                      "됐다는 알림을 받았는데, 이 거래를 확인하고 싶어요."),
        "script": {
            "vague": False, "clarify_reply": None,
            "no_info_reply": "지금 알려드릴 수 있는 정보는 그게 전부예요.",
            "tool_outcomes": ["success"], "risk_flag": True, "kyc_result": "verified",
        },
        "expected_path": "D1 -> D2 -> D5 -> D3 -> D6(goal_met) -> 요약 -> D7(needs_verification) -> KYC tool -> D8(verified) -> D4",
    },
]


def total_spend():
    return json.loads(SPEND_FILE.read_text())["usd"] if SPEND_FILE.exists() else 0.0


def add_spend(usd):
    SPEND_FILE.write_text(json.dumps({"usd": total_spend() + usd}))


def run(graph, sc, arm):
    t0 = time.perf_counter()
    out = graph.invoke(
        {"conv_id": f"{sc['id']}-{arm}", "arm": arm, "lang": "ko",
         "script": sc["script"],
         "dialogue": [{"role": "customer", "text": sc["text_open"]}],
         "trace": []},
        config={"recursion_limit": 60})
    e2e = (time.perf_counter() - t0) * 1000.0
    trace = out.get("trace", [])
    timeline = [{"node": ev["node"], "kind": ev["kind"],
                 "label": ev.get("label"),
                 "confidence": ev.get("confidence"),
                 "latency_ms": round(ev["latency_ms"], 1)} for ev in trace]
    dec_ms = sum(ev["latency_ms"] for ev in trace if ev["kind"] == "decision")
    llm_ms = sum(ev["latency_ms"] for ev in trace if ev["kind"] == "llm")
    cost = sum(ev.get("cost_usd", 0.0) for ev in trace)
    add_spend(cost)
    return {
        "arm": arm,
        "timeline": timeline,
        "dialogue": out.get("dialogue", []),
        "n_decisions": sum(1 for ev in trace if ev["kind"] == "decision"),
        "decision_ms": round(dec_ms, 1), "llm_ms": round(llm_ms, 1),
        "e2e_ms": round(e2e, 1),
        "final_response": out.get("final_response", ""),
        "bedrock_cost_usd": round(cost, 6),
    }


def main():
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region="us-west-2", max_retries=8)
    llm = BedrockLLM(client, BEDROCK_MODEL_IDS["sonnet-4-5"], PRICING["sonnet-4-5"])
    graphs = {
        "A_haiku": build_graph_v2(
            HaikuDecision(client, BEDROCK_MODEL_IDS["haiku-4-5"],
                          PRICING["haiku-4-5"]), llm),
        "B_sglang": build_graph_v2(SGLangDecision(model=SGLANG_MODEL), llm),
    }

    results = []
    for sc in SCENARIOS:
        # warm-up both arms once (uncounted) to exclude cold-start effects
        for arm, g in graphs.items():
            run(g, {**sc, "id": f"warm-{sc['id']}"}, arm)
        entry = {"id": sc["id"], "title": sc["title"],
                 "question": sc["text_open"], "expected_path": sc["expected_path"],
                 "runs": {}}
        for arm, g in graphs.items():
            r = run(g, sc, arm)
            entry["runs"][arm] = r
            log(PHASE, f"{sc['id']} [{arm}] n_dec={r['n_decisions']} "
                       f"dec={r['decision_ms']}ms llm={r['llm_ms']}ms "
                       f"e2e={r['e2e_ms']}ms")
        results.append(entry)

    out = {"scenarios": results, "session_spend_usd": round(total_spend(), 4)}
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / "workshop_workflow.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log(PHASE, f"wrote {path}; session spend=${total_spend():.3f}")
    set_status(PHASE, "demo", "done", note=f"spend ${total_spend():.3f}")


if __name__ == "__main__":
    main()
