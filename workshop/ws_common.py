"""Workshop 공용 helper. / Shared workshop helpers.

- make_clients()/make_graphs(): Bedrock/SGLang 클라이언트와 두 arm의 v2 graph 생성
- run_verbose(): graph.stream(stream_mode="updates")로 노드 실행을 실시간 수신,
  "STEP n" 형식으로 각 노드의 결정(label/confidence/latency)·생성 텍스트·tool 결과를
  출력하고 수행시간 집계를 반환
- compare(): 두 arm 실행 결과의 수행시간 비교표(markdown) 출력
- vlm_sglang()/vlm_haiku(): 이미지 판단 1회 호출 (notebook 05)

언어: DMWS_LANG 환경변수 ('ko' 기본 | 'en') — import 전에 설정해야 합니다.
Language: set env DMWS_LANG ('ko' default | 'en') BEFORE importing this module;
it localizes step output, comparison tables and graph captions.
"""

import base64
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

# 프로젝트 루트: DMWS_ROOT 환경변수 > 이 파일 기준 상위 폴더 (repo clone 위치 무관)
ROOT = Path(os.environ.get("DMWS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# 모델 서버 호스트: SGLANG_HOST 환경변수 > localhost (SGLang :8001 전용)
# strands-decider(:8002)는 별도 EC2일 수 있어 STRANDS_HOST를 우선 사용
# (agent/decisions.py의 SGLangDecision(port=8002)이 자동 처리)
SERVER_HOST = os.environ.get("SGLANG_HOST", "localhost")
STRANDS_HOST = os.environ.get("STRANDS_HOST", SERVER_HOST)

# 출력/그래프 캡션 언어 ('ko' | 'en')
LANG = os.environ.get("DMWS_LANG", "ko")
if LANG not in ("ko", "en"):
    LANG = "ko"

from common import BEDROCK_MODEL_IDS, PRICING                      # noqa: E402
from agent.bedrock_llm import BedrockLLM                           # noqa: E402
from agent.decisions import HaikuDecision, SGLangDecision          # noqa: E402
from agent.graph_v2 import build_graph_v2                          # noqa: E402

SGLANG_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NO_INFO = "지금 알려드릴 수 있는 정보는 그게 전부예요."
NO_INFO_EN = "That's all the information I have right now."

# scripts/94에서 검증된 상담 시나리오 3종 (고객 발화는 한국어 능력 시연을 위해 항상 한국어;
# *_en 필드는 EN 노트북 표시용 번역)
SCENARIOS = {
    1: {
        "title": "모호한 문의 -> clarification loop",
        "title_en": "Vague inquiry -> clarification loop",
        "question": "카드가 안 긁히는데 어떻게 해야 하나요?",
        "question_en": '"My card is not swiping. What should I do?"',
        "script": {"vague": True,
                   "clarify_reply": ("어제부터 오프라인 매장 단말기에서만 승인이 거절돼요. "
                                     "관련 정보: Visa 체크카드(끝자리 7712), 마지막 시도 "
                                     "2026-10-27 TechMart 강남점, 금액 45,000원입니다."),
                   "no_info_reply": NO_INFO,
                   "tool_outcomes": ["success"], "risk_flag": False, "kyc_result": None},
    },
    2: {
        "title": "시스템 오류 -> tool retry loop",
        "title_en": "System error -> tool retry loop",
        "question": ("어제 오후에 보낸 해외송금이 아직 수취인에게 도착하지 않았어요. "
                     "송금 번호는 TR-20261026-8841, 금액은 USD 2,000입니다. 확인해 주세요."),
        "question_en": ('"The overseas transfer I sent yesterday has not reached the '
                        'recipient. Reference TR-20261026-8841, USD 2,000. Please check."'),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO,
                   "tool_outcomes": ["tool_error", "success"],
                   "risk_flag": False, "kyc_result": None},
    },
    3: {
        "title": "이상거래 모니터링 flag -> KYC sub-flow",
        "title_en": "Fraud-monitoring flag -> KYC sub-flow",
        "question": ("카드 끝자리 7712로 해외 결제(USD 1,284, LUXE GADGETS) 승인이 "
                     "됐다는 알림을 받았는데, 이 거래를 확인하고 싶어요."),
        "question_en": ('"I got an approval alert for an overseas charge (USD 1,284, '
                        'LUXE GADGETS) on my card ending 7712. I would like to verify '
                        'this transaction."'),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO,
                   "tool_outcomes": ["success"], "risk_flag": True,
                   "kyc_result": "verified"},
    },
}

# EN 워크샵용 영어 고객 발화 (KR과 동일한 시나리오 구조, script 분기 동일)
SCENARIOS_EN = {
    1: {
        "title": "Vague inquiry -> clarification loop",
        "question": "My card doesn't swipe. What should I do?",
        "script": {"vague": True,
                   "clarify_reply": ("Since yesterday the card won't go through at "
                                     "in-store terminals only. Details: Visa debit "
                                     "card ending 7712, last attempt on 2026-10-27 at "
                                     "TechMart Gangnam, amount USD 34.60."),
                   "no_info_reply": NO_INFO_EN,
                   "tool_outcomes": ["success"], "risk_flag": False, "kyc_result": None},
    },
    2: {
        "title": "System error -> tool retry loop",
        "question": ("The overseas transfer I sent yesterday afternoon has not reached "
                     "the recipient yet. Reference number TR-20261026-8841, amount "
                     "USD 2,000. Please check what happened."),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO_EN,
                   "tool_outcomes": ["tool_error", "success"],
                   "risk_flag": False, "kyc_result": None},
    },
    3: {
        "title": "Fraud-monitoring flag -> KYC sub-flow",
        "question": ("I received an approval alert for an overseas charge "
                     "(USD 1,284, LUXE GADGETS) on my card ending 7712, and I would "
                     "like to verify this transaction."),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO_EN,
                   "tool_outcomes": ["success"], "risk_flag": True,
                   "kyc_result": "verified"},
    },
}


def scenario(n, lang=None):
    """언어에 맞는 시나리오 반환 (lang 기본: DMWS_LANG). EN은 영어 고객 발화 사용."""
    return (SCENARIOS_EN if (lang or LANG) == "en" else SCENARIOS)[n]

_NODE_DESC = {
    "ko": {
        "guardrail_check": "D1 접수 심사: 정상 문의 / 범위 밖 / compliance 위반",
        "intent_router": "D2 업무 분류: 12개 전문 상담 category 중 선택",
        "clarify_gate": "D5 정보 충분성 판정: 바로 처리 가능한가, 되물어야 하는가",
        "ask_clarify": "되묻기: Sonnet이 질문 생성 -> 고객 답변 수신",
        "tool_step": "D3 tool 선택 -> tool 실행 -> D6 결과 검토(성공/재시도/포기)",
        "give_up": "tool 처리 포기 -> 상담사 이관 경로",
        "summarize": "Sonnet이 tool 결과를 고객 응대용 사실 요약으로 정리",
        "risk_check": "D7 리스크 점검: fraud/AML 모니터링 flag 유무",
        "kyc_verify": "KYC 재검증 tool 실행 -> D8 통과 여부 판정",
        "escalation_gate": "D4 처리 방침: 종결 / 추가 질문 / 상담사 이관",
        "respond": "Sonnet이 최종 고객 응답 작성",
    },
    "en": {
        "guardrail_check": "D1 intake screening: in-scope / out-of-scope / compliance",
        "intent_router": "D2 routing: pick one of 12 specialist teams",
        "clarify_gate": "D5 sufficiency check: can we act now, or must we ask back?",
        "ask_clarify": "Clarify: Sonnet writes a question -> customer reply received",
        "tool_step": "D3 tool choice -> run tool -> D6 result review (done/retry/give up)",
        "give_up": "Give up automated handling -> human handoff path",
        "summarize": "Sonnet condenses the tool result into customer-facing facts",
        "risk_check": "D7 risk check: fraud/AML monitoring flag present?",
        "kyc_verify": "Run KYC re-verification tool -> D8 pass/fail decision",
        "escalation_gate": "D4 final action: resolve / clarify / human handoff",
        "respond": "Sonnet writes the final customer reply",
    },
}
NODE_DESC = _NODE_DESC[LANG]

# run_verbose/compare/compare_image 출력 문자열
_UI = {
    "ko": {
        "customer": "고객", "agent": "상담 agent",
        "decide": "판정", "gen": "생성(Sonnet 4.5)", "tool": "tool 실행",
        "auto": "tool 자동 선택(남은 후보 1개)",
        "summary": "요약(첫 줄)", "final": "--- 최종 응답 ---",
        "totals": ">>> [{arm}] 결정 {n}회 합계 {dec:,.0f}ms | 생성 {llm:,.0f}ms | e2e {e2e:,.0f}ms",
        "h_metric": "지표", "h_ratio": "배수", "h_ndec": "결정 횟수",
        "h_decsum": "**결정 latency 합계**", "h_llm": "생성(LLM) latency",
        "speedup": "**결정 계층 속도 차이: {r:,.1f}x** (e2e 단축 {d:,.0f}ms)",
        "per_dec": "**결정별 latency**",
        "h_node": "노드", "h_dec": "결정", "h_label": "판정 label",
        "same": "{l} (동일)",
        "img_hdr": "[이미지 판단] {t}", "img_q": "질문", "img_verdict": "판정",
        "team_cluster": "12개 전문 상담팀 (D2 선택지)", "team_assign": "1개 팀 배정",
    },
    "en": {
        "customer": "customer", "agent": "support agent",
        "decide": "decision", "gen": "generate (Sonnet 4.5)", "tool": "tool run",
        "auto": "tool auto-selected (single remaining option)",
        "summary": "summary (first line)", "final": "--- final reply ---",
        "totals": ">>> [{arm}] {n} decisions, total {dec:,.0f}ms | generation {llm:,.0f}ms | e2e {e2e:,.0f}ms",
        "h_metric": "metric", "h_ratio": "ratio", "h_ndec": "decisions",
        "h_decsum": "**decision latency total**", "h_llm": "generation (LLM) latency",
        "speedup": "**decision-layer speedup: {r:,.1f}x** (e2e saved {d:,.0f}ms)",
        "per_dec": "**per-decision latency**",
        "h_node": "node", "h_dec": "decision", "h_label": "label",
        "same": "{l} (same)",
        "img_hdr": "[image decision] {t}", "img_q": "question", "img_verdict": "label",
        "team_cluster": "12 specialist teams (D2 choices)", "team_assign": "assign 1 team",
    },
}
UI = _UI[LANG]


def make_clients():
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region="us-west-2", max_retries=8)
    llm = BedrockLLM(client, BEDROCK_MODEL_IDS["sonnet-4-5"], PRICING["sonnet-4-5"])
    return client, llm


def make_graphs():
    client, llm = make_clients()
    return {
        "Haiku": build_graph_v2(
            HaikuDecision(client, BEDROCK_MODEL_IDS["haiku-4-5"],
                          PRICING["haiku-4-5"]), llm),
        "SGLang": build_graph_v2(SGLangDecision(model=SGLANG_MODEL), llm),
    }, client


def _print_event(ev):
    kind = ev["kind"]
    if kind == "decision":
        conf = f", confidence {ev['confidence']:.3f}" if ev.get("confidence") is not None else ""
        print(f"      {UI['decide']}[{ev.get('decision','?').upper()}] -> {ev['label']}"
              f"   ({ev['latency_ms']:,.0f}ms{conf})")
    elif kind == "llm":
        print(f"      {UI['gen']}   ({ev['latency_ms']:,.0f}ms)")
    elif kind == "tool":
        print(f"      {UI['tool']}: {ev['node'].replace('tool:', '')}"
              f"   ({ev['latency_ms']:,.1f}ms)")
    elif kind == "auto":
        print(f"      {UI['auto']}: {ev['label']}")


def run_verbose(graph, question, script, arm_name):
    print(f"\n{'=' * 86}\n[{arm_name}] {UI['customer']}: {question}\n{'=' * 86}")
    t0 = time.perf_counter()
    step = 0
    events = []
    stream = graph.stream(
        {"conv_id": f"ws-{arm_name}-{int(t0)}", "arm": arm_name, "lang": LANG,
         "script": script, "trace": [],
         "dialogue": [{"role": "customer", "text": question}]},
        config={"recursion_limit": 60}, stream_mode="updates")
    for update in stream:
        for node, delta in update.items():
            delta = delta or {}
            step += 1
            print(f"\nSTEP {step} ▶ {node} — {NODE_DESC.get(node, '')}")
            for ev in delta.get("trace", []):
                events.append(ev)
                _print_event(ev)
            for turn in delta.get("dialogue", []):
                who = UI["agent"] if turn["role"] == "agent" else UI["customer"]
                print(f"      {who}: {turn['text']}")
            if delta.get("specialist_summary"):
                head = delta["specialist_summary"].strip().splitlines()[0]
                print(f"      {UI['summary']}: {head[:80]}")
            if delta.get("final_response"):
                print(f"      {UI['final']}\n      "
                      + delta["final_response"].strip().replace("\n", "\n      "))
    e2e = (time.perf_counter() - t0) * 1000.0
    dec = [e for e in events if e["kind"] == "decision"]
    res = {
        "arm": arm_name,
        "n_decisions": len(dec),
        "decision_ms": sum(e["latency_ms"] for e in dec),
        "llm_ms": sum(e["latency_ms"] for e in events if e["kind"] == "llm"),
        "e2e_ms": e2e,
        "decisions": [(e["node"], e.get("decision"), e["label"], e["latency_ms"])
                      for e in dec],
    }
    print("\n" + UI["totals"].format(arm=arm_name, n=res["n_decisions"],
                                     dec=res["decision_ms"], llm=res["llm_ms"],
                                     e2e=res["e2e_ms"]))
    return res


def compare(res_a, res_b):
    """두 arm 수행시간 비교를 markdown 표로 display (res_a=Haiku, res_b=SGLang 권장)."""
    from IPython.display import Markdown, display
    a, b = res_a, res_b
    ratio = a["decision_ms"] / b["decision_ms"] if b["decision_ms"] else float("nan")
    def x(va, vb):
        return f"**{va / vb:,.1f}x**" if vb else "-"
    md = [f"| {UI['h_metric']} | {a['arm']} | {b['arm']} | {UI['h_ratio']}({a['arm']}/{b['arm']}) |",
          "|---|---:|---:|---:|",
          f"| {UI['h_ndec']} | {a['n_decisions']} | {b['n_decisions']} | |",
          f"| {UI['h_decsum']} | **{a['decision_ms']:,.0f}ms** | **{b['decision_ms']:,.0f}ms** "
          f"| {x(a['decision_ms'], b['decision_ms'])} |",
          f"| {UI['h_llm']} | {a['llm_ms']:,.0f}ms | {b['llm_ms']:,.0f}ms "
          f"| {x(a['llm_ms'], b['llm_ms'])} |",
          f"| e2e | {a['e2e_ms']:,.0f}ms | {b['e2e_ms']:,.0f}ms "
          f"| {x(a['e2e_ms'], b['e2e_ms'])} |",
          "",
          UI["speedup"].format(r=ratio, d=a["e2e_ms"] - b["e2e_ms"]),
          "",
          UI["per_dec"],
          "",
          f"| {UI['h_node']} | {UI['h_dec']} | {a['arm']} | {b['arm']} "
          f"| {UI['h_ratio']} | {UI['h_label']} |",
          "|---|---|---:|---:|---:|---|"]
    for (na, da, la, ma), (nb, db, lb, mb) in zip(a["decisions"], b["decisions"]):
        same = UI["same"].format(l=la) if la == lb \
            else f"{a['arm']}={la} / {b['arm']}={lb}"
        md.append(f"| {na} | {(da or '?').upper()} | {ma:,.0f}ms | {mb:,.0f}ms "
                  f"| {x(ma, mb)} | {same} |")
    display(Markdown("\n".join(md)))


# ------------------------------------------------------------ graph drawing --
# 노드 캡션 (그래프 시각화 전용, 언어별)
_NODES = {
    "ko": {
        "START": None, "END": None,
        "guardrail_check": "guardrail_check\n[D1] 접수 심사\n정상 / 범위밖 / 컴플라이언스",
        "intent_router": "intent_router\n[D2] 업무 분류\n(12개 전문 상담팀)",
        "clarify_gate": "clarify_gate\n[D5] 정보 충분성 판정",
        "ask_clarify": "ask_clarify\n되묻기 질문 생성(Sonnet)\n+ 고객 답변 수신",
        "tool_step": "tool_step\n[D3] tool 선택 -> 실행\n[D6] 결과 검토",
        "give_up": "give_up\n자동 처리 포기",
        "summarize": "summarize\n사실 요약 (Sonnet)",
        "risk_check": "risk_check\n[D7] 리스크 점검\n(fraud/AML flag)",
        "kyc_verify": "kyc_verify\nKYC 재검증 tool\n[D8] 통과 판정",
        "escalation_gate": "escalation_gate\n[D4] 처리 방침\n종결 / 추가질문 / 이관",
        "respond": "respond\n최종 응답 작성 (Sonnet)",
    },
    "en": {
        "START": None, "END": None,
        "guardrail_check": "guardrail_check\n[D1] intake screening\nin-scope / out-of-scope / compliance",
        "intent_router": "intent_router\n[D2] routing\n(12 specialist teams)",
        "clarify_gate": "clarify_gate\n[D5] information sufficiency",
        "ask_clarify": "ask_clarify\nwrite clarifying question (Sonnet)\n+ receive customer reply",
        "tool_step": "tool_step\n[D3] pick tool -> run\n[D6] review result",
        "give_up": "give_up\nabandon automated handling",
        "summarize": "summarize\nfact summary (Sonnet)",
        "risk_check": "risk_check\n[D7] risk check\n(fraud/AML flag)",
        "kyc_verify": "kyc_verify\nKYC re-verification tool\n[D8] pass decision",
        "escalation_gate": "escalation_gate\n[D4] final action\nresolve / clarify / handoff",
        "respond": "respond\nwrite final reply (Sonnet)",
    },
}
_DECISION_NODES = {"guardrail_check", "intent_router", "clarify_gate", "tool_step",
                   "risk_check", "kyc_verify", "escalation_gate"}
_LLM_NODES = {"ask_clarify", "summarize", "respond"}

# (src, dst, edge label, style) — style: "", "dashed", "loop"(빨간 굵은선)
_EDGES = {
    "ko": [
        ("START", "guardrail_check", "", ""),
        ("guardrail_check", "intent_router", "정상 문의", ""),
        ("guardrail_check", "respond", "범위밖/민감\n(거절 응답 직행)", "dashed"),
        ("intent_router", "clarify_gate", "", ""),
        ("clarify_gate", "ask_clarify", "정보 부족", "loop"),
        ("ask_clarify", "clarify_gate", "재판정 (loop, 최대 2회)", "loop"),
        ("clarify_gate", "tool_step", "정보 충분 /\n되묻기 상한 도달", ""),
        ("tool_step", "tool_step", "재시도 (loop,\n대체 tool, 최대 2회)", "loop"),
        ("tool_step", "summarize", "tool 성공", ""),
        ("tool_step", "give_up", "포기", "dashed"),
        ("give_up", "escalation_gate", "", "dashed"),
        ("summarize", "risk_check", "", ""),
        ("risk_check", "kyc_verify", "검증 필요", ""),
        ("kyc_verify", "escalation_gate", "", ""),
        ("risk_check", "escalation_gate", "이상 없음", ""),
        ("escalation_gate", "respond", "", ""),
        ("respond", "END", "", ""),
    ],
    "en": [
        ("START", "guardrail_check", "", ""),
        ("guardrail_check", "intent_router", "in-scope", ""),
        ("guardrail_check", "respond", "out-of-scope/sensitive\n(straight to refusal)", "dashed"),
        ("intent_router", "clarify_gate", "", ""),
        ("clarify_gate", "ask_clarify", "info missing", "loop"),
        ("ask_clarify", "clarify_gate", "re-check (loop, max 2)", "loop"),
        ("clarify_gate", "tool_step", "sufficient /\nask budget exhausted", ""),
        ("tool_step", "tool_step", "retry (loop,\nalternative tool, max 2)", "loop"),
        ("tool_step", "summarize", "tool success", ""),
        ("tool_step", "give_up", "give up", "dashed"),
        ("give_up", "escalation_gate", "", "dashed"),
        ("summarize", "risk_check", "", ""),
        ("risk_check", "kyc_verify", "verification needed", ""),
        ("kyc_verify", "escalation_gate", "", ""),
        ("risk_check", "escalation_gate", "no risk", ""),
        ("escalation_gate", "respond", "", ""),
        ("respond", "END", "", ""),
    ],
}

# 12개 전문 상담팀 캡션 (언어별)
_TEAMS = {
    "ko": {
        "cards": "카드 발급/배송", "card_security": "분실/도난/보안",
        "disputes_fees": "분쟁/수수료 이의", "transfers": "이체/송금",
        "accounts_balance": "잔액/입금 반영", "top_up": "계좌 충전",
        "exchange_fx": "환전/환율", "refunds": "환불",
        "declined_pay": "승인 거절/보류", "identity_verif": "신원확인(KYC)",
        "account_mgmt": "계정 관리", "general_faq": "일반 FAQ",
    },
    "en": {
        "cards": "card issuing/delivery", "card_security": "lost/stolen/security",
        "disputes_fees": "disputes/fees", "transfers": "transfers/remittance",
        "accounts_balance": "balance/deposits", "top_up": "account top-up",
        "exchange_fx": "FX/exchange rate", "refunds": "refunds",
        "declined_pay": "declined/pending pay", "identity_verif": "identity (KYC)",
        "account_mgmt": "account management", "general_faq": "general FAQ",
    },
}

# 하위 호환 별칭 (기존 노트북이 참조)
KO_NODES = _NODES["ko"]
KO_EDGES = _EDGES["ko"]
KO_TEAMS = _TEAMS["ko"]

# 01 노트북의 조립 단계: 단계별로 보이는 노드 집합 (누적)
BUILD_STAGES = [
    ["START", "guardrail_check", "intent_router", "respond", "END"],
    ["clarify_gate", "ask_clarify"],
    ["tool_step", "give_up"],
    ["summarize", "risk_check", "kyc_verify"],
    ["escalation_gate"],
]


def draw_graph(visible=None, highlight=None, new=None, width=None, show_teams=False,
               lang=None):
    """workflow 그래프를 그려 IPython Image로 반환 (캡션은 lang/DMWS_LANG 언어).

    visible    : 표시할 노드 집합 (None=전체). edge는 양끝이 모두 보일 때만 그림
    highlight  : 시나리오 경로 강조(빨간 두꺼운 테두리, team_<cat> 포함 가능)
    new        : 이번 단계에 추가된 노드 강조(초록 두꺼운 테두리)
    show_teams : intent_router의 12개 전문 상담팀을 점선 cluster 노드로 표시
    lang       : 'ko' | 'en' (기본: DMWS_LANG)
    """
    import graphviz
    from IPython.display import Image
    lang = lang or LANG
    nodes, edges, teams = _NODES[lang], _EDGES[lang], _TEAMS[lang]
    ui = _UI[lang]
    visible = set(visible) if visible else set(nodes)
    highlight = set(highlight or [])
    new = set(new or [])
    g = graphviz.Digraph(format="png")
    g.attr(rankdir="TB", fontname="Noto Sans CJK KR", splines="spline",
           nodesep="0.3", ranksep="0.4", dpi="110", compound="true")
    g.attr("node", fontname="Noto Sans CJK KR", fontsize="11")
    g.attr("edge", fontname="Noto Sans CJK KR", fontsize="9")
    for n in visible:
        label = nodes[n]
        kw = {}
        if n in ("START", "END"):
            kw = dict(shape="circle" if n == "START" else "doublecircle",
                      style="filled", fillcolor="#333333", fontcolor="white", width="0.45")
            label = n
        elif n in _DECISION_NODES:
            kw = dict(shape="diamond", style="filled", fillcolor="#FFE08A")
        elif n in _LLM_NODES:
            kw = dict(shape="box", style="filled,rounded", fillcolor="#BFD7FF")
        else:
            kw = dict(shape="box", style="filled", fillcolor="#EEEEEE")
        if n in new:
            kw.update(color="#1E8449", penwidth="3")
        elif n in highlight:
            kw.update(color="#C0392B", penwidth="3")
        g.node(n, label, **kw)
    for src, dst, lbl, style in edges:
        if src not in visible or dst not in visible:
            continue
        kw = {"label": lbl}
        if style == "dashed":
            kw["style"] = "dashed"
        elif style == "loop":
            kw.update(color="#CC6677", penwidth="2.2", fontcolor="#CC6677")
        if src in highlight and dst in highlight and style != "dashed":
            kw.setdefault("penwidth", "2.2")
        if show_teams and (src, dst) == ("intent_router", "clarify_gate"):
            continue   # 팀 cluster를 경유하도록 아래에서 대체
        g.edge(src, dst, **kw)
    if show_teams and "intent_router" in visible:
        with g.subgraph(name="cluster_teams") as c:
            c.attr(label=ui["team_cluster"], fontname="Noto Sans CJK KR",
                   fontsize="10", style="dashed", color="#888888")
            for cat, cap in teams.items():
                kw = dict(shape="box", style="filled", fillcolor="#F6EBD9", fontsize="9")
                if f"team_{cat}" in highlight:
                    kw.update(color="#C0392B", penwidth="3")
                c.node(f"team_{cat}", f"{cat}\n{cap}", **kw)
        g.edge("intent_router", "team_exchange_fx", lhead="cluster_teams",
               label=ui["team_assign"])
        if "clarify_gate" in visible:
            g.edge("team_exchange_fx", "clarify_gate", ltail="cluster_teams")
    return Image(data=g.pipe(), width=width)


def draw_build_stage(stage, width=640, show_teams=True, lang=None):
    """01 노트북용: stage(1부터) 단계까지 누적된 그래프, 이번 단계 노드는 초록 강조.
    show_teams=True면 intent_router 아래 12개 상담팀 cluster를 함께 표시."""
    visible, new = [], []
    for i, nodes in enumerate(BUILD_STAGES[:stage], start=1):
        visible += nodes
        if i == stage:
            new = nodes
    return draw_graph(visible=visible, new=new, width=width, show_teams=show_teams,
                      lang=lang)


# 시나리오별 통과 경로 (전체 그래프에서 빨간 강조; team_* 는 배정되는 상담팀 노드)
# EN은 발화 표현상 시나리오 1이 declined_pay 팀으로 라우팅됨 (KR=cards와 대비 포인트)
def scenario_path(n, lang=None):
    path = list(SCENARIO_PATHS[n])
    if (lang or LANG) == "en" and n == 1:
        path[path.index("team_cards")] = "team_declined_pay"
    return path


SCENARIO_PATHS = {
    1: ["START", "guardrail_check", "intent_router", "team_cards", "clarify_gate",
        "ask_clarify", "tool_step", "summarize", "risk_check", "escalation_gate",
        "respond", "END"],
    2: ["START", "guardrail_check", "intent_router", "team_transfers", "clarify_gate",
        "tool_step", "summarize", "risk_check", "escalation_gate", "respond", "END"],
    3: ["START", "guardrail_check", "intent_router", "team_card_security",
        "clarify_gate", "tool_step", "summarize", "risk_check", "kyc_verify",
        "escalation_gate", "respond", "END"],
}


# ------------------------------------------------------------- notebook 05 --
_VLM_PROMPT = {
    "ko": ('위 이미지를 보고 다음 질문에 답하라. {q}\n'
           '아래 Labels 중 정확히 하나를 골라 {{"label": "<label>"}} JSON 객체로만 '
           '응답하라. 다른 텍스트는 출력하지 마라.\n\nLabels:\n{labels}'),
    "en": ('Look at the image above and answer the question. {q}\n'
           'Pick exactly one of the Labels below and respond with ONLY the JSON '
           'object {{"label": "<label>"}}. No other text.\n\nLabels:\n{labels}'),
}
VLM_PROMPT = _VLM_PROMPT[LANG]


def _parse(text, valid):
    try:
        lbl = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))["label"]
        return lbl if lbl in valid else f"INVALID:{lbl}"
    except Exception:
        return "INVALID"


def vlm_sglang(img: Path, q: str, labels):
    body = json.dumps({
        "model": SGLANG_MODEL, "temperature": 0, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
             base64.b64encode(img.read_bytes()).decode()}},
            {"type": "text", "text": VLM_PROMPT.format(
                q=q, labels="\n".join(f"- {l}" for l in labels))}]}]},
        ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://{SERVER_HOST}:8001/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    ms = (time.perf_counter() - t0) * 1000.0
    return _parse(resp["choices"][0]["message"]["content"] or "", set(labels)), ms


def make_vlm_haiku(client):
    def call(img: Path, q: str, labels):
        blocks = [{"type": "image", "source": {
                      "type": "base64", "media_type": "image/png",
                      "data": base64.b64encode(img.read_bytes()).decode()}},
                  {"type": "text", "text": VLM_PROMPT.format(
                      q=q, labels="\n".join(f"- {l}" for l in labels))}]
        t0 = time.perf_counter()
        resp = client.messages.create(model=BEDROCK_MODEL_IDS["haiku-4-5"],
                                      max_tokens=64,
                                      messages=[{"role": "user", "content": blocks}])
        ms = (time.perf_counter() - t0) * 1000.0
        return _parse(resp.content[0].text if resp.content else "", set(labels)), ms
    return call


def compare_image(img: Path, title: str, q: str, labels, haiku_fn, reps=3):
    """이미지 판단 1건을 양 backend로 실행, 과정 출력 후 (p50 결과) 반환."""
    print(f"\n{'=' * 86}\n{UI['img_hdr'].format(t=title)}\n{UI['img_q']}: {q}\n{'=' * 86}")
    out = {}
    for name, fn in (("SGLang(Qwen3-VL)", vlm_sglang), ("Haiku", haiku_fn)):
        fn(img, q, labels)  # warm-up
        runs = [fn(img, q, labels) for _ in range(reps)]
        lats = sorted(m for _, m in runs)
        p50 = statistics.median(lats)
        print(f"  [{name:<17s}] {UI['img_verdict']} -> {runs[0][0]:<34s} p50 {p50:>8,.1f}ms"
              f"  (reps={reps}: {', '.join(f'{m:,.0f}' for _, m in runs)}ms)")
        out[name] = {"label": runs[0][0], "p50_ms": p50}
    return out
