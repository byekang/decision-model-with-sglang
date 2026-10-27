"""Workshop 공용 helper.

- make_clients()/make_graphs(): Bedrock/SGLang 클라이언트와 두 arm의 v2 graph 생성
- run_verbose(): graph.stream(stream_mode="updates")로 노드 실행을 실시간 수신,
  "STEP n ▶ 노드명" 형식으로 각 노드의 결정(label/confidence/latency)·생성 텍스트·
  tool 결과를 출력하고 수행시간 집계를 반환
- compare(): 두 arm 실행 결과의 수행시간 비교표 출력
- vlm_sglang()/vlm_haiku(): 이미지 판단 1회 호출 (notebook 05)
"""

import base64
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/ubuntu/LLMRouter/OpenSource")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common import BEDROCK_MODEL_IDS, PRICING                      # noqa: E402
from agent.bedrock_llm import BedrockLLM                           # noqa: E402
from agent.decisions import HaikuDecision, SGLangDecision          # noqa: E402
from agent.graph_v2 import build_graph_v2                          # noqa: E402

SGLANG_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NO_INFO = "지금 알려드릴 수 있는 정보는 그게 전부예요."

# scripts/94에서 검증된 상담 시나리오 3종 (문구/script 동일)
SCENARIOS = {
    1: {
        "title": "모호한 문의 -> clarification loop",
        "question": "카드가 안 긁히는데 어떻게 해야 하나요?",
        "script": {"vague": True,
                   "clarify_reply": ("어제부터 오프라인 매장 단말기에서만 승인이 거절돼요. "
                                     "관련 정보: Visa 체크카드(끝자리 7712), 마지막 시도 "
                                     "2026-10-27 TechMart 강남점, 금액 45,000원입니다."),
                   "no_info_reply": NO_INFO,
                   "tool_outcomes": ["success"], "risk_flag": False, "kyc_result": None},
    },
    2: {
        "title": "시스템 오류 -> tool retry loop",
        "question": ("어제 오후에 보낸 해외송금이 아직 수취인에게 도착하지 않았어요. "
                     "송금 번호는 TR-20261026-8841, 금액은 USD 2,000입니다. 확인해 주세요."),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO,
                   "tool_outcomes": ["tool_error", "success"],
                   "risk_flag": False, "kyc_result": None},
    },
    3: {
        "title": "이상거래 모니터링 flag -> KYC sub-flow",
        "question": ("카드 끝자리 7712로 해외 결제(USD 1,284, LUXE GADGETS) 승인이 "
                     "됐다는 알림을 받았는데, 이 거래를 확인하고 싶어요."),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO,
                   "tool_outcomes": ["success"], "risk_flag": True,
                   "kyc_result": "verified"},
    },
}

NODE_DESC = {
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
}


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
        print(f"      판정[{ev.get('decision','?').upper()}] -> {ev['label']}"
              f"   ({ev['latency_ms']:,.0f}ms{conf})")
    elif kind == "llm":
        print(f"      생성(Sonnet 4.5)   ({ev['latency_ms']:,.0f}ms)")
    elif kind == "tool":
        print(f"      tool 실행: {ev['node'].replace('tool:', '')}"
              f"   ({ev['latency_ms']:,.1f}ms)")
    elif kind == "auto":
        print(f"      tool 자동 선택(남은 후보 1개): {ev['label']}")


def run_verbose(graph, question, script, arm_name):
    print(f"\n{'=' * 86}\n[{arm_name}] 고객: {question}\n{'=' * 86}")
    t0 = time.perf_counter()
    step = 0
    events = []
    stream = graph.stream(
        {"conv_id": f"ws-{arm_name}-{int(t0)}", "arm": arm_name, "lang": "ko",
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
                who = "상담 agent" if turn["role"] == "agent" else "고객"
                print(f"      {who}: {turn['text']}")
            if delta.get("specialist_summary"):
                head = delta["specialist_summary"].strip().splitlines()[0]
                print(f"      요약(첫 줄): {head[:80]}")
            if delta.get("final_response"):
                print(f"      --- 최종 응답 ---\n      "
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
    print(f"\n>>> [{arm_name}] 결정 {res['n_decisions']}회 합계 {res['decision_ms']:,.0f}ms"
          f" | 생성 {res['llm_ms']:,.0f}ms | e2e {res['e2e_ms']:,.0f}ms")
    return res


def compare(res_a, res_b):
    """두 arm 수행시간 비교를 markdown 표로 display (res_a=Haiku, res_b=SGLang 권장)."""
    from IPython.display import Markdown, display
    a, b = res_a, res_b
    ratio = a["decision_ms"] / b["decision_ms"] if b["decision_ms"] else float("nan")
    def x(va, vb):
        return f"**{va / vb:,.1f}x**" if vb else "-"
    md = [f"| 지표 | {a['arm']} | {b['arm']} | 배수({a['arm']}/{b['arm']}) |",
          "|---|---:|---:|---:|",
          f"| 결정 횟수 | {a['n_decisions']} | {b['n_decisions']} | |",
          f"| **결정 latency 합계** | **{a['decision_ms']:,.0f}ms** | **{b['decision_ms']:,.0f}ms** "
          f"| {x(a['decision_ms'], b['decision_ms'])} |",
          f"| 생성(LLM) latency | {a['llm_ms']:,.0f}ms | {b['llm_ms']:,.0f}ms "
          f"| {x(a['llm_ms'], b['llm_ms'])} |",
          f"| e2e | {a['e2e_ms']:,.0f}ms | {b['e2e_ms']:,.0f}ms "
          f"| {x(a['e2e_ms'], b['e2e_ms'])} |",
          "",
          f"**결정 계층 속도 차이: {ratio:,.1f}x** (e2e 단축 {a['e2e_ms'] - b['e2e_ms']:,.0f}ms)",
          "",
          "**결정별 latency**",
          "",
          f"| 노드 | 결정 | {a['arm']} | {b['arm']} | 배수 | 판정 label |",
          "|---|---|---:|---:|---:|---|"]
    for (na, da, la, ma), (nb, db, lb, mb) in zip(a["decisions"], b["decisions"]):
        same = f"{la} (동일)" if la == lb else f"{a['arm']}={la} / {b['arm']}={lb}"
        md.append(f"| {na} | {(da or '?').upper()} | {ma:,.0f}ms | {mb:,.0f}ms "
                  f"| {x(ma, mb)} | {same} |")
    display(Markdown("\n".join(md)))


# ------------------------------------------------------------ graph drawing --
# 한글 캡션 노드 정의 (그래프 시각화 전용)
KO_NODES = {
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
}
_DECISION_NODES = {"guardrail_check", "intent_router", "clarify_gate", "tool_step",
                   "risk_check", "kyc_verify", "escalation_gate"}
_LLM_NODES = {"ask_clarify", "summarize", "respond"}

# (src, dst, 한글 edge label, style) — style: "", "dashed", "loop"(빨간 굵은선)
KO_EDGES = [
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
]

# 01 노트북의 조립 단계: 단계별로 보이는 노드 집합 (누적)
BUILD_STAGES = [
    ["START", "guardrail_check", "intent_router", "respond", "END"],
    ["clarify_gate", "ask_clarify"],
    ["tool_step", "give_up"],
    ["summarize", "risk_check", "kyc_verify"],
    ["escalation_gate"],
]


# 12개 전문 상담팀 한글 캡션 (D2 선택지를 노드로 표시할 때 사용)
KO_TEAMS = {
    "cards": "카드 발급/배송", "card_security": "분실/도난/보안",
    "disputes_fees": "분쟁/수수료 이의", "transfers": "이체/송금",
    "accounts_balance": "잔액/입금 반영", "top_up": "계좌 충전",
    "exchange_fx": "환전/환율", "refunds": "환불",
    "declined_pay": "승인 거절/보류", "identity_verif": "신원확인(KYC)",
    "account_mgmt": "계정 관리", "general_faq": "일반 FAQ",
}


def draw_graph(visible=None, highlight=None, new=None, width=None, show_teams=False):
    """한글 캡션 workflow 그래프를 그려 IPython Image로 반환.

    visible    : 표시할 노드 집합 (None=전체). edge는 양끝이 모두 보일 때만 그림
    highlight  : 시나리오 경로 강조(빨간 두꺼운 테두리)
    new        : 이번 단계에 추가된 노드 강조(초록 두꺼운 테두리)
    show_teams : intent_router의 12개 전문 상담팀을 점선 cluster 노드로 표시
    """
    import graphviz
    from IPython.display import Image
    visible = set(visible) if visible else set(KO_NODES)
    highlight = set(highlight or [])
    new = set(new or [])
    g = graphviz.Digraph(format="png")
    g.attr(rankdir="TB", fontname="Noto Sans CJK KR", splines="spline",
           nodesep="0.3", ranksep="0.4", dpi="110", compound="true")
    g.attr("node", fontname="Noto Sans CJK KR", fontsize="11")
    g.attr("edge", fontname="Noto Sans CJK KR", fontsize="9")
    for n in visible:
        label = KO_NODES[n]
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
    for src, dst, lbl, style in KO_EDGES:
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
            c.attr(label="12개 전문 상담팀 (D2 선택지)", fontname="Noto Sans CJK KR",
                   fontsize="10", style="dashed", color="#888888")
            for cat, ko in KO_TEAMS.items():
                kw = dict(shape="box", style="filled", fillcolor="#F6EBD9", fontsize="9")
                if f"team_{cat}" in highlight:
                    kw.update(color="#C0392B", penwidth="3")
                c.node(f"team_{cat}", f"{cat}\n{ko}", **kw)
        g.edge("intent_router", "team_exchange_fx", lhead="cluster_teams",
               label="1개 팀 배정")
        if "clarify_gate" in visible:
            g.edge("team_exchange_fx", "clarify_gate", ltail="cluster_teams")
    return Image(data=g.pipe(), width=width)


def draw_build_stage(stage, width=640, show_teams=True):
    """01 노트북용: stage(1부터) 단계까지 누적된 그래프, 이번 단계 노드는 초록 강조.
    show_teams=True면 intent_router 아래 12개 상담팀 cluster를 함께 표시."""
    visible, new = [], []
    for i, nodes in enumerate(BUILD_STAGES[:stage], start=1):
        visible += nodes
        if i == stage:
            new = nodes
    return draw_graph(visible=visible, new=new, width=width, show_teams=show_teams)


# 시나리오별 통과 경로 (전체 그래프에서 빨간 강조; team_* 는 배정되는 상담팀 노드)
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
VLM_PROMPT = ('위 이미지를 보고 다음 질문에 답하라. {q}\n'
              '아래 Labels 중 정확히 하나를 골라 {{"label": "<label>"}} JSON 객체로만 '
              '응답하라. 다른 텍스트는 출력하지 마라.\n\nLabels:\n{labels}')


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
    req = urllib.request.Request("http://localhost:8001/v1/chat/completions",
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
    print(f"\n{'=' * 86}\n[이미지 판단] {title}\n질문: {q}\n{'=' * 86}")
    out = {}
    for name, fn in (("SGLang(Qwen3-VL)", vlm_sglang), ("Haiku", haiku_fn)):
        fn(img, q, labels)  # warm-up
        runs = [fn(img, q, labels) for _ in range(reps)]
        lats = sorted(m for _, m in runs)
        p50 = statistics.median(lats)
        print(f"  [{name:<17s}] 판정 -> {runs[0][0]:<34s} p50 {p50:>8,.1f}ms"
              f"  (reps={reps}: {', '.join(f'{m:,.0f}' for _, m in runs)}ms)")
        out[name] = {"label": runs[0][0], "p50_ms": p50}
    return out
