"""LangGraph StateGraph v2: multi-stage banking case handling with two recursive
loops (clarification, tool retry) and a conditional risk-verification sub-flow.

Identical topology for both arms; only the DecisionClient implementation differs.
Decision points: D1 guardrail, D2 routing, D5 clarify gate (loop, max 2 asks),
D3 tool choice + D6 tool review (loop, max 2 attempts), D7 risk check,
D8 KYC review (conditional), D4 escalation. Customer clarify replies are
scripted (dataset), so loop gold stays objective.

START -> guardrail_check [D1] -> (out_of_scope|sensitive -> respond)
      -> intent_router [D2]
      -> clarify_gate [D5] <-> ask_clarify (Sonnet question + scripted reply)
      -> tool_step [D3 + tool + D6] -(try_alternative)-> tool_step
                                     -(give_up)--------> escalation_gate
      -> summarize (Sonnet) -> risk_check [D7] -(needs_verification)-> kyc_verify [D8]
      -> escalation_gate [D4] -> respond (Sonnet) -> END
"""

import json
import operator
import time
from typing import Annotated, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from .prompts import (D1_QUESTION, D2_QUESTION, D3_QUESTION, D4_QUESTION,
                      RESPOND_SYSTEM, SPECIALIST_SYSTEM)
from .prompts_v2 import (ASK_CLARIFY_SYSTEM, D5_CHOICES, D5_QUESTION, D6_CHOICES,
                         D6_QUESTION, D7_CHOICES, D7_QUESTION, D8_CHOICES, D8_QUESTION)
from .state import CATEGORIES, D1_CHOICES, D4_CHOICES
from .tools_v2 import CATEGORY_TOOLS, run_kyc, run_tool_v2

MAX_CLARIFY_ASKS = 2
MAX_TOOL_ATTEMPTS = 2


class AgentStateV2(TypedDict, total=False):
    conv_id: str
    arm: str
    lang: str
    script: dict
    dialogue: Annotated[List[dict], operator.add]   # {"role": customer|agent, "text"}
    trace: Annotated[List[dict], operator.add]
    d1: str
    d2: str
    d4: str
    d5_labels: Annotated[List[str], operator.add]
    clarify_asks: int
    d3_labels: Annotated[List[str], operator.add]
    d6_labels: Annotated[List[str], operator.add]
    tools_used: Annotated[List[str], operator.add]
    tool_attempts: int
    tool_result: Optional[dict]
    gave_up: bool
    d7: str
    d8: str
    kyc_result: Optional[dict]
    specialist_summary: str
    final_response: str


def _conv_text(state) -> str:
    return "\n".join(f"{t['role']}: {t['text']}" for t in state["dialogue"])


def _dec_event(node, decision_id, dr, backend, forced=None):
    ev = {"node": node, "kind": "decision", "decision": decision_id, "backend": backend,
          "label": dr.label, "confidence": dr.confidence, "valid": dr.valid,
          "latency_ms": dr.latency_ms, "usage": dr.usage, "cost_usd": dr.cost_usd}
    if forced:
        ev["forced"] = forced
    return ev


def _llm_event(node, lr, backend):
    return {"node": node, "kind": "llm", "backend": backend, "latency_ms": lr.latency_ms,
            "usage": lr.usage, "cost_usd": lr.cost_usd}


def _tool_event(node, t0):
    return {"node": node, "kind": "tool",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)}


def build_graph_v2(decision, llm):
    def guardrail_check(state: AgentStateV2):
        dr = decision.decide(state_text=state["dialogue"][0]["text"], question=D1_QUESTION,
                             choices=D1_CHOICES, decision_id="d1")
        return {"d1": dr.label,
                "trace": [_dec_event("guardrail_check", "d1", dr, decision.name)]}

    def intent_router(state: AgentStateV2):
        dr = decision.decide(state_text=state["dialogue"][0]["text"], question=D2_QUESTION,
                             choices=CATEGORIES, decision_id="d2")
        return {"d2": dr.label,
                "trace": [_dec_event("intent_router", "d2", dr, decision.name)]}

    def clarify_gate(state: AgentStateV2):
        dr = decision.decide(state_text="Conversation so far:\n" + _conv_text(state),
                             question=D5_QUESTION, choices=D5_CHOICES, decision_id="d5")
        return {"d5_labels": [dr.label],
                "trace": [_dec_event("clarify_gate", "d5", dr, decision.name)]}

    def route_clarify(state: AgentStateV2):
        if (state["d5_labels"][-1] == "need_clarification"
                and state.get("clarify_asks", 0) < MAX_CLARIFY_ASKS):
            return "ask_clarify"
        return "tool_step"   # info_sufficient, or ask budget exhausted (forced proceed)

    def ask_clarify(state: AgentStateV2):
        lr = llm.generate(system=ASK_CLARIFY_SYSTEM,
                          user="Conversation so far:\n" + _conv_text(state), max_tokens=120)
        asks = state.get("clarify_asks", 0)
        script = state["script"]
        if script.get("vague") and asks == 0:
            reply = script["clarify_reply"]
        else:
            reply = script["no_info_reply"]
        return {"clarify_asks": asks + 1,
                "dialogue": [{"role": "agent", "text": lr.text},
                             {"role": "customer", "text": reply}],
                "trace": [_llm_event("ask_clarify", lr, llm.name)]}

    def tool_step(state: AgentStateV2):
        category = state["d2"]
        attempt = state.get("tool_attempts", 0)
        used = state.get("tools_used", [])
        choices = [t for t in CATEGORY_TOOLS[category] if t not in used]
        events = []
        if len(choices) == 1:
            tool = choices[0]
            events.append({"node": "tool_select", "kind": "auto", "decision": "d3",
                           "label": tool, "latency_ms": 0.0, "cost_usd": 0.0})
        else:
            dr = decision.decide(
                state_text="Conversation so far:\n" + _conv_text(state)
                           + (f"\n\nAlready tried (failed): {', '.join(used)}" if used else ""),
                question=D3_QUESTION.format(category=category),
                choices=choices, decision_id="d3")
            tool = dr.label
            events.append(_dec_event("tool_select", "d3", dr, decision.name))
        t0 = time.perf_counter()
        tool_result = run_tool_v2(tool, state["conv_id"], state["script"], attempt)
        events.append(_tool_event(f"tool:{tool}", t0))
        dr6 = decision.decide(
            state_text=(f"Customer request:\n{state['dialogue'][0]['text']}\n\n"
                        f"Tool executed: {tool}\nTool result:\n"
                        f"{json.dumps(tool_result, ensure_ascii=False)}"),
            question=D6_QUESTION, choices=D6_CHOICES, decision_id="d6")
        events.append(_dec_event("tool_review", "d6", dr6, decision.name))
        return {"d3_labels": [tool], "d6_labels": [dr6.label], "tools_used": [tool],
                "tool_attempts": attempt + 1, "tool_result": tool_result, "trace": events}

    def route_tool(state: AgentStateV2):
        label = state["d6_labels"][-1]
        category = state["d2"]
        remaining = [t for t in CATEGORY_TOOLS[category]
                     if t not in state.get("tools_used", [])]
        if (label == "try_alternative"
                and state.get("tool_attempts", 0) < MAX_TOOL_ATTEMPTS and remaining):
            return "tool_step"
        if label == "goal_met":
            return "summarize"
        return "give_up"   # give_up, or try_alternative with budget/tools exhausted

    def mark_give_up(state: AgentStateV2):
        return {"gave_up": True}

    def summarize(state: AgentStateV2):
        lr = llm.generate(
            system=SPECIALIST_SYSTEM.format(category=state["d2"]),
            user=(f"Conversation:\n{_conv_text(state)}\n\n"
                  f"Tool executed: {state['tool_result']['tool']}\nTool result:\n"
                  f"{json.dumps(state['tool_result'], ensure_ascii=False)}"),
            max_tokens=300)
        return {"specialist_summary": lr.text,
                "trace": [_llm_event("summarize", lr, llm.name)]}

    def risk_check(state: AgentStateV2):
        ctx = (f"Case facts:\n{state.get('specialist_summary', '')}\n\n"
               f"Tool result:\n{json.dumps(state.get('tool_result'), ensure_ascii=False)}")
        dr = decision.decide(state_text=ctx, question=D7_QUESTION,
                             choices=D7_CHOICES, decision_id="d7")
        return {"d7": dr.label,
                "trace": [_dec_event("risk_check", "d7", dr, decision.name)]}

    def kyc_verify(state: AgentStateV2):
        t0 = time.perf_counter()
        kyc = run_kyc(state["conv_id"], state["script"])
        events = [_tool_event("tool:run_kyc_check", t0)]
        dr = decision.decide(
            state_text=f"KYC result:\n{json.dumps(kyc, ensure_ascii=False)}",
            question=D8_QUESTION, choices=D8_CHOICES, decision_id="d8")
        events.append(_dec_event("kyc_review", "d8", dr, decision.name))
        return {"d8": dr.label, "kyc_result": kyc, "trace": events}

    def escalation_gate(state: AgentStateV2):
        parts = [f"Conversation:\n{_conv_text(state)}",
                 f"Tool result:\n{json.dumps(state.get('tool_result'), ensure_ascii=False)}"]
        if state.get("specialist_summary"):
            parts.append(f"Specialist facts:\n{state['specialist_summary']}")
        if state.get("kyc_result") is not None:
            parts.append(f"KYC result:\n{json.dumps(state['kyc_result'], ensure_ascii=False)}")
        dr = decision.decide(state_text="\n\n".join(parts), question=D4_QUESTION,
                             choices=D4_CHOICES, decision_id="d4")
        return {"d4": dr.label,
                "trace": [_dec_event("escalation_gate", "d4", dr, decision.name)]}

    def respond(state: AgentStateV2):
        action = state["d1"] if state["d1"] != "in_scope" else state.get("d4", "resolve")
        parts = [f"Decided action: {action}",
                 f"Conversation:\n{_conv_text(state)}"]
        if state.get("specialist_summary"):
            parts.append(f"Resolution facts:\n{state['specialist_summary']}")
        if state.get("tool_result"):
            parts.append(f"Tool result:\n{json.dumps(state['tool_result'], ensure_ascii=False)}")
        if state.get("kyc_result") is not None:
            parts.append(f"KYC result:\n{json.dumps(state['kyc_result'], ensure_ascii=False)}")
        lr = llm.generate(system=RESPOND_SYSTEM, user="\n\n".join(parts), max_tokens=400)
        return {"final_response": lr.text, "trace": [_llm_event("respond", lr, llm.name)]}

    g = StateGraph(AgentStateV2)
    g.add_node("guardrail_check", guardrail_check)
    g.add_node("intent_router", intent_router)
    g.add_node("clarify_gate", clarify_gate)
    g.add_node("ask_clarify", ask_clarify)
    g.add_node("tool_step", tool_step)
    g.add_node("give_up", mark_give_up)
    g.add_node("summarize", summarize)
    g.add_node("risk_check", risk_check)
    g.add_node("kyc_verify", kyc_verify)
    g.add_node("escalation_gate", escalation_gate)
    g.add_node("respond", respond)

    g.add_edge(START, "guardrail_check")
    g.add_conditional_edges(
        "guardrail_check",
        lambda s: "intent_router" if s["d1"] == "in_scope" else "respond",
        ["intent_router", "respond"])
    g.add_edge("intent_router", "clarify_gate")
    g.add_conditional_edges("clarify_gate", route_clarify, ["ask_clarify", "tool_step"])
    g.add_edge("ask_clarify", "clarify_gate")
    g.add_conditional_edges("tool_step", route_tool, ["tool_step", "summarize", "give_up"])
    g.add_edge("give_up", "escalation_gate")
    g.add_edge("summarize", "risk_check")
    g.add_conditional_edges(
        "risk_check",
        lambda s: "kyc_verify" if s["d7"] == "needs_verification" else "escalation_gate",
        ["kyc_verify", "escalation_gate"])
    g.add_edge("kyc_verify", "escalation_gate")
    g.add_edge("escalation_gate", "respond")
    g.add_edge("respond", END)
    return g.compile()
