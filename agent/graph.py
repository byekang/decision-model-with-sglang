"""LangGraph StateGraph for the banking support agent.

Identical topology for both arms; only the DecisionClient implementation differs.
Every node appends a trace event (decision label / backend / latency / cost) so the
report can reconstruct per-request flows and attribute latency per node.

START -> guardrail_check [D1] -> (out_of_scope|sensitive -> respond)
      -> intent_router [D2] -> specialist_<cat> [D3 + stub tool + Sonnet summary]
      -> escalation_gate [D4] -> respond (Sonnet) -> END
"""

import json
import time

from langgraph.graph import StateGraph, START, END

from .prompts import (D1_QUESTION, D2_QUESTION, D3_QUESTION, D4_QUESTION,
                      RESPOND_SYSTEM, SPECIALIST_SYSTEM)
from .state import AgentState, CATEGORIES, D1_CHOICES, D4_CHOICES
from .tools import CATEGORY_TOOLS, run_tool


def _decision_event(node, kind, dr, backend):
    return {"node": node, "kind": kind, "backend": backend, "label": dr.label,
            "confidence": dr.confidence, "valid": dr.valid, "latency_ms": dr.latency_ms,
            "usage": dr.usage, "cost_usd": dr.cost_usd}


def _llm_event(node, lr, backend):
    return {"node": node, "kind": "llm", "backend": backend, "latency_ms": lr.latency_ms,
            "usage": lr.usage, "cost_usd": lr.cost_usd}


def build_graph(decision, llm):
    def guardrail_check(state: AgentState):
        dr = decision.decide(state_text=state["text"], question=D1_QUESTION,
                             choices=D1_CHOICES, decision_id="d1")
        return {"d1": dr.label,
                "trace": [_decision_event("guardrail_check", "decision", dr, decision.name)]}

    def intent_router(state: AgentState):
        dr = decision.decide(state_text=state["text"], question=D2_QUESTION,
                             choices=CATEGORIES, decision_id="d2")
        return {"d2": dr.label,
                "trace": [_decision_event("intent_router", "decision", dr, decision.name)]}

    def make_specialist(category: str):
        node_name = f"specialist_{category}"
        tools = CATEGORY_TOOLS[category]

        def specialist(state: AgentState):
            dr = decision.decide(state_text=state["text"],
                                 question=D3_QUESTION.format(category=category),
                                 choices=tools, decision_id="d3")
            events = [_decision_event(node_name, "decision", dr, decision.name)]
            t0 = time.perf_counter()
            tool_result = run_tool(dr.label, state["conv_id"], state.get("tool_script"))
            events.append({"node": f"tool:{dr.label}", "kind": "tool",
                           "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)})
            lr = llm.generate(
                system=SPECIALIST_SYSTEM.format(category=category),
                user=(f"Customer message:\n{state['text']}\n\n"
                      f"Tool executed: {dr.label}\nTool result:\n"
                      f"{json.dumps(tool_result, ensure_ascii=False)}"),
                max_tokens=300)
            events.append(_llm_event(node_name, lr, llm.name))
            return {"d3": dr.label, "tool_result": tool_result,
                    "specialist_summary": lr.text, "trace": events}

        return specialist

    def escalation_gate(state: AgentState):
        ctx = (f"Customer message:\n{state['text']}\n\n"
               f"Tool result:\n{json.dumps(state.get('tool_result'), ensure_ascii=False)}\n\n"
               f"Specialist facts:\n{state.get('specialist_summary', '')}")
        dr = decision.decide(state_text=ctx, question=D4_QUESTION,
                             choices=D4_CHOICES, decision_id="d4")
        return {"d4": dr.label,
                "trace": [_decision_event("escalation_gate", "decision", dr, decision.name)]}

    def respond(state: AgentState):
        action = state["d1"] if state["d1"] != "in_scope" else state.get("d4", "resolve")
        parts = [f"Decided action: {action}", f"Customer message:\n{state['text']}"]
        if state.get("specialist_summary"):
            parts.append(f"Resolution facts:\n{state['specialist_summary']}")
        if state.get("tool_result"):
            parts.append(f"Tool result:\n{json.dumps(state['tool_result'], ensure_ascii=False)}")
        lr = llm.generate(system=RESPOND_SYSTEM, user="\n\n".join(parts), max_tokens=400)
        return {"final_response": lr.text, "trace": [_llm_event("respond", lr, llm.name)]}

    g = StateGraph(AgentState)
    g.add_node("guardrail_check", guardrail_check)
    g.add_node("intent_router", intent_router)
    for cat in CATEGORIES:
        g.add_node(f"specialist_{cat}", make_specialist(cat))
    g.add_node("escalation_gate", escalation_gate)
    g.add_node("respond", respond)

    g.add_edge(START, "guardrail_check")
    g.add_conditional_edges(
        "guardrail_check",
        lambda s: "intent_router" if s["d1"] == "in_scope" else "respond",
        ["intent_router", "respond"])
    g.add_conditional_edges(
        "intent_router",
        lambda s: f"specialist_{s['d2']}",
        [f"specialist_{c}" for c in CATEGORIES])
    for cat in CATEGORIES:
        g.add_edge(f"specialist_{cat}", "escalation_gate")
    g.add_edge("escalation_gate", "respond")
    g.add_edge("respond", END)
    return g.compile()
