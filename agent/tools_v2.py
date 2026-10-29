"""V2 stub banking tools: per-attempt scripted outcomes + risk flag + KYC check.

Outcome of the n-th tool attempt comes from the conversation's script
(script["tool_outcomes"][attempt]), so the D6 (tool review), D7 (risk) and
D8 (KYC) gold labels are objective by construction, like D4 in v1.
"""

from .tools import CATEGORY_TOOLS, INTENT_TOOL  # noqa: F401  (reused by scripts)

_OUTCOMES = {
    "success": {
        "status": "ok",
        "result": "The requested operation completed successfully in the banking system.",
        "reference_id": "REF-{cid}",
    },
    "tool_error": {
        "status": "error",
        "result": "The banking system returned an internal error; the operation did not "
                  "complete. The case data is intact and another tool may still succeed.",
        "error_code": "SYS-5xx-{cid}",
    },
    "blocked": {
        "status": "blocked",
        "result": "The operation is flagged for manual review and cannot be completed "
                  "automatically.",
        "reason": "Bank policy requires a human specialist to review this case "
                  "(risk/compliance threshold).",
    },
}

RISK_NOTE = ("Transaction-monitoring flag: this transaction pattern matched a fraud/AML "
             "monitoring rule (unusual geography or amount).")

_KYC = {
    "verified": {
        "tool": "run_kyc_check", "status": "ok",
        "result": "Identity verified: document and liveness checks passed.",
        "kyc_ref": "KYC-{cid}",
    },
    "verification_failed": {
        "tool": "run_kyc_check", "status": "failed",
        "result": "Identity could not be verified: document check did not pass. "
                  "Manual review by a human specialist is required.",
        "kyc_ref": "KYC-{cid}",
    },
}


def _fmt(payload: dict, cid: str) -> dict:
    return {k: (v.format(cid=cid) if isinstance(v, str) else v) for k, v in payload.items()}


def run_tool_v2(tool_name: str, conv_id: str, script: dict, attempt: int) -> dict:
    outcomes = script.get("tool_outcomes", ["success"])
    key = outcomes[min(attempt, len(outcomes) - 1)]
    payload = _fmt(_OUTCOMES[key], conv_id)
    if key == "success" and script.get("risk_flag"):
        payload["risk_note"] = RISK_NOTE
    return {"tool": tool_name, "outcome_key": key, **payload}


def run_kyc(conv_id: str, script: dict) -> dict:
    # script에 KYC 결과가 없는데(D7 오판 등) 이 노드에 도달할 수 있음 -> verified로 통과
    key = script.get("kyc_result") or "verified"
    return {"outcome_key": key, **_fmt(_KYC[key], conv_id)}
