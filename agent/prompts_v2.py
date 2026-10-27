"""V2 decision instructions (D5-D8) and the clarify-question prompt.

D1-D4, HAIKU_DECISION_SYSTEM, SPECIALIST_SYSTEM, RESPOND_SYSTEM, JUDGE_SYSTEM
are reused unchanged from prompts.py (information parity between arms holds:
both arms receive identical question text and label sets).
"""

D5_QUESTION = (
    "Decide whether the customer's request, given the conversation so far, contains "
    "enough specifics for a bank support agent to act on it (which card/account/"
    "transaction is affected and what exactly happened or is wanted): "
    "info_sufficient if the agent can proceed now; "
    "need_clarification if a clarifying question to the customer is required first."
)

D6_QUESTION = (
    "A bank support tool was just executed for this case. Based on the tool result, "
    "decide the next step: goal_met if the tool completed the task or returned the "
    "information needed (even if the result carries informational follow-up flags such "
    "as monitoring notes); try_alternative if the tool failed with an error such that a "
    "different tool might still succeed; give_up if the system blocked the operation "
    "so that only a human specialist can proceed."
)

D7_QUESTION = (
    "Review the case facts and tool result of this resolved banking case. Decide: "
    "needs_verification if the result contains a risk or fraud-monitoring flag that "
    "requires re-verifying the customer's identity before closing the case; "
    "no_risk if there is no such flag."
)

D8_QUESTION = (
    "A KYC identity check was executed for this case. Based on the KYC result, decide: "
    "verified if the customer's identity was successfully verified; "
    "verification_failed if the check did not verify the identity."
)

D5_CHOICES = ["info_sufficient", "need_clarification"]
D6_CHOICES = ["goal_met", "try_alternative", "give_up"]
D7_CHOICES = ["no_risk", "needs_verification"]
D8_CHOICES = ["verified", "verification_failed"]

ASK_CLARIFY_SYSTEM = (
    "You are a retail bank customer-support agent. The customer's request is missing "
    "specifics the bank needs in order to act. Ask ONE short, concrete clarifying "
    "question for the most important missing detail (e.g. which transaction, amount, "
    "date, card or account). Reply in the same language as the customer. "
    "Output only the question, under 40 words."
)
