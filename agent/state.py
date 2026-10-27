"""Agent state and routing constants for the banking support A/B experiment.

The 77 Banking77 intents are mapped to 12 specialist categories (all decision
choice sets stay well under the 26-option SGLang accuracy cliff). Ambiguous
intents carry an accept_also set used for lenient D2 scoring.
"""

import operator
from typing import Annotated, Any, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    conv_id: str
    arm: str                 # "A" (haiku decisions) | "B" (sglang decisions)
    lang: str                # "en" | "ko"
    text: str                # customer message
    d1: str                  # guardrail: in_scope | out_of_scope | sensitive_compliance
    d2: str                  # specialist category
    d3: str                  # tool name
    d4: str                  # resolve | clarify | human_handoff
    tool_result: Optional[dict]
    specialist_summary: str
    final_response: str
    trace: Annotated[List[dict], operator.add]
    tool_script: Optional[dict]   # {"outcome_key": success|needs_more_info|requires_human}


D1_CHOICES = ["in_scope", "out_of_scope", "sensitive_compliance"]
D4_CHOICES = ["resolve", "clarify", "human_handoff"]

CATEGORIES = [
    "cards", "card_security", "disputes_fees", "transfers", "accounts_balance",
    "top_up", "exchange_fx", "refunds", "declined_pay", "identity_verif",
    "account_mgmt", "general_faq",
]

# Banking77 intent -> primary specialist category (77 entries, validated in 80_).
INTENT_CATEGORY = {
    # cards (15)
    "activate_my_card": "cards",
    "card_about_to_expire": "cards",
    "card_arrival": "cards",
    "card_delivery_estimate": "cards",
    "card_linking": "cards",
    "card_not_working": "cards",
    "card_swallowed": "cards",
    "contactless_not_working": "cards",
    "disposable_card_limits": "cards",
    "get_disposable_virtual_card": "cards",
    "get_physical_card": "cards",
    "getting_spare_card": "cards",
    "getting_virtual_card": "cards",
    "order_physical_card": "cards",
    "virtual_card_not_working": "cards",
    # card_security (6)
    "change_pin": "card_security",
    "compromised_card": "card_security",
    "lost_or_stolen_card": "card_security",
    "lost_or_stolen_phone": "card_security",
    "passcode_forgotten": "card_security",
    "pin_blocked": "card_security",
    # disputes_fees (9)
    "card_payment_fee_charged": "disputes_fees",
    "card_payment_not_recognised": "disputes_fees",
    "cash_withdrawal_charge": "disputes_fees",
    "cash_withdrawal_not_recognised": "disputes_fees",
    "direct_debit_payment_not_recognised": "disputes_fees",
    "extra_charge_on_statement": "disputes_fees",
    "transaction_charged_twice": "disputes_fees",
    "transfer_fee_charged": "disputes_fees",
    "wrong_amount_of_cash_received": "disputes_fees",
    # transfers (7)
    "beneficiary_not_allowed": "transfers",
    "cancel_transfer": "transfers",
    "failed_transfer": "transfers",
    "receiving_money": "transfers",
    "transfer_into_account": "transfers",
    "transfer_not_received_by_recipient": "transfers",
    "transfer_timing": "transfers",
    # accounts_balance (2)
    "balance_not_updated_after_bank_transfer": "accounts_balance",
    "balance_not_updated_after_cheque_or_cash_deposit": "accounts_balance",
    # top_up (10)
    "automatic_top_up": "top_up",
    "pending_top_up": "top_up",
    "top_up_by_bank_transfer_charge": "top_up",
    "top_up_by_card_charge": "top_up",
    "top_up_by_cash_or_cheque": "top_up",
    "top_up_failed": "top_up",
    "top_up_limits": "top_up",
    "top_up_reverted": "top_up",
    "topping_up_by_card": "top_up",
    "verify_top_up": "top_up",
    # exchange_fx (6)
    "card_payment_wrong_exchange_rate": "exchange_fx",
    "exchange_charge": "exchange_fx",
    "exchange_rate": "exchange_fx",
    "exchange_via_app": "exchange_fx",
    "fiat_currency_support": "exchange_fx",
    "wrong_exchange_rate_for_cash_withdrawal": "exchange_fx",
    # refunds (3)
    "Refund_not_showing_up": "refunds",
    "request_refund": "refunds",
    "reverted_card_payment?": "refunds",
    # declined_pay (6)
    "declined_card_payment": "declined_pay",
    "declined_cash_withdrawal": "declined_pay",
    "declined_transfer": "declined_pay",
    "pending_card_payment": "declined_pay",
    "pending_cash_withdrawal": "declined_pay",
    "pending_transfer": "declined_pay",
    # identity_verif (4)
    "unable_to_verify_identity": "identity_verif",
    "verify_my_identity": "identity_verif",
    "verify_source_of_funds": "identity_verif",
    "why_verify_identity": "identity_verif",
    # account_mgmt (3)
    "age_limit": "account_mgmt",
    "edit_personal_details": "account_mgmt",
    "terminate_account": "account_mgmt",
    # general_faq (6)
    "apple_pay_or_google_pay": "general_faq",
    "atm_support": "general_faq",
    "card_acceptance": "general_faq",
    "country_support": "general_faq",
    "supported_cards_and_currencies": "general_faq",
    "visa_or_mastercard": "general_faq",
}

# Lenient D2 scoring: these secondary categories also count as correct.
INTENT_ACCEPT_ALSO = {
    "balance_not_updated_after_bank_transfer": {"transfers"},
    "transfer_fee_charged": {"transfers"},
    "declined_transfer": {"transfers"},
    "pending_transfer": {"transfers"},
    "failed_transfer": {"declined_pay"},
    "receiving_money": {"accounts_balance"},
    "card_payment_wrong_exchange_rate": {"disputes_fees"},
    "wrong_exchange_rate_for_cash_withdrawal": {"disputes_fees"},
    "reverted_card_payment?": {"disputes_fees"},
    "wrong_amount_of_cash_received": {"declined_pay"},
    "pending_card_payment": {"disputes_fees"},
    "pending_cash_withdrawal": {"disputes_fees"},
    "verify_top_up": {"identity_verif"},
    "card_swallowed": {"card_security"},
    "visa_or_mastercard": {"cards"},
    "card_acceptance": {"cards"},
    "supported_cards_and_currencies": {"cards", "exchange_fx"},
}
