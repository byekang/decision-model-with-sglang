"""Stub banking tools: deterministic canned results keyed by the conversation's
seeded outcome_key, so the D4 (escalation) gold label is objective by construction.
"""

CATEGORY_TOOLS = {
    "cards": ["card_order_status", "activate_card", "card_troubleshoot", "issue_new_card"],
    "card_security": ["freeze_card", "reset_credentials", "replace_card"],
    "disputes_fees": ["get_transaction_details", "open_dispute", "explain_fees"],
    "transfers": ["track_transfer", "cancel_transfer_tool", "transfer_info"],
    "accounts_balance": ["check_balance", "track_deposit"],
    "top_up": ["topup_status", "retry_topup", "topup_info"],
    "exchange_fx": ["get_exchange_rate", "review_fx_transaction", "fx_info"],
    "refunds": ["track_refund", "initiate_refund"],
    "declined_pay": ["check_payment_status", "explain_decline"],
    "identity_verif": ["run_kyc_check", "kyc_info"],
    "account_mgmt": ["update_profile", "close_account", "account_policy_info"],
    "general_faq": ["kb_search", "product_info_lookup"],
}

# Expected (gold) tool per Banking77 intent, used to score D3.
INTENT_TOOL = {
    # cards
    "card_arrival": "card_order_status", "card_delivery_estimate": "card_order_status",
    "activate_my_card": "activate_card",
    "card_not_working": "card_troubleshoot", "contactless_not_working": "card_troubleshoot",
    "virtual_card_not_working": "card_troubleshoot", "card_linking": "card_troubleshoot",
    "disposable_card_limits": "card_troubleshoot",
    "card_swallowed": "issue_new_card", "card_about_to_expire": "issue_new_card",
    "get_physical_card": "issue_new_card", "order_physical_card": "issue_new_card",
    "getting_spare_card": "issue_new_card", "getting_virtual_card": "issue_new_card",
    "get_disposable_virtual_card": "issue_new_card",
    # card_security
    "compromised_card": "freeze_card", "lost_or_stolen_card": "freeze_card",
    "lost_or_stolen_phone": "freeze_card",
    "change_pin": "reset_credentials", "pin_blocked": "reset_credentials",
    "passcode_forgotten": "reset_credentials",
    # disputes_fees
    "card_payment_not_recognised": "open_dispute", "cash_withdrawal_not_recognised": "open_dispute",
    "direct_debit_payment_not_recognised": "open_dispute", "transaction_charged_twice": "open_dispute",
    "extra_charge_on_statement": "open_dispute", "wrong_amount_of_cash_received": "open_dispute",
    "card_payment_fee_charged": "explain_fees", "cash_withdrawal_charge": "explain_fees",
    "transfer_fee_charged": "explain_fees",
    # transfers
    "cancel_transfer": "cancel_transfer_tool",
    "failed_transfer": "track_transfer", "transfer_not_received_by_recipient": "track_transfer",
    "transfer_timing": "transfer_info", "beneficiary_not_allowed": "transfer_info",
    "receiving_money": "transfer_info", "transfer_into_account": "transfer_info",
    # accounts_balance
    "balance_not_updated_after_bank_transfer": "track_deposit",
    "balance_not_updated_after_cheque_or_cash_deposit": "track_deposit",
    # top_up
    "pending_top_up": "topup_status", "top_up_reverted": "topup_status",
    "verify_top_up": "topup_status",
    "top_up_failed": "retry_topup",
    "automatic_top_up": "topup_info", "top_up_by_bank_transfer_charge": "topup_info",
    "top_up_by_card_charge": "topup_info", "top_up_by_cash_or_cheque": "topup_info",
    "top_up_limits": "topup_info", "topping_up_by_card": "topup_info",
    # exchange_fx
    "exchange_rate": "get_exchange_rate",
    "card_payment_wrong_exchange_rate": "review_fx_transaction",
    "wrong_exchange_rate_for_cash_withdrawal": "review_fx_transaction",
    "exchange_charge": "fx_info", "exchange_via_app": "fx_info",
    "fiat_currency_support": "fx_info",
    # refunds
    "Refund_not_showing_up": "track_refund", "reverted_card_payment?": "track_refund",
    "request_refund": "initiate_refund",
    # declined_pay
    "pending_card_payment": "check_payment_status", "pending_cash_withdrawal": "check_payment_status",
    "pending_transfer": "check_payment_status",
    "declined_card_payment": "explain_decline", "declined_cash_withdrawal": "explain_decline",
    "declined_transfer": "explain_decline",
    # identity_verif
    "unable_to_verify_identity": "run_kyc_check", "verify_my_identity": "run_kyc_check",
    "verify_source_of_funds": "run_kyc_check",
    "why_verify_identity": "kyc_info",
    # account_mgmt
    "edit_personal_details": "update_profile",
    "terminate_account": "close_account",
    "age_limit": "account_policy_info",
    # general_faq
    "apple_pay_or_google_pay": "kb_search", "atm_support": "kb_search",
    "card_acceptance": "kb_search", "country_support": "kb_search",
    "supported_cards_and_currencies": "product_info_lookup",
    "visa_or_mastercard": "product_info_lookup",
}

_OUTCOMES = {
    "success": {
        "status": "ok",
        "result": "The requested operation completed successfully in the banking system.",
        "reference_id": "REF-{cid}",
    },
    "needs_more_info": {
        "status": "incomplete",
        "result": "The system could not complete the operation with the information provided.",
        "missing": "The customer must confirm the exact transaction date/amount or the account in question.",
    },
    "requires_human": {
        "status": "blocked",
        "result": "The operation is flagged for manual review and cannot be completed automatically.",
        "reason": "Bank policy requires a human specialist to review this case (risk/compliance threshold).",
    },
}


def run_tool(tool_name: str, conv_id: str, tool_script: dict | None) -> dict:
    outcome_key = (tool_script or {}).get("outcome_key", "success")
    payload = {k: (v.format(cid=conv_id) if isinstance(v, str) else v)
               for k, v in _OUTCOMES[outcome_key].items()}
    return {"tool": tool_name, **payload}
