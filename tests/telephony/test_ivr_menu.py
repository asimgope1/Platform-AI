"""
Unit tests for IVR Menu State Machine and DTMF routing.
"""
import pytest
from telephony.ivr_menu import (
    IVRState,
    IVRMenuStateMachine,
    SUPPORTED_LANGUAGES,
    SERVICE_CATEGORIES,
)
from telephony.ivr_prompts import (
    WELCOME_AND_LANGUAGE,
    SERVICE_MENUS,
    SERVICE_NAMES,
    get_service_ack_prompt,
)


def test_initial_state():
    sm = IVRMenuStateMachine()
    assert sm.state == IVRState.WELCOME
    assert sm.language == "en"
    assert sm.retries == 0


def test_start_call_transition():
    sm = IVRMenuStateMachine()
    state = sm.start_call()
    assert state == IVRState.LANGUAGE_SELECTION
    assert sm.retries == 0


def test_language_selection_valid_digits():
    # Test 1 -> English
    sm = IVRMenuStateMachine()
    sm.start_call()
    state, action = sm.process_dtmf("1")
    assert state == IVRState.SERVICE_SELECTION
    assert action == "LANGUAGE_SELECTED"
    assert sm.language == "en"

    # Test 2 -> Hindi
    sm = IVRMenuStateMachine()
    sm.start_call()
    state, action = sm.process_dtmf("2")
    assert state == IVRState.SERVICE_SELECTION
    assert action == "LANGUAGE_SELECTED"
    assert sm.language == "hi"

    # Test 3 -> Odia
    sm = IVRMenuStateMachine()
    sm.start_call()
    state, action = sm.process_dtmf("3")
    assert state == IVRState.SERVICE_SELECTION
    assert action == "LANGUAGE_SELECTED"
    assert sm.language == "or"


def test_language_selection_repeat_menu():
    sm = IVRMenuStateMachine()
    sm.start_call()
    state, action = sm.process_dtmf("0")
    assert state == IVRState.LANGUAGE_SELECTION
    assert action == "REPEAT_MENU"
    assert sm.retries == 0


def test_language_selection_invalid_and_max_retries():
    sm = IVRMenuStateMachine()
    sm.start_call()

    # Retry 1
    state, action = sm.process_dtmf("8")
    assert state == IVRState.LANGUAGE_SELECTION
    assert action == "INVALID_INPUT"
    assert sm.retries == 1

    # Retry 2
    state, action = sm.process_dtmf("9")
    assert state == IVRState.LANGUAGE_SELECTION
    assert action == "INVALID_INPUT"
    assert sm.retries == 2

    # Retry 3 -> Max exceeded, defaults to English and moves to SERVICE_SELECTION
    state, action = sm.process_dtmf("*")
    assert state == IVRState.SERVICE_SELECTION
    assert action == "MAX_RETRIES_EXCEEDED"
    assert sm.language == "en"
    assert sm.retries == 0


def test_service_selection_valid_keys():
    expected_mappings = {
        "1": "customer_support",
        "2": "complaint",
        "3": "upi_payment",
        "4": "account_services",
        "5": "card_services",
        "6": "loans_emi",
        "7": "other_enquiries",
    }

    for digit, expected_key in expected_mappings.items():
        sm = IVRMenuStateMachine()
        sm.start_call()
        sm.process_dtmf("1")  # select English
        assert sm.state == IVRState.SERVICE_SELECTION

        state, action = sm.process_dtmf(digit)
        assert state == IVRState.AI_CONVERSATION
        assert action == "SERVICE_SELECTED"
        assert sm.service_key == expected_key
        assert sm.service_info is not None


def test_service_selection_human_transfer():
    sm = IVRMenuStateMachine()
    sm.start_call()
    sm.process_dtmf("2")  # select Hindi
    state, action = sm.process_dtmf("9")
    assert state == IVRState.HUMAN_TRANSFER
    assert action == "HUMAN_TRANSFER"


def test_service_selection_repeat():
    sm = IVRMenuStateMachine()
    sm.start_call()
    sm.process_dtmf("3")  # select Odia
    state, action = sm.process_dtmf("0")
    assert state == IVRState.SERVICE_SELECTION
    assert action == "REPEAT_MENU"


def test_service_selection_max_retries_fallback():
    sm = IVRMenuStateMachine()
    sm.start_call()
    sm.process_dtmf("1")

    sm.process_dtmf("8")
    sm.process_dtmf("8")
    state, action = sm.process_dtmf("8")
    assert state == IVRState.AI_CONVERSATION
    assert action == "MAX_RETRIES_EXCEEDED"
    assert sm.service_key == "customer_support"


def test_ai_conversation_state_transfer():
    sm = IVRMenuStateMachine()
    sm.start_call()
    sm.process_dtmf("1")
    sm.process_dtmf("3")  # UPI
    assert sm.state == IVRState.AI_CONVERSATION

    # In AI conversation, pressing 9 requests transfer
    state, action = sm.process_dtmf("9")
    assert state == IVRState.HUMAN_TRANSFER
    assert action == "HUMAN_TRANSFER"

    # Other digits are ignored
    state, action = sm.process_dtmf("5")
    assert action == "IGNORED"


def test_timeout_progression():
    sm = IVRMenuStateMachine()
    sm.start_call()

    # Timeout 1
    state, action = sm.handle_timeout()
    assert state == IVRState.LANGUAGE_SELECTION
    assert action == "TIMEOUT_RETRY"
    assert sm.retries == 1

    # Timeout 2
    state, action = sm.handle_timeout()
    assert state == IVRState.LANGUAGE_SELECTION
    assert action == "TIMEOUT_RETRY"
    assert sm.retries == 2

    # Timeout 3 -> defaults to English and advances
    state, action = sm.handle_timeout()
    assert state == IVRState.SERVICE_SELECTION
    assert action == "MAX_RETRIES_EXCEEDED"
    assert sm.language == "en"
    assert sm.retries == 0


def test_prompt_rendering():
    # Test service ack prompt formatting in all 3 languages
    en_ack = get_service_ack_prompt("en", "upi_payment")
    assert "UPI and Payment Support" in en_ack
    assert "Please explain your issue" in en_ack

    hi_ack = get_service_ack_prompt("hi", "upi_payment")
    assert "यूपीआई" in hi_ack
    assert "धन्यवाद" in hi_ack or "Dhanyawad" in hi_ack

    or_ack = get_service_ack_prompt("or", "upi_payment")
    assert "ୟୁପିଆଇ" in or_ack
    assert "ଧନ୍ୟବାଦ" in or_ack or "Dhanyabad" in or_ack

