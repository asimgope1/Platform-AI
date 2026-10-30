"""
IVR Menu State Machine & Routing Definitions
Supports keypad DTMF navigation for Multilingual Customer Assistance.
"""
from enum import Enum
from typing import Optional, Dict, Tuple


class IVRState(str, Enum):
    WELCOME = "WELCOME"
    LANGUAGE_SELECTION = "LANGUAGE_SELECTION"
    SERVICE_SELECTION = "SERVICE_SELECTION"
    AI_CONVERSATION = "AI_CONVERSATION"
    HUMAN_TRANSFER = "HUMAN_TRANSFER"
    CALL_END = "CALL_END"


SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "1": {"code": "en", "name": "English", "whisper_lang": "en"},
    "2": {"code": "hi", "name": "Hindi", "whisper_lang": "hi"},
    "3": {"code": "or", "name": "Odia", "whisper_lang": "or"},
}

SERVICE_CATEGORIES: Dict[str, Dict[str, str]] = {
    "1": {
        "key": "customer_support",
        "name_en": "Customer Support",
        "name_hi": "ग्राहक सेवा (Customer Support)",
        "name_or": "ଗ୍ରାହକ ସେବା (Customer Support)",
        "rag_scope": "customer_support",
    },
    "2": {
        "key": "complaint",
        "name_en": "Register a Complaint",
        "name_hi": "शिकायत पंजीकरण (Complaint)",
        "name_or": "ଅଭିଯୋଗ ପଞ୍ଜୀକରଣ (Complaint)",
        "rag_scope": "complaint",
    },
    "3": {
        "key": "upi_payment",
        "name_en": "UPI & Payment Issues",
        "name_hi": "यूपीआई और भुगतान समस्याएं (UPI & Payments)",
        "name_or": "ୟୁପିଆଇ ଏବଂ ପେମେଣ୍ଟ (UPI & Payments)",
        "rag_scope": "upi_payment",
    },
    "4": {
        "key": "account_services",
        "name_en": "Account & Banking Services",
        "name_hi": "खाता और बैंकिंग सेवाएं (Account Services)",
        "name_or": "ଖାତା ଏବଂ ବ୍ୟାଙ୍କିଙ୍ଗ ସେବା (Account Services)",
        "rag_scope": "account_services",
    },
    "5": {
        "key": "card_services",
        "name_en": "Card-Related Issues",
        "name_hi": "कार्ड संबंधित समस्याएं (Card Services)",
        "name_or": "କାର୍ଡ ସମ୍ବନ୍ଧୀୟ ସମସ୍ୟା (Card Services)",
        "rag_scope": "card_services",
    },
    "6": {
        "key": "loans_emi",
        "name_en": "Loan & EMI Queries",
        "name_hi": "ऋण और ईएमआई पूछताछ (Loans & EMI)",
        "name_or": "ଋଣ ଏବଂ ଇଏମଆଇ (Loans & EMI)",
        "rag_scope": "loans_emi",
    },
    "7": {
        "key": "other_enquiries",
        "name_en": "Other Enquiries",
        "name_hi": "अन्य पूछताछ (Other Enquiries)",
        "name_or": "ଅନ୍ୟାନ୍ୟ ସୂଚନା (Other Enquiries)",
        "rag_scope": "other_enquiries",
    },
}

SERVICE_BY_KEY: Dict[str, Dict[str, str]] = {v["key"]: v for v in SERVICE_CATEGORIES.values()}

MAX_MENU_RETRIES = 3
MENU_TIMEOUT_SECONDS = 9.0


class IVRMenuStateMachine:
    """Deterministic IVR menu state machine for handling DTMF transitions."""

    def __init__(self):
        self.state: IVRState = IVRState.WELCOME
        self.language: str = "en"
        self.service_key: Optional[str] = None
        self.service_info: Optional[Dict[str, str]] = None
        self.retries: int = 0

    def start_call(self) -> IVRState:
        """Transitions call into LANGUAGE_SELECTION."""
        self.state = IVRState.LANGUAGE_SELECTION
        self.retries = 0
        return self.state

    def process_dtmf(self, digit: str) -> Tuple[IVRState, str]:
        """
        Processes a keypad DTMF digit according to current state.
        Returns (new_state, action_code):
          action_code can be:
            'LANGUAGE_SELECTED'
            'SERVICE_SELECTED'
            'HUMAN_TRANSFER'
            'REPEAT_MENU'
            'INVALID_INPUT'
            'MAX_RETRIES_EXCEEDED'
            'IGNORED'
        """
        digit = digit.strip()

        if self.state == IVRState.LANGUAGE_SELECTION:
            if digit in SUPPORTED_LANGUAGES:
                self.language = SUPPORTED_LANGUAGES[digit]["code"]
                self.state = IVRState.SERVICE_SELECTION
                self.retries = 0
                return self.state, "LANGUAGE_SELECTED"

            if digit == "0":
                # Repeat language menu
                return self.state, "REPEAT_MENU"

            # Invalid selection
            self.retries += 1
            if self.retries >= MAX_MENU_RETRIES:
                # Default to English and proceed
                self.language = "en"
                self.state = IVRState.SERVICE_SELECTION
                self.retries = 0
                return self.state, "MAX_RETRIES_EXCEEDED"
            return self.state, "INVALID_INPUT"

        elif self.state == IVRState.SERVICE_SELECTION:
            if digit in SERVICE_CATEGORIES:
                self.service_key = SERVICE_CATEGORIES[digit]["key"]
                self.service_info = SERVICE_CATEGORIES[digit]
                self.state = IVRState.AI_CONVERSATION
                self.retries = 0
                return self.state, "SERVICE_SELECTED"

            if digit == "9":
                self.state = IVRState.HUMAN_TRANSFER
                self.retries = 0
                return self.state, "HUMAN_TRANSFER"

            if digit == "0":
                # Repeat service menu
                return self.state, "REPEAT_MENU"

            # Invalid selection
            self.retries += 1
            if self.retries >= MAX_MENU_RETRIES:
                # Default to general customer support
                self.service_key = "customer_support"
                self.service_info = SERVICE_CATEGORIES["1"]
                self.state = IVRState.AI_CONVERSATION
                self.retries = 0
                return self.state, "MAX_RETRIES_EXCEEDED"
            return self.state, "INVALID_INPUT"

        elif self.state == IVRState.AI_CONVERSATION:
            # During AI conversation, pressing 9 requests human transfer
            if digit == "9":
                self.state = IVRState.HUMAN_TRANSFER
                return self.state, "HUMAN_TRANSFER"
            return self.state, "IGNORED"

        return self.state, "IGNORED"

    def handle_timeout(self) -> Tuple[IVRState, str]:
        """
        Handles menu timeout when no keypad input was received.
        Returns (state, action_code):
          'TIMEOUT_RETRY' or 'MAX_RETRIES_EXCEEDED'
        """
        self.retries += 1
        if self.retries >= MAX_MENU_RETRIES:
            if self.state == IVRState.LANGUAGE_SELECTION:
                self.language = "en"
                self.state = IVRState.SERVICE_SELECTION
                self.retries = 0
                return self.state, "MAX_RETRIES_EXCEEDED"
            elif self.state == IVRState.SERVICE_SELECTION:
                self.service_key = "customer_support"
                self.service_info = SERVICE_CATEGORIES["1"]
                self.state = IVRState.AI_CONVERSATION
                self.retries = 0
                return self.state, "MAX_RETRIES_EXCEEDED"

        return self.state, "TIMEOUT_RETRY"

    def reset(self) -> None:
        """Resets state machine for new call."""
        self.state = IVRState.WELCOME
        self.language = "en"
        self.service_key = None
        self.service_info = None
        self.retries = 0
