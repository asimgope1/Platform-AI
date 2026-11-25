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
    ESCALATED = "ESCALATED"
    FEEDBACK = "FEEDBACK"
    CALL_END = "CALL_END"


SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "1": {"code": "en", "name": "English", "whisper_lang": "en"},
    "2": {"code": "hi", "name": "Hindi", "whisper_lang": "hi"},
    "3": {"code": "or", "name": "Odia", "whisper_lang": "or"},
}

SERVICE_CATEGORIES: Dict[str, Dict[str, str]] = {
    "1": {
        "key": "citizen_services",
        "name_en": "Citizen Services & Schemes",
        "name_hi": "नागरिक सेवाएं और योजनाएं (Citizen Services)",
        "name_or": "ନାଗରିକ ସେବା ଓ ଯୋଜନା (Citizen Services)",
        "rag_scope": "citizen_services",
    },
    "2": {
        "key": "complaint",
        "name_en": "Complaint Status and Registration",
        "name_hi": "शिकायत स्थिति और पंजीकरण (Complaint)",
        "name_or": "ଅଭିଯୋଗ ସ୍ଥିତି ଏବଂ ପଞ୍ଜୀକରଣ (Complaint)",
        "rag_scope": "complaint",
    },
    "3": {
        "key": "subhadra_scheme",
        "name_en": "Subhadra Scheme Assistance",
        "name_hi": "सुभद्रा योजना सहायता (Subhadra Scheme)",
        "name_or": "ସୁଭଦ୍ରା ଯୋଜନା ସହାୟତା (Subhadra Scheme)",
        "rag_scope": "subhadra_scheme",
    },
    "4": {
        "key": "certificates",
        "name_en": "Certificate Services (Caste, Income, Residence)",
        "name_hi": "प्रमाण पत्र सेवाएं (जाति, आय, निवास)",
        "name_or": "ପ୍ରମାଣପତ୍ର ସେବା (ଜାତି, ଆୟ, ବାସସ୍ଥାନ)",
        "rag_scope": "certificates",
    },
    "5": {
        "key": "identity_cards",
        "name_en": "Identity Cards (Aadhaar, PAN, Voter ID)",
        "name_hi": "पहचान पत्र (आधार, पैन, वोटर आईडी)",
        "name_or": "ପରିଚୟ ପତ୍ର (ଆଧାର, ପ୍ୟାନ୍, ଭୋଟର ଆଇଡି)",
        "rag_scope": "identity_cards",
    },
    "6": {
        "key": "legal_documents",
        "name_en": "Legal Documents & Affidavits",
        "name_hi": "कानूनी दस्तावेज और शपथ पत्र (Legal Documents)",
        "name_or": "ଆଇନଗତ ଦଲିଲ ଏବଂ ସତ୍ୟପାଠ (Legal Documents)",
        "rag_scope": "legal_documents",
    },
    "7": {
        "key": "kendra_helpdesk",
        "name_en": "Jana Seva Kendra Enquiries & Timings",
        "name_hi": "जन सेवा केंद्र पूछताछ और समय (Kendra Helpdesk)",
        "name_or": "ଜନ ସେବା କେନ୍ଦ୍ର ସୂଚନା (Kendra Helpdesk)",
        "rag_scope": "kendra_helpdesk",
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
                # Default to general citizen services
                self.service_key = "citizen_services"
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
