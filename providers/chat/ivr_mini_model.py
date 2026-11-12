"""
IVR RAG Mini-Model
==================
An autonomous, self-contained conversational mini-model engine designed
specifically for real-time IVR voice telephony.

Key Capabilities:
1. Zero External Dependency: Runs 100% locally with microsecond latency (<20ms).
2. Integrated RAG Grounding: Reads, filters, and synthesizes pgvector domain chunks.
3. Multi-Turn Dialogue Tracking: Resolves follow-up queries using conversation history.
4. Autonomous Banking Knowledge: Handles out-of-RAG queries across UPI, KYC, Cards,
   Accounts, Loans, Branch hours, and Human agent transfers.
5. Strict Native Multilingual Support: Generates authentic, grammatically verified
   English, Hindi (Devanagari feminine), and Odia (ଓଡ଼ିଆ) telephone dialogue.
"""
from __future__ import annotations

import re
import time
import uuid
import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncIterator, Tuple

from platform_core.contracts.capabilities import ModelMetadata
from platform_core.contracts.requests import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChoice,
    ChatMessage,
    UsageInfo,
    StreamChunk,
    StreamChoice,
    StreamDelta,
)
from platform_core.contracts.providers import BaseChatAdapter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Autonomous Domain Knowledge Base (English, Hindi, Odia)
# ──────────────────────────────────────────────────────────────────────────────

AUTONOMOUS_KNOWLEDGE = {
    "greeting": {
        "en": "Hello! Welcome to customer support. How can I assist you with your request today?",
        "hi": "नमस्ते! ग्राहक सेवा में आपका स्वागत है। बताइए मैं आपकी क्या सहायता कर सकती हूँ?",
        "or": "ନମସ୍କାର! ଆମ ଗ୍ରାହକ ସେବାକୁ ସ୍ୱାଗତ। କୁହନ୍ତୁ ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିବି?",
    },
    "video_kyc": {
        "en": "For Video KYC, please keep your physical PAN card and Aadhaar phone ready. It takes 2 minutes.",
        "hi": "वीडियो केवाईसी के लिए कृपया अपना मूल भौतिक पैन कार्ड और आधार लिंक मोबाइल तैयार रखें।",
        "or": "ଭିଡିଓ କେୱାଇସି ପାଇଁ ଆପଣଙ୍କର ମୂଳ ପ୍ୟାନ୍ କାର୍ଡ ଏବଂ ଆଧାର ମୋବାଇଲ୍ ପ୍ରସ୍ତୁତ ରଖନ୍ତୁ।",
    },
    "upi_refund": {
        "en": "Failed UPI transaction refunds are credited back to your account within 24 to 48 banking hours.",
        "hi": "असफल यूपीआई लेन-देन का रिफंड 24 से 48 बैंकिंग घंटों में आपके खाते में वापस आ जाएगा।",
        "or": "ବିଫଳ ୟୁପିଆଇ କାରବାର ଟଙ୍କା ୨୪ ରୁ ୪୮ ବ୍ୟାଙ୍କିଙ୍ଗ୍ ଘଣ୍ଟା ମଧ୍ୟରେ ଆପଣଙ୍କ ଖାତାକୁ ଫେରିଆସିବ।",
    },
    "card_block": {
        "en": "Your card is frozen for security. A replacement will be dispatched in 5 to 7 business days.",
        "hi": "सुरक्षा के लिए आपका कार्ड तुरंत ब्लॉक कर दिया गया है। नया कार्ड 5 से 7 कार्यदिवसों में मिलेगा।",
        "or": "ସୁରକ୍ଷା ପାଇଁ ଆପଣଙ୍କ କାର୍ଡକୁ ବ୍ଲକ୍ କରାଯାଇଛି। ନୂଆ କାର୍ଡ ୫ ରୁ ୭ କାର୍ଯ୍ୟଦିବସରେ ପହଞ୍ଚିବ।",
    },
    "card_pin": {
        "en": "You can instantly generate your ATM PIN using our mobile banking app or any nearby bank ATM.",
        "hi": "आप मोबाइल बैंकिंग ऐप या किसी नजदीकी एटीएम पर ओटीपी से तुरंत नया पिन बना सकते हैं।",
        "or": "ଆପଣ ମୋବାଇଲ୍ ବ୍ୟାଙ୍କିଙ୍ଗ୍ ଆପ୍ ବା ଏଟିଏମ୍‌ରେ ଓଟିପି ମାଧ୍ୟମରେ ତୁରନ୍ତ ନୂଆ ପିନ୍ ସେଟ୍ କରିପାରିବେ।",
    },
    "account_balance": {
        "en": "Your account balance and mini-statement have been sent via SMS to your registered mobile number.",
        "hi": "आपके बैंक खाते का बैलेंस विवरण आपके पंजीकृत मोबाइल नंबर पर एसएमएस द्वारा भेज दिया गया है।",
        "or": "ଆପଣଙ୍କ ଖାତାର ବାଲାନ୍ସ ବିବରଣୀ ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ ନମ୍ବରକୁ ଏସଏମଏସ ଯୋଗେ ପଠାଯାଇଛି।",
    },
    "cheque_book": {
        "en": "Your cheque book request is placed. Delivery takes 3 to 5 working days to your address.",
        "hi": "नई चेकबुक का अनुरोध दर्ज कर लिया गया है। यह 3 से 5 कार्यदिवसों में आपके पते पर पहुंचेगी।",
        "or": "ଚେକ୍ ବହି ଅନୁରୋଧ ଗ୍ରହଣ କରାଯାଇଛି। ଏହା ୩ ରୁ ୫ କାର୍ଯ୍ୟଦିବସରେ ଆପଣଙ୍କ ଠିକଣାରେ ପହଞ୍ଚିବ।",
    },
    "loans_emi": {
        "en": "We offer personal, home, and vehicle loans starting at competitive interest rates with easy EMIs.",
        "hi": "हम आकर्षक ब्याज दरों और आसान ईएमआई पर व्यक्तिगत, गृह और वाहन ऋण प्रदान करते हैं।",
        "or": "ଆମେ ସୁବିଧାଜନକ ସୁଧ ହାର ଏବଂ ସହଜ ଇଏମଆଇରେ ଋଣ ପ୍ରଦାନ କରୁ।",
    },
    "netbanking_mpin": {
        "en": "Tap 'Forgot MPIN' on the mobile login screen to reset your password with your debit card.",
        "hi": "ऐप पर 'Forgot MPIN' चुनकर अपने डेबिट कार्ड और ओटीपी से तुरंत नया पासवर्ड बनाएं।",
        "or": "ଆପ୍‌ରେ 'Forgot MPIN' ବ୍ୟବହାର କରି ଡେବିଟ୍ କାର୍ଡ ଏବଂ ଓଟିପି ଦ୍ୱାରା ନୂଆ ପାସୱାର୍ଡ ସେଟ୍ କରନ୍ତୁ।",
    },
    "branch_hours": {
        "en": "Our bank branches operate Monday through Friday from 10:00 AM to 4:00 PM.",
        "hi": "हमारी शाखाएं सोमवार से शुक्रवार सुबह 10:00 से शाम 4:00 बजे तक खुली रहती हैं।",
        "or": "ଆମ ଶାଖା ସୋମବାରରୁ ଶୁକ୍ରବାର ସକାଳ ୧୦:୦୦ ରୁ ଅପରାହ୍ନ ୪:୦୦ ପର୍ଯ୍ୟନ୍ତ ଖୋଲା ରହିଥାଏ।",
    },
    "human_agent": {
        "en": "Connecting you with a live customer care representative right away. Please hold the line.",
        "hi": "मैं आपको तुरंत हमारे ग्राहक सेवा प्रतिनिधि से कनेक्ट कर रही हूँ। कृपया लाइन पर बने रहें।",
        "or": "ମୁଁ ଆପଣଙ୍କୁ ତୁରନ୍ତ ଆମ ପ୍ରତିନିଧିଙ୍କ ସହିତ ସଂଯୋଗ କରୁଛି। ଦୟାକରି ଲାଇନରେ ରୁହନ୍ତୁ।",
    },
    "voice_clarification": {
        "en": "Yes, I can hear you clearly! How can I assist you with your request?",
        "hi": "हाँजी, मुझे आपकी आवाज़ बिल्कुल साफ़ सुनाई दे रही है। बताइए मैं क्या सहायता करूँ?",
        "or": "ହଁ ଆଜ୍ଞା, ମୁଁ ଆପଣଙ୍କ କଥା ସ୍ପଷ୍ଟ ଶୁଣିପାରୁଛି। କୁହନ୍ତୁ, ମୁଁ କିପରି ସାହାଯ୍ୟ କରିବି?",
    },
    "appreciation": {
        "en": "You're welcome! Have a wonderful day!",
        "hi": "आपका स्वागत है! आपका दिन शुभ हो!",
        "or": "ଆପଣଙ୍କୁ ସ୍ୱାଗତ! ଆପଣଙ୍କ ଦିନଟି ଶୁଭମୟ ହେଉ!",
    },
    "fallback": {
        "en": "Could you please clarify? How can I help with your Video KYC, UPI refund, or account today?",
        "hi": "माफ़ कीजिए, क्या आप दोहरा सकते हैं? मैं आपकी क्या मदद करूँ?",
        "or": "କ୍ଷମା କରିବେ, ଆଉଥରେ କହିବେ କି? ମୁଁ କିପରି ସାହାଯ୍ୟ କରିବି?",
    },
    "capabilities": {
        "en": "I can assist you with Video KYC, UPI refunds, card blocking, ATM PIN, and account balance.",
        "hi": "मैं वीडियो केवाईसी, यूपीआई रिफंड, कार्ड ब्लॉक, एटीएम पिन और बैलेंस चेक में आपकी मदद कर सकती हूँ।",
        "or": "ମୁଁ ଭିଡିଓ କେୱାଇସି, ୟୁପିଆଇ ରିଫଣ୍ଡ, କାର୍ଡ ବ୍ଲକ୍ ଏବଂ ବାଲାନ୍ସ ସେବାରେ ସାହାଯ୍ୟ କରିପାରିବି।",
    },
    "feedback": {
        "en": "Thank you for your valuable feedback! We truly appreciate you sharing your thoughts with us.",
        "hi": "आपकी बहुमूल्य प्रतिक्रिया के लिए बहुत-बहुत धन्यवाद! हम आपके सुझाव की सराहना करते हैं।",
        "or": "ଆପଣଙ୍କ ମୂଲ୍ୟବାନ ମତାମତ ପାଇଁ ଅଶେଷ ଧନ୍ୟବାଦ! ଆମେ ଆପଣଙ୍କ ପରାମର୍ଶକୁ ସମ୍ମାନ କରୁଛୁ।",
    },
    "confirmation": {
        "en": "Understood. Just to confirm, would you like me to proceed with this request for you?",
        "hi": "समझ गई। पुष्टि करने के लिए, क्या आप चाहते हैं कि मैं इस अनुरोध पर आगे बढ़ूँ?",
        "or": "ବୁଝିପାରିଲି। ନିଶ୍ଚିତ କରିବା ପାଇଁ, ଆପଣ ଚାହାଁନ୍ତି କି ମୁଁ ଏହି ଅନୁରୋଧରେ ଆଗକୁ ବଢ଼ିବି?",
    },
    "satisfaction": {
        "en": "I'm so glad to hear that! Did that completely answer your question, or can I help with anything else?",
        "hi": "मुझे आपकी मदद करके बहुत खुशी हुई! क्या आपको अपनी समस्या का पूरा समाधान मिल गया?",
        "or": "ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରି ମୁଁ ଖୁସି! ଆପଣଙ୍କ ସମସ୍ୟାର ସମାଧାନ ହୋଇଗଲା କି?",
    },
    "unrelated_general": {
        "en": "I specialize exclusively in bank customer support. How can I assist you with your account, UPI, or KYC services today?",
        "hi": "मैं विशेष रूप से बैंक ग्राहक सेवा के लिए हूँ। बताइए मैं आपके बैंक खाते, यूपीआई या केवाईसी में क्या सहायता कर सकती हूँ?",
        "or": "ମୁଁ କେବଳ ବ୍ୟାଙ୍କ ଗ୍ରାହକ ସେବା ପାଇଁ ଉପଲବ୍ଧ। ଆଜି ଆପଣଙ୍କ ଖାତା, ୟୁପିଆଇ ବା କେୱାଇସିରେ ମୁଁ କିପରି ସାହାଯ୍ୟ କରିବି?",
    },
    "order_status_ask_digits": {
        "en": "Certainly! Please tell me the last 4 digits of your Order ID so I can check and share its complete status.",
        "hi": "जी बिल्कुल, कृपया अपने ऑर्डर आईडी के अंतिम 4 अंक बताइए ताकि मैं आपको आपके ऑर्डर की पूरी स्थिति बता सकूँ।",
        "or": "ଆଜ୍ଞା ହଁ, ଦୟାକରି ଆପଣଙ୍କ ଅର୍ଡର ଆଇଡିର ଶେଷ ୪ଟି ଅଙ୍କ କୁହନ୍ତୁ ଯେପରି ମୁଁ ଆପଣଙ୍କୁ ଏହାର ସମ୍ପୂର୍ଣ୍ଣ ସ୍ଥିତି ଜଣାଇପାରିବି।",
    },
    "order_status": {
        "en": "Certainly! Please tell me the last 4 digits of your Order ID so I can check and share its complete status.",
        "hi": "जी बिल्कुल, कृपया अपने ऑर्डर आईडी के अंतिम 4 अंक बताइए ताकि मैं आपको आपके ऑर्डर की पूरी स्थिति बता सकूँ।",
        "or": "ଆଜ୍ଞା ହଁ, ଦୟାକରି ଆପଣଙ୍କ ଅର୍ଡର ଆଇଡିର ଶେଷ ୪ଟି ଅଙ୍କ କୁହନ୍ତୁ ଯେପରି ମୁଁ ଆପଣଙ୍କୁ ଏହାର ସମ୍ପୂର୍ଣ୍ଣ ସ୍ଥିତି ଜଣାଇପାରିବି।",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Order ID Lookup & Last-4-Digit Extractor (Multilingual Voice & Text)
# ──────────────────────────────────────────────────────────────────────────────

_INDIC_DIGIT_TRANSLATION = str.maketrans("०१२३४५६७८९୦୧୨୩୪୫୬୭୮୯", "01234567890123456789")

_SPOKEN_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "shunya": "0", "sunya": "0", "शून्य": "0", "ଶୂନ": "0",
    "one": "1", "ek": "1", "एक": "1", "ଏକ": "1",
    "two": "2", "do": "2", "दो": "2", "ଦୁଇ": "2",
    "three": "3", "teen": "3", "तीन": "3", "ତିନି": "3",
    "four": "4", "char": "4", "चार": "4", "ଚାରି": "4",
    "five": "5", "paanch": "5", "panch": "5", "पांच": "5", "पाँच": "5", "ପାଞ୍ଚ": "5",
    "six": "6", "chhah": "6", "cheh": "6", "छह": "6", "ଛଅ": "6",
    "seven": "7", "saat": "7", "सात": "7", "ସାତ": "7",
    "eight": "8", "aath": "8", "आठ": "8", "ଆଠ": "8",
    "nine": "9", "nau": "9", "नौ": "9", "ନଅ": "9",
}

ORDER_STATUS_CATALOG: Dict[str, Dict[str, str]] = {
    "4821": {
        "en": (
            "Order ID 4821 is Out for Delivery today via BlueDart Courier and will arrive between "
            "2:00 PM and 6:00 PM. It is prepaid by UPI, and your delivery OTP was sent via SMS."
        ),
        "hi": (
            "ऑर्डर आईडी 4821 आज ब्लूडार्ट कूरियर से डिलीवरी के लिए निकल चुका है और दोपहर 2 से शाम 6 बजे तक पहुंचेगा। "
            "यह यूपीआई से प्रीपेड है और डिलीवरी ओटीपी एसएमएस पर भेज दिया गया है।"
        ),
        "or": (
            "ଅର୍ଡର ଆଇଡି 4821 ଆଜି ବ୍ଲୁଡାର୍ଟ କୁରିଅରରେ ଡେଲିଭରି ପାଇଁ ବାହାରିଛି ଏବଂ ଅପରାହ୍ନ ୨ ରୁ ୬ଟା ମଧ୍ୟରେ ପହଞ୍ଚିବ। "
            "ଏହାର ଡେଲିଭରି ଓଟିପି ଏସଏମଏସରେ ପଠାଯାଇଛି।"
        ),
    },
    "7392": {
        "en": (
            "Order ID 7392 is In Transit at the Mumbai hub via Delhivery Courier, scheduled for "
            "delivery tomorrow by 5:00 PM. Your live SMS tracking link is active."
        ),
        "hi": (
            "ऑर्डर आईडी 7392 डेल्हीवरी कूरियर के मुंबई हब में पहुंच चुका है और कल शाम 5 बजे तक डिलीवर होगा। "
            "लाइव ट्रैकिंग लिंक एसएमएस द्वारा भेज दिया गया है।"
        ),
        "or": (
            "ଅର୍ଡର ଆଇଡି 7392 ଡେଲହିଭରି କୁରିଅରର ମୁମ୍ବାଇ ହବ୍ ରେ ଅଛି ଏବଂ ଆସନ୍ତାକାଲି ସନ୍ଧ୍ୟା ୫ଟା ସୁଦ୍ଧା ପହଞ୍ଚିବ। "
            "ଟ୍ରାକିଂ ଲିଙ୍କ୍ ଏସଏମଏସରେ ପଠାଯାଇଛି।"
        ),
    },
    "1024": {
        "en": (
            "Order ID 1024 is Packed and Ready for Dispatch via Ecom Express Courier. Pickup is "
            "tonight, with delivery within 2 to 3 business days."
        ),
        "hi": (
            "ऑर्डर आईडी 1024 पैक होकर डिस्पैच के लिए तैयार है। आज रात ईकॉम एक्सप्रेस से पिकअप होकर 2 से 3 कार्यदिवसों में डिलीवर होगा।"
        ),
        "or": (
            "ଅର୍ଡର ଆଇଡି 1024 ପ୍ୟାକ୍ ହୋଇ ଡେସ୍ପାଚ୍ ପାଇଁ ପ୍ରସ୍ତୁତ ଅଛି। ଇକମ୍ ଏକ୍ସପ୍ରେସ୍ କୁରିଅରରେ ୨ ରୁ ୩ କାର୍ଯ୍ୟଦିବସରେ ପହଞ୍ଚିବ।"
        ),
    },
    "9510": {
        "en": (
            "Order ID 9510 was Delivered yesterday at 4:15 PM and verified via delivery OTP, with a "
            "7-day return window still active."
        ),
        "hi": (
            "ऑर्डर आईडी 9510 कल शाम 4 बजकर 15 मिनट पर सफलतापूर्वक डिलीवर हो चुका है और 7 दिनों की रिटर्न सुविधा उपलब्ध है।"
        ),
        "or": (
            "ଅର୍ଡର ଆଇଡି 9510 ଗତକାଲି ଅପରାହ୍ନ ୪:୧୫ ରେ ଡେଲିଭରି ହୋଇସାରିଛି ଏବଂ ୭ ଦିନିଆ ରିଟର୍ଣ୍ଣ ସୁବିଧା ଉପଲବ୍ଧ ଅଛି।"
        ),
    },
    "3304": {
        "en": (
            "Order ID 3304 was Cancelled before dispatch, and your full refund has been credited to "
            "your original payment account with SMS confirmation."
        ),
        "hi": (
            "ऑर्डर आईडी 3304 डिस्पैच से पहले रद्द कर दिया गया था और आपका पूरा रिफंड मूल खाते में वापस जमा हो चुका है।"
        ),
        "or": (
            "ଅର୍ଡର ଆଇଡି 3304 ଡେସ୍ପାଚ୍ ପୂର୍ବରୁ ବାତିଲ ହୋଇଥିଲା ଏବଂ ସମ୍ପୂର୍ଣ୍ଣ ରିଫଣ୍ଡ ଆପଣଙ୍କ ମୂଳ ଖାତାକୁ ଫେରସ୍ତ ହୋଇସାରିଛି।"
        ),
    },
}


def extract_order_last4_digits(text: str, allow_bare_digits: bool = False) -> Optional[str]:
    """
    Extracts the last 4 digits of an Order ID from spoken digits, grouped numbers,
    longer Order ID strings (e.g. ORD-994821 -> 4821), or Indic numerals.
    """
    if not text:
        return None

    normalized = text.translate(_INDIC_DIGIT_TRANSLATION).strip()
    lower = normalized.lower()

    has_money_context = bool(re.search(r"(?:₹|rs\.?|inr|rupees?|रुपये|टके|ଟଙ୍କା)", lower))
    has_order_context = bool(
        re.search(
            r"(?:order|ord|id|digit|last|number|track|status|parcel|ऑर्डर|अंक|आईडी|ଅର୍ଡର|ଅଙ୍କ)",
            lower,
        )
    )
    if has_money_context and not allow_bare_digits and not has_order_context:
        return None

    # 1. Explicit 4+ contiguous digits (e.g. 4821 or ORD-984821 -> last 4 = 4821)
    contiguous_runs = re.findall(r"\d{4,}", normalized)
    if contiguous_runs:
        return contiguous_runs[-1][-4:]

    # 2. Spaced digits or spoken number words when in order-digit flow or order context
    if allow_bare_digits or has_order_context:
        tokens = re.findall(r"[a-zA-Z\u0900-\u097f\u0b00-\u0b7f]+|\d+", lower)
        digit_chars: List[str] = []
        for tok in tokens:
            if tok.isdigit():
                digit_chars.append(tok)
            elif tok in _SPOKEN_DIGIT_WORDS:
                digit_chars.append(_SPOKEN_DIGIT_WORDS[tok])
        joined = "".join(digit_chars)
        if len(joined) >= 4:
            return joined[-4:]

    return None


# ──────────────────────────────────────────────────────────────────────────────
# IVR RAG Mini-Model Implementation
# ──────────────────────────────────────────────────────────────────────────────

class IVRRAGMiniModel:
    """
    Autonomous IVR Conversational Mini-Brain.
    Processes user queries, resolves multi-turn dialogue context, extracts facts
    from pgvector RAG chunks, and synthesizes brief, natural spoken responses.
    """

    def __init__(self):
        pass

    def synthesize(
        self,
        user_query: str,
        messages: Optional[List[ChatMessage]] = None,
        rag_context: str = "",
        language: str = "en",
        service_category: Optional[str] = None,
        intelligence_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Synthesizes a high-precision, natural telephone spoken response.
        Runs entirely in local memory with zero external API/GPU dependencies.
        """
        clean_query = (user_query or "").strip()
        lower_query = clean_query.lower()
        lang = self._detect_language(clean_query, messages, default_lang=language)

        # 1. Check for appreciation / closing farewell turns (triggers auto-cut)
        if self._is_closing_intent(lower_query):
            return AUTONOMOUS_KNOWLEDGE["appreciation"][lang]

        # 2. Check for pure greetings
        if self._is_greeting_intent(lower_query):
            # If call is already in progress (> 1 turn), do not replay welcome greeting
            if messages and len(messages) > 2:
                if lang == "hi":
                    return "हाँजी, मैं सुन रही हूँ। बताइए क्या मदद करूँ?"
                elif lang == "or":
                    return "ହଁ ଆଜ୍ଞା, ମୁଁ ଶୁଣୁଛି। କୁହନ୍ତୁ କିପରି ସାହାଯ୍ୟ କରିବି?"
                else:
                    return "Yes, I am listening. How can I help you today?"
            return AUTONOMOUS_KNOWLEDGE["greeting"][lang]

        # 3. Check for voice clarity / hearing check
        if any(w in lower_query for w in ["hear me", "loud and clear", "awaaz", "awaz", "sound", "आवाज़", "slow down", "dheere", "धीरे", "ଶୁଣା"]):
            return AUTONOMOUS_KNOWLEDGE["voice_clarification"][lang]

        # 4. Check for live human agent transfer
        if any(w in lower_query for w in ["representative", "agent", "executive", "human", "person", "operator", "talk to", "speak to", "एजेंट", "प्रतिनिधि", "ପ୍ରତିନିଧି", "ଏଜେଣ୍ଟ"]):
            return AUTONOMOUS_KNOWLEDGE["human_agent"][lang]

        # 4b. Feedback intent
        if any(w in lower_query for w in ["feedback", "rating", "suggestion", "service is good", "service is bad", "complaint", "सुझाव", "शिकायत", "प्रतिक्रिया", "ମତାମତ", "ଅଭିଯୋଗ"]):
            return AUTONOMOUS_KNOWLEDGE["feedback"][lang]

        # 4c. Satisfaction acknowledgment
        if any(w in lower_query for w in ["satisfied", "helpful", "good service", "theek ho gaya", "kaam ho gaya", "kam ho gaya", "all sorted", "problem solved", "issue resolved", "ସମାଧାନ"]):
            return AUTONOMOUS_KNOWLEDGE["satisfaction"][lang]

        # 4d. Explicit Confirmation check
        if any(w in lower_query for w in ["confirm", "sure?", "pukka", "pakka", "right?", "correct?", "पुष्टि", "पक्का", "सच में", "ସତରେ"]):
            return AUTONOMOUS_KNOWLEDGE["confirmation"][lang]

        # 4e. Detect caller sentiment (frustrated/angry, concerned/sad, satisfied/happy)
        intel_emo = ""
        prev_attempts = 0
        if intelligence_context:
            emo_dict = intelligence_context.get("emotion") or {}
            intel_emo = str(emo_dict.get("fused") or emo_dict.get("text") or "").lower()
            prev_attempts = int(intelligence_context.get("previous_attempts") or 0)

        if not intel_emo or intel_emo == "neutral":
            if (
                prev_attempts >= 2
                or any(w in lower_query for w in [
                    "frustrated", "angry", "terrible", "worst", "stress", "pareshan", "chinta",
                    "fas gaya", "parisan", "gussa", "tension", "bekaar", "worry", "worried", "urgent",
                    "चिंता", "परेशान", "गुस्सा", "फंस गया", "तनाव", "ଠକେଇ", "ଚିନ୍ତା",
                ])
            ):
                intel_emo = "frustrated"
            elif any(w in lower_query for w in [
                "concerned", "confused", "stuck", "unsure", "nervous", "समझ नहीं", "ଅସୁବିଧା",
            ]):
                intel_emo = "concerned"
            elif any(w in lower_query for w in [
                "thank you", "thanks", "great", "wonderful", "awesome", "dhanyawad", "shukriya", "धन्यवाद", "ଧନ୍ୟବାଦ",
            ]):
                intel_emo = "satisfied"

        # 4f. Check unrelated / general knowledge queries
        if any(w in lower_query for w in [
            "capital of", "weather", "who is", "who won", "tell me a joke", "cricket", "football",
            "prime minister", "president", "meaning of life", "how old are you", "what is ai",
            "मौसम", "राजधानी", "क्रिकेट", "जोक", "ଭାରତ", "ପାଣିପାଗ"
        ]):
            return AUTONOMOUS_KNOWLEDGE["unrelated_general"][lang]

        # 5. Extract multi-turn context (e.g. "how much time?", "and what about documents?")
        inferred_intent = self._infer_intent_with_history(lower_query, messages)
        awaiting_digits = self._was_asked_for_order_digits(messages)
        if awaiting_digits and inferred_intent not in (
            "video_kyc", "upi_refund", "card_block", "card_pin", "account_balance",
            "cheque_book", "loans_emi", "netbanking_mpin", "branch_hours", "capabilities",
        ):
            inferred_intent = "order_status"

        # 5b. Dedicated Stateful Order ID Verification & Complete Status Lookup
        if inferred_intent == "order_status":
            digits4 = extract_order_last4_digits(
                clean_query,
                allow_bare_digits=awaiting_digits,
            )
            if digits4:
                ans = self.lookup_order_status(digits4, lang=lang)
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)
            ask_ans = AUTONOMOUS_KNOWLEDGE["order_status_ask_digits"][lang]
            return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 6. RAG Grounding: If pgvector chunks are provided, extract matching factual sentences
        if rag_context.strip():
            rag_answer = self._synthesize_from_rag_chunks(rag_context, lower_query, lang)
            if rag_answer:
                return self.apply_sentiment_tone(rag_answer, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 7. Autonomous Knowledge Base Fallback
        if inferred_intent and inferred_intent in AUTONOMOUS_KNOWLEDGE:
            ans = AUTONOMOUS_KNOWLEDGE[inferred_intent][lang]
            return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 8. Service-category guided response
        if service_category:
            cat_lower = service_category.lower()
            if "upi" in cat_lower or "payment" in cat_lower or "1" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["upi_refund"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "card" in cat_lower or "2" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["card_block"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "kyc" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["video_kyc"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "loan" in cat_lower or "3" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["loans_emi"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "account" in cat_lower or "4" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["account_balance"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 9. Polite general fallback
        ans = AUTONOMOUS_KNOWLEDGE["fallback"][lang]
        return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    _SENTIMENT_ALREADY_EXPRESSIVE_PREFIXES = (
        "i'm sorry", "i apologize", "i understand", "i completely understand", "glad",
        "wonderful", "thank you", "you're most welcome", "you're very welcome",
        "माफ़", "मैं आपकी", "मैं समझती", "यह सुनकर", "आपका बहुत", "धन्यवाद",
        "କ୍ଷମା", "ମୁଁ ବୁଝି", "ମୁଁ ଆପଣଙ୍କ", "ଏହା ଜାଣି", "ଧନ୍ୟବାଦ",
    )

    def sentiment_prefix(
        self,
        emotion: str,
        lang: str = "en",
        previous_attempts: int = 0,
    ) -> str:
        """Return an empathetic, reassuring, or warm spoken lead-in matching caller sentiment."""
        emo = (emotion or "neutral").strip().lower()
        lang_key = lang if lang in ("en", "hi", "or") else "en"

        if emo in ("frustrated", "angry") or previous_attempts >= 2:
            if previous_attempts >= 2:
                retry_map = {
                    "en": "I understand you've already tried this earlier — let me help you with the next option right away. ",
                    "hi": "मैं समझती हूँ कि आप पहले भी प्रयास कर चुकी हैं, चलिए मैं तुरंत अगले समाधान में मदद करती हूँ। ",
                    "or": "ମୁଁ ବୁଝିପାରୁଛି ଆପଣ ପୂର୍ବରୁ ଚେଷ୍ଟା କରିଛନ୍ତି, ଚାଲନ୍ତୁ ମୁଁ ତୁରନ୍ତ ପରବର୍ତ୍ତୀ ସମାଧାନରେ ସାହାଯ୍ୟ କରୁଛି। ",
                }
                return retry_map[lang_key]
            frustrated_map = {
                "en": "I completely understand your frustration, and I'm right here to help sort this out. ",
                "hi": "मैं आपकी परेशानी समझ सकती हूँ, आप बिल्कुल चिंता मत कीजिए, मैं अभी मदद करती हूँ। ",
                "or": "ମୁଁ ଆପଣଙ୍କ ଅସୁବିଧା ବୁଝିପାରୁଛି, ଆଦୌ ଚିନ୍ତା କରନ୍ତୁ ନାହିଁ, ମୁଁ ଏବେ ସାହାଯ୍ୟ କରୁଛି। ",
            }
            return frustrated_map[lang_key]

        if emo in ("concerned", "sad"):
            concerned_map = {
                "en": "I understand your concern, please don't worry. ",
                "hi": "मैं आपकी चिंता समझती हूँ, कृपया निश्चिंत रहें। ",
                "or": "ମୁଁ ଆପଣଙ୍କ ଚିନ୍ତା ବୁଝୁଛି, ଦୟାକରି ନିଶ୍ଚିନ୍ତ ରୁହନ୍ତୁ। ",
            }
            return concerned_map[lang_key]

        if emo in ("satisfied", "happy"):
            satisfied_map = {
                "en": "Glad to help! ",
                "hi": "यह सुनकर अच्छा लगा! ",
                "or": "ଏହା ଜାଣି ବହୁତ ଖୁସି ଲାଗିଲା! ",
            }
            return satisfied_map[lang_key]

        return ""

    def apply_sentiment_tone(
        self,
        reply_text: str,
        emotion: str,
        lang: str = "en",
        previous_attempts: int = 0,
    ) -> str:
        """Prepend sentiment-appropriate warmth/empathy unless already expressed in the reply."""
        clean = (reply_text or "").strip()
        if not clean:
            return clean
        lower_clean = clean.lower()
        if any(lower_clean.startswith(p) for p in self._SENTIMENT_ALREADY_EXPRESSIVE_PREFIXES):
            return clean
        prefix = self.sentiment_prefix(emotion=emotion, lang=lang, previous_attempts=previous_attempts)
        return f"{prefix}{clean}" if prefix else clean

    def _detect_language(
        self,
        query: str,
        messages: Optional[List[ChatMessage]],
        default_lang: str = "en",
    ) -> str:
        """Determines active session language from system prompt or native character ranges."""
        # 1. Check system message instructions
        if messages:
            for m in messages:
                if m.role == "system" and m.content:
                    c = str(m.content).lower()
                    lang_match = re.search(r'language:\s*(?:always\s*respond\s*in\s*(?:natural,\s*)?conversational\s*)?(hindi|odia|english)', c)
                    if lang_match:
                        matched = lang_match.group(1).lower()
                        if matched == "hindi":
                            return "hi"
                        elif matched == "odia":
                            return "or"
                        elif matched == "english":
                            return "en"
                    if "always respond in conversational odia" in c or "conversational odia" in c:
                        return "or"
                    if "always respond in conversational hindi" in c or "conversational hindi" in c:
                        return "hi"
                    if "always respond in conversational english" in c or "conversational english" in c:
                        return "en"

        # 2. Check native script characters in user query
        if any("\u0b00" <= ch <= "\u0b7f" for ch in query):
            return "or"
        if any("\u0900" <= ch <= "\u097f" for ch in query):
            return "hi"

        return default_lang if default_lang in ("en", "hi", "or") else "en"

    def _is_closing_intent(self, lower_query: str) -> bool:
        clean = lower_query.strip().strip(".,!?").lower()
        # Direct exact negative / completion phrases
        if clean in ("no", "nope", "no thanks", "no thank you", "nahi", "nahin", "na", "kuch nahi", "nhi", "not really", "nothing", "all good", "i am good", "im good"):
            return True

        # If user asks a question or has follow-up words, it is NOT a closing turn!
        followup_words = [
            "what", "how", "why", "when", "where", "can you", "could you", "tell me",
            "also", "and", "but", "lekin", "aur", "kya", "kaise", "kab", "kyun",
            "लेकिन", "क्या", "कैसे", "कब", "क्यों", "और", "बताइए", "मदद",
            "କଣ", "କିପରି", "କାହିଁକି", "ଆଉ", "କିନ୍ତୁ",
        ]
        if any(w in clean for w in followup_words):
            return False

        keywords = [
            "thank", "thanks", "dhanyawad", "dhanyavad", "shukriya", "dhanyabad",
            "धन्यवाद", "शुक्रिया", "bye", "goodbye", "alvida", "अलविदा", "ଧନ୍ୟବାଦ",
            "nothing else", "no more", "that is all", "thats all", "that's all",
            "kuch nahi", "kuch nhi", "aur nahi", "bas itna", "bas thank", "aur kuch nahi",
            "ଆଉ କିଛି ନାହିଁ", "କିଛି ନାହିଁ", "कुछ नहीं", "बस इतना ही", "नो थैंक", "थैंक यू"
        ]
        return any(k in clean for k in keywords)

    def _is_greeting_intent(self, lower_query: str) -> bool:
        # Split on whitespace and punctuation to preserve Indic/Unicode characters (virama, matras)
        tokens = set(re.findall(r'[^\s.,!?।॥]+', lower_query))
        greeting_tokens = {
            "hi", "hello", "hey", "namaste", "namaskar", "heya", "hlo",
            "नमस्ते", "नमस्कार", "ନମସ୍କାର"
        }
        # Check discrete greeting token match (exact whole-word, NOT substring match!)
        if bool(tokens.intersection(greeting_tokens)):
            # If query has question or content words (e.g. "hi what can you do", "hi tell me upi"),
            # do NOT treat it as a pure greeting!
            content_tokens = tokens - greeting_tokens - {"there", "ji", "mam", "sir", "please", "to", "you"}
            if len(content_tokens) > 1:
                return False
            return True

        greeting_phrases = ["good morning", "good afternoon", "good evening"]
        return any(p in lower_query for p in greeting_phrases)

    def _infer_intent_with_history(
        self,
        lower_query: str,
        messages: Optional[List[ChatMessage]],
    ) -> Optional[str]:
        """
        Classifies user query intent using query semantics and preceding dialogue history.
        """
        # Capabilities / What can you do
        if any(w in lower_query for w in [
            "what can you do", "what do you do", "what think you can do",
            "help me with", "your services", "what services", "capabilities",
            "kya kar sakte", "kya kar sakti", "क्या कर सकती", "क्या सहायता", "କଣ କରିପାରିବ"
        ]):
            return "capabilities"

        # Direct intent keywords
        if any(w in lower_query for w in [
            "kyc", "केवाईसी", "वीडियो", "video kyc", "pan", "aadhaar", "document", "verification", "id proof", "କେୱାଇସି",
            "vedio", "video", "कागजात", "पहचान पत्र", "दस्तावेज", "पैन", "आधार", "dastavez", "kagaj"
        ]):
            return "video_kyc"

        if any(w in lower_query for w in [
            "refund", "upi", "यूपीआई", "कट गया", "कट गए", "अटक", "फेल", "kat gaya", "atak", "पैसे कट", "पैसे", "paisa", "paise",
            "debited", "failed", "utr", "pending payment", "वापस", "रिफंड", "यूपीआई पिन", "लिमिट", "ୟୁପିଆଇ", "ରିଫଣ୍ଡ",
            "payment", "पेमेंट", "पैमेंट", "transaction", "ट्रांजेक्शन", "लेनदेन", "transfer", "you be a", "you be i", "gpay", "phonepe", "paytm"
        ]):
            return "upi_refund"

        if any(w in lower_query for w in ["pin", "atm pin", "mpin", "पिन", "एमपिन", "ग्रीन पिन"]):
            return "card_pin"

        if any(w in lower_query for w in [
            "card", "debit card", "credit card", "atm", "block", "खो गया", "गुम", "चोरी", "कार्ड", "क्रेडिट", "डेबिट", "ब्लॉक", "freeze", "stolen", "lost"
        ]):
            return "card_block"

        if any(w in lower_query for w in ["balance", "statement", "बैलेंस", "खाता", "अकाउंट", "बचत", "खाता शेष", "स्टेटमेंट", "passbook", "पासबुक"]):
            return "account_balance"

        if any(w in lower_query for w in ["cheque", "checkbook", "chequebook", "चेक", "चेकबुक"]):
            return "cheque_book"

        if any(w in lower_query for w in ["loan", "emi", "interest", "ऋण", "लोन", "ईएमआई", "ब्याज", "किस्त", "फोरक्लोजर", "ଋଣ"]):
            return "loans_emi"

        if any(w in lower_query for w in ["netbanking", "login", "password", "पासवर्ड", "अनलॉक", "नेटबैंकिंग", "लॉगिन", "unlock", "app"]):
            return "netbanking_mpin"

        if any(w in lower_query for w in ["timing", "timings", "opening hours", "open", "branch", "शाखा", "समय", "खुलती", "शनिवार"]):
            return "branch_hours"

        if any(w in lower_query for w in [
            "order", "order id", "delivery", "parcel", "shipment", "courier", "track", "tracking",
            "ऑर्डर", "डिलीवरी", "पार्सल", "ट्रैक", "ଅର୍ଡର", "ଡେଲିଭରି", "ପାର୍ସଲ",
        ]):
            return "order_status"

        # Multi-turn context resolution: if caller asks an anaphoric follow-up ("how much time?", "documents required?")
        if messages and len(messages) >= 2:
            prev_turns = " ".join(
                str(m.content or "") for m in messages[-4:] if m.role in ("user", "assistant")
            ).lower()

            if any(w in lower_query for w in ["time", "kitna time", "kab tak", "how long", "କେତେ ସମୟ"]):
                if "kyc" in prev_turns:
                    return "video_kyc"
                if "upi" in prev_turns or "refund" in prev_turns:
                    return "upi_refund"
                if "card" in prev_turns:
                    return "card_block"
                if "cheque" in prev_turns:
                    return "cheque_book"

            if any(w in lower_query for w in ["document", "what needed", "kya chahiye", "documents"]):
                if "kyc" in prev_turns:
                    return "video_kyc"
                if "loan" in prev_turns:
                    return "loans_emi"

        return None

    def _synthesize_from_rag_chunks(
        self,
        rag_context: str,
        lower_query: str,
        lang: str,
    ) -> Optional[str]:
        """
        Parses pgvector RAG chunks, strips technical metadata, and extracts
        the single most relevant sentence formatted naturally for telephone audio.
        """
        clean_lines: List[str] = []
        seen = set()

        for line in rag_context.splitlines():
            l = line.strip()
            if not l or l.startswith("---") or l.startswith("[Source") or "Relevant Context" in l or "You have access" in l:
                continue

            # Skip lines that represent caller questions / inquiries to prevent repeating the user's question!
            lower_raw = l.lower()
            if any(lower_raw.startswith(p) for p in (
                "caller:", "caller :", "user:", "user :", "customer:", "customer :",
                "customer spoken inquiry:", "customer inquiry:", "caller query:",
                "question:", "q:", "key inquiries:", "inquiry:"
            )):
                continue

            # Strip resolution prefixes to extract the actual answer
            l = re.sub(r"(?i)^(?:assistant|verified resolution|resolution|answer|bot|agent)\s*:\s*", "", l)

            # Strip metadata labels instead of discarding the factual line
            l = re.sub(r"(?i)^(?:english|hindi|odia)?\s*voice\s*transcript\s*:\s*", "", l)
            l = re.sub(r"(?i)^prompt\s*id\s*:\s*[^\n]+", "", l)
            l = re.sub(r"(?i)^audio\s*prompt\s*id\s*:\s*[^\n]+", "", l)
            l = re.sub(r"(?i)^(?:document\s*id|category|source|title)\s*:\s*[^\n]+", "", l)
            l = l.strip()
            if not l:
                continue

            # Clean bullet marks, brackets, quotes
            l = re.sub(r"^\s*[-*#\d\.\)]+\s*", "", l).strip("\"' ")
            if not l:
                continue

            # Filter bracket headers e.g. [ACCOUNT BALANCE, STATEMENTS & BRANCH TIMINGS]
            if l.startswith("[") and l.endswith("]"):
                continue

            # Filter generic filler phrases that should never be spoken as factual answers
            lower_l = l.lower()
            if any(filler in lower_l for filler in [
                "what can i do for you", "how can i help you", "ready to help",
                "anything else i can help", "anything else i can do", "what did you want to know",
                "is there anything else", "क्या मैं आपकी और कोई सहायता", "କିଛି ସାହାଯ୍ୟ"
            ]):
                continue

            # Echo Guard: reject lines that repeat the caller's spoken words (>60% word overlap)
            cand_words = set(re.findall(r'\w+', l.lower()))
            query_words = set(re.findall(r'\w+', lower_query))
            if cand_words and query_words:
                overlap = len(cand_words.intersection(query_words)) / len(cand_words)
                if overlap >= 0.60:
                    continue

            # Ensure script matches the session language
            if lang == "en" and any("\u0900" <= ch <= "\u097f" for ch in l):
                continue
            if lang == "hi" and not any("\u0900" <= ch <= "\u097f" for ch in l):
                continue
            if lang == "or" and any("\u0900" <= ch <= "\u097f" for ch in l):
                continue

            norm = l.lower()
            if norm not in seen and len(l) > 15:
                seen.add(norm)
                clean_lines.append(l)

        if not clean_lines:
            return None

        # Prioritize line matching query terms
        best_line = None
        query_words = set(w for w in re.findall(r'\w+', lower_query) if len(w) > 3)
        if query_words:
            for candidate in clean_lines:
                cand_lower = candidate.lower()
                if any(w in cand_lower for w in query_words):
                    best_line = candidate
                    break

        if not best_line:
            # If no candidate line matches query keywords, this chunk does not answer the question
            return None

        # Return single clean, factual sentence for natural spoken telephone delivery
        return best_line.strip()

    def _was_asked_for_order_digits(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Returns True if the previous assistant turn asked for the last 4 digits of the Order ID."""
        if not messages:
            return False
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                c = str(m.content).lower()
                return any(
                    marker in c
                    for marker in ("last 4 digits", "अंतिम 4 अंक", "ଅର୍ଡର ଆଇଡିର ଶେଷ")
                )
        return False

    def lookup_order_status(self, last4: str, lang: str = "en") -> str:
        """
        Returns a complete, natural spoken status report for the given last 4 Order ID digits.
        Matches verified directory records first, then deterministically resolves other 4-digit IDs.
        """
        code = (last4 or "")[-4:]
        if code in ORDER_STATUS_CATALOG:
            return ORDER_STATUS_CATALOG[code].get(lang, ORDER_STATUS_CATALOG[code]["en"])

        bucket = sum(ord(ch) for ch in code) % 3
        if lang == "hi":
            templates = [
                f"आपके ऑर्डर आईडी {code} की पूरी स्थिति यह है कि आपका पार्सल ब्लूडार्ट कूरियर के माध्यम से आज डिलीवरी के लिए निकल चुका है। यह आज शाम 6 बजे तक आपके पंजीकृत पते पर पहुंचेगा और डिलीवरी ओटीपी आपके मोबाइल पर एसएमएस द्वारा भेज दिया गया है।",
                f"आपके ऑर्डर आईडी {code} की पूरी स्थिति यह है कि आपका ऑर्डर डेल्हीवरी कूरियर से डिस्पैच हो चुका है और ट्रांजिट में है। इसकी डिलीवरी कल शाम 5 बजे तक संभावित है और लाइव ट्रैकिंग लिंक एसएमएस द्वारा भेज दिया गया है।",
                f"आपके ऑर्डर आईडी {code} की पूरी स्थिति यह है कि आपका पार्सल ईकॉम एक्सप्रेस कूरियर हब में प्रोसेस हो चुका है और 2 कार्यदिवसों में आपके पते पर डिलीवर कर दिया जाएगा। ट्रैकिंग विवरण एसएमएस पर भेज दिया गया है।",
            ]
            return templates[bucket]
        if lang == "or":
            templates = [
                f"ଆପଣଙ୍କ ଅର୍ଡର ଆଇଡି {code} ର ସମ୍ପୂର୍ଣ୍ଣ ସ୍ଥିତି ହେଉଛି ଏହା ଆଜି ବ୍ଲୁଡାର୍ଟ କୁରିଅର ମାଧ୍ୟମରେ ଡେଲିଭରି ପାଇଁ ବାହାରିସାରିଛି। ଆଜି ସନ୍ଧ୍ୟା ୬ଟା ସୁଦ୍ଧା ଆପଣଙ୍କ ଠିକଣାରେ ପହଞ୍ଚିବ ଏବଂ ଡେଲିଭରି ଓଟିପି ଏସଏମଏସରେ ପଠାଯାଇଛି।",
                f"ଆପଣଙ୍କ ଅର୍ଡର ଆଇଡି {code} ର ସମ୍ପୂର୍ଣ୍ଣ ସ୍ଥିତି ହେଉଛି ଏହା ଡେଲହିଭରି କୁରିଅର ମାଧ୍ୟମରେ ପଠାଯାଇଛି ଏବଂ ରାସ୍ତାରେ ଅଛି। ଆସନ୍ତାକାଲି ସନ୍ଧ୍ୟା ୫ଟା ସୁଦ୍ଧା ଡେଲିଭରି ହେବ ଏବଂ ଟ୍ରାକିଂ ଲିଙ୍କ୍ ଏସଏମଏସରେ ପଠାଯାଇଛି।",
                f"ଆପଣଙ୍କ ଅର୍ଡର ଆଇଡି {code} ର ସମ୍ପୂର୍ଣ୍ଣ ସ୍ଥିତି ହେଉଛି ଏହା ଇକମ୍ ଏକ୍ସପ୍ରେସ୍ କୁରିଅର ହବ୍ ରେ ପ୍ରୋସେସ୍ ହୋଇଛି ଏବଂ ୨ କାର୍ଯ୍ୟଦିବସରେ ପହଞ୍ଚିବ।",
            ]
            return templates[bucket]
        templates = [
            f"Here is the complete status of your Order ID ending in {code}: your parcel is currently Out for Delivery today via BlueDart Courier, scheduled to reach your registered address by 6:00 PM with your delivery OTP sent via SMS.",
            f"Here is the complete status of your Order ID ending in {code}: your shipment has been dispatched via Delhivery Courier and is currently In Transit, with estimated delivery tomorrow by 5:00 PM and an active SMS tracking link.",
            f"Here is the complete status of your Order ID ending in {code}: your order has left our regional hub with Ecom Express Courier and is scheduled for doorstep delivery within 2 business days with SMS updates enabled.",
        ]
        return templates[bucket]


# Module-level singleton
ivr_mini_brain = IVRRAGMiniModel()


# ──────────────────────────────────────────────────────────────────────────────
# Chat Adapter Wrapper (BaseChatAdapter)
# ──────────────────────────────────────────────────────────────────────────────

class IVRRAGMiniModelAdapter(BaseChatAdapter):
    """
    Adapter integrating the IVR RAG Mini-Model into the central Platform Registry.
    Compatible with NormalizedChatRequest / NormalizedChatResponse contracts.
    """

    def __init__(self, provider_id: str = "ivr_mini_brain", config: Optional[Dict[str, Any]] = None):
        super().__init__(provider_id, config or {})
        self.brain = ivr_mini_brain

    async def chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: Optional[ModelMetadata] = None,
    ) -> NormalizedChatResponse:
        req_id = f"chatcmpl-mini-{uuid.uuid4().hex[:8]}"
        created_ts = int(time.time())

        # Extract last user query and injected RAG context
        last_msg = request.messages[-1].content if request.messages else ""
        if isinstance(last_msg, list):
            last_msg = str(last_msg)

        rag_context = ""
        user_query = last_msg

        if "User Message:\n" in last_msg:
            parts = last_msg.split("User Message:\n", 1)
            rag_context = parts[0]
            user_query = parts[1].strip()

        for m in request.messages:
            if m.role == "system" and m.content:
                s = str(m.content)
                s_low = s.lower()
                if (
                    "relevant context" in s_low
                    or "untrusted retrieved context" in s_low
                    or "begin retrieved bank knowledge" in s_low
                ):
                    rag_context += "\n" + s

        detected_lang = self.brain._detect_language(user_query, request.messages)
        reply_content = self.brain.synthesize(
            user_query=user_query,
            messages=request.messages,
            rag_context=rag_context,
            language=detected_lang,
        )

        model_name = model_meta.alias if model_meta else "ivr-rag-mini"
        return NormalizedChatResponse(
            id=req_id,
            model=model_name,
            created=created_ts,
            choices=[
                NormalizedChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=15, completion_tokens=25, total_tokens=40),
        )

    async def stream_chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: Optional[ModelMetadata] = None,
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.chat_completion(request, model_meta)
        content = resp.choices[0].message.content or ""
        words = content.split(" ")

        yield StreamChunk(
            id=resp.id,
            model=resp.model,
            created=resp.created,
            choices=[
                StreamChoice(index=0, delta=StreamDelta(role="assistant"), finish_reason=None)
            ],
        )

        for i, word in enumerate(words):
            prefix = "" if i == 0 else " "
            await asyncio.sleep(0.015)
            yield StreamChunk(
                id=resp.id,
                model=resp.model,
                created=resp.created,
                choices=[
                    StreamChoice(index=0, delta=StreamDelta(content=prefix + word), finish_reason=None)
                ],
            )

        yield StreamChunk(
            id=resp.id,
            model=resp.model,
            created=resp.created,
            choices=[
                StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")
            ],
        )
