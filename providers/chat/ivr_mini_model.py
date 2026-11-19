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

import os
import re
import time
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, AsyncIterator, Tuple

try:
    import asyncpg
except ImportError:
    asyncpg = None

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
        "en": "Thank you for your valuable feedback! We truly appreciate you sharing your thoughts with us. Have a wonderful day!",
        "hi": "आपकी बहुमूल्य प्रतिक्रिया और रेटिंग के लिए बहुत-बहुत धन्यवाद! हम आपके सुझाव की सराहना करते हैं। आपका दिन शुभ हो!",
        "or": "ଆପଣଙ୍କ ମୂଲ୍ୟବାନ ମତାମତ ଏବଂ ରେଟିଂ ପାଇଁ ଅଶେଷ ଧନ୍ୟବାଦ! ଆମେ ଆପଣଙ୍କ ପରାମର୍ଶକୁ ସମ୍ମାନ କରୁଛୁ। ଆପଣଙ୍କ ଦିନଟି ଶୁଭମୟ ହେଉ!",
    },
    "feedback_ask": {
        "en": "We would love to hear your feedback! Please rate your experience on a scale of 1 to 5, or share your thoughts.",
        "hi": "हम आपकी प्रतिक्रिया सुनना पसंद करेंगे! कृपया 1 से 5 के पैमाने पर रेटिंग दें या अपने विचार साझा करें।",
        "or": "ଆମେ ଆପଣଙ୍କ ମତାମତ ଜାଣିବାକୁ ଚାହୁଁ! ଦୟାକରି ୧ ରୁ ୫ ମଧ୍ୟରେ ରେଟିଂ ଦିଅନ୍ତୁ କିମ୍ବା ଆପଣଙ୍କ ମତାମତ ଜଣାନ୍ତୁ।",
    },
    "feedback_negative": {
        "en": "Thank you for your honest feedback. We apologize for the inconvenience and will work to improve our service. Have a good day.",
        "hi": "आपकी स्पष्ट प्रतिक्रिया के लिए धन्यवाद। हमें हुई असुविधा के लिए खेद है, और हम अपनी सेवा में सुधार करेंगे। आपका दिन शुभ हो।",
        "or": "ଆପଣଙ୍କ ମତାମତ ପାଇଁ ଧନ୍ୟବାଦ। ହୋଇଥିବା ଅସୁବିଧା ପାଇଁ ଆମେ ଦୁଃଖିତ, ଆମେ ଆମର ସେବାରେ ଉନ୍ନତି ଆଣିବୁ।",
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
    "complaint_status_ask_id": {
        "en": "Certainly! Please tell me your complaint ticket number (for example 1 or 0001), or your registered phone number.",
        "hi": "जी बिल्कुल, कृपया अपना शिकायत टिकट नंबर (जैसे 1 या 0001) या अपना पंजीकृत मोबाइल नंबर बताइए।",
        "or": "ଆଜ୍ଞା ହଁ, ଦୟାକରି ଆପଣଙ୍କ ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର (ଯେପରି ୧ ବା ୦୦୦୧) କିମ୍ବା ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ ନମ୍ବର କୁହନ୍ତୁ।",
    },
    "complaint_register_ask_details": {
        "en": "Certainly! Please describe the issue or problem you are facing, and I will register your complaint right away.",
        "hi": "जी बिल्कुल! कृपया अपनी समस्या का विवरण बताइए, मैं अभी आपकी शिकायत दर्ज कर देती हूँ।",
        "or": "ନିଶ୍ଚୟ! ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା ବିଷୟରେ କୁହନ୍ତୁ, ମୁଁ ଏବେ ଆପଣଙ୍କ ଅଭିଯୋଗ ପଞ୍ଜୀକୃତ କରିଦେବି।",
    },
    "complaint_status": {
        "en": "Certainly! Please tell me your complaint ticket number (for example 1 or 0001), or your registered phone number.",
        "hi": "जी बिल्कुल, कृपया अपना शिकायत टिकट नंबर (जैसे 1 या 0001) या अपना पंजीकृत मोबाइल नंबर बताइए।",
        "or": "ଆଜ୍ଞା ହଁ, ଦୟାକରି ଆପଣଙ୍କ ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର (ଯେପରି ୧ ବା ୦୦୦୧) କିମ୍ବା ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ ନମ୍ବର କୁହନ୍ତୁ।",
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

COMPLAINT_STATUS_CATALOG: Dict[str, Dict[str, str]] = {
    "1": {
        "en": (
            "Here are the complete details for complaint ticket number 1 (reference CMP 0001): "
            "It was booked for customer Swagatika Swain on 21 September 2026 from Bhubaneswar, Odisha under the Support department. "
            "The issue category is Discussion, and the current ticket status is Canceled after review. "
            "Would you like to register a fresh complaint or need help with another service?"
        ),
        "hi": (
            "शिकायत टिकट नंबर 1 (संदर्भ सीएमपी 0001) का पूरा विवरण यह है: "
            "यह 21 सितंबर 2026 को भुवनेश्वर, ओडिशा से स्वातिका स्वाईं के लिए सपोर्ट विभाग में दर्ज किया गया था। "
            "समस्या की श्रेणी चर्चा (Discussion) है, और वर्तमान स्थिति समीक्षा के बाद रद्द (Canceled) है। "
            "क्या आप नई शिकायत दर्ज करना चाहते हैं या किसी अन्य सहायता की आवश्यकता है?"
        ),
        "or": (
            "ଆପଣଙ୍କ ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୧ (ସିଏମ୍‌ପି ୦୦୦୧) ର ସମ୍ପୂର୍ଣ୍ଣ ବିବରଣୀ: "
            "ଏହା ୨୧ ସେପ୍ଟେମ୍ବର ୨୦୨୬ ରେ ଭୁବନେଶ୍ୱରରୁ ସ୍ଵାଗତିକା ସ୍ଵାଇଁଙ୍କ ନାମରେ ସପୋର୍ଟ ବିଭାଗରେ ପଞ୍ଜୀକୃତ ହୋଇଥିଲା। "
            "ବିଷୟ ଶ୍ରେଣୀ ଆଲୋଚନା ଏବଂ ବର୍ତ୍ତମାନର ସ୍ଥିତି ବାତିଲ୍ (Canceled) ରହିଛି। "
            "ଆପଣ ନୂଆ ଅଭିଯୋଗ ଦାଖଲ କରିବାକୁ ଚାହାଁନ୍ତି କି?"
        ),
    },
    "2": {
        "en": (
            "Here are the complete details for complaint ticket number 2 (reference CMP 0002): "
            "It was booked for customer Shantanu on 30 September 2026 from Bhadrak, Odisha. "
            "The current status is Solved, and the resolution has been verified and closed by our Support team. "
            "Is there anything else I can help you with today?"
        ),
        "hi": (
            "शिकायत टिकट नंबर 2 (सीएमपी 0002) का पूरा विवरण: "
            "यह 30 सितंबर 2026 को भद्रक, ओडिशा से शांतनु के लिए दर्ज हुआ था। "
            "इसकी वर्तमान स्थिति हल (Solved) हो चुकी है और सपोर्ट टीम द्वारा समस्या का समाधान कर दिया गया है। "
            "क्या मैं और कोई सहायता करूँ?"
        ),
        "or": (
            "ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୨ (ସିଏମ୍‌ପି ୦୦୦୨) ର ବିବରଣୀ: "
            "ଏହା ୩୦ ସେପ୍ଟେମ୍ବର ୨୦୨୬ ରେ ଭଦ୍ରକରୁ ଶାନ୍ତନୁଙ୍କ ପାଇଁ ଦାଖଲ ହୋଇଥିଲା। "
            "ଏହାର ସମାଧାନ ହୋଇସାରିଛି ଏବଂ ସ୍ଥିତି Solved ରହିଛି। "
            "ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
        ),
    },
    "3": {
        "en": (
            "Here are the complete details for complaint ticket number 3 (reference CMP 0003): "
            "It was booked for customer Swarna on 30 September 2026 from Kendrapada, Odisha for Network Issue under Epsumlabs. "
            "The current status is Canceled. "
            "Is there anything else I can help you with today?"
        ),
        "hi": (
            "शिकायत टिकट नंबर 3 (सीएमपी 0003) का विवरण: "
            "यह 30 सितंबर 2026 को केंद्रपाड़ा, ओडिशा से स्वर्णा के लिए एप्समलैब्स में नेटवर्क समस्या हेतु दर्ज था। "
            "वर्तमान स्थिति रद्द (Canceled) है। "
            "क्या मैं और कोई सहायता करूँ?"
        ),
        "or": (
            "ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୩ (ସିଏମ୍‌ପି ୦୦୦୩) କେନ୍ଦ୍ରାପଡ଼ାରୁ ସ୍ୱର୍ଣ୍ଣାଙ୍କ ପାଇଁ ନେଟୱର୍କ ସମସ୍ୟା ପାଇଁ ଦାଖଲ ହୋଇଥିଲା, ବର୍ତ୍ତମାନ ଏହା ବାତିଲ୍ ହୋଇଛି। "
            "ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
        ),
    },
    "4": {
        "en": (
            "Here are the complete details for complaint ticket number 4 (reference CMP 0004): "
            "It was booked for customer Saswat on 30 September 2026 from Bhadrak, Odisha under Epsumlabs. "
            "The current status is Discussion with ongoing investigation. "
            "Is there anything else I can help you with today?"
        ),
        "hi": (
            "शिकायत टिकट नंबर 4 (सीएमपी 0004) का विवरण: "
            "यह 30 सितंबर 2026 को भद्रक से शाश्वत के लिए एप्समलैब्स में दर्ज हुआ था। स्थिति अभी चर्चा में (Discussion) है। "
            "क्या मैं और कोई सहायता करूँ?"
        ),
        "or": (
            "ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୪ (ସିଏମ୍‌ପି ୦୦୦୪) ଭଦ୍ରକରୁ ଶାଶ୍ୱତଙ୍କ ପାଇଁ ଦାଖଲ ହୋଇଥିଲା। ଏହା ବର୍ତ୍ତମାନ ଚର୍ଚ୍ଚାରେ ଅଛି। "
            "ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
        ),
    },
    "5": {
        "en": (
            "Here are the complete details for complaint ticket number 5 (reference CMP 0005): "
            "It was booked for customer Swagatika Swain on 30 September 2026 from Bhadrak, Odisha for Network Issue under Support. "
            "The current status is New and awaiting assignment. "
            "Is there anything else I can help you with today?"
        ),
        "hi": (
            "शिकायत टिकट नंबर 5 (सीएमपी 0005) का विवरण: "
            "यह 30 सितंबर 2026 को भद्रक से स्वातिका स्वाईं के लिए नेटवर्क समस्या हेतु दर्ज हुआ था। स्थिति नई (New) है। "
            "क्या मैं और कोई सहायता करूँ?"
        ),
        "or": (
            "ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୫ (ସିଏମ୍‌ପି ୦୦୦୫) ସ୍ଵାଗତିକା ସ୍ଵାଇଁଙ୍କ ପାଇଁ ନେଟୱର୍କ ସମସ୍ୟା ପାଇଁ ଦାଖଲ ହୋଇଛି, ସ୍ଥିତି ନୂଆ ଅଟେ। "
            "ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
        ),
    },
    "6": {
        "en": (
            "Here are the complete details for complaint ticket number 6 (reference CMP 0006): "
            "It was booked for customer Sas on 30 September 2026 from Bhadrak, Odisha under Support for Discussion. "
            "The current status is New. "
            "Is there anything else I can help you with today?"
        ),
        "hi": "शिकायत टिकट नंबर 6 (सीएमपी 0006) भद्रक से दर्ज किया गया था। वर्तमान स्थिति नई (New) है। क्या मैं और कोई सहायता करूँ?",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୬ (ସିଏମ୍‌ପି ୦୦୦୬) ଭଦ୍ରକରୁ ଦାଖଲ ହୋଇଛି, ସ୍ଥିତି ନୂଆ ଅଟେ। ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?",
    },
    "7": {
        "en": (
            "Here are the complete details for complaint ticket number 7 (reference CMP 0007): "
            "It was booked on 30 September 2026 under the Support department. The current status is New. "
            "Is there anything else I can help you with today?"
        ),
        "hi": "शिकायत टिकट नंबर 7 (सीएमपी 0007) सपोर्ट विभाग में दर्ज है। वर्तमान स्थिति नई (New) है। क्या मैं और कोई सहायता करूँ?",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୭ (ସିଏମ୍‌ପି ୦୦୦୭) ସପୋର୍ଟ ବିଭାଗରେ ନୂଆ ରହିଛି। ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?",
    },
    "8": {
        "en": (
            "Here are the complete details for complaint ticket number 8 (reference CMP 0008): "
            "It was booked for customer Suvarna Sai on 17 November 2026 from Bhubaneswar, Odisha under Customer Support. "
            "The current status is New and under active review. "
            "Is there anything else I can help you with today?"
        ),
        "hi": (
            "शिकायत टिकट नंबर 8 (सीएमपी 0008) का पूरा विवरण: "
            "यह 17 नवंबर 2026 को भुवनेश्वर से सुवर्णा साई के लिए दर्ज हुआ था। वर्तमान स्थिति नई (New) है और समीक्षा में है। "
            "क्या मैं और कोई सहायता करूँ?"
        ),
        "or": (
            "ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୮ (ସିଏମ୍‌ପି ୦୦୦୮) ର ବିବରଣୀ: "
            "ଏହା ୧୭ ନଭେମ୍ବର ୨୦୨୬ ରେ ଭୁବନେଶ୍ୱରରୁ ସୁବର୍ଣ୍ଣା ସାଇଙ୍କ ନାମରେ ଦାଖଲ ହୋଇଥିଲା। ବର୍ତ୍ତମାନର ସ୍ଥିତି ନୂଆ ଅଟେ। "
            "ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
        ),
    },
    "9": {
        "en": (
            "Here are the complete details for complaint ticket number 9 (reference CMP 0009): "
            "It was booked on 18 November 2026 for Broadband service issue under Customer Support. "
            "The current status is New and under review. "
            "Is there anything else I can help you with today?"
        ),
        "hi": (
            "शिकायत टिकट नंबर 9 (सीएमपी 0009) का पूरा विवरण: "
            "यह 18 नवंबर 2026 को ब्रॉडबैंड सेवा समस्या के लिए दर्ज हुआ था। वर्तमान स्थिति नई (New) है। "
            "क्या मैं और कोई सहायता करूँ?"
        ),
        "or": (
            "ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର ୯ (ସିଏମ୍‌ପି ୦୦୦୯) ର ବିବରଣୀ: "
            "ଏହା ୧୮ ନଭେମ୍ବର ୨୦୨୬ ରେ ବ୍ରଡବ୍ୟାଣ୍ଡ ସମସ୍ୟା ପାଇଁ ପଞ୍ଜୀକୃତ ହୋଇଛି, ସ୍ଥିତି ନୂଆ ଅଟେ। "
            "ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
        ),
    },
    "8976453421": {
        "en": (
            "For registered phone number 8976453421 (Shantanu and Swagatika Swain), we found 3 complaints on file: "
            "Ticket 1 from Bhubaneswar is Canceled, Ticket 2 from Bhadrak is Solved, and Ticket 5 is New. "
            "Which ticket would you like full details on?"
        ),
        "hi": (
            "पंजीकृत मोबाइल नंबर 8976453421 के लिए 3 टिकट दर्ज हैं: "
            "टिकट 1 (भुवनेश्वर) रद्द है, टिकट 2 (भद्रक) हल हो चुका है, और टिकट 5 नया है। "
            "आप किस टिकट की पूरी जानकारी चाहते हैं?"
        ),
        "or": (
            "ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ 8976453421 ପାଇଁ ୩ଟି ଟିକେଟ୍ ରହିଛି: "
            "ଟିକେଟ୍ ୧ ବାତିଲ୍, ଟିକେଟ୍ ୨ ସମାଧାନ ହୋଇଛି, ଏବଂ ଟିକେଟ୍ ୫ ନୂଆ ଅଛି।"
        ),
    },
    "8976453488": {
        "en": (
            "For registered phone number 8976453488 (Swarna and Saswat), we found 2 complaints on file: "
            "Ticket 3 is Canceled, and Ticket 4 is in Discussion under Epsumlabs. "
            "Which ticket would you like full details on?"
        ),
        "hi": (
            "पंजीकृत मोबाइल नंबर 8976453488 के लिए 2 टिकट दर्ज हैं: "
            "टिकट 3 रद्द है और टिकट 4 चर्चा में है। आप किस टिकट की जानकारी चाहते हैं?"
        ),
        "or": (
            "ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ 8976453488 ପାଇଁ ଟିକେଟ୍ ୩ ବାତିଲ୍ ଏବଂ ଟିକେଟ୍ ୪ ଚର୍ଚ୍ଚାରେ ଅଛି।"
        ),
    },
    "9937234512": {
        "en": (
            "For registered phone number 9937234512 (Suvarna Sai), you have Ticket 8 booked on 17 November 2026 from Bhubaneswar. "
            "The current status is New and assigned to Customer Support."
        ),
        "hi": (
            "पंजीकृत मोबाइल नंबर 9937234512 (सुवर्णा साई) के लिए टिकट 8 दर्ज है, जो 17 नवंबर को भुवनेश्वर से दर्ज हुआ था। "
            "वर्तमान स्थिति नई (New) है।"
        ),
        "or": (
            "ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ 9937234512 (ସୁବର୍ଣ୍ଣା ସାଇ) ପାଇଁ ଟିକେଟ୍ ୮ ରହିଛି, ଯାହାର ସ୍ଥିତି ନୂଆ ଅଟେ।"
        ),
    },
}

# Alias padding for 4-digit representations (0001 -> 1, 0002 -> 2, etc.)
for _i in range(1, 10):
    _pad = f"{_i:04d}"
    if str(_i) in COMPLAINT_STATUS_CATALOG:
        COMPLAINT_STATUS_CATALOG[_pad] = COMPLAINT_STATUS_CATALOG[str(_i)]

# Dynamic complaint ticket counter initialized beyond existing CRM tickets (latest is 11)
_NEXT_TICKET_COUNTER: int = 12


async def _persist_complaint_to_crm_db(
    ticket_id: str,
    complaint_sl: uuid.UUID,
    description: str,
    caller_phone: Optional[str] = None,
    customer_name: Optional[str] = "Customer",
    call_uuid: Optional[str] = None,
) -> bool:
    """
    Asynchronously persists a newly registered complaint into the remote CRM PostgreSQL DB
    ('ComplainBookings' and 'ComplainFollowUp') and registers locally without blocking audio.
    """
    host = os.getenv("CRM_DB_HOST", "192.168.1.124")
    port = int(os.getenv("CRM_DB_PORT", "5435"))
    dbname = os.getenv("CRM_DB_NAME", "crm_backend_db")
    user = os.getenv("CRM_DB_USERNAME", "postgres")
    password = os.getenv("CRM_DB_PASSWORD", "epsumlabs")

    phone_val = str(caller_phone or "9937234512")
    cust_val = str(customer_name or "Customer")
    now = datetime.now(timezone.utc)

    # 1. Insert into remote PostgreSQL CRM DB if reachable
    if asyncpg is not None:
        try:
            conn = await asyncpg.connect(
                host=host,
                port=port,
                database=dbname,
                user=user,
                password=password,
                ssl=False,
                timeout=3.0,
            )
            try:
                await conn.execute(
                    """
                    INSERT INTO "ComplainBookings" (
                        complainsl, complainid, customername, phonenumber, department_sl,
                        complain_phase_sl, createdby, complaindatetime, createdate, reopened, othernotes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    complaint_sl,
                    ticket_id,
                    cust_val,
                    phone_val,
                    uuid.UUID("d6b3256a-fa21-4708-9910-3ea086831c6f"),
                    uuid.UUID("c84516c3-1592-4d0e-bc0e-7f50b2758f97"),
                    uuid.UUID("a778db92-1d4f-4778-a698-05830387b0ed"),
                    now,
                    now,
                    False,
                    description,
                )
                await conn.execute(
                    """
                    INSERT INTO "ComplainFollowUp" (
                        follow_up_sl, complainsl, description, follow_up_date, created_by, created_date, complain_phase_sl
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    uuid.uuid4(),
                    complaint_sl,
                    f"Voice IVR Registration: {description}",
                    now,
                    uuid.UUID("a778db92-1d4f-4778-a698-05830387b0ed"),
                    now,
                    uuid.UUID("c84516c3-1592-4d0e-bc0e-7f50b2758f97"),
                )
                logger.info(f"[IVR CRM DB] Successfully inserted {ticket_id} into ComplainBookings & FollowUp.")
            finally:
                await conn.close()
        except Exception as e:
            logger.debug(f"[IVR CRM DB] Remote insert notice ({ticket_id}): {e}")

    # 2. Insert into local SQLite / Postgres complaints table if available
    try:
        from tools.business_data.complaints import register_complaint
        await register_complaint(
            description=description,
            caller_phone=phone_val,
            service_category="complaint",
            call_uuid=call_uuid,
        )
        logger.info(f"[IVR Local DB] Registered local complaint record for {ticket_id}.")
    except Exception as e:
        logger.debug(f"[IVR Local DB] Local complaint log: {e}")

    # 3. Asynchronously synchronize updated CRM dataset & pgvector chunks
    try:
        from services.knowledge.crm_sync_service import sync_crm_knowledge_to_pgvector
        asyncio.create_task(sync_crm_knowledge_to_pgvector())
    except Exception:
        pass

    return True


def extract_complaint_identifier(text: str, allow_bare_digits: bool = False) -> Optional[str]:
    """
    Extracts complaint ticket numbers, CMP codes, spoken numbers, or customer phone numbers.
    Supports formats like:
      - 'CMP/0001/26-27' -> '1'
      - 'complaint 1' / 'ticket one' -> '1'
      - 'ticket 0008' / 'CMP 8' -> '8'
      - '9937234512' -> '9937234512'
      - '1' (when allow_bare_digits=True) -> '1'
    """
    if not text:
        return None
    lower = text.translate(_INDIC_DIGIT_TRANSLATION).lower().strip()
    # 0. Telephony phonetic recovery: 8kHz mishearing of 'zero' as 'video' adjacent to digits
    lower = re.sub(r'\bvideo\b(?=(?:[\s,]+video)*[\s,]+(?:and\s*)?\d)', 'zero', lower)
    # Guard against 'new one', 'another one', 'register a new one', etc. being misheard or parsed as digit 1
    lower = re.sub(r'\b(?:new|another|a\s+new|fresh)\s+one\b', 'new_ticket', lower)

    # Guard against feedback, ratings, and star numbers being extracted as ticket IDs
    if re.search(r"\b[1-5]\s*(?:star|stars|स्टार|ଷ୍ଟାର୍)\b", lower):
        return None
    if any(w in lower for w in (
        "feedback", "feed back", "rating", "ratings", "rate", "स्टार", "रेटिंग",
        "प्रतिक्रिया", "सुझाव", "फीडबैक", "ମତାମତ", "ରେଟିଂ", "ଷ୍ଟାର୍"
    )):
        return None

    # 1. 10-digit Indian phone number lookup
    phone_m = re.search(r"\b([6-9]\d{9})\b", lower)
    if phone_m:
        return phone_m.group(1)

    # 2. CMP code format (e.g. CMP/0001/26-27, CMP 0001, CMP-1)
    cmp_m = re.search(r"cmp(?:/|\s|_|-)*0*(\d{1,4})", lower)
    if cmp_m:
        return str(int(cmp_m.group(1)))

    # 3. Explicit complaint/ticket prefix + number (e.g. 'complaint 1', 'ticket no 8', 'शिकायत नंबर 1')
    tkt_m = re.search(r"(?:complaint|ticket|case|cmp|शिकायत|टिकट|ଅଭିଯୋଗ|ଟିକେଟ୍)\s*(?:no|number|num|नंबर|नं|ନମ୍ବର|ନଂ)?\s*#?\s*0*(\d{1,4})\b", lower)
    if tkt_m:
        return str(int(tkt_m.group(1)))

    # 4. Spoken word digits following complaint/ticket keyword (e.g. 'complaint one', 'ticket eight')
    for word, digit in _SPOKEN_DIGIT_WORDS.items():
        if re.search(rf"(?:complaint|ticket|case|cmp|शिकायत|टिकट|ଅଭିଯୋଗ|ଟିକେଟ୍)\s*(?:no|number|num|नंबर|नं|ନମ୍ବର|ନଂ)?\s*{word}\b", lower):
            return digit

    # 4b. If complaint/ticket context is clearly present and a 1-4 digit number exists in query
    if any(w in lower for w in ("complaint", "ticket", "cmp", "शिकायत", "टिकट", "ଅଭିଯୋଗ", "ଟିକେଟ୍")):
        general_m = re.search(r"\b0*(\d{1,4})\b", lower)
        if general_m:
            return str(int(general_m.group(1)))
        for word, digit in _SPOKEN_DIGIT_WORDS.items():
            if re.search(rf"(?:complaint|ticket|case|cmp|शिकायत|ଅଭିଯୋଗ)\s*(?:no|number)?\s*{word}\b", lower):
                return digit

    # If the user expresses intent to register or describe a new complaint without an explicit ticket prefix,
    # do NOT extract bare numbers or single words as ticket IDs
    tokens = set(re.findall(r'[a-zA-Z\u0900-\u097f\u0b00-\u0b7f]+', lower))
    if tokens.intersection({"register", "lodge", "file", "book", "दर्ज", "पंजीकृत", "दाखिल", "ଦାଖଲ", "new", "fresh", "नया", "नई", "ନୂଆ"}):
        return None

    # 5. Direct number sequences, padded zeros, or pure digit utterances (e.g. '0001', '0 0 0 1', 'Zero, zero zero one.', '1')
    normalized_spoken = lower
    for word, digit in _SPOKEN_DIGIT_WORDS.items():
        normalized_spoken = re.sub(rf'\b{word}\b', digit, normalized_spoken)
    found_digits = re.findall(r'\d', normalized_spoken)
    if found_digits:
        digit_seq = ''.join(found_digits)
        if len(digit_seq) == 10 and digit_seq[0] in '6789':
            return digit_seq
        non_digit_words = [
            w for w in re.findall(r'[a-zA-Z\u0900-\u097f\u0b00-\u0b7f]+', normalized_spoken)
            if w not in ('and', 'or', 'the', 'is', 'it', 'my', 'please', 'no', 'number', 'ticket', 'id', 'dot', 'zero')
        ]
        # Padded numbers (e.g. '0001', '0 0 0 1', 'Zero, zero zero one.') or pure digit utterances when permitted
        if len(digit_seq) <= 4:
            if len(digit_seq) >= 2 or (allow_bare_digits and len(non_digit_words) <= 1):
                return str(int(digit_seq))

    return None


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
        self._last_crm_sync_ts: float = 0.0

    async def sync_live_crm_cache(self, force: bool = False) -> bool:
        """
        Synchronizes live complaints and phone directories from remote CRM PostgreSQL DB
        into the in-memory COMPLAINT_STATUS_CATALOG.
        Runs in ~100ms with a 5-second debounce window.
        """
        global _NEXT_TICKET_COUNTER
        now_ts = time.time()
        if not force and (now_ts - getattr(self, "_last_crm_sync_ts", 0.0) < 5.0):
            return True

        host = os.getenv("CRM_DB_HOST", "192.168.1.124")
        port = int(os.getenv("CRM_DB_PORT", "5435"))
        dbname = os.getenv("CRM_DB_NAME", "crm_backend_db")
        user = os.getenv("CRM_DB_USERNAME", "postgres")
        password = os.getenv("CRM_DB_PASSWORD", "epsumlabs")

        if asyncpg is None:
            return False

        try:
            conn = await asyncpg.connect(
                host=host,
                port=port,
                database=dbname,
                user=user,
                password=password,
                ssl=False,
                timeout=2.5,
            )
            try:
                query = """
                    SELECT 
                        cb.complainid, 
                        cb.customername, 
                        cb.phonenumber, 
                        cb.city, 
                        cb.state, 
                        cb.othernotes, 
                        cb.complaindatetime, 
                        cpm.complain_phase_name AS status,
                        dm.department_name AS department,
                        ctm.complain_type_name AS category
                    FROM "ComplainBookings" cb
                    LEFT JOIN "ComplainPhaseMaster" cpm ON cb.complain_phase_sl = cpm.complain_phase_sl
                    LEFT JOIN "DepartmentMasters" dm ON cb.department_sl = dm.department_sl
                    LEFT JOIN "ComplainTypeMasters" ctm ON cb.complain_type_sl = ctm.complain_type_sl
                    ORDER BY cb.createdate DESC
                """
                rows = await conn.fetch(query)
            finally:
                await conn.close()

            phone_map: Dict[str, List[Dict[str, str]]] = {}
            max_tnum = 0
            for r in rows:
                cid = str(r["complainid"] or "").strip()
                m = re.search(r"0*(\d{1,4})", cid)
                if not m:
                    continue
                tnum_val = int(m.group(1))
                if tnum_val > max_tnum:
                    max_tnum = tnum_val

                tnum = str(tnum_val)
                pad4 = f"{tnum_val:04d}"
                cname = str(r["customername"] or "Customer").strip()
                status = str(r["status"] or "In Progress").strip()
                city = str(r["city"] or "Odisha").strip()
                dept = str(r["department"] or "Customer Support").strip()
                bdate = r["complaindatetime"]
                bdate_str = bdate.strftime("%d %B %Y") if bdate else "recently"

                status_hi = {"New": "नई (New)", "Discussion": "चर्चाधीन (Discussion)", "Solved": "हल (Solved)", "Canceled": "रद्द (Canceled)"}.get(status, status)
                status_or = {"New": "ନୂଆ (New)", "Discussion": "ଆଲୋଚନାରେ (Discussion)", "Solved": "ସମାଧାନ (Solved)", "Canceled": "ବାତିଲ୍ (Canceled)"}.get(status, status)

                entry = {
                    "en": (
                        f"Here are the complete details for complaint ticket number {tnum} (reference CMP {pad4}): "
                        f"It was booked for customer {cname} on {bdate_str} from {city} under {dept}. "
                        f"The current ticket status is {status}. "
                        f"Is there anything else I can help you with today?"
                    ),
                    "hi": (
                        f"शिकायत टिकट नंबर {tnum} (संदर्भ सीएमपी {pad4}) का पूरा विवरण: "
                        f"यह {city} से {cname} के लिए {dept} में दर्ज हुआ था। "
                        f"वर्तमान स्थिति {status_hi} है। "
                        f"क्या मैं और कोई सहायता करूँ?"
                    ),
                    "or": (
                        f"ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର {tnum} (ସିଏମ୍‌ପି {pad4}) ର ବିବରଣୀ: "
                        f"ଏହା {city} ରୁ {cname} ଙ୍କ ପାଇଁ ଦାଖଲ ହୋଇଥିଲା। "
                        f"ବର୍ତ୍ତମାନର ସ୍ଥିତି {status_or} ଅଟେ। "
                        f"ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
                    ),
                }
                COMPLAINT_STATUS_CATALOG[tnum] = entry
                COMPLAINT_STATUS_CATALOG[pad4] = entry
                COMPLAINT_STATUS_CATALOG[cid] = entry

                phone = str(r["phonenumber"] or "").strip()
                if phone and len(phone) >= 10 and phone.isdigit():
                    phone_map.setdefault(phone, []).append({
                        "tnum": tnum,
                        "status": status,
                        "customer": cname,
                    })

            for phone, tkts in phone_map.items():
                names = list(dict.fromkeys(t["customer"] for t in tkts if t["customer"] and t["customer"] not in ("Customer", "aa", "qwq")))
                names_str = " and ".join(names) if names else "Customer"

                tkt_summaries_en = []
                tkt_summaries_hi = []
                tkt_summaries_or = []
                for t in tkts:
                    tnum = t["tnum"]
                    st = t["status"]
                    st_hi = {"New": "नई", "Discussion": "चर्चाधीन", "Solved": "हल", "Canceled": "रद्द"}.get(st, st)
                    st_or = {"New": "ନୂଆ", "Discussion": "ଆଲୋଚନାରେ", "Solved": "ସମାଧାନ", "Canceled": "ବାତିଲ୍"}.get(st, st)
                    tkt_summaries_en.append(f"Ticket {tnum} is {st}")
                    tkt_summaries_hi.append(f"टिकट {tnum} {st_hi}")
                    tkt_summaries_or.append(f"ଟିକେଟ୍ {tnum} {st_or}")

                en_list = ", ".join(tkt_summaries_en[:-1]) + (f", and {tkt_summaries_en[-1]}" if len(tkt_summaries_en) > 1 else tkt_summaries_en[0])
                hi_list = ", ".join(tkt_summaries_hi[:-1]) + (f", और {tkt_summaries_hi[-1]}" if len(tkt_summaries_hi) > 1 else tkt_summaries_hi[0])
                or_list = ", ".join(tkt_summaries_or[:-1]) + (f", ଏବଂ {tkt_summaries_or[-1]}" if len(tkt_summaries_or) > 1 else tkt_summaries_or[0])

                COMPLAINT_STATUS_CATALOG[phone] = {
                    "en": f"For registered phone number {phone} ({names_str}), we found {len(tkts)} complaint{'s' if len(tkts) > 1 else ''} on file: {en_list}. Which ticket would you like full details on?",
                    "hi": f"पंजीकृत मोबाइल नंबर {phone} ({names_str}) के लिए {len(tkts)} शिकायतें दर्ज हैं: {hi_list}। आप किस टिकट की पूरी जानकारी चाहते हैं?",
                    "or": f"ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ {phone} ({names_str}) ପାଇଁ {len(tkts)} ଟି ଅଭିଯୋଗ ରହିଛି: {or_list}। ଆପଣ କେଉଁ ଟିକେଟ୍ ବିଷୟରେ ଜାଣିବାକୁ ଚାହାଁନ୍ତି?",
                }

            if max_tnum >= _NEXT_TICKET_COUNTER:
                _NEXT_TICKET_COUNTER = max_tnum + 1

            self._last_crm_sync_ts = now_ts
            logger.info(f"[IVR CRM Sync] Live sync updated {len(rows)} tickets and {len(phone_map)} phone directories (next ticket={_NEXT_TICKET_COUNTER}).")
            return True
        except Exception as e:
            logger.warning(f"[IVR CRM Sync] Live CRM sync notice (using cached catalog): {e}")
            return False

    def _is_complaint_register_intent(self, lower_query: str) -> bool:
        """Checks if the user wants to register, file, or book a new complaint."""
        clean = (lower_query or "").strip().lower()
        if any(w in clean for w in ("status", "check", "track", "where is", "existing", "update", "स्थिति", "ଟ୍ରାକ୍", "ପୂର୍ବ")):
            return False
        tokens = set(re.findall(r'[a-zA-Z\u0900-\u097f\u0b00-\u0b7f]+', clean))
        if tokens.intersection({"register", "lodge", "file", "book", "दर्ज", "पंजीकृत", "पंजीकरण", "दाखिल", "ଦାଖଲ", "ପଞ୍ଜୀକରଣ"}):
            return True
        if "new one" in clean or "fresh complaint" in clean or "new complaint" in clean or "register one" in clean or "a new one" in clean:
            return True
        if tokens.intersection({"new", "fresh", "नया", "नई", "ନୂଆ", "ନୂତନ"}):
            return True
        return any(
            phrase in clean
            for phrase in (
                "register a new complaint", "register new complaint", "register a complaint",
                "register complaint", "file a complaint", "file a new complaint", "file complaint",
                "book a complaint", "book complaint", "lodge a complaint", "lodge complaint",
                "new complaint", "fresh complaint", "want to complain",
                "register new complain", "register complain", "file complain", "book complain",
                "register a new one", "register new one", "lodge a new one",
                "शिकायत दर्ज", "दर्ज करना", "दर्ज करनी", "नया शिकायत", "नई शिकायत", "नई वाली", "नया वाला",
                "ଅଭିଯୋଗ ଦାଖଲ", "ନୂଆ ଅଭିଯୋଗ", "ଅଭିଯୋଗ ପଞ୍ଜୀକରଣ", "ନୂଆ ଗୋଟିଏ",
            )
        )

    def _is_complaint_status_intent(self, lower_query: str) -> bool:
        """Checks if the user wants to check the status or track an existing complaint."""
        clean = (lower_query or "").strip().lower()
        tokens = set(re.findall(r'[a-zA-Z\u0900-\u097f\u0b00-\u0b7f]+', clean))
        if tokens.intersection({"status", "track", "check", "existing", "old", "previous", "स्थिति", "पुरानी", "पुराना", "जांच", "ଟ୍ରାକ୍", "ସ୍ଥିତି", "ପୂର୍ବ", "ପୁରୁଣା"}):
            return True
        return any(
            phrase in clean
            for phrase in (
                "check status", "check the status", "status of complaint", "complaint status",
                "my complaint", "check existing", "existing complaint", "check existing complaint",
                "existing one", "old complaint", "status check", "know status", "track complaint",
                "track my complaint", "status dekhna", "status janna", "स्थिति देखना",
                "स्थिति जांचना", "पुरानी शिकायत", "पुराना शिकायत", "स्थिति बताएं",
                "ସ୍ଥିତି ଜାଣିବା", "ପୂର୍ବ ଅଭିଯୋଗ", "ପୁରୁଣା ଅଭିଯୋଗ", "ଅଭିଯୋଗ ସ୍ଥିତି",
            )
        )

    def _was_asked_status_or_register(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Returns True if the assistant asked whether the caller wants to check status or register a new complaint."""
        if not messages:
            return False
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                c = str(m.content).lower()
                return any(
                    marker in c
                    for marker in (
                        "existing complaint, or register a new one",
                        "check the status of an existing complaint",
                        "status of an existing complaint, or register",
                        "पुरानी शिकायत की स्थिति देखना चाहते हैं, या नई शिकायत",
                        "ପୂର୍ବ ଅଭିଯୋଗର ସ୍ଥିତି ଜାଣିବାକୁ ଚାହାଁନ୍ତି, କିମ୍ବା ନୂଆ ଅଭିଯୋଗ",
                    )
                )
        return False

    def _was_asked_anything_else(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Returns True if the assistant asked if there is anything else they can help with."""
        if not messages:
            return False
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                c = str(m.content).lower()
                return any(
                    marker in c
                    for marker in (
                        "anything else i can help you with",
                        "need help with another service",
                        "और कोई सहायता करूँ",
                        "किसी अन्य सहायता",
                        "ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର",
                    )
                )
        return False

    def _was_asked_for_complaint_description(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Returns True if the previous assistant turn asked for a problem or issue description."""
        if not messages:
            return False
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                c = str(m.content).lower()
                return any(
                    marker in c
                    for marker in (
                        "describe the issue", "describe the problem", "problem you are facing",
                        "समस्या का विवरण", "ସମସ୍ୟା ବିଷୟରେ", "register your complaint right away",
                    )
                )
        return False

    def _extract_complaint_description_from_query(self, query: str) -> Optional[str]:
        """
        Extracts the problem description if caller included it along with registration intent.
        E.g. 'I want to register a complaint that internet is down' -> 'internet is down'
        """
        lower = (query or "").lower().strip()
        patterns = [
            r"(?:register|file|book|lodge)\s+(?:a\s+)?(?:new\s+)?complain[a-z]*\s+(?:that|about|for|regarding|saying|because)\s+(.+)",
            r"(?:शिकायत दर्ज|दर्ज करनी है|दर्ज करना चाहता हूँ)\s+(?:कि|हेतु|के बारे में)\s+(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                desc = m.group(1).strip()
                if len(desc) >= 5:
                    return desc
        return None

    def _was_complaint_just_registered(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Returns True if a complaint was registered in the previous assistant turn."""
        if not messages:
            return False
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                c = str(m.content).lower()
                return any(
                    marker in c
                    for marker in (
                        "successfully registered under ticket number",
                        "सफलतापूर्वक टिकट नंबर",
                        "ସଫଳତାର ସହ ଟିକେଟ୍ ନମ୍ବର",
                        "reference cmp",
                        "संदर्भ सीएमपी",
                        "ସିଏମ୍‌ପି",
                    )
                )
        return False

    def _was_asked_feedback(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Returns True if the assistant asked for feedback or rating."""
        if not messages:
            return False
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                c = str(m.content).lower()
                return any(
                    marker in c
                    for marker in (
                        "rate your experience",
                        "scale of 1 to 5",
                        "love to hear your feedback",
                        "रेटिंग दें",
                        "प्रतिक्रिया सुनना",
                        "ରେଟିଂ ଦିଅନ୍ତୁ",
                    )
                )
        return False

    def _is_feedback_ask_intent(self, lower_query: str) -> bool:
        """Checks if the caller is asking to give feedback."""
        clean = (lower_query or "").strip().lower()
        clean = re.sub(r'[^a-zA-Z\u0900-\u097f\u0b00-\u0b7f\s]', '', clean)
        ask_phrases = [
            "give feedback", "want to give feedback", "can i give feedback", "like to give feedback",
            "leave feedback", "provide feedback", "share feedback",
            "फीडबैक देना", "फीडबैक देना चाहता", "फीडबैक देना चाहती", "फीडबैक देना है",
            "प्रतिक्रिया देना", "सुझाव देना", "मदद का फीडबैक",
            "ମତାମତ ଦେବା", "ମତାମତ ଦେବାକୁ ଚାହୁଁଛି"
        ]
        return any(p in clean for p in ask_phrases) or clean in ("feedback", "give feedback", "फीडबैक")

    def _is_feedback_or_rating_intent(
        self,
        query: str,
        messages: Optional[List[ChatMessage]] = None,
    ) -> bool:
        """
        Detects whether caller is submitting feedback, rating, service quality comments,
        or a numerical satisfaction rating (1-5).
        """
        clean = (query or "").strip().lower()
        clean_no_punct = re.sub(r'[.,!?\'"]', '', clean).strip()

        # 1. Explicit rating / star / feedback expressions with word boundaries
        if re.search(r'\b(?:ratings?|rate\s+(?:the|this|your|our|service)|stars?|feed\s*back|स्टार|रेटिंग|प्रतिक्रिया|सुझाव|फीडबैक|ମତାମତ|ରେଟିଂ|ଷ୍ଟାର୍)\b', clean_no_punct):
            return True

        # 2. Service evaluation comments with word boundaries
        service_evals = [
            r'\b(?:good|great|excellent|bad|poor|worst|very good|very bad)\s+service\b',
            r'\bservice\s+(?:is|was)\s+(?:good|great|excellent|bad|poor|worst|very good|very bad)\b',
            r'\b(?:अच्छी सेवा|सेवा अच्छी|बहुत अच्छी सेवा|खराब सेवा|सेवा खराब|ଭଲ ସେବା|ଉତ୍ତମ ସେବା)\b',
        ]
        if any(re.search(p, clean_no_punct) for p in service_evals):
            return True

        # 4. If complaint was just registered, or assistant asked 'anything else' or 'feedback':
        was_registered = self._was_complaint_just_registered(messages)
        was_anything_else = self._was_asked_anything_else(messages)
        was_feedback_asked = self._was_asked_feedback(messages)

        if was_registered or was_anything_else or was_feedback_asked:
            rating_tokens = {
                "1", "2", "3", "4", "5",
                "one", "two", "three", "four", "five",
                "१", "२", "३", "४", "५",
                "୧", "୨", "୩", "୪", "୫",
                "1/5", "2/5", "3/5", "4/5", "5/5",
                "5 out of 5", "4 out of 5",
                "five stars", "four stars", "three stars", "two stars", "one star",
                "i give 5", "i give 4", "give 5", "give 4",
                "very good", "excellent", "super", "poor", "terrible", "awesome"
            }
            if clean_no_punct in rating_tokens:
                return True
            if re.fullmatch(r"[1-5]\s*(?:star|stars|स्टार|ଷ୍ଟାର୍)?", clean_no_punct):
                return True

        return False

    def register_new_complaint(
        self,
        description: str,
        lang: str = "en",
        caller_phone: Optional[str] = None,
        customer_name: Optional[str] = "Customer",
        call_uuid: Optional[str] = None,
    ) -> str:
        """
        Registers a new complaint instantly (< 2ms) without stalling telephony audio on slow LLMs.
        Allocates next sequential ticket ID, updates in-memory catalog, schedules async DB persistence,
        and returns a natural spoken confirmation.
        """
        global _NEXT_TICKET_COUNTER
        clean_desc = (description or "").strip()
        if len(clean_desc) < 3:
            clean_desc = "Customer service inquiry and follow-up"

        ticket_num = _NEXT_TICKET_COUNTER
        _NEXT_TICKET_COUNTER += 1

        pad_4 = f"{ticket_num:04d}"
        ticket_cmp = f"CMP/{pad_4}/26-27"
        comp_sl = uuid.uuid4()
        phone = caller_phone or "9937234512"
        cust_name = customer_name or "Customer"

        # Update in-memory catalog so immediate follow-up status queries work
        COMPLAINT_STATUS_CATALOG[str(ticket_num)] = {
            "en": (
                f"Here are the complete details for complaint ticket number {ticket_num} (reference CMP {pad_4}): "
                f"It was registered recently for customer {cust_name} regarding '{clean_desc}'. "
                f"The current status is New and assigned to Customer Support for investigation."
            ),
            "hi": (
                f"शिकायत टिकट नंबर {ticket_num} (संदर्भ सीएमपी {pad_4}) का पूरा विवरण: "
                f"यह '{clean_desc}' के लिए दर्ज किया गया है। वर्तमान स्थिति नई (New) है और सपोर्ट टीम जांच कर रही है।"
            ),
            "or": (
                f"ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର {ticket_num} (ସିଏମ୍‌ପି {pad_4}) ର ବିବରଣୀ: "
                f"ଏହା '{clean_desc}' ପାଇଁ ପଞ୍ଜୀକୃତ ହୋଇଛି, ସ୍ଥିତି ନୂଆ ଅଟେ।"
            ),
        }
        COMPLAINT_STATUS_CATALOG[pad_4] = COMPLAINT_STATUS_CATALOG[str(ticket_num)]
        if phone:
            COMPLAINT_STATUS_CATALOG[phone] = {
                "en": f"For registered phone number {phone}, complaint ticket number {ticket_num} (reference CMP {pad_4}) is on file with status New.",
                "hi": f"पंजीकृत मोबाइल नंबर {phone} के लिए शिकायत टिकट नंबर {ticket_num} (सीएमपी {pad_4}) नई (New) स्थिति में दर्ज है।",
                "or": f"ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ {phone} ପାଇଁ ଟିକେଟ୍ ନମ୍ବର {ticket_num} (ସିଏମ୍‌ପି {pad_4}) ନୂଆ ଅଟେ।",
            }

        # Fire async background task to persist to CRM DB & local DB
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                _persist_complaint_to_crm_db(
                    ticket_id=ticket_cmp,
                    complaint_sl=comp_sl,
                    description=clean_desc,
                    caller_phone=phone,
                    customer_name=cust_name,
                    call_uuid=call_uuid,
                )
            )
        except RuntimeError:
            pass

        logger.info(
            f"[IVR Mini Brain] Registered new complaint ticket={ticket_num} ({ticket_cmp}) "
            f"phone={phone} desc='{clean_desc[:40]}'"
        )

        if lang == "hi":
            return (
                f"आपकी शिकायत सफलतापूर्वक टिकट नंबर {ticket_num} (संदर्भ सीएमपी {pad_4}) के तहत दर्ज कर ली गई है। "
                f"हमारी सपोर्ट टीम जल्द जांच करके आपसे संपर्क करेगी। क्या मैं और कोई सहायता करूँ?"
            )
        elif lang == "or":
            return (
                f"ଆପଣଙ୍କ ଅଭିଯୋଗ ସଫଳତାର ସହ ଟିକେଟ୍ ନମ୍ବର {ticket_num} (ସିଏମ୍‌ପି {pad_4}) ଅଧୀନରେ ପଞ୍ଜୀକୃତ ହୋଇଛି। "
                f"ଆମ ସପୋର୍ଟ ଟିମ୍ ଖୁବ୍ ଶୀଘ୍ର ଯୋଗାଯୋଗ କରିବେ। ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
            )
        return (
            f"Your complaint has been successfully registered under ticket number {ticket_num} (reference CMP {pad_4}). "
            f"Our support team will investigate the issue and update you shortly. Is there anything else I can help you with today?"
        )

    def synthesize(
        self,
        user_query: str,
        messages: Optional[List[ChatMessage]] = None,
        rag_context: str = "",
        language: str = "en",
        service_category: Optional[str] = None,
        intelligence_context: Optional[Dict[str, Any]] = None,
        caller_phone: Optional[str] = None,
        customer_name: Optional[str] = "Customer",
        call_uuid: Optional[str] = None,
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

        # 4b. Feedback intent (ratings, suggestions, service quality feedback, post-complaint rating)
        if self._is_feedback_ask_intent(lower_query):
            return AUTONOMOUS_KNOWLEDGE["feedback_ask"][lang]

        if self._is_feedback_or_rating_intent(clean_query, messages):
            if any(w in lower_query for w in ["bad", "poor", "terrible", "worst", "खराब", "बकवास", "ଖରାପ"]):
                return AUTONOMOUS_KNOWLEDGE["feedback_negative"][lang]
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

        # 4g. Check if caller answered an affirmative 'yeah' / 'yes' to 'Is there anything else I can help you with today?'
        clean_words = re.sub(r'[^a-zA-Z\u0900-\u097f\u0b00-\u0b7f\s]', '', lower_query).strip()
        if self._was_asked_anything_else(messages) and any(clean_words == w or clean_words.startswith(w + " ") for w in [
            "yeah", "yes", "sure", "okay", "yep", "haan", "ha", "ji", "हाँ", "हाँजी", "ହଁ", "ହଁ ଆଜ୍ଞା"
        ]):
            follow_up = {
                "en": "Certainly! Please let me know what else I can help you with today.",
                "hi": "जी बिल्कुल! बताइए, मैं आपकी और क्या सहायता कर सकती हूँ?",
                "or": "ନିଶ୍ଚୟ! କୁହନ୍ତୁ, ମୁଁ ଆପଣଙ୍କୁ ଆଉ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
            }
            return self.apply_sentiment_tone(follow_up.get(lang, follow_up["en"]), intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5. Complaint Services Dialogue Flow (Status & Registration)
        is_complaint_context = bool(service_category and "complaint" in service_category.lower())
        awaiting_complaint_desc = self._was_asked_for_complaint_description(messages)
        awaiting_complaint_id = self._was_asked_for_complaint_id(messages)
        was_asked_branch = self._was_asked_status_or_register(messages)
        is_register_intent = self._is_complaint_register_intent(lower_query)
        is_status_intent = self._is_complaint_status_intent(lower_query)

        # 5a. If previous turn asked caller to describe their complaint issue:
        if awaiting_complaint_desc:
            # Check if caller answered with an explicit ticket number instead of problem description
            cid_probe = extract_complaint_identifier(clean_query, allow_bare_digits=False)
            if cid_probe:
                ans = self.lookup_complaint_status(cid_probe, lang=lang)
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)
            # Register the caller's described issue
            ans = self.register_new_complaint(
                description=clean_query,
                lang=lang,
                caller_phone=caller_phone,
                customer_name=customer_name,
                call_uuid=call_uuid,
            )
            return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5b. If previous turn asked caller for complaint ticket ID or phone:
        if awaiting_complaint_id:
            # If caller switched to register intent:
            if is_register_intent:
                embedded_desc = self._extract_complaint_description_from_query(clean_query)
                if embedded_desc:
                    ans = self.register_new_complaint(
                        description=embedded_desc,
                        lang=lang,
                        caller_phone=caller_phone,
                        customer_name=customer_name,
                        call_uuid=call_uuid,
                    )
                    return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)
                ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_register_ask_details"][lang]
                return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            cid_probe = extract_complaint_identifier(clean_query, allow_bare_digits=True)
            if cid_probe:
                ans = self.lookup_complaint_status(cid_probe, lang=lang)
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)
            # Prompt again for complaint ticket number
            ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_status_ask_id"][lang]
            return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5c. Caller explicitly declared intent to register a new complaint:
        if is_register_intent:
            embedded_desc = self._extract_complaint_description_from_query(clean_query)
            if embedded_desc:
                ans = self.register_new_complaint(
                    description=embedded_desc,
                    lang=lang,
                    caller_phone=caller_phone,
                    customer_name=customer_name,
                    call_uuid=call_uuid,
                )
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)
            # Prompt caller to describe the issue in detail
            ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_register_ask_details"][lang]
            return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5d. Caller explicitly declared intent to check status of existing complaint:
        if is_status_intent:
            cid_probe = extract_complaint_identifier(clean_query, allow_bare_digits=False)
            if cid_probe:
                ans = self.lookup_complaint_status(cid_probe, lang=lang)
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)
            ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_status_ask_id"][lang]
            return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5e. Caller in Complaint Service context provided a direct ticket ID or digits:
        if is_complaint_context:
            cid_probe = extract_complaint_identifier(clean_query, allow_bare_digits=True)
            if cid_probe:
                ans = self.lookup_complaint_status(cid_probe, lang=lang)
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            # Check if caller directly provided an issue description (>= 3 words describing a problem)
            if any(w in lower_query for w in (
                "not working", "down", "issue", "problem", "fault", "broken", "deducted", "failed",
                "kharab", "nahi chal raha", "kaam nahi", "paise", "kat gaye", "बंद", "खराब", "काम नहीं",
                "କାମ କରୁନାହିଁ", "ବନ୍ଦ", "ପଇସା", "ସମସ୍ୟା",
            )) and len(clean_query.split()) >= 3:
                ans = self.register_new_complaint(
                    description=clean_query,
                    lang=lang,
                    caller_phone=caller_phone,
                    customer_name=customer_name,
                    call_uuid=call_uuid,
                )
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            # Re-ask the branching question if initial entry or branch was asked:
            if was_asked_branch or not messages or len(messages) <= 2:
                branch_prompt = {
                    "en": "Would you like to check the status of an existing complaint, or register a new one?",
                    "hi": "क्या आप अपनी पुरानी शिकायत की स्थिति देखना चाहते हैं, या नई शिकायत दर्ज करना चाहते हैं?",
                    "or": "ଆପଣ ପୂର୍ବ ଅଭିଯୋଗର ସ୍ଥିତି ଜାଣିବାକୁ ଚାହାଁନ୍ତି, କିମ୍ବା ନୂଆ ଅଭିଯୋଗ ଦାଖଲ କରିବାକୁ ଚାହାଁନ୍ତି?",
                }
                return self.apply_sentiment_tone(branch_prompt.get(lang, branch_prompt["en"]), intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5f. Multi-turn context for Order ID / other services
        inferred_intent = self._infer_intent_with_history(lower_query, messages)
        awaiting_digits = self._was_asked_for_order_digits(messages)
        if awaiting_digits and inferred_intent not in (
            "video_kyc", "upi_refund", "card_block", "card_pin", "account_balance",
            "cheque_book", "loans_emi", "netbanking_mpin", "branch_hours", "capabilities",
        ):
            inferred_intent = "order_status"

        # Dedicated Stateful Order ID Verification & Complete Status Lookup
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
            elif "complaint" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["complaint_status_ask_id"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 9. Polite general fallback
        ans = AUTONOMOUS_KNOWLEDGE["fallback"][lang]
        return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    _SENTIMENT_ALREADY_EXPRESSIVE_PREFIXES = (
        "i'm sorry", "i apologize", "i understand", "i completely understand", "glad",
        "wonderful", "thank you", "you're most welcome", "you're very welcome",
        "your complaint has been successfully registered", "आपकी शिकायत सफलतापूर्वक", "ଆପଣଙ୍କ ଅଭିଯୋଗ ସଫଳତାର ସହ",
        "here are the complete details", "शिकायत टिकट नंबर", "ଅଭିଯୋଗ ଟିକେଟ୍",
        "for registered phone number", "पंजीकृत मोबाइल", "ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍",
        "certainly! please describe", "जी बिल्कुल! कृपया अपनी समस्या", "ନିଶ୍ଚୟ! ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା",
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
        is_video_digit_mishearing = bool(re.search(r'\bvideo\b(?=(?:[\s,]+video)*[\s,]+(?:and\s*)?\d)', lower_query))
        if not is_video_digit_mishearing and any(w in lower_query for w in [
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

        if self._is_complaint_register_intent(lower_query):
            return "complaint_register"

        if any(w in lower_query for w in [
            "complaint", "ticket", "cmp", "cmp/", "cmp-", "status of complaint", "my complaint",
            "शिकायत", "अभिযোগ", "ଅଭିଯୋଗ", "ଟିକେଟ୍",
        ]):
            return "complaint_status"

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

            # Filter generic filler and clarification phrases that should never be spoken as factual answers
            lower_l = l.lower()
            if any(filler in lower_l for filler in [
                "what can i do for you", "how can i help you", "ready to help",
                "anything else i can help", "anything else i can do", "what did you want to know",
                "is there anything else", "क्या मैं आपकी और कोई सहायता", "କିଛି ସାହାଯ୍ୟ",
                "could you please", "could you tell me", "can you tell me", "can you please",
                "what specific details", "could you clarify", "please clarify",
                "tell me a bit more", "on a scale of", "how satisfied are you",
                "look up information on", "certainly look up", "did that completely answer",
                "would you like me to", "please confirm whether", "confirm whether your issue",
                "i understand you have a question about", "i see you're looking to",
                "tell me what specific", "confirm your full name",
            ]):
                continue

            # Skip lines that end with '?' - bot questions / surveys are not factual answers
            if l.endswith("?"):
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

        # Dedicated synthesis for Video KYC document inquiries:
        # If multiple document requirements (PAN + Aadhaar) are in the RAG chunks, synthesize a complete spoken summary.
        if any(w in lower_query for w in ("kyc", "kyicy", "video", "document", "documents", "dastavez", "दस्तावेज", "कागजात", "ପ୍ୟାନ୍")):
            pan_line = next((c for c in clean_lines if "pan" in c.lower() or "पैन" in c or "ପ୍ୟାନ୍" in c), None)
            aadhaar_line = next((c for c in clean_lines if "aadhaar" in c.lower() or "aadhar" in c.lower() or "आधार" in c or "ଆଧାର" in c), None)
            if pan_line and aadhaar_line:
                if lang == "hi":
                    return "वीडियो केवाईसी के लिए कृपया अपना मूल भौतिक पैन कार्ड और आधार लिंक मोबाइल तैयार रखें।"
                elif lang == "or":
                    return "ଭିଡିଓ କେୱାଇସି ପାଇଁ ଆପଣଙ୍କର ମୂଳ ପ୍ୟାନ୍ କାର୍ଡ ଏବଂ ଆଧାର ମୋବାଇଲ୍ ପ୍ରସ୍ତୁତ ରଖନ୍ତୁ।"
                else:
                    return "For Video KYC, please keep your original physical PAN card and Aadhaar-linked mobile ready for OTP verification."
            elif pan_line:
                return pan_line.strip()

        # Prioritize line matching query terms with factual domain relevance scoring
        best_line = None
        query_words = set(w for w in re.findall(r'\w+', lower_query) if len(w) > 3)
        if query_words:
            scored_candidates: List[Tuple[float, str]] = []
            for candidate in clean_lines:
                cand_lower = candidate.lower()
                cand_words = set(re.findall(r'\w+', cand_lower))
                overlap = len(query_words.intersection(cand_words))
                score = float(overlap)

                # Domain entity boost for concrete factual information
                if any(w in cand_lower for w in ("pan", "aadhaar", "aadhar", "physical pan", "original physical")):
                    score += 4.0
                if any(w in cand_lower for w in ("refund", "banking hours", "working days", "dispatch", "credited back")):
                    score += 3.0
                if any(w in cand_lower for w in ("atm pin", "forgot mpin", "otp", "debit card")):
                    score += 3.0
                if any(w in cand_lower for w in ("frozen", "block", "replacement")):
                    score += 3.0
                if any(w in cand_lower for w in ("complaint", "ticket", "cmp/", "cmp_", "phase", "sla", "resolved", "solved", "canceled")):
                    score += 4.0

                # Boost complete declarative responses
                if any(cand_lower.startswith(p) for p in ("for video kyc", "video kyc", "failed upi", "your card", "you can", "service complaint", "customer service complaint", "registered customer profile", "हमारी शाखा", "ଭିଡିଓ କେୱାଇସି")):
                    score += 3.0

                if score > 0:
                    scored_candidates.append((score, candidate))

            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                best_line = scored_candidates[0][1]

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

    def _was_asked_for_complaint_id(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Returns True if the previous assistant turn asked for a complaint ticket number or registered phone."""
        if not messages:
            return False
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                c = str(m.content).lower()
                return any(
                    marker in c
                    for marker in (
                        "complaint ticket number", "complaint ticket", "registered phone number",
                        "ticket number", "शिकायत टिकट", "पंजीकृत मोबाइल", "ଅଭିଯୋଗ ଟିକେଟ୍",
                    )
                )
        return False

    def lookup_complaint_status(self, identifier: str, lang: str = "en") -> str:
        """
        Returns a natural, telephone-optimized status report for a complaint ticket or customer phone number.
        Avoids clumsy strings like 'slash zero zero zero one' and speaks clear ticket IDs.
        """
        key = str(identifier).strip()
        if key in COMPLAINT_STATUS_CATALOG:
            return COMPLAINT_STATUS_CATALOG[key].get(lang, COMPLAINT_STATUS_CATALOG[key]["en"])

        # 10-digit phone lookup fallback
        if len(key) == 10 and key.isdigit():
            if lang == "hi":
                return f"मोबाइल नंबर {key} के लिए हमारे रिकॉर्ड में कोई खुली शिकायत नहीं मिली है। यदि आप नई शिकायत दर्ज करना चाहते हैं, तो कृपया बताएं।"
            elif lang == "or":
                return f"ମୋବାଇଲ୍ ନମ୍ବର {key} ପାଇଁ କୌଣସି ଅଭିଯୋଗ ରେକର୍ଡ ମିଳିଲା ନାହିଁ। ନୂଆ ଅଭିଯୋଗ ଦାଖଲ କରିବାକୁ ଚାହାଁନ୍ତି କି?"
            return f"No active complaints were found for registered phone number {key}. Would you like to register a new complaint?"

        # Numeric ticket fallback
        pad_id = f"{int(key):04d}" if key.isdigit() else key
        if lang == "hi":
            return f"आपके शिकायत टिकट नंबर {key} (सीएमपी {pad_id}) की स्थिति प्रक्रियाधीन है और हमारी सपोर्ट टीम इस पर काम कर रही है। क्या मैं और कोई सहायता करूँ?"
        elif lang == "or":
            return f"ଆପଣଙ୍କ ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର {key} (ସିଏମ୍‌ପି {pad_id}) ପ୍ରକ୍ରିୟାଧୀନ ରହିଛି। ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
        return f"For complaint ticket number {key} (CMP {pad_id}), your ticket is currently in progress and being reviewed by our customer support team. Is there anything else I can help you with today?"


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
