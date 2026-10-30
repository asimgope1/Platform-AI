"""
Natural, Conversational Voice Scripts & Prompts for Keypad IVR.
Polite, human-like telephone phrasing in English, Hindi (Devanagari), and Odia (ଓଡ଼ିଆ).
"""
from typing import Dict

# ── 1. Initial Welcome & Language Selection ──────────────────────────────────
# Measured pacing with authentic Indian phonetics & native Odia VITS
WELCOME_AND_LANGUAGE = (
    "Namaste! Thank you for calling customer support. "
    "Please choose your preferred language to continue. ... "
    "Press one for English. ... "
    "हिंदी के लिए दो दबाएं। ... "
    "ଓଡ଼ିଆ ପାଇଁ ତିନି ଦବାନ୍ତୁ। ... "
    "To repeat this menu, press zero."
)

# ── 2. Service Selection Menus ───────────────────────────────────────────────
SERVICE_MENUS: Dict[str, str] = {
    "en": (
        "How can I help you today? "
        "Press 1 for Customer Support, "
        "2 to register a complaint, "
        "3 for UPI and payment issues, "
        "4 for account and banking services, "
        "5 for card-related issues, "
        "6 for loans and EMI queries, "
        "7 for other enquiries, "
        "or press 9 to speak with a representative."
    ),
    "hi": (
        "नमस्ते! हम आपकी क्या मदद कर सकते हैं? ... "
        "कस्टमर सपोर्ट के लिए 1 दबाएं, ... "
        "शिकायत दर्ज करने के लिए 2, ... "
        "यूपीआई और पेमेंट समस्या के लिए 3, ... "
        "खाता और बैंकिंग सेवा के लिए 4, ... "
        "कार्ड संबंधित समस्या के लिए 5, ... "
        "लोन और ईएमआई के लिए 6, ... "
        "अन्य जानकारी के लिए 7, ... "
        "या प्रतिनिधि से बात करने के लिए 9 दबाएं।"
    ),
    "or": (
        "ନମସ୍କାର! ଆମେ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରି ପାରିବୁ? ... "
        "ଗ୍ରାହକ ସେବା ପାଇଁ ଏକ ଦବାନ୍ତୁ, ... "
        "ଅଭିଯୋଗ ପାଇଁ ଦୁଇ, ... "
        "ୟୁପିଆଇ ଏବଂ ପେମେଣ୍ଟ ପାଇଁ ତିନି, ... "
        "ଖାତା ଏବଂ ବ୍ୟାଙ୍କିଙ୍ଗ ପାଇଁ ଚାରି, ... "
        "କାର୍ଡ ପାଇଁ ପାଞ୍ଚ, ... "
        "ଋଣ ପାଇଁ ଛଅ, ... "
        "ଅନ୍ୟାନ୍ୟ ସୂଚନା ପାଇଁ ସାତ, ... "
        "ବା ପ୍ରତିନିଧିଙ୍କ ସହିତ କଥା ହେବା ପାଇଁ ନଅ ଦବାନ୍ତୁ।"
    ),
}

# ── 3. Service Names & Friendly Acknowledgements ─────────────────────────────
SERVICE_NAMES: Dict[str, Dict[str, str]] = {
    "customer_support": {
        "en": "Customer Support",
        "hi": "ग्राहक सेवा",
        "or": "ଗ୍ରାହକ ସେବା",
    },
    "complaint": {
        "en": "Complaint Registration",
        "hi": "शिकायत पंजीकरण",
        "or": "ଅଭିଯୋଗ ପଞ୍ଜୀକରଣ",
    },
    "upi_payment": {
        "en": "UPI and Payment Support",
        "hi": "यूपीआई और भुगतान सहायता",
        "or": "ୟୁପିଆଇ ଏବଂ ପେମେଣ୍ଟ ସହାୟତା",
    },
    "account_services": {
        "en": "Account and Banking Services",
        "hi": "खाता और बैंकिंग सेवाएं",
        "or": "ଖାତା ଏବଂ ବ୍ୟାଙ୍କିଙ୍ଗ ସେବା",
    },
    "card_services": {
        "en": "Card Services",
        "hi": "कार्ड सेवाएं",
        "or": "କାର୍ଡ ସେବା",
    },
    "loans_emi": {
        "en": "Loans and EMI Support",
        "hi": "ऋण और ईएमआई सहायता",
        "or": "ଋଣ ଏବଂ ଇଏମଆଇ ସହାୟତା",
    },
    "other_enquiries": {
        "en": "General Enquiries",
        "hi": "सामान्य पूछताछ",
        "or": "ସାଧାରଣ ପଚରାଉଚରା",
    },
}

SERVICE_ACKS: Dict[str, str] = {
    "en": "Sure, I can help you with {service_name}. Please explain your issue, I'm listening.",
    "hi": "धन्यवाद। मैं आपकी {service_name} में मदद कर सकती हूँ। कृपया अपनी समस्या बताएं, मैं सुन रही हूँ।",
    "or": "ଧନ୍ୟବାଦ। ମୁଁ ଆପଣଙ୍କ {service_name} ରେ ସାହାଯ୍ୟ କରି ପାରିବି। ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା କୁହନ୍ତୁ, ମୁଁ ଶୁଣୁଛି।",
}

# ── 4. Natural Error, Timeout & Transfer Prompts ─────────────────────────────
INVALID_INPUT: Dict[str, str] = {
    "en": "Sorry, that was an invalid choice. Please choose an option from the menu.",
    "hi": "क्षमा करें, यह गलत विकल्प है। कृपया मेन्यू से सही विकल्प चुनें।",
    "or": "କ୍ଷମା କରିବେ, ଏହା ଭୁଲ ବିକଳ୍ପ ଅଟେ। ଦୟାକରି ମେନୁରୁ ସଠିକ ବିକଳ୍ପ ବାଛନ୍ତୁ।",
}

TIMEOUT_PROMPTS: Dict[str, str] = {
    "en": "I didn't hear your response. Let me repeat the options for you.",
    "hi": "हमें आपका कोई इनपुट प्राप्त नहीं हुआ। मैं विकल्प फिर से दोहरा रही हूँ।",
    "or": "ଆମକୁ କୌଣସି ଇନପୁଟ ମିଳିଲା ନାହିଁ। ମୁଁ ବିକଳ୍ପ ଦୋହରାଉଛି।",
}

MAX_RETRIES_EXCEEDED: Dict[str, str] = {
    "en": "Connecting you directly to our support assistant. Please stay on the line.",
    "hi": "मैं आपको सीधे हमारे सहायक से जोड़ रही हूँ। कृपया लाइन पर बने रहें।",
    "or": "ମୁଁ ଆପଣଙ୍କୁ ସିଧା ସହାୟକ ଟିମ୍ ସହିତ ଯୋଡୁଛି। ଦୟାକରି ଲାଇନରେ ରୁହନ୍ତୁ।",
}

HUMAN_TRANSFER: Dict[str, str] = {
    "en": "Connecting your call to a customer care representative right away. Please stay on the line.",
    "hi": "जी बिल्कुल, आपकी कॉल तुरंत हमारे ग्राहक सेवा प्रतिनिधि को ट्रांसफर की जा रही है। कृपया लाइन पर बने रहें।",
    "or": "ଆଜ୍ଞା, ଆପଣଙ୍କ କଲ୍ ପ୍ରତିନିଧିଙ୍କୁ ଟ୍ରାନ୍ସଫର କରାଯାଉଛି। ଦୟାକରି ଲାଇନରେ ରୁହନ୍ତୁ।",
}


def get_service_ack_prompt(language: str, service_key: str) -> str:
    """Returns a natural, warm service acknowledgement prompt in the selected language."""
    lang = language if language in SERVICE_ACKS else "en"
    names = SERVICE_NAMES.get(service_key, {})
    service_name = names.get(lang, names.get("en", "Customer Support"))
    template = SERVICE_ACKS.get(lang, SERVICE_ACKS["en"])
    return template.format(service_name=service_name)
