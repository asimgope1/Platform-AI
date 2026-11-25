"""
Natural, Conversational Voice Scripts & Prompts for Keypad IVR.
Polite, human-like telephone phrasing in English, Hindi (Devanagari), and Odia (ଓଡ଼ିଆ).
"""
from typing import Dict

# ── 1. Initial Welcome & Language Selection ──────────────────────────────────
# Measured pacing with authentic Indian phonetics & native Odia VITS
WELCOME_AND_LANGUAGE = (
    "Namaste! Thank you for calling Jana Seva Citizen Helpline. "
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
        "Press 1 for Citizen Services and Schemes, "
        "2 for Complaint Status and Registration, "
        "3 for Subhadra Scheme, "
        "4 for Certificates like Caste, Income, or Residence, "
        "5 for Aadhaar, PAN card, or Voter ID, "
        "6 for Legal Documents and Affidavits, "
        "7 for Jana Seva Kendra helpdesk, "
        "or press 9 to speak with a representative."
    ),
    "hi": (
        "नमस्ते! जन सेवा नागरिक सहायता में आपका स्वागत है। ... "
        "नागरिक सेवाओं और सरकारी योजनाओं के लिए 1 दबाएं, ... "
        "शिकायत स्थिति और पंजीकरण के लिए 2, ... "
        "सुभद्रा योजना के लिए 3, ... "
        "जाति, आय या निवास प्रमाण पत्र के लिए 4, ... "
        "आधार, पैन कार्ड या वोटर आईडी के लिए 5, ... "
        "कानूनी दस्तावेज और शपथ पत्र के लिए 6, ... "
        "जन सेवा केंद्र सहायता के लिए 7, ... "
        "या प्रतिनिधि से बात करने के लिए 9 दबाएं।"
    ),
    "or": (
        "ନମସ୍କାର! ଜନ ସେବା ଓଡ଼ିଶା ସହାୟତା କେନ୍ଦ୍ରକୁ ସ୍ୱାଗତ। ... "
        "ନାଗରିକ ସେବା ଏବଂ ସରକାରୀ ଯୋଜନା ପାଇଁ ଏକ ଦବାନ୍ତୁ, ... "
        "ଅଭିଯୋଗ ସ୍ଥିତି ଏବଂ ପଞ୍ଜୀକରଣ ପାଇଁ ଦୁଇ, ... "
        "ସୁଭଦ୍ରା ଯୋଜନା ପାଇଁ ତିନି, ... "
        "ଜାତି, ଆୟ ବା ବାସସ୍ଥାନ ପ୍ରମାଣପତ୍ର ପାଇଁ ଚାରି, ... "
        "ଆଧାର, ପ୍ୟାନ୍ ବା ଭୋଟର ଆଇଡି ପାଇଁ ପାଞ୍ଚ, ... "
        "ଆଇନଗତ ଦଲିଲ ଏବଂ ସତ୍ୟପାଠ ପାଇଁ ଛଅ, ... "
        "ଜନ ସେବା କେନ୍ଦ୍ର ସୂଚନା ପାଇଁ ସାତ, ... "
        "ବା ପ୍ରତିନିଧିଙ୍କ ସହିତ କଥା ହେବା ପାଇଁ ନଅ ଦବାନ୍ତୁ।"
    ),
}

# ── 3. Service Names & Friendly Acknowledgements ─────────────────────────────
SERVICE_NAMES: Dict[str, Dict[str, str]] = {
    "citizen_services": {
        "en": "Citizen Services & Welfare Schemes",
        "hi": "नागरिक सेवाएं और योजनाएं",
        "or": "ନାଗରିକ ସେବା ଓ ଯୋଜନା",
    },
    "complaint": {
        "en": "Complaint Status and Registration",
        "hi": "शिकायत स्थिति और पंजीकरण",
        "or": "ଅଭିଯୋଗ ଏବଂ ସ୍ଥିତି",
    },
    "subhadra_scheme": {
        "en": "Subhadra Scheme Assistance",
        "hi": "सुभद्रा योजना सहायता",
        "or": "ସୁଭଦ୍ରା ଯୋଜନା ସହାୟତା",
    },
    "certificates": {
        "en": "Certificate Services",
        "hi": "प्रमाण पत्र सेवाएं",
        "or": "ପ୍ରମାଣପତ୍ର ସେବା",
    },
    "identity_cards": {
        "en": "Identity Cards and Documents",
        "hi": "पहचान पत्र सेवाएं",
        "or": "ପରିଚୟ ପତ୍ର ସେବା",
    },
    "legal_documents": {
        "en": "Legal Documents and Affidavits",
        "hi": "कानूनी दस्तावेज और शपथ पत्र",
        "or": "ଆଇନଗତ ଦଲିଲ ଏବଂ ସତ୍ୟପାଠ",
    },
    "kendra_helpdesk": {
        "en": "Jana Seva Kendra Enquiries",
        "hi": "जन सेवा केंद्र पूछताछ",
        "or": "ଜନ ସେବା କେନ୍ଦ୍ର ସୂଚନା",
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
    lang = language if language in ("en", "hi", "or") else "en"
    
    if service_key == "complaint":
        complaint_acks = {
            "en": "Welcome to Complaint Services. Would you like to check the status of an existing complaint, or register a new one?",
            "hi": "शिकायत सेवा में आपका स्वागत है। क्या आप अपनी पुरानी शिकायत की स्थिति देखना चाहते हैं, या नई शिकायत दर्ज करना चाहते हैं?",
            "or": "ଅଭିଯୋଗ ସେବାକୁ ସ୍ୱାଗତ। ଆପଣ ପୂର୍ବ ଅଭିଯୋଗର ସ୍ଥିତି ଜାଣିବାକୁ ଚାହାଁନ୍ତି, କିମ୍ବା ନୂଆ ଅଭିଯୋଗ ଦାଖଲ କରିବାକୁ ଚାହାଁନ୍ତି?",
        }
        return complaint_acks.get(lang, complaint_acks["en"])

    names = SERVICE_NAMES.get(service_key, {})
    service_name = names.get(lang, names.get("en", "Customer Support"))
    template = SERVICE_ACKS.get(lang, SERVICE_ACKS["en"])
    return template.format(service_name=service_name)
