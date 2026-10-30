import time
import uuid
import json
import asyncio
from typing import AsyncIterator, Dict, Any, List, Optional
from platform_core.contracts.capabilities import ModelMetadata
from platform_core.contracts.requests import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChoice,
    ChatMessage,
    UsageInfo,
    ToolCall,
    FunctionCall,
    StreamChunk,
    StreamChoice,
    StreamDelta,
)
from platform_core.contracts.providers import BaseChatAdapter


class MockChatAdapter(BaseChatAdapter):
    """Deterministic, high-performance mock chat adapter for tests, CI, and local verification."""

    async def chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> NormalizedChatResponse:
        req_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
        created_ts = int(time.time())

        # Check if the user's last message is triggering a tool call
        last_msg = request.messages[-1].content if request.messages else ""
        if isinstance(last_msg, list):
            last_msg = str(last_msg)

        # If tools are provided and user asks for purchase requests / report, simulate tool call
        if request.tools and ("purchase request" in last_msg.lower() or "report" in last_msg.lower() or "customer" in last_msg.lower()):
            tool_name = "count_open_purchase_requests"
            if "customer" in last_msg.lower():
                tool_name = "get_customer_info"
            elif "report" in last_msg.lower():
                tool_name = "generate_summary_report"

            tool_call = ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=tool_name,
                    arguments=json.dumps({"filter_status": "open" if "open" in last_msg.lower() else "all"}),
                ),
            )
            return NormalizedChatResponse(
                id=req_id,
                model=model_meta.alias,
                created=created_ts,
                choices=[
                    NormalizedChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[tool_call],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=UsageInfo(prompt_tokens=15, completion_tokens=20, total_tokens=35),
            )

        # Context-aware intelligent mock responses
        reply_content = self._generate_mock_reply(request, model_meta)

        return NormalizedChatResponse(
            id=req_id,
            model=model_meta.alias,
            created=created_ts,
            choices=[
                NormalizedChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=12, completion_tokens=18, total_tokens=30),
        )

    def _generate_mock_reply(self, request: NormalizedChatRequest, model_meta: Optional[ModelMetadata] = None) -> str:
        import re
        if model_meta and model_meta.alias == "dynamic-chat":
            return f"Mock response for [{model_meta.alias}]: Processed request successfully."
        last_raw = request.messages[-1].content if request.messages else ""
        if isinstance(last_raw, list):
            last_raw = str(last_raw)

        # 1. Separate injected RAG context from the actual user query
        context_corpus = ""
        user_query = last_raw

        # ContextBuilder injects: "Relevant Context (if applicable to query):\n{rag_context}\n\nUser Message:\n{user_msg.content}"
        if "User Message:\n" in last_raw:
            parts = last_raw.split("User Message:\n", 1)
            context_corpus = parts[0]
            user_query = parts[1].strip()

        # Also check system messages for RAG context
        for m in request.messages:
            if m.role == "system" and m.content:
                sys_content = str(m.content)
                if "relevant context" in sys_content.lower():
                    context_corpus += "\n" + sys_content

        clean_query = user_query.strip()
        lower_query = clean_query.lower()
        lower_context = context_corpus.lower()

        # Detect if caller/user asks in Hindi or Hinglish or if system requested Hindi
        has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in clean_query)
        has_hinglish = any(re.search(rf"\b{w}\b", lower_query) for w in ["jubiai", "istiti", "sthiti", "paisa", "paise", "kat", "gaya", "karo", "batao", "batae", "kya", "kaise", "kab", "kripya", "namaste", "madad", "chahiye", "hai", "hu", "kijiye"])
        wants_hindi = (
            has_devanagari
            or has_hinglish
            or "reply in natural conversational hindi" in lower_context
            or any(m.role == "system" and "hindi" in str(m.content).lower() for m in request.messages)
        )

        # 2. Check for pure greetings using strict word boundaries on isolated user query
        is_greeting = bool(re.search(r"\b(hi|hello|hey|namaste|नमस्ते|hii|heya|hlo|good morning|good afternoon|good evening)\b", lower_query))
        if is_greeting and len(clean_query.split()) <= 4:
            if wants_hindi:
                return "नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ? आप यूपीआई रिफंड, वीडियो केवाईसी, या कस्टमर केयर से बात करने के लिए कह सकते हैं।"
            return "Hello! How can I help you today? You can ask me about your UPI refund status, Video KYC documents, or speak with an agent."

        # 3. Domain & Knowledge Base matching:
        # IVR Keypad / Live Support
        if any(w in lower_query for w in ["ivr", "keypad", "key", "press", "representative", "agent", "live support", "executive", "human", "person", "operator", "talk to", "speak to", "दबाएं", "एजेंट", "प्रतिनिधि"]):
            if wants_hindi:
                return "कस्टमर केयर प्रतिनिधि से सीधे बात करने के लिए अपने डायल पैड पर '9' दबाएं या 'एजेंट' बोलें। औसत प्रतीक्षा समय 2 मिनट से कम है।"
            return "To connect directly with a live customer care representative, press 9 on your dial pad or say agent. The wait time is under 2 minutes."

        # Video KYC / Documents Required
        if any(w in lower_query for w in ["kyc", "केवाईसी", "video kyc", "documents", "pan", "aadhaar", "document", "verification", "id proof"]):
            if wants_hindi:
                return "डिजिटल वीडियो केवाईसी के लिए आपको मूल पैन कार्ड और आधार से लिंक सक्रिय मोबाइल नंबर चाहिए। यह सत्यापन सिर्फ 2 मिनट में पूरा हो जाता है।"
            return "For Video KYC, you will need your original physical PAN card and an active Aadhaar-linked mobile number for OTP. The verification takes just 2 minutes."

        # UPI Refund / Payment failure
        if any(w in lower_query for w in [
            "refund", "upi", "jubiai", "रिफंड", "cut", "kat gaya", "paisa", "paise", "debited",
            "failed", "विफलता", "istiti", "sthiti", "you pay", "hundred acre", "you pay a hundred acre",
            "u p", "up", "money deducted", "amount deducted", "deducted", "transaction failed",
            "payment failed", "utr", "pending payment"
        ]):
            if wants_hindi:
                return "यदि असफल यूपीआई ट्रांजेक्शन में आपके पैसे कट गए हैं, तो 24 से 48 बैंकिंग घंटों में सीधे आपके बैंक खाते में रिफंड आ जाता है। आप अपने 12 अंकों के यूटीआर नंबर से भी स्टेटस ट्रैक कर सकते हैं।"
            return "If your money was debited for a failed UPI transaction, it will be automatically refunded to your bank account within 24 to 48 banking hours. You can also track the status using your 12-digit UPI UTR reference number."

        # Aurora Sensor / Quantum specs
        if "aurora" in lower_query and any(w in lower_query for w in ["quantum", "temperature", "kelvin", "sensor"]):
            return "The Aurora quantum sensor operates at 4 Kelvin cryogenic temperature and detects magnetic anomalies with sub-nanotesla precision."

        # Order Tracking & Delivery
        if any(w in lower_query for w in ["order", "delivery", "track", "shipment", "कूरियर", "ऑर्डर", "डिलीवरी"]):
            if wants_hindi:
                return "आपका ऑर्डर प्रोसेस हो चुका है और सामान्य डिलीवरी में 2 से 4 कार्यदिवस लगते हैं। आप अपने पंजीकृत मोबाइल नंबर से कभी भी स्टेटस ट्रैक कर सकते हैं।"
            return "Your order has been processed and standard delivery takes 2 to 4 business days. You can track your shipment anytime with your registered mobile number or Order ID."

        # Debit & Credit Card Blocking / ATM Fraud
        if any(w in lower_query for w in ["card", "debit card", "credit card", "atm", "block", "खो गया", "चोरी", "कार्ड", "ब्लॉक", "unauthorized"]):
            if wants_hindi:
                return "खोए या चोरी हुए कार्ड को तुरंत ब्लॉक करने के लिए IVR पर 0 दबाएं या 'ब्लॉक' बोलें। आपका कार्ड तुरंत फ्रीज हो जाएगा और 5 से 7 दिनों में नया कार्ड आपके पते पर भेज दिया जाएगा।"
            return "To immediately block your lost or compromised card, press 0 on your dial pad or say 'Block'. Your card will be instantly frozen and a free replacement dispatched within 5 to 7 business days."

        # Account Balance & Statement
        if any(w in lower_query for w in ["balance", "statement", "बैलेंस", "खाता शेष", "स्टेटमेंट", "पासबुक", "passbook"]):
            if wants_hindi:
                return "आपके बैंक खाते का उपलब्ध बैलेंस और अंतिम 3 लेन-देन का विवरण आपके पंजीकृत मोबाइल नंबर पर एसएमएस द्वारा भेज दिया गया है। विस्तृत स्टेटमेंट के लिए ईमेल चेक करें।"
            return "Your current account balance and last 3 transactions summary have been sent via SMS to your registered mobile number. You can also receive an e-Statement on your email."

        # Branch Timings & Working Hours
        if any(w in lower_query for w in ["timing", "timings", "opening hours", "open", "branch", "शाखा", "समय", "खुलती", "holiday", "शनिवार"]):
            if wants_hindi:
                return "हमारी सभी बैंक शाखाएं सोमवार से शुक्रवार सुबह 10:00 बजे से शाम 4:00 बजे तक खुली रहती हैं। पहले, तीसरे और पांचवें शनिवार को भी शाखा खुली रहती है।"
            return "Our bank branches operate Monday to Friday from 10:00 AM to 4:00 PM IST, and on the 1st, 3rd, and 5th Saturdays of each month."

        # Netbanking & MPIN Reset
        if any(w in lower_query for w in ["netbanking", "mpin", "pin", "login", "password", "पासवर्ड", "पिन", "अनलॉक", "unlock"]):
            if wants_hindi:
                return "अपना MPIN या पासवर्ड रीसेट करने के लिए मोबाइल ऐप लॉगिन स्क्रीन पर 'Forgot MPIN' पर जाएं। आप अपने एटीएम कार्ड और ओटीपी से 1 मिनट में नया पिन बना सकते हैं।"
            return "To reset your MPIN or NetBanking password, tap 'Forgot MPIN' on the app login screen. You can set a new 6-digit MPIN in 1 minute using your debit card and OTP."

        # Conversational Audio & Voice Clarification
        if any(w in lower_query for w in ["hear me", "loud and clear", "awaaz", "awaz", "sound", "आवाज़", "slow down", "dheere", "धीरे"]):
            if wants_hindi:
                return "हाँजी, मुझे आपकी आवाज़ बिल्कुल साफ़ और स्पष्ट आ रही है। मैं धीरे-धीरे बात करूँगा, बताइए मैं आपकी क्या सहायता करूँ?"
            return "Yes, I can hear you loud and clear! I will speak slowly and clearly. How can I assist you with your request today?"

        # Appreciation & Closing
        if any(w in lower_query for w in ["thank", "thanks", "dhanyawad", "shukriya", "धन्यवाद", "शुक्रिया", "bye", "alvida", "अलविदा"]):
            if wants_hindi:
                return "आपका बहुत-बहुत धन्यवाद! हमारी सहायता सेवा में संपर्क करने के लिए आभार। आपका दिन शुभ हो!"
            return "You are most welcome! Thank you for contacting our support service. Have a wonderful day ahead!"

        # If RAG context was injected with verified text, synthesize naturally without bot boilerplate
        if context_corpus.strip():
            clean_lines = []
            for line in context_corpus.splitlines():
                l = line.strip()
                if not l or l.startswith("---") or l.startswith("[") or "Relevant Context" in l or "You have access" in l:
                    continue
                clean_lines.append(l)
            if clean_lines:
                summary = " ".join(clean_lines[:2])
                if wants_hindi:
                    return f"{summary} क्या मैं इसमें आपकी और कोई सहायता कर सकता हूँ?"
                return f"{summary} Is there anything else I can help you with today?"

        # Fallback response - natural, polite IVR prompt
        if wants_hindi:
            return "माफ़ कीजिए, मैं आपकी बात पूरी तरह समझ नहीं पाया। क्या आप कृपया दोहरा सकते हैं? आप मुझसे यूपीआई रिफंड, वीडियो केवाईसी, या एजेंट से बात करने के लिए कह सकते हैं।"
        return "I'm sorry, I didn't quite catch that. Could you please repeat? You can ask me about your UPI refund status, Video KYC documents, or speak with an agent."

    async def stream_chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> AsyncIterator[StreamChunk]:
        req_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
        created_ts = int(time.time())

        # Yield role chunk
        yield StreamChunk(
            id=req_id,
            model=model_meta.alias,
            created=created_ts,
            choices=[
                StreamChoice(
                    index=0,
                    delta=StreamDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )

        reply_content = self._generate_mock_reply(request)
        words = reply_content.split(" ")
        for i, word in enumerate(words):
            prefix = "" if i == 0 else " "
            await asyncio.sleep(0.015)
            yield StreamChunk(
                id=req_id,
                model=model_meta.alias,
                created=created_ts,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(content=prefix + word),
                        finish_reason=None,
                    )
                ],
            )

        # Final stop chunk
        yield StreamChunk(
            id=req_id,
            model=model_meta.alias,
            created=created_ts,
            choices=[
                StreamChoice(
                    index=0,
                    delta=StreamDelta(),
                    finish_reason="stop",
                )
            ],
        )
