import time
import uuid
import json
import asyncio
from typing import AsyncIterator, Dict, Any, List
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
        reply_content = self._generate_mock_reply(request)

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

    def _generate_mock_reply(self, request: NormalizedChatRequest) -> str:
        import re
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
        is_greeting = bool(re.search(r"\b(hi|hello|hey|namaste|नमस्ते|hii|heya|hlo)\b", lower_query))
        if is_greeting and len(clean_query.split()) <= 4:
            if wants_hindi:
                return "नमस्ते! मैं आज आपकी क्या सहायता कर सकता हूँ? आप मुझसे यूपीआई रिफंड, वीडियो केवाईसी, या ऑर्डर ट्रैकिंग के बारे में पूछ सकते हैं।"
            return "Hello! How can I assist you today? You can ask me about UPI refunds, Video KYC requirements, order tracking, or test PostgreSQL RAG retrieval."

        # 3. Domain & Knowledge Base matching:
        # IVR Keypad / Live Support
        if any(w in lower_query for w in ["ivr", "keypad", "key", "press", "representative", "agent", "live support", "executive", "दबाएं", "एजेंट", "प्रतिनिधि"]):
            if wants_hindi:
                return "कस्टमर केयर प्रतिनिधि से सीधे बात करने के लिए अपने डायल पैड पर किसी भी समय '9' दबाएं या 'एजेंट' बोलें। औसत प्रतीक्षा समय 2 मिनट से कम है।"
            return "Based on the IVR routing knowledge base: Press option '9' (or say 'agent' / 'representative') on your dial pad at any time to connect directly with a live customer care representative. Average queue wait time is under 2 minutes."

        # Video KYC / Documents Required
        if any(w in lower_query for w in ["kyc", "केवाईसी", "video kyc", "documents", "pan", "aadhaar", "document"]):
            if wants_hindi:
                return "डिजिटल वीडियो केवाईसी के लिए आवश्यक दस्तावेज: 1) मूल फिजिकल पैन कार्ड, 2) आधार से लिंक सक्रिय मोबाइल नंबर (ओटीपी के लिए), और 3) अच्छी रोशनी वाला कमरा। यह सत्यापन सिर्फ 2 मिनट में पूरा हो जाता है।"
            return "According to verified Video KYC guidelines: You will need: 1) Original physical PAN card, 2) Aadhaar card with active mobile number linked for OTP, and 3) A well-lit room with clear camera access. The verification completes in 2 minutes and unlocks full account privileges within 1 hour."

        # UPI Refund / Payment failure
        if any(w in lower_query for w in ["refund", "upi", "jubiai", "रिफंड", "cut", "kat gaya", "paise", "debited", "failed", "विफलता", "istiti", "sthiti"]):
            if wants_hindi:
                return "आरबीआई और एनपीसीआई के नियमानुसार: यदि यूपीआई से पैसे कट गए हैं, तो 24 से 48 बैंकिंग घंटों में स्वतः आपके बैंक खाते में रिफंड आ जाता है। आप 12 अंकों के यूटीआर (UTR) नंबर से भी स्टेटस चेक कर सकते हैं।"
            return "Per RBI and NPCI auto-refund regulations: Debited amounts from failed UPI transactions are automatically credited back to your original bank account within 24 to 48 banking hours. You can track this in real time using your 12-digit UPI UTR reference number."

        # Aurora Sensor / Quantum specs
        if any(w in lower_query for w in ["aurora", "quantum", "temperature", "kelvin"]):
            return "According to the Aurora technical specifications: The Aurora quantum sensor operates at 4 Kelvin (-269.15°C) cryogenic temperature and detects magnetic anomalies with sub-nanotesla precision."

        # Order Tracking & Delivery
        if any(w in lower_query for w in ["order", "delivery", "track", "shipment", "कूरियर", "ऑर्डर", "डिलीवरी"]):
            if wants_hindi:
                return "आपका ऑर्डर प्रोसेस हो चुका है और सामान्य डिलीवरी में 2 से 4 कार्यदिवस लगते हैं। आप अपने पंजीकृत मोबाइल नंबर या ऑर्डर आईडी से कभी भी स्टेटस ट्रैक कर सकते हैं।"
            return "Your order has been processed and standard delivery takes 2 to 4 business days. You can track the shipment status anytime using your registered mobile number or Order ID."

        # If RAG context was injected with verified text, synthesize an excerpt
        if context_corpus.strip():
            lines = [line.strip() for line in context_corpus.splitlines() if line.strip() and not line.startswith("[") and not line.startswith("Relevant Context") and not line.startswith("You have access")]
            if lines:
                excerpt = "\n".join(lines[:3])
                return f"Based on verified knowledge base documents:\n\n{excerpt}\n\nHow else can I assist you with this?"

        # Fallback response
        if wants_hindi:
            return f"मैंने आपकी पूछताछ दर्ज कर ली है: '{clean_query}'। आप यूपीआई रिफंड, वीडियो केवाईसी, या ऑर्डर ट्रैकिंग के बारे में पूछ सकते हैं।"
        return f"I have processed your inquiry: '{clean_query}'. You can test domain queries like IVR keypad routing, Video KYC document checklist, or UPI refund status."

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
