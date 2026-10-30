#!/usr/bin/env python3
"""
Comprehensive Expanded Knowledge Base Ingestion for Standalone IVR & Web Telephony.
Embeds and persists dozens of bilingual (Hindi & English) chunks into PostgreSQL pgvector
using 'semantic-default' (sentence-transformers/all-MiniLM-L6-v2) so the AI platform
operates 100% independently without requiring external Colab/Gemma GPU fallback.
"""
import os
import sys
import asyncio
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.database import async_session, init_db
from storage.repositories.tenant_repo import TenantRepository
from storage.repositories.document_repo import DocumentRepository
from storage.repositories.api_key_repo import ApiKeyRepository
from platform_core.security.api_keys import hash_api_key
from retrieval.ingestion import DocumentIngestionService
from retrieval.embedding_service import EmbeddingService
from retrieval.rag_pipeline import RAGPipeline
from retrieval.search import SemanticSearchService
from providers.registry import provider_registry


IVR_EXPANDED_KNOWLEDGE = [
    # ── 1. GREETINGS, CAPABILITIES & SMALL TALK ─────────────────────────────
    {
        "title": "Telephony Greetings, Agent Capabilities & Conversational Dialogue",
        "source_type": "conversational_dialogue",
        "content": """
[GREETINGS, IDENTITY & CAPABILITY KNOWLEDGE]

Voice Greeting (Hindi):
"नमस्ते! हमारी एआई सहायता सेवा में आपका स्वागत है। मैं आपका डिजिटल असिस्टेंट हूँ।
मैं आपकी यूपीआई रिफंड स्थिति, बैंक खाता बैलेंस, डेबिट कार्ड ब्लॉकिंग, वीडियो केवाईसी, या ऑर्डर डिलीवरी ट्रैकिंग में पूरी मदद कर सकता हूँ।
यदि आप किसी मानव अधिकारी (Live Agent) से बात करना चाहते हैं, तो आप कभी भी 'एजेंट' कह सकते हैं। आज मैं आपकी क्या सहायता करूँ?"

Voice Greeting (English):
"Hello! Welcome to our 24/7 AI Customer Support helpline. I am your automated virtual assistant.
I can help you with your UPI refund status, account balance inquiries, debit or credit card blocking, digital Video KYC verification, and order shipment tracking.
You can also ask to speak with a human support agent at any time by pressing 9 or saying 'Agent'. How may I assist you today?"

Conversational Speech & Audio Clarifications:
- If caller says 'Awaaz nahi aa rahi' or 'Can you hear me?':
  "हाँजी, मुझे आपकी आवाज़ बिल्कुल साफ़ और स्पष्ट आ रही है। कृपया अपनी समस्या बताइए, मैं तुरंत समाधान करूँगा।"
  "Yes, I can hear you loud and clear! Please go ahead with your question and I will assist you right away."
- If caller says 'Slow down' or 'Thoda dheere bolo':
  "जी बिल्कुल, मैं थोड़ा धीरे बोलता हूँ। आप आराम से अपनी बात बताइए।"
  "Certainly! I will speak more slowly for you. Please tell me how I can help."
- If caller says 'Thank you' or 'Dhanyawad' or 'Shukriya':
  "अरे, आपका बहुत-बहुत धन्यवाद! क्या मैं आपकी किसी अन्य सेवा में सहायता कर सकता हूँ?"
  "You are most welcome! Is there anything else I can assist you with today?"
- If caller says 'Bye' or 'Tata' or 'Alvida':
  "हमारे साथ संपर्क करने के लिए धन्यवाद। आपका दिन शुभ हो! अलविदा।"
  "Thank you for contacting Customer Support. Have a wonderful day ahead! Goodbye."
"""
    },

    # ── 2. DEBIT & CREDIT CARDS, ATM SERVICES & FRAUD ───────────────────────
    {
        "title": "Debit Card, Credit Card, ATM & Emergency Fraud Resolution",
        "source_type": "card_security_policy",
        "content": """
[CARD SERVICES, ATM & EMERGENCY ACCOUNT FREEZE PROTOCOL]

Emergency Lost/Stolen Card Blocking (Hindi):
"खोए या चोरी हुए डेबिट/क्रेडिट कार्ड को तुरंत ब्लॉक करने के लिए:
1. किसी भी अनधिकृत लेनदेन (Fraud Transaction) को रोकने के लिए तुरंत 'BLOCK CARD' बोलें या IVR पर '0' दबाएं। यह कॉल बिना किसी प्रतीक्षा के 24x7 इमरजेंसी सुरक्षा डेस्क पर स्थानांतरित कर दी जाती है।
2. कार्ड ब्लॉक होते ही आपके खाते से सभी एटीएम, ऑनलाइन पीओएस और अंतरराष्ट्रीय लेन-देन तुरंत फ्रीज हो जाते हैं।
3. नया रिप्लेसमेंट कार्ड आपके पंजीकृत डाक पते पर 5 से 7 कार्यदिवसों में बिना किसी अतिरिक्त शुल्क के डिलीवर कर दिया जाएगा।"

Emergency Lost/Stolen Card Blocking (English):
"Emergency Card Blocking and Fraud Prevention:
- To immediately block a lost, stolen, or compromised debit or credit card, press 0 on your dial pad or speak 'Block Card' or 'Fraud'.
- Emergency security requests bypass all queues and immediately freeze all ATM, online eCommerce, POS, and contactless transactions.
- A new chip-and-PIN replacement card will be dispatched to your registered communication address within 5 to 7 business days."

ATM PIN Generation & Daily Limits:
- ATM PIN Reset: You can generate a Green PIN instantly via Mobile Banking app or by visiting any nearby bank ATM with your registered mobile OTP.
- Daily Cash Withdrawal Limit: Standard debit cards have a daily ATM cash limit of ₹40,000 to ₹1,00,000 depending on your card variant.
- Unblocking Temporarily Locked Card: If your card was locked due to 3 incorrect PIN attempts, it unlocks automatically after 24 hours, or you can unlock it immediately in Mobile Banking under 'Manage Cards'.
"""
    },

    # ── 3. ACCOUNT BALANCE, STATEMENTS, PASSBOOK & TIMINGS ──────────────────
    {
        "title": "Account Balance, Mini Statements, Passbook & Bank Timings",
        "source_type": "banking_account_policy",
        "content": """
[ACCOUNT BALANCE, STATEMENTS & BRANCH TIMINGS]

Account Balance & Statement Check (Hindi):
"बैंक खाता शेष और मिनी स्टेटमेंट की जानकारी:
1. खाता शेष (Balance): अपने पंजीकृत मोबाइल नंबर से बैलेंस जानने के लिए 'BALANCE' बोलें या IVR पर 1 दबाएं। आपके खाते का उपलब्ध बैलेंस और अंतिम 3 लेन-देन का विवरण तुरंत एसएमएस द्वारा भेज दिया जाएगा।
2. बैंक स्टेटमेंट (e-Statement): पिछले 1 महीने, 3 महीने या 1 साल का स्टेटमेंट सीधे आपकी पंजीकृत ईमेल आईडी पर पासवर्ड-सुरक्षित पीडीएफ (PDF) के रूप में 5 मिनट के भीतर भेज दिया जाता है।
3. शाखा का समय (Branch Timings): हमारी सभी बैंक शाखाएं सोमवार से शुक्रवार सुबह 10:00 बजे से शाम 4:00 बजे तक और महीने के पहले, तीसरे और पांचवें शनिवार को खुली रहती हैं। दूसरे और चौथे शनिवार तथा रविवार को अवकाश रहता है।"

Account Balance & Statement Check (English):
"Account Balance Inquiries and Bank Working Hours:
- Balance & Mini Statement: Inquire about your available account balance by pressing 1 or speaking 'Balance'. A real-time SMS summary of your available balance and last 3 transactions will be delivered instantly.
- Detailed Statement: Monthly or annual account statements in password-protected PDF format can be delivered to your registered email address within 5 minutes.
- Branch Timings: Physical branches operate Monday through Friday from 10:00 AM to 4:00 PM IST, and on the 1st, 3rd, and 5th Saturdays of the month. Branches remain closed on the 2nd and 4th Saturdays, public holidays, and Sundays."

IFSC Code & Bank Transfers:
- IMPS (Immediate Payment Service): 24x7 instant transfer up to ₹5,00,000 with zero waiting time.
- NEFT & RTGS: Available 24x7 with zero processing charges for retail internet banking users.
- IFSC Code Verification: Check your branch IFSC code printed on your chequebook front page or ask our assistant with your branch city name.
"""
    },

    # ── 4. NETBANKING, MOBILE APP LOGIN & MPIN RESET ────────────────────────
    {
        "title": "Internet Banking, Mobile App Login & MPIN Reset Guidelines",
        "source_type": "digital_banking_faq",
        "content": """
[NETBANKING, MOBILE APP ACCESS & PASSWORD RESET]

Mobile App & NetBanking Troubleshooting (Hindi):
"इंटरनेट बैंकिंग और मोबाइल ऐप लॉगिन समाधान:
1. यदि आपका खाता गलत पासवर्ड या MPIN डालने के कारण लॉक हो गया है, तो सुरक्षा कारणों से यह 24 घंटे के लिए अस्थाई रूप से ब्लॉक हो जाता है।
2. तुरंत अनलॉक करने के लिए: ऐप की लॉगिन स्क्रीन पर 'Forgot MPIN' या 'पासवर्ड भूल गए' पर क्लिक करें।
3. अपने पंजीकृत मोबाइल पर आया 6-अंकों का ओटीपी दर्ज करें और अपने डेबिट कार्ड के अंतिम 4 अंक व एटीएम पिन डालकर 1 मिनट में नया पासवर्ड बना लें।
4. ऐप क्रैश या बायोमेट्रिक फिंगरप्रिंट समस्या: ऐप को प्ले स्टोर से नवीनतम संस्करण में अपडेट करें या 'Clear Cache' करें।"

Mobile App & NetBanking Troubleshooting (English):
"Digital Banking Access & MPIN Recovery:
- Account Lockout: If your mobile banking profile is locked due to consecutive incorrect MPIN entries, it will automatically reset after 24 hours.
- Instant Password Reset: Open the mobile banking login screen, tap 'Forgot Login Password / MPIN', enter your registered mobile number, and authenticate with the SMS OTP.
- Debit Card Verification: Enter your active debit card's last 4 digits and ATM PIN to establish a new 6-digit MPIN in under 60 seconds.
- Biometric & Face ID: Biometric login can be re-enabled under App Settings -> Security -> Biometric Authentication."
"""
    },

    # ── 5. UPI PAYMENT FAILURES, NPCI SLA & REFUNDS ─────────────────────────
    {
        "title": "UPI Payment Failures, NPCI Auto-Refund SLA & UTR Tracking",
        "source_type": "upi_refund_master",
        "content": """
[UPI PAYMENT RESOLUTION, NPCI SETTLEMENT & AUTO-REFUND SLA]

UPI Transaction Failure & Money Deducted (Hindi):
"यूपीआई पेमेंट फेल और पैसे कटने का त्वरित समाधान:
1. यदि PhonePe, Google Pay, Paytm या BHIM से भुगतान करते समय बैंक से पैसे कट गए पर मर्चेंट को नहीं मिले, तो घबराएं नहीं। आपका पैसा एनपीसीआई (NPCI) के सुरक्षित क्लियरिंग हाउस में होल्ड रहता है।
2. आरबीआई और एनपीसीआई की आधिकारिक समय-सीमा के अनुसार, ऐसा पैसा 24 से 48 बैंकिंग घंटों (24 to 48 hours) के भीतर स्वतः आपके उसी बैंक खाते में वापस (Auto-Credit) आ जाता है।
3. रिफंड स्थिति ट्रैक करने के लिए: अपनी ऐप के लेन-देन विवरण में 12-अंकों का यूटीआर नंबर (UTR: 12-digit Unique Transaction Reference) देखें।
4. यदि 48 घंटे के बाद भी पैसा न आए, तो हमें यूटीआर नंबर बताएं, हमारी सिस्टम तुरंत आपके बैंक के नोडल अधिकारी के साथ विवाद टिकट (Dispute Ticket) दर्ज कर देगी।"

UPI Transaction Failure & Money Deducted (English):
"UPI Transaction Failure and Refund SLAs:
- Money Debited for Failed Payment: If money was debited from your bank account but the recipient did not receive credit, the transaction is marked as pending in the NPCI inter-bank settlement network.
- Official SLA: Under RBI and NPCI regulatory guidelines, 100% of failed UPI debits are automatically reversed and credited back to your originating bank account within 24 to 48 business hours.
- Tracking via 12-Digit UTR: Check your UPI transaction receipt for the 12-digit UTR (Unique Transaction Reference) number.
- Escalation: If funds are not credited back after 48 business hours, our automated system generates a chargeback dispute ticket directly with your bank's regulatory nodal desk."
"""
    },

    # ── 6. VIDEO KYC, DOCUMENT VERIFICATION & PROFILE UNLOCK ─────────────────
    {
        "title": "Digital Video KYC, Document Requirements & Account Activation",
        "source_type": "kyc_verification_protocol",
        "content": """
[DIGITAL VIDEO KYC & IDENTITY VERIFICATION STANDARDS]

Video KYC Procedure & Requirements (Hindi):
"डिजिटल वीडियो केवाईसी और खाता सत्यापन:
1. वीडियो केवाईसी क्यों जरूरी है? खाते की दैनिक लेनदेन सीमा बढ़ाने, अंतरराष्ट्रीय भुगतान सक्षम करने और खाते को पूर्ण सक्रिय करने के लिए वीडियो केवाईसी अनिवार्य है।
2. आवश्यक दस्तावेज:
   - आपका मूल भौतिक पैन कार्ड (Original Physical PAN Card) - फोटोकॉपी या डिजिटल कॉपी मान्य नहीं है।
   - आधार से लिंक पंजीकृत मोबाइल नंबर (जिस पर सत्यापन ओटीपी आएगा)।
   - एक सादा सफेद कागज़ और नीला/काला पेन (लाइव हस्ताक्षर के लिए)।
3. प्रक्रिया: चैट में 'केवाईसी शुरू करें' कहें या फोन पर 4 दबाएं। आपके मोबाइल पर एक सुरक्षित वीडियो कॉल लिंक एसएमएस द्वारा भेजा जाएगा।
4. कॉल का समय: केवल 2 से 3 मिनट। कॉल संपन्न होने के 1 घंटे के भीतर खाता पूर्ण सक्रिय हो जाता है।"

Video KYC Procedure & Requirements (English):
"Digital Video KYC Verification Protocol:
- Purpose: Complete mandatory full-KYC compliance to remove account holding caps, upgrade monthly transaction limits, and enable international cards.
- Mandatory Prerequisites:
  - Original physical PAN card (scans, laminations, or soft copies are rejected by regulatory rules).
  - Active Aadhaar-linked mobile device to receive and input the live OTP.
  - Plain white paper and pen for live signature capture during the video call.
  - Well-lit, quiet room with stable front camera connectivity.
- Duration: The video session with our certified verification officer takes 2 to 3 minutes.
- Activation SLA: Your profile is fully approved and unlocked within 1 hour post-call."
"""
    },

    # ── 7. ORDER TRACKING, DELIVERY, RETURNS & REFUNDS ───────────────────────
    {
        "title": "E-Commerce Orders, Live Delivery Tracking, Returns & Cancellations",
        "source_type": "ecommerce_fulfillment_policy",
        "content": """
[ORDERS, LIVE SHIPMENT TRACKING, CANCELLATIONS & RETURNS]

Order Tracking & Delayed Deliveries (Hindi):
"ऑर्डर स्थिति, डिलीवरी ट्रैकिंग और कैंसिलेशन नियम:
1. लाइव डिलीवरी ट्रैकिंग: जैसे ही आपका पार्सल कूरियर पार्टनर (Delhivery, BlueDart, Ecom Express) को सौंपा जाता है, ट्रैकिंग लिंक व ओटीपी एसएमएस द्वारा भेज दिया जाता है। सामान्य डिलीवरी 2 से 4 कार्यदिवसों में हो जाती है।
2. ऑर्डर कैंसल करना: गोदाम से निकलने (Dispatch) से पहले आप बिना किसी पेनल्टी के ऑर्डर रद्द कर सकते हैं।
3. रिफंड की समय-सीमा:
   - यदि भुगतान यूपीआई या वॉलेट से किया था: 24 घंटे के भीतर रिफंड।
   - यदि क्रेडिट या डेबिट कार्ड से किया था: 5 से 7 बैंकिंग दिनों में रिफंड।
   - कैश ऑन डिलीवरी (COD): रिफंड प्राप्त करने के लिए ऐप में अपना यूपीआई आईडी या बैंक खाता विवरण दर्ज करें।
4. 7-दिवसीय रिटर्न व रिप्लेसमेंट: यदि सामान टूटा-फूटा, दोषपूर्ण या गलत निकला है, तो डिलीवरी के 7 दिनों के भीतर निःशुल्क पिकअप व एक्सचेंज का अनुरोध करें।"

Order Tracking & Delayed Deliveries (English):
"Shipment Tracking, Cancellation, and Returns Protocol:
- Tracking: Real-time courier tracking links and delivery OTPs are dispatched via SMS upon shipment. Standard delivery window is 2 to 4 business days.
- Free Cancellation: Orders can be cancelled instantly with zero cancellation fee prior to warehouse dispatch.
- Refund Crediting SLA:
  - UPI and digital wallets: Processed within 24 hours.
  - Credit/Debit cards & Net Banking: Processed within 5 to 7 banking business days.
  - Cash on Delivery (COD): Add your preferred UPI ID or Bank account in the app for instant direct-to-bank transfer.
- 7-Day Hassle-Free Returns: Free door-step reverse pickup and replacement for any damaged, defective, or incorrect items."
"""
    },

    # ── 8. LIVE HUMAN AGENT ESCALATION, QUEUES & OFFICE HOURS ────────────────
    {
        "title": "Live Customer Care Agent Transfer, Support Queues & Operating Hours",
        "source_type": "agent_escalation_protocol",
        "content": """
[LIVE AGENT ESCALATION, HUMAN QUEUE MANAGEMENT & HOURS]

Connecting to a Live Representative (Hindi):
"कस्टमर केयर एक्जीक्यूटिव से लाइव बातचीत (Agent Transfer):
1. यदि आप हमारे किसी मानव सहायता विशेषज्ञ या सुपरवाइजर से सीधे बात करना चाहते हैं, तो:
   - फोन डायल पैड पर '9' दबाएं या स्पष्ट स्वर में 'कस्टमर केयर' या 'एजेंट' बोलें।
2. हमारी लाइव टेलीफोन सपोर्ट टीम सप्ताह के सातों दिन (सोमवार से रविवार) सुबह 8:00 बजे से रात 11:00 बजे तक उपलब्ध रहती है।
3. औसत प्रतीक्षा समय (Wait Time) व्यस्ततम घंटों में भी 2 मिनट से कम रहता है।
4. सत्यापन: कॉल ट्रांसफर होने पर अधिकारी आपकी पहचान के लिए पंजीकृत मोबाइल नंबर या सुरक्षा पिन की पुष्टि कर सकते हैं।
5. कॉल बैक सेवा: यदि लाइनें व्यस्त हैं, तो 'CALL ME' कहें; हमारे विशेषज्ञ 15 मिनट के भीतर आपको स्वतः कॉल बैक करेंगे।"

Connecting to a Live Representative (English):
"Live Human Specialist Escalation Protocol:
- Immediate Transfer: To transfer your voice call or chat session to a live human customer support specialist, press 9 on your dial pad or say 'Agent' or 'Representative'.
- Operating Hours: Live human phone and chat assistance is active 7 days a week from 8:00 AM to 11:00 PM IST.
- Average Queue Wait Time: Under 2 minutes during normal operating hours.
- Automated Callback Option: If queues are busy, simply state 'Call me back' and our next available supervisor will return your call within 15 minutes.
- Verification: Keep the last 4 digits of your registered mobile number or your 4-digit IVR T-PIN handy for quick authorization."
"""
    }
]

IVR_TENANT_NAME = "tenant-ivr-support"
IVR_TENANT_ID = os.getenv("IVR_TENANT_ID", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")


async def train_expanded_knowledge():
    print("\n" + "=" * 75)
    print("INGESTING EXPANDED BILINGUAL KNOWLEDGE CHUNKS (LOCAL PGVECTOR RAG)")
    print("=" * 75)

    await init_db()
    print("✓ PostgreSQL 16 + pgvector database initialized.")

    provider_registry.initialize_from_file("config/providers.yaml")
    emb_adapter = provider_registry.get_embedding_adapter("semantic-default")
    print(f"✓ Local Embedding Adapter loaded: {emb_adapter.__class__.__name__} (Dimension: {emb_adapter.dimension})")

    async with async_session() as session:
        tenant_repo = TenantRepository(session)
        doc_repo = DocumentRepository(session)
        emb_service = EmbeddingService(provider_id="semantic-default")
        ingestion_service = DocumentIngestionService(doc_repo, emb_service)

        tenant = await tenant_repo.get_by_name(IVR_TENANT_NAME)
        if not tenant:
            tenant = await tenant_repo.create(IVR_TENANT_NAME)
            print(f"✓ Created new tenant: {tenant.name} (ID: {tenant.id})")
        else:
            print(f"✓ Using existing tenant: {tenant.name} (ID: {tenant.id})")

        total_chunks = 0
        for doc_item in IVR_EXPANDED_KNOWLEDGE:
            doc = await ingestion_service.ingest_document(
                tenant_id=tenant.id,
                title=doc_item["title"],
                content=doc_item["content"],
                source_type=doc_item["source_type"],
            )
            total_chunks += doc.total_chunks
            print(f"  + Ingested '{doc.title[:45]}...' -> {doc.total_chunks} chunks vectorized.")

        print(f"\n✓ Successfully vectorized and persisted {total_chunks} new chunks in pgvector.")

        # Verification tests across multiple real caller inquiries
        search_service = SemanticSearchService(doc_repo, emb_service)
        rag_pipeline = RAGPipeline(search_service=search_service)

        print("\n" + "-" * 75)
        print("VERIFYING LOCAL RAG RETRIEVAL ACROSS POPULAR CALLER QUERIES")
        print("-" * 75)

        test_queries = [
            ("Greeting / Who are you", "Hello, who are you and what can you help me with?"),
            ("Hindi Greeting", "नमस्ते आप कौन बोल रहे हैं?"),
            ("Lost Debit Card", "Mera ATM debit card kho gaya hai, turant block karna hai"),
            ("English Card Fraud", "I want to block my credit card due to unauthorized transaction"),
            ("Account Balance", "Mera bank balance kitna hai batao"),
            ("Bank Timings", "What are your branch opening hours on Saturday?"),
            ("UPI Refund", "UPI transaction fail ho gaya paise kat gaye"),
            ("Video KYC", "What documents are required to complete Video KYC?"),
            ("Order Tracking", "Mera delivery status kya hai parcel kab aayega"),
            ("Speak to Human", "I want to talk to a human customer care representative"),
            ("Audio Clarification", "Awaaz nahi aa rahi thoda dheere bolo"),
            ("Closing / Thanks", "Bahut bahut shukriya aapka"),
        ]

        passed = 0
        for label, q in test_queries:
            results = await search_service.search(tenant_id=tenant.id, query=q, top_k=2)
            top_score = results[0][1] if results else 0.0
            is_match = top_score >= 0.35
            mark = "✓" if is_match else "✗"
            if is_match:
                passed += 1
            print(f"  {mark} [{label}] Score: {top_score:.3f} | Query: '{q[:40]}...'")

        print(f"\nResults: {passed}/{len(test_queries)} queries successfully resolved by local RAG chunks!")
        print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(train_expanded_knowledge())

