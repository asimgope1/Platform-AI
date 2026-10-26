#!/usr/bin/env python3
"""
Seed and Train the AI Platform with Bilingual IVR Call & Chat Support Knowledge
Supports Hindi, English, and Hinglish for both Text and Audio workflows using Google Embeddings.
"""
import os
import sys
import asyncio
from typing import Dict, Any, List

# Ensure project root is in sys.path
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


IVR_BILINGUAL_DOCUMENTS = [
    {
        "title": "IVR Call Routing & Voice Menu Prompts (Hindi & English)",
        "source_type": "ivr_audio_script",
        "content": """
[IVR SYSTEM GREETING - HINDI & ENGLISH VOICE SCRIPTS]

Audio Prompt ID: IVR-PROMPT-001
Hindi Voice Transcript:
"नमस्ते! हमारी 24x7 ग्राहक सेवा हेल्पलाइन में आपका स्वागत है। 
हिंदी में जारी रखने के लिए 1 दबाएं। 
To continue in English, press 2.
खाता शेष और बिलिंग पूछताछ के लिए 3 दबाएं।
यूपीआई भुगतान विफलता या रिफंड स्थिति के लिए 4 दबाएं।
ऑर्डर और कूरियर ट्रैकिंग के लिए 5 दबाएं।
वीडियो केवाईसी और खाता अनब्लॉक के लिए 6 दबाएं।
हमारे ग्राहक सेवा प्रतिनिधि (Customer Care Executive) से सीधे बात करने के लिए किसी भी समय 9 दबाएं या 'एजेंट' बोलें।"

Audio Prompt ID: IVR-PROMPT-002
English Voice Transcript:
"Thank you for calling Customer Support.
For Account Balance and Billing inquiries, press 1.
For UPI transaction failure, deduction issues, and refund status, press 2.
To track your recent delivery or shipment, press 3.
For Video KYC verification and profile unlock, press 4.
To transfer your call directly to a live customer service specialist, press 9 or say 'Representative' at any prompt."

[DTMF KEYPAD ROUTING LOGIC]
- Key 1 / Hindi Key 3: Routes to Account Balance & Billing sub-menu.
- Key 2 / Hindi Key 4: Routes to Instant UPI & Refund automated resolution engine.
- Key 3 / Hindi Key 5: Requests 10-digit Order ID or Registered Mobile number for live shipment tracking.
- Key 4 / Hindi Key 6: Triggers automated SMS with a secure Video KYC link.
- Key 9: Bypasses automated voice prompts and places call in human executive queue. Average wait time is under 2 minutes.
"""
    },
    {
        "title": "UPI Payment Failures & Refund Timelines (Hindi & English)",
        "source_type": "support_policy",
        "content": """
[UPI PAYMENT FAILURE & AUTO-REFUND RESOLUTION PROTOCOL]

Audio Prompt ID: IVR-PROMPT-UPI-01
Hindi Voice & Chat Guidance:
"यूपीआई पेमेंट विफलता और रिफंड दिशानिर्देश:
यदि आपके बैंक खाते से पैसे कट गए हैं परंतु मर्चेंट या ऑर्डर को भुगतान प्राप्त नहीं हुआ है, तो घबराएं नहीं। 
भारतीय राष्ट्रीय भुगतान निगम (NPCI) और भारतीय रिजर्व बैंक (RBI) के नियमानुसार, ऐसी सभी विफल लेनदेन की राशि 24 से 48 बैंकिंग घंटों (24 to 48 Banking Hours) के भीतर स्वतः आपके मूल बैंक खाते में वापस (Auto-Refund) जमा कर दी जाती है।
रिफंड की स्थिति ट्रैक करने के लिए:
1. अपने बैंक या यूपीआई ऐप (GPay, PhonePe, Paytm) में 12-अंकों का यूपीआई यूटीआर नंबर (UTR - Unique Transaction Reference) देखें।
2. चैट में 'REFUND <12-अंकों का UTR>' लिखकर भेजें या IVR पर 4 दबाएं।
3. यदि 48 घंटे के बाद भी पैसा वापस न आए, तो चैट में 'DISPUTE' टाइप करें या 9 दबाकर बैंक विवाद निपटान अधिकारी से बात करें।"

Audio Prompt ID: IVR-PROMPT-UPI-02
English Voice & Chat Guidance:
"UPI Transaction Failure and Refund Protocol:
If funds have been debited from your account via UPI but the transaction displays as pending or failed, the amount is held securely in the NPCI inter-bank settlement pool.
Under official RBI guidelines, debited amounts from failed UPI transactions are automatically refunded to your originating bank account within 24 to 48 business hours.
How to track or escalate:
- Locate the 12-digit UTR (Unique Transaction Reference) number from your bank statement or UPI application.
- In support chat, message 'REFUND <12-digit UTR>' or enter it on the IVR keypad after selecting option 2.
- If funds are not credited after 48 banking hours, an automated dispute ticket is created and routed directly to the bank nodal officer."
"""
    },
    {
        "title": "Live Agent Transfer & Customer Care Escalation (Hindi & English)",
        "source_type": "escalation_guide",
        "content": """
[LIVE AGENT ESCALATION & CALL TRANSFER PROTOCOL]

Audio Prompt ID: IVR-PROMPT-ESC-01
Hindi Voice & Chat Support Script:
"कस्टमर केयर एक्जीक्यूटिव से लाइव बातचीत (Agent Transfer):
यदि आप हमारे किसी सहायता विशेषज्ञ से सीधे बात करना चाहते हैं, तो:
1. IVR कॉल पर 9 दबाएं या स्पष्ट स्वर में 'कस्टमर केयर' या 'एजेंट' बोलें।
2. हमारी लाइव सपोर्ट टीम सप्ताह के सातों दिन (सोमवार से रविवार) सुबह 8:00 बजे से रात 11:00 बजे तक उपलब्ध रहती है।
3. कॉल ट्रांसफर से पहले कृपया अपना 4-अंकीय टेलीफोन पिन (T-PIN) या पंजीकृत मोबाइल नंबर पर आया ओटीपी (OTP) तैयार रखें।
4. किसी भी वित्तीय धोखाधड़ी या कार्ड/खाता तुरंत ब्लॉक करने के लिए IVR पर 0 दबाएं, यह कॉल बिना किसी प्रतीक्षा के आपातकालीन सुरक्षा डेस्क पर ट्रांसफर हो जाएगी।"

Audio Prompt ID: IVR-PROMPT-ESC-02
English Voice & Chat Support Script:
"Connecting to a Live Customer Support Specialist:
- To speak with a human support agent at any time, press 9 on your dial pad or speak 'Agent' or 'Representative'.
- Operating Hours: Live phone and chat assistance is available 7 days a week, 8:00 AM to 11:00 PM IST.
- Average Queue Wait Time: Under 2 minutes during peak hours.
- Verification Requirement: You will be asked to confirm the last 4 digits of your registered mobile number or enter a 4-digit IVR T-PIN.
- Emergency Fraud Hotkey: Press 0 immediately to report unauthorized activity or freeze your account with zero queue delay."
"""
    },
    {
        "title": "Digital Video KYC & Document Verification (Hindi & English)",
        "source_type": "kyc_procedure",
        "content": """
[VIDEO KYC & IDENTITY VERIFICATION GUIDELINES]

Audio Prompt ID: IVR-PROMPT-KYC-01
Hindi Guidance:
"वीडियो केवाईसी और खाता सत्यापन प्रक्रिया:
अपने खाते की सीमा बढ़ाने या पूर्ण केवाईसी पूरा करने के लिए:
1. आवश्यक दस्तावेज: आपका मूल पैन कार्ड (Original Physical PAN Card) और आधार कार्ड।
2. चैट सपोर्ट में 'START KYC' या 'केवाईसी शुरू करें' लिखकर भेजें।
3. आपको एसएमएस और चैट में एक सुरक्षित वीडियो कॉल लिंक प्राप्त होगा।
4. शांत और अच्छी रोशनी वाले कमरे में लिंक खोलें। हमारे प्रमाणित अधिकारी 2 मिनट की वीडियो कॉल में आपके मूल पैन कार्ड की तस्वीर लेंगे और लाइव फेस मैच करेंगे।
5. सत्यापन पूरा होने के 1 घंटे के भीतर खाता पूर्ण सक्रिय हो जाता है।"

Audio Prompt ID: IVR-PROMPT-KYC-02
English Guidance:
"Digital Video KYC and Identity Verification:
- Purpose: Complete full identity verification to increase monthly transaction limits and enable international services.
- Mandatory Prerequisites: Original physical PAN card, Aadhaar-linked registered mobile for OTP, well-lit environment, and front camera access.
- How to Initiate: Request 'Start Video KYC' in chat or press 4 on the IVR. A secure one-time video conference link will be dispatched via SMS.
- Turnaround Time: The call takes 2 to 3 minutes. Your account is fully unlocked within 1 hour post-verification."
"""
    },
    {
        "title": "Order Tracking, Cancellation & Refund Policy (Hindi & English)",
        "source_type": "order_billing_policy",
        "content": """
[ORDER SHIPMENT, CANCELLATION & BILLING POLICY]

Audio Prompt ID: IVR-PROMPT-ORD-01
Hindi Voice & Chat Rules:
"ऑर्डर डिलीवरी, रद्दीकरण और रिफंड नीतियां:
1. डिलीवरी स्थिति: ऑर्डर डिस्पैच होते ही लाइव जीपीएस ट्रैकिंग लिंक आपके फोन पर एसएमएस द्वारा भेजा जाता है। डिलीवरी का सामान्य समय 2 से 4 कार्यदिवस है।
2. ऑर्डर रद्दीकरण: गोदाम से निकलने (Dispatch) से पहले आप ऐप में 'Cancel Order' दबाकर या चैट में 'CANCEL <OrderID>' लिखकर बिना किसी शुल्क के ऑर्डर रद्द कर सकते हैं।
3. रिफंड की समय-सीमा:
   - यूपीआई (UPI) और वॉलेट भुगतान: 24 घंटे के भीतर रिफंड।
   - क्रेडिट/डेबिट कार्ड या नेट बैंकिंग: 5 से 7 कार्यदिवस (Business Days)।
4. क्षतिग्रस्त उत्पाद या रिटर्न: यदि पार्सल क्षतिग्रस्त है या गलत सामान मिला है, तो 7 दिनों के भीतर चैट में फोटो अपलोड करके मुफ्त रिप्लेसमेंट प्राप्त करें।"

Audio Prompt ID: IVR-PROMPT-ORD-02
English Voice & Chat Rules:
"Shipment Tracking, Cancellation, and Refund Terms:
- Tracking: Type 'TRACK <Order_ID>' in chat or press 3 on the IVR. Real-time courier partner tracking and OTP details are sent via SMS.
- Free Cancellation: You can cancel orders free of charge at any moment prior to dispatch.
- Refund SLAs:
  - UPI & Digital Wallets: Within 24 hours.
  - Credit/Debit Cards & NetBanking: 5 to 7 business days depending on the card-issuing bank.
- Returns: 7-day hassle-free return window for damaged, defective, or incorrect items with zero pickup fees."
"""
    }
]

IVR_TENANT_NAME = "tenant-ivr-support"
IVR_API_KEY_PRESET = "sk-live-tenant-ivr-support-key-101"


async def seed_ivr_bilingual_data():
    print("\n" + "=" * 70)
    print("TRAINING AI PLATFORM WITH BILINGUAL IVR CALL & CHAT SUPPORT CHUNKS")
    print("=" * 70)

    # 1. Initialize Database
    await init_db()
    print("✓ PostgreSQL 16 + pgvector database initialized.")

    # 2. Initialize Provider Registry with Google Embeddings
    provider_registry.initialize_from_file("config/providers.yaml")
    emb_adapter = provider_registry.get_embedding_adapter("google-embeddings")
    print(f"✓ Google Embedding Adapter loaded: {emb_adapter.__class__.__name__} (Dimension: {emb_adapter.dimension})")

    async with async_session() as session:
        tenant_repo = TenantRepository(session)
        api_key_repo = ApiKeyRepository(session)
        doc_repo = DocumentRepository(session)
        emb_service = EmbeddingService(provider_id="google-embeddings")
        ingestion_service = DocumentIngestionService(doc_repo, emb_service)

        # 3. Create or Fetch IVR Tenant
        tenant = await tenant_repo.get_by_name(IVR_TENANT_NAME)
        if not tenant:
            tenant = await tenant_repo.create(IVR_TENANT_NAME)
            print(f"✓ Created new tenant: {tenant.name} (ID: {tenant.id})")
        else:
            print(f"✓ Found existing tenant: {tenant.name} (ID: {tenant.id})")

        # 4. Ensure Tenant API Key exists
        key_hash = hash_api_key(IVR_API_KEY_PRESET)
        existing_key = await api_key_repo.get_by_hash(key_hash)
        if not existing_key:
            await api_key_repo.create(
                tenant_id=tenant.id,
                key_hash=key_hash,
                name="IVR Bilingual Support Key",
                scopes="*",
            )
            print(f"✓ Created API Key for {tenant.name}: {IVR_API_KEY_PRESET}")
        else:
            print(f"✓ API Key verified: {IVR_API_KEY_PRESET}")

        # 5. Ingest all Bilingual IVR Documents
        print("\nIngesting Knowledge Chunks using Google Embeddings:")
        total_chunks = 0
        for doc_item in IVR_BILINGUAL_DOCUMENTS:
            doc = await ingestion_service.ingest_document(
                tenant_id=tenant.id,
                title=doc_item["title"],
                content=doc_item["content"],
                source_type=doc_item["source_type"],
            )
            total_chunks += doc.total_chunks
            print(f"  + Ingested '{doc.title}' -> {doc.total_chunks} chunks vectorized.")

        print(f"\n✓ Successfully vectorized and persisted {total_chunks} IVR chunks in pgvector.")

        # 6. Test RAG Retrieval in Hindi and English
        search_service = SemanticSearchService(doc_repo, emb_service)
        rag_pipeline = RAGPipeline(search_service=search_service)

        print("\n" + "-" * 70)
        print("TESTING RETRIEVAL IN HINDI & ENGLISH")
        print("-" * 70)

        test_queries = [
            ("Hindi UPI Query", "UPI में पैसे कट गए पर ऑर्डर कन्फर्म नहीं हुआ, रिफंड कब तक आएगा?"),
            ("English Escalation Query", "How can I transfer my call to speak to a live customer care agent?"),
            ("Hindi Menu Query", "IVR के मुख्य मेनू में कौन कौन से विकल्प उपलब्ध हैं?"),
            ("English KYC Query", "What documents are required to complete digital Video KYC?"),
        ]

        for label, query in test_queries:
            context = await rag_pipeline.retrieve_context(
                tenant_id=tenant.id,
                query=query,
                top_k=2,
                min_similarity_score=0.25,
            )
            print(f"\n[Query: {label}] \"{query}\"")
            if context:
                first_line = context.split("\n")[1] if len(context.split("\n")) > 1 else context[:100]
                print(f"  ✓ Retrieved Relevant Context: {first_line[:120]}...")
            else:
                print("  ✗ Warning: No context retrieved.")

    print("\n" + "=" * 70)
    print("IVR TRAINING COMPLETE - KNOWLEDGE BASE READY FOR TEXT & AUDIO")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(seed_ivr_bilingual_data())
