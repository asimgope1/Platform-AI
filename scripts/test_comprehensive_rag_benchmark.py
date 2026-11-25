#!/usr/bin/env python3
"""
Comprehensive RAG Benchmark Suite for Jana Seva Odisha
Tests all 9 services, administrative info, legal rules, and multilingual queries (EN, HI, OR).
Evaluates both retrieval precision/confidence and conversational synthesis.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.knowledge.retrieval_service import HybridRetrievalService
from providers.chat.ivr_mini_model import ivr_mini_brain

TENANT_ID = "879e9bda-8049-4824-9aa0-fc5e9f0012d9"

BENCHMARK_CASES = [
    # ── Category 1: Subhadra Scheme ──────────────────────────────────────────
    {
        "id": "EN-01",
        "category": "Subhadra Scheme",
        "lang": "en",
        "query": "What is the financial benefit and total amount provided under the Subhadra Scheme?",
        "expected_facts": ["50,000", "financial assistance"],
    },
    {
        "id": "EN-02",
        "category": "Subhadra Scheme",
        "lang": "en",
        "query": "What are the age criteria and income limit for Subhadra Scheme eligibility?",
        "expected_facts": ["18 to 60", "2.5 lakh"],
    },
    {
        "id": "EN-03",
        "category": "Subhadra Scheme",
        "lang": "en",
        "query": "What kind of bank account is required for receiving Subhadra DBT payments?",
        "expected_facts": ["single-holder", "aadhaar-seeded", "dbt"],
    },
    {
        "id": "EN-04",
        "category": "Subhadra Scheme",
        "lang": "en",
        "query": "Is there any application fee charged by the government for Subhadra Scheme?",
        "expected_facts": ["free of cost", "zero fee"],
    },

    # ── Category 2: Legal Documents & Stamp Papers ───────────────────────────
    {
        "id": "EN-05",
        "category": "Legal Documents",
        "lang": "en",
        "query": "What is the key difference between notarisation and registration of documents?",
        "expected_facts": ["notarisation", "registration"],
    },
    {
        "id": "EN-06",
        "category": "Legal Documents",
        "lang": "en",
        "query": "Which legal instruments can Jana Seva Kendra draft and prepare?",
        "expected_facts": ["affidavits", "agreements"],
    },

    # ── Category 3: Certificates (Caste, Income, Residence, Bonafide) ────────
    {
        "id": "EN-07",
        "category": "Caste Certificate",
        "lang": "en",
        "query": "Who is the competent authority to issue a Caste Certificate in Odisha?",
        "expected_facts": ["tahasildar", "revenue"],
    },
    {
        "id": "EN-08",
        "category": "Caste Certificate",
        "lang": "en",
        "query": "What documents are required to apply for a Caste Certificate?",
        "expected_facts": ["ror", "caste certificate"],
    },
    {
        "id": "EN-09",
        "category": "Caste Certificate",
        "lang": "en",
        "query": "Can a person from General or unreserved category obtain a Caste Certificate?",
        "expected_facts": ["general category", "reserved"],
    },
    {
        "id": "EN-10",
        "category": "Income Certificate",
        "lang": "en",
        "query": "What is the validity period of an Income Certificate in Odisha?",
        "expected_facts": ["3 years", "three financial years"],
    },
    {
        "id": "EN-11",
        "category": "Residence Certificate",
        "lang": "en",
        "query": "What documents can be submitted as address proof for a Residence Certificate?",
        "expected_facts": ["ror", "electricity bill"],
    },
    {
        "id": "EN-12",
        "category": "Bonafide Certificate",
        "lang": "en",
        "query": "Who issues a Bonafide Certificate and what is its purpose?",
        "expected_facts": ["educational institution", "school"],
    },

    # ── Category 4: Identity & Cards (Aadhaar, PAN, Voter ID) ────────────────
    {
        "id": "EN-13",
        "category": "Aadhaar Services",
        "lang": "en",
        "query": "Can biometric updates like fingerprint and iris scans be done online for Aadhaar?",
        "expected_facts": ["in-person", "authorized aadhaar center"],
    },
    {
        "id": "EN-14",
        "category": "PAN Card Services",
        "lang": "en",
        "query": "Can a person legally possess two PAN cards in India?",
        "expected_facts": ["illegal", "section 272b"],
    },
    {
        "id": "EN-15",
        "category": "PAN Card Services",
        "lang": "en",
        "query": "What is the penalty under Income Tax Act for possessing more than one PAN card?",
        "expected_facts": ["10,000", "penalty"],
    },
    {
        "id": "EN-16",
        "category": "Voter ID Services",
        "lang": "en",
        "query": "Which form should be submitted to apply for a new Voter ID card?",
        "expected_facts": ["form 6"],
    },
    {
        "id": "EN-17",
        "category": "Voter ID Services",
        "lang": "en",
        "query": "Which form is used for correction of name, address, or photo in Voter ID?",
        "expected_facts": ["form 8"],
    },

    # ── Category 5: Kendra Administration, Timings & Contact ────────────────
    {
        "id": "EN-18",
        "category": "Kendra Contact",
        "lang": "en",
        "query": "What is the exact physical office address of Jana Seva Kendra?",
        "expected_facts": ["kalabuda", "kendrapara"],
    },
    {
        "id": "EN-19",
        "category": "Kendra Contact",
        "lang": "en",
        "query": "What is the contact telephone number of Jana Seva Kendra?",
        "expected_facts": ["1234567890"],
    },
    {
        "id": "EN-20",
        "category": "Kendra Contact",
        "lang": "en",
        "query": "What is the official email address for citizen support?",
        "expected_facts": ["support@janasevaodisha.com"],
    },
    {
        "id": "EN-21",
        "category": "Kendra Timings",
        "lang": "en",
        "query": "What are the working days and operating hours of Jana Seva Kendra?",
        "expected_facts": ["monday to saturday", "9:00 am to 6:00 pm"],
    },

    # ── Category 6: Multilingual Indic Evaluation (Hindi & Odia) ─────────────
    {
        "id": "HI-01",
        "category": "Subhadra (Hindi)",
        "lang": "hi",
        "query": "सुभद्रा योजना में कुल कितनी आर्थिक सहायता मिलती है?",
        "expected_facts": ["50,000"],
    },
    {
        "id": "HI-02",
        "category": "Kendra Phone (Hindi)",
        "lang": "hi",
        "query": "जन सेवा केंद्र का संपर्क नंबर क्या है?",
        "expected_facts": ["1234567890"],
    },
    {
        "id": "HI-03",
        "category": "Timings (Hindi)",
        "lang": "hi",
        "query": "जन सेवा केंद्र खुलने और बंद होने का समय क्या है?",
        "expected_facts": ["9:00", "6:00"],
    },
    {
        "id": "OR-01",
        "category": "Subhadra (Odia)",
        "lang": "or",
        "query": "ସୁଭଦ୍ରା ଯୋଜନାରେ କେତେ ଟଙ୍କା ମିଳେ?",
        "expected_facts": ["୫୦,୦୦୦", "50,000", "ସହାୟତା"],
    },
    {
        "id": "OR-02",
        "category": "Kendra Phone (Odia)",
        "lang": "or",
        "query": "ଜନ ସେବା କେନ୍ଦ୍ରର ଫୋନ ନମ୍ବର କଣ?",
        "expected_facts": ["1234567890", "+91"],
    },
    {
        "id": "OR-03",
        "category": "Timings (Odia)",
        "lang": "or",
        "query": "ଜନ ସେବା କେନ୍ଦ୍ର କେଉଁ ସମୟରେ ଖୋଲା ରହେ?",
        "expected_facts": ["9:00", "6:00", "ସୋମବାର"],
    },
]


async def run_benchmark():
    print("=" * 78)
    print("RUNNING COMPREHENSIVE RAG BENCHMARK EVALUATION (27 Test Cases)")
    print("=" * 78)

    retriever = HybridRetrievalService()

    passed = 0
    failed = 0

    for case in BENCHMARK_CASES:
        cid = case["id"]
        cat = case["category"]
        lang = case["lang"]
        query = case["query"]
        expected = case["expected_facts"]

        # 1. Retrieve Knowledge Chunks
        ret_ctx = await retriever.search(query=query, tenant_id=TENANT_ID, top_k=4)
        top_score = ret_ctx.evaluation.top_score
        tier = ret_ctx.evaluation.tier.value

        # Build context string
        context_str = "\n".join(ret_ctx.chunk_texts)

        # 2. Synthesize Audio-Ready Answer
        ans = ivr_mini_brain._synthesize_from_rag_chunks(context_str, query.lower(), lang)

        # 3. Check assertions
        ans_lower = (ans or "").lower()
        matched_facts = [f for f in expected if f.lower() in ans_lower]
        ok = len(matched_facts) > 0 and top_score >= 0.55

        if ok:
            passed += 1
            print(f"[{cid}] PASS | Score: {top_score:.2f} ({tier:<8}) | {cat}")
            print(f"       Q: {query}")
            print(f"       A: {ans}")
        else:
            failed += 1
            print(f"[{cid}] FAIL | Score: {top_score:.2f} ({tier:<8}) | {cat}")
            print(f"       Q: {query}")
            print(f"       A: {ans}")
            print(f"       Expected any of: {expected}")
        print("-" * 78)

    total = len(BENCHMARK_CASES)
    pass_pct = (passed / total) * 100.0
    print("\n" + "=" * 78)
    print(f"BENCHMARK RESULTS: {passed}/{total} Passed ({pass_pct:.1f}%)")
    print("=" * 78)
    assert failed == 0, f"{failed} test case(s) failed the benchmark!"


if __name__ == "__main__":
    asyncio.run(run_benchmark())
