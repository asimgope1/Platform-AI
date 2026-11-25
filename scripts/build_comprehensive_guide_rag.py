#!/usr/bin/env python3
"""
Comprehensive Knowledge Generator & Ingestion Engine for Jana Seva Odisha
==========================================================================
Generates high-density, atomic, multi-perspective knowledge chunks covering:
- Every service (Subhadra Scheme, Legal Documents, Caste, Income, Residence,
  Aadhaar, PAN, Bonafide, Voter ID, Tools & Utilities, Banking, Contact, Disclaimers)
- Explicit bilingual / trilingual Q&A pairs matching caller conversational speech
- Persists all vector embeddings (all-MiniLM-L6-v2 384d) in PostgreSQL pgvector and SQLite.
"""
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.embedding_service import EmbeddingService
from storage.database import async_session, init_db
from storage.models.document import Document, DocumentChunk
from storage.repositories.document_repo import DocumentRepository
from sqlalchemy import select, delete

TENANT_ID = "879e9bda-8049-4824-9aa0-fc5e9f0012d9"
SOURCE_NAME = "Jana_Seva_Odisha_Service_Guide.pdf"

# ──────────────────────────────────────────────────────────────────────────────
# ATOMIC KNOWLEDGE UNITS & Q&A PAIRS
# ──────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_ENTRIES: List[Dict[str, Any]] = [
    # ── 1. Contact & Center Directory ──────────────────────────────────────────
    {
        "topic": "kendra_contact_details",
        "title": "Jana Seva Kendra - Official Address, Phone Number, Email, and Website",
        "category": "kendra_helpdesk",
        "content": (
            "Official Contact Details for Jana Seva Kendra Odisha:\n"
            "• Physical Address: Jana Seva Kendra, Kalabuda, Kendrapara, Odisha, PIN 754153.\n"
            "• Helpline Phone Number: +91 1234567890.\n"
            "• Official Email: support@janasevaodisha.com.\n"
            "• Official Website: https://janasevaodisha.com/services.\n"
            "Citizens can visit the Kendra in person at Kalabuda, Kendrapara for all government service applications, "
            "document uploads, biometric appointments, and grievance tracking."
        ),
        "qas": [
            ("What is the phone number of Jana Seva Kendra?", "The helpline phone number of Jana Seva Kendra is +91 1234567890."),
            ("Where is Jana Seva Kendra located in Odisha?", "Jana Seva Kendra is located at Kalabuda, Kendrapara, Odisha, PIN 754153."),
            ("What is the address of Jana Seva Kendra?", "The official address is Jana Seva Kendra, Kalabuda, Kendrapara, Odisha 754153."),
            ("What is the email address of Jana Seva Kendra?", "The official email address is support@janasevaodisha.com."),
            ("What is the website of Jana Seva Odisha?", "The official website is https://janasevaodisha.com."),
            ("ಜನ ಸೇವಾ ಕೇಂದ್ರದ ಫೋನ್ ನಂಬರ್", "ಜನ ಸೇವಾ ಕೇಂದ್ರದ ಫೋನ್ ನಂಬರ್ +91 1234567890."),
            ("ଜନ ସେବା କେନ୍ଦ୍ରର ଫୋନ ନମ୍ବର କଣ?", "ଜନ ସେବା କେନ୍ଦ୍ରର ହେଲ୍ପଲାଇନ ଫୋନ ନମ୍ବର ହେଉଛି +91 1234567890 ଏବଂ ଠିକଣା ହେଉଛି କଳାବୁଦା, କେନ୍ଦ୍ରାପଡ଼ା ୭୫୪୧୫୩।"),
            ("जन सेवा केंद्र का फोन नंबर और पता क्या है?", "जन सेवा केंद्र का हेल्पलाइन नंबर +91 1234567890 है और पता कलाबुदा, केंद्रापड़ा, ओडिशा 754153 है।"),
        ]
    },
    {
        "topic": "kendra_working_hours",
        "title": "Jana Seva Kendra - Operating Hours, Timings, and Working Days",
        "category": "kendra_helpdesk",
        "content": (
            "Operating Timings & Working Hours for Jana Seva Kendra Odisha:\n"
            "• Working Days: Monday to Saturday.\n"
            "• Working Hours: 9:00 AM to 6:00 PM.\n"
            "• Weekly Holiday: Closed on Sundays and gazetted government holidays.\n"
            "Citizens can visit during working hours for Subhadra Scheme assistance, caste, income, residence certificates, "
            "Aadhaar, PAN card, voter ID, and legal document preparation."
        ),
        "qas": [
            ("What are the timings of Jana Seva Kendra?", "Jana Seva Kendra is open Monday through Saturday from 9:00 AM to 6:00 PM. It is closed on Sundays and gazetted holidays."),
            ("Is Jana Seva Kendra open on Sunday?", "No, Jana Seva Kendra is closed on Sundays and gazetted public holidays."),
            ("जन सेवा केंद्र का समय क्या है?", "जन सेवा केंद्र सोमवार से शनिवार सुबह 9:00 बजे से शाम 6:00 बजे तक खुला रहता है। रविवार को बंद रहता है।"),
            ("ଜନ ସେବା କେନ୍ଦ୍ର କେତେବେଳେ ଖୋଲେ?", "ଜନ ସେବା କେନ୍ଦ୍ର ସୋମବାରରୁ ଶନିବାର ସକାଳ ୯:୦୦ ରୁ ସନ୍ଧ୍ୟା ୬:୦୦ ପର୍ଯ୍ୟନ୍ତ ଖୋଲା ରହିଥାଏ। ରବିବାର ବନ୍ଦ ରହେ।"),
        ]
    },
    {
        "topic": "organisation_status_disclaimer",
        "title": "Jana Seva Odisha - Organisational Status, Scope, and Government Disclaimer",
        "category": "kendra_helpdesk",
        "content": (
            "Organisational Status and Legal Disclaimer for Jana Seva Odisha:\n"
            "• Independent Facilitation Service: Jana Seva Odisha is an independent service provider and is NOT a government organisation or directly connected with the Government of India or the Government of Odisha.\n"
            "• Role: Helping citizens access online government portals, prepare applications, and process documentation.\n"
            "• Issuing Authority: A service centre only facilitates applications. Official approval, issuance of government certificates, and welfare benefit disbursements are strictly executed by the competent government authority (such as Tahasil, UIDAI, Income Tax Department, or Election Commission of India)."
        ),
        "qas": [
            ("Is Jana Seva Odisha a government organisation?", "No, Jana Seva Odisha is an independent private facilitation centre, not a government body. It assists citizens with applications, while official certificates are issued solely by competent government authorities."),
            ("Can Jana Seva Kendra approve my certificate?", "No, Jana Seva Kendra only assists with application submission. Official certificates can only be verified and approved by government revenue authorities such as the Tahasildar."),
            ("क्या जन सेवा केंद्र सरकारी विभाग है?", "नहीं, जन सेवा केंद्र एक स्वतंत्र नागरिक सुविधा केंद्र है। यह आवेदन में सहायता करता है, जबकि प्रमाण पत्र जारी करने का अधिकार केवल सरकारी अधिकारियों (जैसे तहसीलदार) के पास है।"),
        ]
    },
    {
        "topic": "website_tools_and_utilities",
        "title": "Jana Seva Odisha - Free Digital Document Tools (PDF & Photo Tools)",
        "category": "citizen_services",
        "content": (
            "Tools and Utilities on Jana Seva Odisha Website:\n"
            "The website provides digital document utility tools to assist citizens with online applications:\n"
            "1. JPG to PDF: Converts image scans into standard PDF files required for portal uploads.\n"
            "2. Word to PDF: Converts editable documents into PDF format.\n"
            "3. Resize Photos: Compresses passport-size photos to match government file size limits (e.g. under 50 KB or 100 KB).\n"
            "These tools are document conversion utilities, not certificate-issuing services."
        ),
        "qas": [
            ("What tools are available on the Jana Seva website?", "The website offers document utilities including JPG to PDF, Word to PDF, and photo resizing tools for government application uploads."),
            ("Can I resize my photo on the Jana Seva website?", "Yes, the website has a Resize Photos tool to compress passport-size photographs to meet government portal upload requirements."),
        ]
    },
    {
        "topic": "banking_service_guidelines",
        "title": "Jana Seva Odisha - Banking Service Guidelines and Limitations",
        "category": "citizen_services",
        "content": (
            "Banking Service Information for Jana Seva Odisha:\n"
            "• The site navigation references Banking Service as a facilitation service.\n"
            "• Citizens and operators must verify the actual service menu, bank Business Correspondent (BC) authorisation, transaction limits, charges, and receipt process before transacting.\n"
            "• Safety Rule: Never share ATM PINs, OTPs, or net banking passwords with any service centre operator or intermediary. Always obtain an official transaction slip or receipt for cash deposits or withdrawals."
        ),
        "qas": [
            ("What banking services are provided at Jana Seva Kendra?", "The Kendra offers Business Correspondent banking assistance. Always verify the BC authorisation, transaction limits, and obtain official receipts. Never share your ATM PIN or OTP with anyone."),
        ]
    },
    {
        "topic": "citizen_preparation_checklist",
        "title": "General Citizen Preparation Checklist Before Visiting Jana Seva Kendra",
        "category": "citizen_services",
        "content": (
            "Citizen Preparation Checklist Before Visiting Jana Seva Kendra:\n"
            "1. Original Documents: Carry original documents for inspection along with self-attested photocopies.\n"
            "2. Mobile Linked with Aadhaar: Keep the mobile phone linked to Aadhaar active and accessible for OTP authentication.\n"
            "3. Name and DOB Consistency: Ensure consistent spelling of your full name, father's name, and date of birth across all identity documents.\n"
            "4. Receipts and Reference Numbers: Always request and retain the official application acknowledgement slip and fee receipt.\n"
            "5. Confidentiality: Never disclose OTPs, passwords, or bank PINs to any intermediary.\n"
            "6. Realistic Expectations: Do not assume guaranteed approval or fixed processing times; all approvals depend on statutory government verification."
        ),
        "qas": [
            ("What should I bring when visiting Jana Seva Kendra?", "Bring original documents for verification, self-attested photocopies, your Aadhaar-linked mobile phone for OTP verification, and passport-size photographs."),
            ("Can I share my OTP with the Kendra operator?", "No, never share sensitive banking PINs, passwords, or unauthorized OTPs. Only authorize official e-KYC prompts displayed in your presence."),
        ]
    },

    # ── 2. Subhadra Scheme ─────────────────────────────────────────────────────
    {
        "topic": "subhadra_scheme_overview",
        "title": "Subhadra Scheme - Financial Assistance Overview and ₹50,000 Benefit",
        "category": "subhadra_scheme",
        "content": (
            "Subhadra Scheme Overview & Financial Assistance:\n"
            "• Programme: Government of Odisha flagship financial-assistance scheme aimed at supporting women's financial independence and empowerment.\n"
            "• Financial Benefit: Total financial assistance of up to ₹50,000 per eligible beneficiary, disbursed directly into the beneficiary's Aadhaar-seeded bank account through Direct Benefit Transfer (DBT).\n"
            "• Note: The statement of 'up to ₹50,000' is an indicative scheme benefit, not an unconditional guarantee of payment. Disbursal depends on approval under official scheme guidelines."
        ),
        "qas": [
            ("What is Subhadra Scheme?", "Subhadra Scheme is an Odisha Government financial assistance initiative providing eligible women up to ₹50,000 via Direct Benefit Transfer (DBT) to support their financial independence."),
            ("Tell me about Subhadra Scheme", "Subhadra Scheme is an Odisha Government financial assistance initiative providing eligible women up to ₹50,000 via Direct Benefit Transfer (DBT) to support their financial independence."),
            ("Tell me about Subhadra team", "Subhadra Scheme is an Odisha Government financial assistance initiative providing eligible women up to ₹50,000 via Direct Benefit Transfer (DBT) to support their financial independence."),
            ("How much money is given under Subhadra Yojana?", "Under Subhadra Yojana, eligible beneficiaries receive financial assistance of up to ₹50,000 disbursed directly into their bank account."),
            ("How much money can be given in Subhadra Scheme?", "Under Subhadra Scheme, eligible women receive total financial assistance of up to ₹50,000 disbursed directly into their bank account via Direct Benefit Transfer (DBT)."),
            ("How much money will I get in Subhadra?", "Eligible beneficiaries receive total financial assistance of up to ₹50,000 disbursed directly into their Aadhaar-seeded bank account."),
            ("सुभद्रा योजना क्या है और इसमें कितना पैसा मिलता है?", "सुभद्रा योजना ओडिशा सरकार की योजना है जिसमें पात्र महिलाओं को ₹50,000 तक की वित्तीय सहायता डीबीटी के माध्यम से सीधे बैंक खाते में दी जाती है।"),
            ("ସୁଭଦ୍ରା ଯୋଜନାରେ କେତେ ଟଙ୍କା ମିଳେ?", "ସୁଭଦ୍ରା ଯୋଜନାରେ ଓଡ଼ିଶାର ଯୋଗ୍ୟ ମହିଳାମାନଙ୍କୁ ସର୍ବାଧିକ ୫୦,୦୦୦ ଟଙ୍କା ପର୍ଯ୍ୟନ୍ତ ସହାୟତା ସିଧାସଳଖ ବ୍ୟାଙ୍କ ଖାତାକୁ ପ୍ରଦାନ କରାଯାଏ।"),
        ]
    },
    {
        "topic": "subhadra_scheme_eligibility",
        "title": "Subhadra Scheme - Eligibility Criteria, Age Limits, and Income Cap",
        "category": "subhadra_scheme",
        "content": (
            "Subhadra Scheme Eligibility Criteria:\n"
            "1. Domicile: Applicant must be a permanent resident of Odisha.\n"
            "2. Gender: Exclusively for women applicants.\n"
            "3. Age Bracket: Women aged between 18 and 60 years.\n"
            "4. Family Income Limit: Annual family household income must be strictly below ₹2,50,000 (2.5 Lakh rupees).\n"
            "5. Banking & Aadhaar: Must have an individual bank account seeded with Aadhaar for DBT transfers.\n"
            "Exclusion: Women from families with annual income exceeding ₹2.5 Lakh, or residents of other states, are not eligible."
        ),
        "qas": [
            ("Who is eligible for Subhadra Scheme?", "Eligible applicants are women permanent residents of Odisha aged 18 to 60 years whose annual family income is below ₹2,50,000 (2.5 Lakh) and who have an Aadhaar-linked bank account."),
            ("Can government employees apply for Subhadra Scheme?", "No. Women from families with regular government employees or income tax payees are excluded from Subhadra Scheme benefits."),
            ("Can unmarried or single women apply for Subhadra Scheme?", "Yes, all eligible women permanent residents of Odisha aged 18 to 60 who meet the income criteria can apply."),
            ("What is the age limit for Subhadra Yojana?", "The age limit for Subhadra Yojana is 18 to 60 years."),
            ("What is the income limit for Subhadra Scheme?", "The annual family income must be below ₹2,50,000 (2.5 Lakh)."),
            ("Can a person from outside Odisha apply for Subhadra?", "No, the applicant must be a permanent resident of Odisha."),
            ("सुभद्रा योजना के लिए उम्र और आय सीमा क्या है?", "सुभद्रा योजना के लिए महिला की उम्र 18 से 60 वर्ष और वार्षिक पारिवारिक आय 2.5 लाख रुपये से कम होनी चाहिए।"),
            ("ସୁଭଦ୍ରା ଯୋଜନା ପାଇଁ ବୟସ ଏବଂ ଆୟ ସୀମା କେତେ?", "ସୁଭଦ୍ରା ଯୋଜନା ପାଇଁ ବୟସ ୧୮ ରୁ ୬୦ ବର୍ଷ ଏବଂ ପାରିବାରିକ ବାର୍ଷିକ ଆୟ ୨.୫ ଲକ୍ଷ ଟଙ୍କାରୁ କମ୍ ହେବା ଆବଶ୍ୟକ।"),
        ]
    },
    {
        "topic": "subhadra_scheme_documents",
        "title": "Subhadra Scheme - Complete Document Checklist and Application Steps",
        "category": "subhadra_scheme",
        "content": (
            "Subhadra Scheme Document Checklist and Application Steps:\n"
            "Required Documents:\n"
            "1. Aadhaar Card (with mobile number linked for OTP verification).\n"
            "2. Income Certificate (proving annual family income below ₹2.5 Lakh).\n"
            "3. Residence / Domicile Certificate (proving permanent Odisha residency).\n"
            "4. Bank Account Details (bank passbook or statement with clear IFSC and account number, seeded with Aadhaar).\n"
            "5. Recent Passport-size Photograph.\n"
            "Application Steps:\n"
            "• Step 1: Visit nearest Jana Seva Kendra or apply online via official Subhadra portal.\n"
            "• Step 2: Submit required documents and complete biometric/OTP e-KYC.\n"
            "• Step 3: Application verified by local revenue and welfare authorities.\n"
            "• Step 4: DBT benefit released upon official approval."
        ),
        "qas": [
            ("What documents are required for Subhadra Scheme?", "The required documents are: 1) Aadhaar card, 2) Income certificate, 3) Residence certificate, 4) Bank passbook details (Aadhaar-seeded), and 5) Passport-size photograph."),
            ("What kind of bank account is required for receiving Subhadra DBT payments?", "For Subhadra Scheme, the applicant must have an individual single-holder bank account seeded with Aadhaar and enabled for Direct Benefit Transfer (DBT)."),
            ("Is there any application fee charged by the government for Subhadra Scheme?", "No. Subhadra Scheme application forms and submission are completely free of cost (zero fee). No government fee is charged."),
            ("Where can I submit the Subhadra application form?", "You can submit the Subhadra application form at your nearest Jana Seva Kendra, Mo Seva Kendra, or local Anganwadi centre completely free of cost."),
            ("How to apply for Subhadra Scheme in Odisha?", "Visit your nearest Jana Seva Kendra or the official Subhadra portal with your Aadhaar, income certificate, residence certificate, bank passbook, and photo to submit your application."),
            ("सुभद्रा योजना के लिए कौन से दस्तावेज चाहिए?", "सुभद्रा योजना के लिए आधार कार्ड, आय प्रमाण पत्र (2.5 लाख से कम), निवास प्रमाण पत्र, बैंक पासबुक और पासपोर्ट साइज फोटो चाहिए।"),
            ("ସୁଭଦ୍ରା ଯୋଜନା ପାଇଁ କ'ଣ ଡକ୍ୟୁମେଣ୍ଟ ଦରକାର?", "ସୁଭଦ୍ରା ଯୋଜନା ପାଇଁ ଆଧାର କାର୍ଡ, ଆୟ ପ୍ରମାଣପତ୍ର, ବାସସ୍ଥାନ ପ୍ରମାଣପତ୍ର, ବ୍ୟାଙ୍କ ପାସବୁକ୍ ଏବଂ ଫଟୋ ଆବଶ୍ୟକ।"),
        ]
    },

    # ── 3. Legal Documents ─────────────────────────────────────────────────────
    {
        "topic": "legal_documents_overview",
        "title": "Legal Documents - Affidavits, Agreements, Power of Attorney, Wills, and Notary",
        "category": "legal_documents",
        "content": (
            "Legal Documents Services at Jana Seva Odisha:\n"
            "• Services Offered: Drafting and execution assistance for affidavits, agreements, general/special power of attorney (POA), wills and testaments, deed drafting, stamp paper procurement, and legal notarisation.\n"
            "• Eligibility: The executing parties must have legal capacity to contract, be of sound mind, and have all required consenting parties available.\n"
            "• Documents Required: Government photo identity proof and address proof of all executing parties, property/relationship title records where relevant, stamp paper of appropriate denomination, photographs, and identity proofs of two independent witnesses."
        ),
        "qas": [
            ("What legal documents can Jana Seva Kendra assist with?", "Jana Seva Kendra assists with name change affidavits, income affidavits, rental agreements, power of attorney (POA), wills, and notary services."),
            ("What documents are needed for an affidavit or agreement?", "You need government photo ID and address proof for all parties, correct draft details, stamp paper, and witness identification."),
            ("क्या जन सेवा केंद्र में एफिडेविट और एग्रीमेंट बनता है?", "हाँ, जन सेवा केंद्र में नाम परिवर्तन शपथ पत्र (Affidavit), एग्रीमेंट, पावर ऑफ अटॉर्नी और नोटरी कार्य में सहायता दी जाती है।"),
        ]
    },
    {
        "topic": "notarisation_vs_registration",
        "title": "Legal Documents - Distinction Between Notarisation and Mandatory Registration",
        "category": "legal_documents",
        "content": (
            "Important Distinction: Notarisation vs. Statutory Registration:\n"
            "• Notarisation does NOT automatically substitute for legal registration where registration is compulsory by law.\n"
            "• Documents like property sales, gift deeds, mortgage deeds, and leases exceeding 11 months require formal registration at the Sub-Registrar office with applicable stamp duty.\n"
            "• A notary stamp only authenticates the identity and signature of the person executing the document; it does not register title. For high-value property or corporate transactions, consult a qualified advocate."
        ),
        "qas": [
            ("Does notarisation replace registration of a legal document?", "No, notarisation does not replace legal registration where registration is compulsory by law, such as for property sale deeds or leases over 11 months."),
            ("What is the difference between notary and registration?", "A notary verifies the signer's identity and signature on an affidavit or agreement, whereas statutory registration at the Sub-Registrar office legally registers title and ownership."),
        ]
    },

    # ── 4. Caste Certificate ───────────────────────────────────────────────────
    {
        "topic": "caste_certificate_overview",
        "title": "Caste Certificate - SC, ST, OBC, SEBC Categories and Eligibility",
        "category": "certificates",
        "content": (
            "Caste Certificate Services in Odisha:\n"
            "• Purpose: Official proof of caste identity used to access educational admissions, scholarships, government recruitment quotas, and welfare schemes.\n"
            "• Recognized Categories: Scheduled Caste (SC), Scheduled Tribe (ST), Other Backward Classes (OBC), and Socially and Educationally Backward Classes (SEBC).\n"
            "• General Category Note: General category applicants do not receive a reserved caste certificate. The applicant must belong to a recognized community listed under Odisha or Central statutory schedules.\n"
            "• Authority: The local Tahasildar / Revenue Authority determines eligibility after field enquiry; the service centre only facilitates application filing."
        ),
        "qas": [
            ("What is Caste Certificate?", "A Caste Certificate is official proof of caste identity used to access educational admissions, scholarships, government recruitment quotas, and welfare schemes."),
            ("Tell me about Caste Certificate", "A Caste Certificate in Odisha provides official proof of belonging to SC, ST, OBC, or SEBC category, issued by the Tahasildar for education, jobs, and welfare benefits."),
            ("How long does it take to get a Caste Certificate?", "A Caste Certificate is typically issued within 15 to 30 days following local inquiry by the Revenue Inspector and approval by the Tahasildar."),
            ("Which categories can get a Caste Certificate?", "Caste certificates are issued to recognized reserved categories: SC (Scheduled Caste), ST (Scheduled Tribe), OBC (Other Backward Class), and SEBC (Socially and Educationally Backward Classes)."),
            ("Can General category applicants get a Caste Certificate?", "No, caste certificates are meant for recognized reserved categories (SC, ST, OBC, SEBC). General category individuals do not receive a reserved caste certificate."),
            ("जाति प्रमाण पत्र किन श्रेणियों के लिए बनता है?", "जाति प्रमाण पत्र SC, ST, OBC और SEBC वर्ग के नागरिकों के लिए बनता है। सामान्य वर्ग के लिए आरक्षित जाति प्रमाण पत्र नहीं बनता।"),
            ("ଜାତି ପ୍ରମାଣପତ୍ର କାହା ପାଇଁ ହୁଏ?", "ଜାତି ପ୍ରମାଣପତ୍ର SC, ST, OBC ଏବଂ SEBC ବର୍ଗର ଲୋକଙ୍କ ପାଇଁ ପ୍ରଦାନ କରାଯାଏ। ଜେନେରାଲ ବର୍ଗ ପାଇଁ ସଂରକ୍ଷିତ ଜାତି ପ୍ରମାଣପତ୍ର ହୁଏ ନାହିଁ।"),
        ]
    },
    {
        "topic": "caste_certificate_documents_procedure",
        "title": "Caste Certificate - Required Documents, RoR Land Records, and Tahasil Process",
        "category": "certificates",
        "content": (
            "Caste Certificate Document Checklist & Verification Steps:\n"
            "Required Documents:\n"
            "1. Aadhaar Card of the applicant.\n"
            "2. Residence / Domicile Certificate or accepted local residence proof.\n"
            "3. Father's or blood relative's caste certificate (if available).\n"
            "4. Birth Certificate or School Leaving Certificate (SLC).\n"
            "5. RoR Land Record (Record of Rights / Land Patta) establishing family lineage and ancestral residence in Odisha.\n"
            "6. Passport-size photograph and signature.\n"
            "Application Steps:\n"
            "• Submit online application via Odisha e-District portal or at Jana Seva Kendra.\n"
            "• Revenue Inspector (RI) conducts local enquiry and lineage scrutiny.\n"
            "• Tahasildar approves and issues the digitally signed certificate."
        ),
        "qas": [
            ("What documents are required for a Caste Certificate?", "To apply for a Caste Certificate, submit your Aadhaar card, residence proof, father's caste certificate (if available), birth certificate, RoR land record (patta), and passport-size photo."),
            ("Is land record (RoR) required for Caste Certificate?", "Yes, the RoR land record (patta) or ancestral revenue record is required by the Revenue Inspector to verify family lineage and local roots in Odisha."),
            ("जाति प्रमाण पत्र के लिए कौन से दस्तावेज चाहिए?", "आधार कार्ड, निवास प्रमाण पत्र, पिता का जाति प्रमाण पत्र, जन्म प्रमाण पत्र, जमीन का खतियान (RoR) और फोटो आवश्यक है।"),
            ("ଜାତି ପ୍ରମାଣପତ୍ର ପାଇଁ କ'ଣ ଦରକାର?", "ଆଧାର କାର୍ଡ, ବାସସ୍ଥାନ ପ୍ରମାଣ, ବାପାଙ୍କ ଜାତି ପ୍ରମାଣପତ୍ର, ଜନ୍ମ ପ୍ରମାଣପତ୍ର, ଜମି ପଟ୍ଟା (RoR) ଏବଂ ଫଟୋ ଆବଶ୍ୟକ।"),
        ]
    },

    # ── 5. Income Certificate ──────────────────────────────────────────────────
    {
        "topic": "income_certificate_details",
        "title": "Income Certificate - Proof of Annual Income, Salary/Agriculture Proof, and Verification",
        "category": "certificates",
        "content": (
            "Income Certificate Services in Odisha:\n"
            "• Purpose: Official certification of an individual's or family's annual income from all sources (salary, business, agriculture, pension). Required for scholarships, fee concessions, social security pensions, and Subhadra Scheme.\n"
            "• Required Documents:\n"
            "  1. Aadhaar Card.\n"
            "  2. Residence / Domicile Certificate.\n"
            "  3. Income Proof: Salary slip / Form 16 (for salaried), ITR copy (for business/taxpayers), land revenue receipt (for farmers), or pension passbook.\n"
            "  4. Self-declaration of annual household income.\n"
            "  5. Passport-size photograph and applicant's signature.\n"
            "• Issuance Process: Revenue Inspector (RI) conducts field inquiry to verify income before the Tahasildar approves and signs the certificate."
        ),
        "qas": [
            ("Tell me about Income Certificate", "An Income Certificate in Odisha certifies your family's annual income, is valid for 3 financial years, and is required for Subhadra Scheme, scholarships, and fee concessions."),
            ("How long does it take to get an Income Certificate?", "An Income Certificate is typically issued within 10 to 15 days following inquiry by the Revenue Inspector and Tahasildar approval."),
            ("What documents are needed for an Income Certificate?", "Required documents: Aadhaar card, residence certificate, salary slips or ITR or land revenue receipt, photo, and a self-declaration of annual family income."),
            ("What is the validity period of an Income Certificate in Odisha?", "In Odisha, an Income Certificate is typically valid for three financial years (3 years) from the date of issue."),
            ("Who issues the Income Certificate in Odisha?", "The local Tahasil / Revenue Authority issues the Income Certificate after inquiry and verification by the Revenue Inspector."),
            ("What is the use of an Income Certificate?", "It is used to prove family income for welfare schemes like Subhadra Yojana, college scholarships, school fee concessions, and government social benefits."),
            ("आय प्रमाण पत्र की वैधता कितनी होती है?", "ओडिशा में आय प्रमाण पत्र जारी होने की तारीख से 3 वित्तीय वर्षों (3 साल) के लिए मान्य होता है।"),
            ("ଆୟ ପ୍ରମାଣପତ୍ର କେତେ ବର୍ଷ ବୈଧ ରହେ?", "ଓଡ଼ିଶାରେ ଆୟ ପ୍ରମାଣପତ୍ର ଜାରି ହେବା ତାରିଖରୁ ୩ ଆର୍ଥିକ ବର୍ଷ (୩ ବର୍ଷ) ପର୍ଯ୍ୟନ୍ତ ବୈଧ ରହିଥାଏ।"),
            ("आय प्रमाण पत्र के लिए कौन से दस्तावेज चाहिए?", "आधार कार्ड, निवास प्रमाण पत्र, वेतन पर्ची (Salary Slip) या जमीन का लगान रसीद, आय की स्व-घोषणा और फोटो आवश्यक है।"),
            ("ଆୟ ପ୍ରମାଣପତ୍ର ପାଇଁ କ'ଣ ଡକ୍ୟୁମେଣ୍ଟ ଆବଶ୍ୟକ?", "ଆଧାର କାର୍ଡ, ବାସସ୍ଥାନ ପ୍ରମାଣପତ୍ର, ଦରମା ସ୍ଲିପ୍ ବା ଜମି ଖଜଣା ରସିଦ, ସ୍ୱ-ଘୋଷଣା ପତ୍ର ଏବଂ ଫଟୋ ଆବଶ୍ୟକ।"),
        ]
    },

    # ── 6. Residence / Domicile Certificate ────────────────────────────────────
    {
        "topic": "residence_certificate_details",
        "title": "Residence Certificate - Domicile Proof, Local Inquiry, and Required Documents",
        "category": "certificates",
        "content": (
            "Residence (Domicile) Certificate Services in Odisha:\n"
            "• Purpose: Confirms that an applicant is a permanent or continuous resident of Odisha. Mandatory for state education quotas, state government jobs, Subhadra Yojana, and local welfare schemes.\n"
            "• Required Documents:\n"
            "  1. Aadhaar Card or Voter ID.\n"
            "  2. Address Evidence: RoR land record (patta), house tax receipt, utility bill (electricity bill), or ration card.\n"
            "  3. Educational Records: School Leaving Certificate or marksheets showing continuous study in Odisha.\n"
            "  4. Passport-size photograph and self-declaration of residency.\n"
            "• Process: Submitted via e-District or Kendra; Revenue Inspector conducts local verification of continuous stay before Tahasildar issues certificate."
        ),
        "qas": [
            ("What documents are required for a Residence Certificate?", "Documents required: Aadhaar card, electricity bill or house tax receipt, RoR land record, school certificates showing study in Odisha, and a passport-size photo."),
            ("What documents can be submitted as address proof for a Residence Certificate?", "For Residence Certificate address proof, you can submit RoR land record (patta), electricity bill, house tax receipt, or holding tax receipt."),
            ("Why is a Residence Certificate needed?", "It proves permanent residency in Odisha, which is mandatory for state government jobs, Subhadra Scheme, and Odisha college admissions."),
            ("निवास प्रमाण पत्र के लिए क्या दस्तावेज चाहिए?", "आधार कार्ड, बिजली बिल, जमीन का खतियान (RoR) या हाउस टैक्स रसीद, शैक्षणिक प्रमाण पत्र और फोटो आवश्यक है।"),
            ("ବାସସ୍ଥାନ ପ୍ରମାଣପତ୍ର ପାଇଁ କ'ଣ ଦରକାର?", "ଆଧାର କାର୍ଡ, ବିଦ୍ୟୁତ ବିଲ୍, ଜମି ପଟ୍ଟା (RoR), ସ୍କୁଲ ସାର୍ଟିଫିକେଟ୍ ଏବଂ ଫଟୋ ଆବଶ୍ୟକ।"),
        ]
    },

    # ── 7. Aadhaar Card Services ───────────────────────────────────────────────
    {
        "topic": "aadhaar_card_services",
        "title": "Aadhaar Card Services - Enrolment, Demographic & Biometric Updates (UIDAI)",
        "category": "identity_cards",
        "content": (
            "Aadhaar Card Services (UIDAI Guidelines):\n"
            "• Services Offered: New Aadhaar enrolment, demographic updates (address, mobile number linking, name, date of birth), and e-Aadhaar download.\n"
            "• Biometric Update Rule: Biometric updates (fingerprints, iris scan, facial photograph) CANNOT be performed remotely or online. The resident must physically visit an authorized Aadhaar Seva Kendra.\n"
            "• Document Requirements:\n"
            "  - Proof of Identity (POI): PAN card, voter ID, passport, or driving licence.\n"
            "  - Proof of Address (POA): Utility bills, bank passbook, voter ID, or domicile certificate.\n"
            "  - Proof of Date of Birth (PDB): Birth certificate, matriculation marksheet, or passport.\n"
            "• Authority: Only UIDAI generates or modifies Aadhaar records."
        ),
        "qas": [
            ("Tell me about Aadhaar services", "Jana Seva Kendra assists with demographic updates, address change, mobile number assistance, and PVC card printing. Biometric updates require an authorized Aadhaar center."),
            ("Can Aadhaar biometrics be updated online or over the phone?", "No. Biometric updates (fingerprints, iris, photo) require physical in-person presence at an authorized Aadhaar Seva Kendra. They cannot be updated online or over the phone."),
            ("How to update mobile number in Aadhaar?", "Visit an authorized Aadhaar Seva Kendra in person with your original Aadhaar card for biometric verification to link or update your mobile number."),
            ("What documents are needed for new Aadhaar card?", "You need Proof of Identity (POI), Proof of Address (POA), Proof of Date of Birth (PDB), and relationship proof for minors."),
            ("आधार कार्ड में मोबाइल नंबर कैसे लिंक करें?", "मोबाइल नंबर लिंक करवाने के लिए मूल आधार कार्ड लेकर नजदीकी अधिकृत आधार सेवा केंद्र पर बायोमेट्रिक सत्यापन कराना होता है।"),
            ("ଆଧାର କାର୍ଡରେ ମୋବାଇଲ ନମ୍ବର କିପରି ଲିଙ୍କ୍ କରିବେ?", "ମୋବାଇଲ୍ ନମ୍ବର ଲିଙ୍କ୍ ପାଇଁ ଅନୁମୋଦିତ ଆଧାର କେନ୍ଦ୍ରକୁ ଯାଇ ବାୟୋମେଟ୍ରିକ୍ ଅପଡେଟ୍ କରାଇବା ଆବଶ୍ୟକ।"),
        ]
    },

    # ── 8. PAN Card Services ───────────────────────────────────────────────────
    {
        "topic": "pan_card_services",
        "title": "PAN Card Services - Form 49A, e-KYC, Delivery, and Strict Two-PAN Prohibition",
        "category": "identity_cards",
        "content": (
            "PAN Card Services (Income Tax Department Guidelines):\n"
            "• Services Offered: New PAN application (Form 49A), details correction (name, father's name, DOB), and duplicate/reprint requests.\n"
            "• Instant e-PAN & Physical Card: Paperless Aadhaar e-KYC generates digital e-PAN in 2 hours; physical plastic card delivered by Speed Post within 10 to 15 days.\n"
            "• STRICT LEGAL PROHIBITION: Never apply for a second PAN card if one has already been allotted. Possessing more than one PAN is illegal under Section 272B of the Income Tax Act and attracts a mandatory ₹10,000 penalty.\n"
            "• Duplicate / Lost PAN: If your PAN card is lost or damaged, apply for a 'Reprint / Duplicate PAN Card' using your existing PAN number. Do NOT apply for a new card."
        ),
        "qas": [
            ("Tell me about PAN Card services", "Jana Seva Kendra assists with new PAN applications, name/DOB corrections, and duplicate reprints. Possessing two PAN cards is strictly illegal under Section 272B."),
            ("Can a person possess two PAN cards?", "No, possessing more than one PAN card is strictly illegal under Section 272B of the Income Tax Act and attracts a mandatory fine of ₹10,000. If lost, apply for a duplicate reprint, never a second PAN."),
            ("What to do if I lost my PAN card?", "If your PAN card is lost or damaged, apply for a Reprint / Duplicate PAN card using your existing 10-digit PAN number. Do not apply for a new PAN card."),
            ("How much time does it take to get a PAN card?", "With Aadhaar e-KYC, an instant digital e-PAN is generated within 2 hours, and the physical plastic card arrives by post within 10 to 15 days."),
            ("क्या दो पैन कार्ड रख सकते हैं?", "नहीं, एक से अधिक पैन कार्ड रखना आयकर अधिनियम की धारा 272B के तहत गैरकानूनी है और 10,000 रुपये का जुर्माना लग सकता है। खोने पर डुप्लीकेट पैन कार्ड रिप्रिंट कराएं।"),
            ("ପ୍ୟାନ୍ କାର୍ଡ ହଜିଗଲେ କ'ଣ କରିବେ?", "ପ୍ୟାନ୍ କାର୍ଡ ହଜିଗଲେ ପୁରୁଣା ପ୍ୟାନ୍ ନମ୍ବର ଦେଇ ଡୁପ୍ଲିକେଟ୍ ରିପ୍ରିଣ୍ଟ କରନ୍ତୁ। ଦୁଇଟି ପ୍ୟାନ୍ କାର୍ଡ ରଖିବା ବେଆଇନ।"),
        ]
    },

    # ── 9. Bonafide Certificate ────────────────────────────────────────────────
    {
        "topic": "bonafide_certificate_details",
        "title": "Bonafide Certificate - Purpose, Issuing Authorities, and Document Checklist",
        "category": "certificates",
        "content": (
            "Bonafide Certificate Services:\n"
            "• Purpose: Verifies genuine student enrollment or employee status. Used for scholarship applications, education loans, student bus/train passes, visa processing, and employment verification.\n"
            "• Issuing Authority: Bonafide certificates are NOT issued by a single statewide revenue portal. They must be obtained directly from the applicant's educational institution (school/college/university principal or registrar) or employer.\n"
            "• Required Documents: Formal application letter stating purpose, institutional ID card, admission fee receipt or enrollment number, and identity proof."
        ),
        "qas": [
            ("Tell me about Bonafide Certificate", "A Bonafide Certificate verifies enrollment or employment for scholarships, bus passes, and education loans, and is issued directly by your school, college, or employer."),
            ("Who issues a Bonafide Certificate?", "A Bonafide Certificate is issued directly by your educational institution (school, college, or university) or employer, not by a statewide revenue portal."),
            ("What documents are needed for a Bonafide Certificate?", "You need an application letter stating the purpose, student/employee ID card, admission fee receipt, and identity proof."),
            ("बोनाफाइड प्रमाण पत्र कौन जारी करता है?", "बोनाफाइड प्रमाण पत्र आपके स्कूल, कॉलेज, विश्वविद्यालय या नियोक्ता (Employer) द्वारा जारी किया जाता है।"),
        ]
    },

    # ── 10. Voter ID (EPIC) Services ───────────────────────────────────────────
    {
        "topic": "voter_id_services",
        "title": "Voter ID (EPIC) Services - Form 6, Form 8, and Electoral Roll Guidelines",
        "category": "identity_cards",
        "content": (
            "Voter ID (EPIC) Services (Election Commission of India):\n"
            "• Services Offered: New voter registration, details correction, constituency shifting, and duplicate/replacement card requests.\n"
            "• Form 6: For new registration of eligible Indian citizens who are 18 years of age or older on the qualifying date.\n"
            "• Form 8: For correction of electoral entries (name, age, photo, address shifting) and duplicate card replacement.\n"
            "• Required Documents: Recent passport photograph, age proof (birth certificate, Aadhaar, class 10 certificate), and ordinary residence proof (utility bill, bank passbook, Aadhaar).\n"
            "• Crucial Electoral Roll Rule: Possessing a physical voter card does not guarantee voting rights. Your name MUST be officially enrolled in the electoral roll of your polling constituency on election day."
        ),
        "qas": [
            ("Tell me about Voter ID services", "Jana Seva Kendra assists with new voter registration using Form 6 and corrections or replacements using Form 8 for Indian citizens 18 years or older."),
            ("What form is used to apply for a new Voter ID card?", "Form 6 is used for new voter registration by Indian citizens who are 18 years of age or older."),
            ("What form is used for Voter ID correction or address change?", "Form 8 is used for correction of entries, address change, and duplicate voter card replacement."),
            ("Can I vote if I have a Voter ID card but my name is not in the voter list?", "No. To cast a vote, your name must be registered in the official electoral roll of your polling station, regardless of whether you hold a physical card."),
            ("नया वोटर कार्ड बनवाने के लिए कौन सा फॉर्म भरना होता है?", "नया वोटर कार्ड बनवाने के लिए फॉर्म 6 भरा जाता है।"),
            ("वोटर कार्ड सुधार के लिए कौन सा फॉर्म होता है?", "वोटर कार्ड में नाम, पता या अन्य सुधार के लिए फॉर्म 8 भरा जाता है।"),
            ("ନୂଆ ଭୋଟର କାର୍ଡ ପାଇଁ କେଉଁ ଫର୍ମ ଦରକାର?", "ନୂଆ ଭୋଟର ପଞ୍ଜୀକରଣ ପାଇଁ ଫର୍ମ ୬ ଏବଂ ସଂଶୋଧନ ପାଇଁ ଫର୍ମ ୮ ବ୍ୟବହାର କରାଯାଏ।"),
        ]
    },

    # ── 11. Complaint & Grievance Registration ────────────────────────────────
    {
        "topic": "complaint_status_and_tracking",
        "title": "Complaint Status Tracking - Ticket Formats (CMP/0001/26-27), Phone Lookup, and SLA",
        "category": "complaint",
        "content": (
            "Complaint Status Tracking and Grievance Guidelines:\n"
            "• Identifier Formats: Complaint ticket format is CMP/XXXX/26-27 (for example, CMP/0001/26-27, CMP/0022/26-27). Callers can simply say the numeric digits (e.g. '1', '22', or '0001').\n"
            "• Phone Number Lookup: Callers can also look up all registered complaints using their 10-digit registered mobile phone number.\n"
            "• Complaint Statuses: 'New' (recently booked, assigned to support), 'In Progress' (under review by officer), 'Discussion' (in review with citizen/officer), 'Resolved' (action completed), or 'Canceled'.\n"
            "• SLA & Escalation: Standard resolution SLA is 24 to 48 hours. If unresolved, complaints escalate automatically to the district supervisor."
        ),
        "qas": [
            ("How can I check the status of my complaint?", "You can check your complaint status by stating your complaint ticket number (e.g. 1, 22, or CMP/0001/26-27) or your registered 10-digit mobile number."),
            ("What are the complaint statuses?", "Complaint statuses are: New, In Progress, Discussion, Resolved, and Canceled."),
            ("शिकायत की स्थिति कैसे चेक करें?", "आप अपना शिकायत टिकट नंबर (जैसे 1 या 0001) या अपना पंजीकृत मोबाइल नंबर बताकर स्थिति जान सकते हैं।"),
            ("ଅଭିଯୋଗ ସ୍ଥିତି କିପରି ଜାଣିବେ?", "ଆପଣଙ୍କ ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର (ଯେପରି ୧ ବା ୦୦୦୧) କିମ୍ବା ମୋବାଇଲ୍ ନମ୍ବର କହି ସ୍ଥିତି ଜାଣିପାରିବେ।"),
        ]
    },
    {
        "topic": "complaint_registration_flow",
        "title": "New Complaint Registration - Steps, Information Needed, and Instant Ticketing",
        "category": "complaint",
        "content": (
            "New Complaint Registration Procedure via Voice IVR:\n"
            "• Callers can register a citizen grievance or service issue over the phone.\n"
            "• Information Collected: Caller phone number, applicant name, city/district in Odisha, and a brief description of the issue (e.g. delayed certificate, employee conduct, payment deduction, portal error).\n"
            "• Instant Ticketing: The system assigns a sequential ticket number (CMP/XXXX/26-27), logs it into the live CRM database, and assigns it to the relevant department with a 24-48 hour resolution SLA."
        ),
        "qas": [
            ("How do I register a new complaint over the phone?", "To register a new complaint, tell the assistant you want to lodge a complaint, describe your problem, and provide your location. A ticket number will be generated immediately."),
            ("नई शिकायत कैसे दर्ज करें?", "शिकायत दर्ज करने के लिए अपनी समस्या और अपना शहर बताएं। आपका टिकट नंबर तुरंत जनरेट हो जाएगा।"),
        ]
    },
]


async def build_comprehensive_rag():
    print("\n" + "=" * 75)
    print("BUILDING COMPREHENSIVE JANA SEVA ODISHA RAG KNOWLEDGE BASE")
    print("=" * 75)

    await init_db()
    emb_service = EmbeddingService(provider_id="semantic-default")
    print("✓ Initialized EmbeddingService ('semantic-default' 384-dimensional all-MiniLM-L6-v2)")

    total_chunks = 0
    total_qas = 0

    async with async_session() as session:
        doc_repo = DocumentRepository(session)

        # Clean existing guide chunks to prevent duplicates and stale data
        del_stmt = delete(Document).where(
            Document.tenant_id == TENANT_ID,
            Document.source_type == "guide_pdf",
        )
        await session.execute(del_stmt)
        await session.commit()
        print("✓ Purged legacy guide_pdf documents for clean ingestion.")

        for idx, entry in enumerate(KNOWLEDGE_ENTRIES, start=1):
            topic = entry["topic"]
            doc_title = f"Jana Seva Odisha Guide - {entry['title']}"
            category = entry["category"]
            body = entry["content"]
            qas = entry.get("qas", [])

            # 1. Main Dense Factual Narrative Chunk
            searchable_text = f"[{category} / {topic}] {doc_title}\n{body}".strip()
            emb = await emb_service.embed_query(searchable_text)

            meta = {
                "domain": "government_services",
                "category": category,
                "topic": topic,
                "title": doc_title,
                "language": ["en", "hi", "or"],
                "source": SOURCE_NAME,
                "source_type": "guide_pdf",
                "source_version": "2.0_comprehensive",
                "verified": True,
                "effective_date": "2026-11-25",
                "ask_count": 5,
            }

            doc = await doc_repo.get_or_create_document(
                tenant_id=TENANT_ID,
                title=doc_title,
                source_type="guide_pdf",
            )

            chunk = DocumentChunk(
                document_id=doc.id,
                tenant_id=TENANT_ID,
                chunk_index=0,
                content=searchable_text,
                embedding=emb if len(emb) == 384 else None,
                embedding_768=emb if len(emb) == 768 else None,
                embedding_json=json.dumps(emb),
                meta_json=json.dumps(meta, ensure_ascii=False),
            )
            session.add(chunk)
            total_chunks += 1

            # 2. Add Explicit High-Precision Q&A Chunks for direct questions
            for q_idx, (q_text, a_text) in enumerate(qas, start=1):
                qa_content = (
                    f"Customer Spoken Inquiry: \"{q_text}\"\n"
                    f"Verified Resolution: \"{a_text}\"\n"
                    f"Context: {doc_title}"
                )
                qa_emb = await emb_service.embed_query(f"{q_text} {a_text} {doc_title} {topic}")
                qa_meta = {
                    "domain": "government_services",
                    "category": category,
                    "topic": f"{topic}_qa_{q_idx}",
                    "title": f"FAQ: {q_text}",
                    "language": ["en", "hi", "or"],
                    "source": SOURCE_NAME,
                    "source_type": "guide_pdf",
                    "verified": True,
                    "ask_count": 10,
                }
                qa_chunk = DocumentChunk(
                    document_id=doc.id,
                    tenant_id=TENANT_ID,
                    chunk_index=q_idx,
                    content=qa_content,
                    embedding=qa_emb if len(qa_emb) == 384 else None,
                    embedding_768=qa_emb if len(qa_emb) == 768 else None,
                    embedding_json=json.dumps(qa_emb),
                    meta_json=json.dumps(qa_meta, ensure_ascii=False),
                )
                session.add(qa_chunk)
                total_chunks += 1
                total_qas += 1

            await session.commit()
            print(f"[{idx}/{len(KNOWLEDGE_ENTRIES)}] Ingested '{topic}': 1 core chunk + {len(qas)} Q&As")

    print("\n" + "=" * 75)
    print(f"✓ INGESTION COMPLETE! Total Knowledge Chunks: {total_chunks} (including {total_qas} Q&As)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(build_comprehensive_rag())
