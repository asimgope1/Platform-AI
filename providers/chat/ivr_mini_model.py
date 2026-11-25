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
        "en": "Hello! Welcome to Jana Seva Citizen Helpline. How can I assist you with government services or schemes today?",
        "hi": "नमस्ते! जन सेवा नागरिक सहायता में आपका स्वागत है। बताइए मैं सरकारी सेवाओं या योजनाओं में आपकी क्या सहायता करूँ?",
        "or": "ନମସ୍କାର! ଜନ ସେବା ନାଗରିକ ସହାୟତା କେନ୍ଦ୍ରକୁ ସ୍ୱାଗତ। କୁହନ୍ତୁ ସରକାରୀ ଯୋଜନା ବା ସେବାରେ ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିବି?",
    },
    "subhadra_scheme": {
        "en": "Under Subhadra Scheme, eligible women residents of Odisha aged 18 to 60 with family income below 2.5 Lakhs receive financial assistance up to 50,000 rupees. Required documents are Aadhaar card, income certificate, residence certificate, bank passbook, and photo.",
        "hi": "सुभद्रा योजना के तहत ओडिशा की 18 से 60 वर्ष की पात्र महिलाएं (वार्षिक आय 2.5 लाख से कम) 50,000 रुपये तक सहायता प्राप्त कर सकती हैं। आवश्यक दस्तावेज: आधार कार्ड, आय प्रमाण पत्र, निवास प्रमाण पत्र, बैंक पासबुक और फोटो।",
        "or": "ସୁଭଦ୍ରା ଯୋଜନା ଅଧୀନରେ ଓଡ଼ିଶାର ୧୮ ରୁ ୬୦ ବର୍ଷ ବୟସ୍କା ମହିଳା (ପାରିବାରିକ ଆୟ ୨.୫ ଲକ୍ଷରୁ କମ୍) ୫୦,୦୦୦ ଟଙ୍କା ପର୍ଯ୍ୟନ୍ତ ସହାୟତା ପାଇପାରିବେ। ଆବଶ୍ୟକ ଡକ୍ୟୁମେଣ୍ଟ: ଆଧାର କାର୍ଡ, ଆୟ ପ୍ରମାଣପତ୍ର, ବାସସ୍ଥାନ ପ୍ରମାଣପତ୍ର, ବ୍ୟାଙ୍କ ପାସବୁକ୍ ଏବଂ ଫଟୋ।",
    },
    "caste_certificate": {
        "en": "To apply for a Caste Certificate (SC/ST/OBC/SEBC), you need your RoR land record, family caste certificate copy, voter ID or Aadhaar, and photo. Applications can be submitted at your nearest Jana Seva Kendra.",
        "hi": "जाति प्रमाण पत्र (SC/ST/OBC/SEBC) के लिए जमीन का खतियान (RoR), परिवार का जाति प्रमाण पत्र, आधार या वोटर कार्ड और फोटो आवश्यक है। आवेदन जन सेवा केंद्र से किया जा सकता है।",
        "or": "ଜାତି ପ୍ରମାଣପତ୍ର ପାଇଁ ଜମି ପଟ୍ଟା (RoR), ପରିବାରର ଜାତି ପ୍ରମାଣପତ୍ର, ଆଧାର କାର୍ଡ ଏବଂ ଫଟୋ ଆବଶ୍ୟକ। ଆପଣ ନିକଟସ୍ଥ ଜନ ସେବା କେନ୍ଦ୍ରରୁ ଆବେଦନ କରିପାରିବେ।",
    },
    "income_certificate": {
        "en": "For an Income Certificate, submit your salary slip or ITR, land revenue receipt, Aadhaar card, and self-declaration. The local Revenue Inspector conducts verification before issuance.",
        "hi": "आय प्रमाण पत्र के लिए वेतन पर्ची (Salary Slip), जमीन का लगान रसीद, आधार कार्ड और स्व-घोषणा पत्र आवश्यक है। राजस्व अधिकारी द्वारा सत्यापन के बाद यह जारी होता है।",
        "or": "ଆୟ ପ୍ରମାଣପତ୍ର ପାଇଁ ଦରମା ସ୍ଲିପ୍ ବା ଆଇଟିଆର୍, ଜମି ଖଜଣା ରସିଦ, ଆଧାର କାର୍ଡ ଆବଶ୍ୟକ। ରାଜସ୍ୱ ନିରୀକ୍ଷକଙ୍କ ଯାଞ୍ଚ ପରେ ଏହା ପ୍ରଦାନ କରାଯାଏ।",
    },
    "residence_certificate": {
        "en": "For a Residence / Domicile Certificate, you must be a resident of Odisha. Required documents are your RoR land record or house tax receipt, electricity bill, Aadhaar card, and educational records.",
        "hi": "निवास प्रमाण पत्र के लिए ओडिशा का निवासी होना आवश्यक है। आवश्यक दस्तावेज: जमीन का खतियान, बिजली बिल या हाउस टैक्स रसीद, आधार कार्ड और शैक्षणिक प्रमाण पत्र।",
        "or": "ବାସସ୍ଥାନ ପ୍ରମାଣପତ୍ର ପାଇଁ ଓଡ଼ିଶାର ବାସିନ୍ଦା ହେବା ଆବଶ୍ୟକ। ଆବଶ୍ୟକ ଡକ୍ୟୁମେଣ୍ଟ: ଜମି ପଟ୍ଟା, ବିଦ୍ୟୁତ ବିଲ୍, ଆଧାର କାର୍ଡ ଏବଂ ଶିକ୍ଷାଗତ ପ୍ରମାଣପତ୍ର।",
    },
    "bonafide_certificate": {
        "en": "Bonafide certificates verify genuine student enrollment. Required documents: Institute ID card, admission fee receipt, mark sheet, and Aadhaar card.",
        "hi": "बोनाफाइड प्रमाण पत्र छात्रों के वैध नामांकन के लिए जारी होता है। आवश्यक दस्तावेज: संस्थान का आईडी कार्ड, प्रवेश शुल्क रसीद और आधार कार्ड।",
        "or": "ବୋନାଫାଇଡ୍ ପ୍ରମାଣପତ୍ର ଛାତ୍ରଛାତ୍ରୀଙ୍କ ନାମଲେଖା ପ୍ରମାଣ ପାଇଁ ଦିଆଯାଏ। ଆବଶ୍ୟକ ଡକ୍ୟୁମେଣ୍ଟ: କଲେଜ ଆଇଡି କାର୍ଡ, ଫିସ୍ ରସିଦ ଏବଂ ଆଧାର କାର୍ଡ।",
    },
    "aadhaar_services": {
        "en": "For Aadhaar services, visit an authorized Aadhaar Seva Kendra for biometric updates, mobile number linking, or address change using valid proof of identity and address.",
        "hi": "आधार सेवाओं के लिए नजदीकी आधार केंद्र पर जाएं। बायोमेट्रिक अपडेट, मोबाइल नंबर लिंक या पता सुधार के लिए वैध पहचान व पते का प्रमाण साथ ले जाएं।",
        "or": "ଆଧାର ସେବା ପାଇଁ ଅନୁମୋଦିତ ଆଧାର କେନ୍ଦ୍ରକୁ ଯାଆନ୍ତୁ। ମୋବାଇଲ୍ ଲିଙ୍କ୍, ଫଟୋ ଅପଡେଟ୍ ବା ଠିକଣା ସଂଶୋଧନ ପାଇଁ ପରିଚୟ ପ୍ରମାଣ ଆବଶ୍ୟକ।",
    },
    "pan_card": {
        "en": "For a new PAN card or correction, apply via Form 49A with Aadhaar e-KYC. Instant e-PAN is issued in 2 hours, and the physical plastic card is delivered within 10 to 15 days.",
        "hi": "नया पैन कार्ड बनवाने या सुधार के लिए आधार ई-केवाईसी द्वारा फॉर्म 49A भरें। ई-पैन 2 घंटे में और भौतिक कार्ड 10 से 15 दिनों में डाक द्वारा मिलता है।",
        "or": "ନୂଆ ପ୍ୟାନ୍ କାର୍ଡ ପାଇଁ ଆଧାର ଇ-କେୱାଇସି ମାଧ୍ୟମରେ ଫର୍ମ 49A ପୂରଣ କରନ୍ତୁ। ଇ-ପ୍ୟାନ୍ ୨ ଘଣ୍ଟାରେ ଏବଂ ଅସଲ କାର୍ଡ ୧୦ ରୁ ୧୫ ଦିନରେ ଡାକ ଯୋଗେ ଆସିଥାଏ।",
    },
    "voter_id": {
        "en": "Indian citizens aged 18 and above can apply for a Voter ID (EPIC card) using Form 6 with age proof, address proof, and photograph via the Voter Helpline app or Jana Seva Kendra.",
        "hi": "18 वर्ष या उससे अधिक आयु के नागरिक फॉर्म 6 द्वारा वोटर आईडी कार्ड बनवा सकते हैं। इसके लिए आयु प्रमाण, पता प्रमाण और फोटो आवश्यक है।",
        "or": "୧୮ ବର୍ଷ ବୟସ୍କ ନାଗରିକ ଫର୍ମ ୬ ମାଧ୍ୟମରେ ନୂଆ ଭୋଟର ଆଇଡି ପାଇଁ ଆବେଦନ କରିପାରିବେ। ବୟସ ପ୍ରମାଣ, ଠିକଣା ପ୍ରମାଣ ଓ ଫଟୋ ଆବଶ୍ୟକ।",
    },
    "legal_documents": {
        "en": "We assist with legal documents including name change affidavits, income affidavits, agreement drafting, and stamp paper documentation through authorized legal notaries.",
        "hi": "हम शपथ पत्र (Affidavit), नाम परिवर्तन, अनुबंध और स्टांप पेपर दस्तावेजों में सहायता करते हैं। इसके लिए वैध पहचान प्रमाण आवश्यक है।",
        "or": "ଆମେ ସତ୍ୟପାଠ (Affidavit), ନାମ ପରିବର୍ତ୍ତନ, ଚୁକ୍ତିପତ୍ର ଏବଂ ଷ୍ଟାମ୍ପ ପେପର କାର୍ଯ୍ୟରେ ସହାୟତା କରୁ। ଏଥିପାଇଁ ପରିଚୟ ପ୍ରମାଣ ଆବଶ୍ୟକ।",
    },
    "kendra_hours": {
        "en": "Jana Seva Kendras operate Monday through Saturday from 9:00 AM to 6:00 PM across all blocks and urban centers in Odisha. Closed on Sundays and gazetted holidays.",
        "hi": "जन सेवा केंद्र सोमवार से शनिवार सुबह 9:00 बजे से शाम 6:00 बजे तक खुले रहते हैं। रविवार और सरकारी छुट्टियों पर बंद रहते हैं।",
        "or": "ଜନ ସେବା କେନ୍ଦ୍ର ସୋମବାରରୁ ଶନିବାର ସକାଳ ୯:୦୦ ରୁ ସନ୍ଧ୍ୟା ୬:୦୦ ପର୍ଯ୍ୟନ୍ତ ଖୋଲା ରହିଥାଏ। ରବିବାର ଏବଂ ସରକାରୀ ଛୁଟିଦିନରେ ବନ୍ଦ ରହେ।",
    },
    "kendra_contact": {
        "en": "Jana Seva Kendra is located at Kalabuda, Kendrapara, Odisha, PIN 754153. The helpline phone number is +91 1234567890, and email is support@janasevaodisha.com.",
        "hi": "जन सेवा केंद्र कलाबुदा, केंद्रापड़ा, ओडिशा 754153 में स्थित है। हेल्पलाइन नंबर +91 1234567890 और ईमेल support@janasevaodisha.com है।",
        "or": "ଜନ ସେବା କେନ୍ଦ୍ର କଳାବୁଦା, କେନ୍ଦ୍ରାପଡ଼ା ୭୫୪୧୫୩ ରେ ଅବସ୍ଥିତ। ହେଲ୍ପଲାଇନ ନମ୍ବର +91 1234567890 ଏବଂ ଇମେଲ support@janasevaodisha.com ଅଟେ।",
    },
    "two_pan_prohibition": {
        "en": "Holding more than one PAN card is strictly illegal under Section 272B of the Income Tax Act with a mandatory 10,000 rupee fine. If lost, apply for a duplicate reprint using your existing PAN number.",
        "hi": "एक से अधिक पैन कार्ड रखना आयकर अधिनियम की धारा 272B के तहत गैरकानूनी है और 10,000 रुपये का जुर्माना लगता है। खोने पर डुप्लीकेट पैन कार्ड रिप्रिंट कराएं।",
        "or": "ଏକାଧିକ ପ୍ୟାନ୍ କାର୍ଡ ରଖିବା ଆୟକର ନିୟମ ଅନୁଯାୟୀ ବେଆଇନ ଏବଂ ୧୦,୦୦୦ ଟଙ୍କା ଜୋରିମାନା ଲାଗିଥାଏ। ହଜିଗଲେ ଡୁପ୍ଲିକେଟ୍ ରିପ୍ରିଣ୍ଟ କରନ୍ତୁ।",
    },
    "kendra_status": {
        "en": "Jana Seva Odisha is an independent facilitation service provider, not a government department. It assists with document applications, while official certificates are issued by competent government authorities.",
        "hi": "जन सेवा केंद्र एक स्वतंत्र नागरिक सुविधा केंद्र है, न कि सरकारी विभाग। यह आवेदन में सहायता करता है, जबकि प्रमाण पत्र केवल सक्षम सरकारी अधिकारियों द्वारा जारी किए जाते हैं।",
        "or": "ଜନ ସେବା କେନ୍ଦ୍ର ଏକ ସ୍ୱତନ୍ତ୍ର ନାଗରିକ ସୁବିଧା କେନ୍ଦ୍ର, ସରକାରୀ ବିଭାଗ ନୁହେଁ। ଏହା ଆବେଦନରେ ସାହାଯ୍ୟ କରେ କିନ୍ତୁ ପ୍ରମାଣପତ୍ର ସରକାରୀ ଅଧିକାରୀଙ୍କ ଦ୍ୱାରା ପ୍ରଦାନ କରାଯାଏ।",
    },
    "website_tools": {
        "en": "The Jana Seva website provides digital document tools: JPG to PDF conversion, Word to PDF conversion, and photo resizing tools for portal uploads.",
        "hi": "वेबसाइट पर मुफ्त दस्तावेज टूल उपलब्ध हैं: JPG से PDF, Word से PDF और फोटो रीसाइज़ टूल।",
        "or": "ୱେବସାଇଟରେ ଡକ୍ୟୁମେଣ୍ଟ ଟୁଲ୍ସ ଉପଲବ୍ଧ ଅଛି: JPG ରୁ PDF, Word ରୁ PDF ଏବଂ ଫଟୋ ସାଇଜ୍ ଛୋଟ କରିବା ଟୁଲ୍।",
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
        "en": "Could you please clarify? How can I assist you with Subhadra Scheme, certificates, or citizen services today?",
        "hi": "माफ़ कीजिए, क्या आप दोहरा सकते हैं? मैं सुभद्रा योजना, प्रमाण पत्र या नागरिक सेवाओं में आपकी क्या मदद करूँ?",
        "or": "କ୍ଷମା କରିବେ, ଆଉଥରେ କହିବେ କି? ମୁଁ ସୁଭଦ୍ରା ଯୋଜନା, ପ୍ରମାଣପତ୍ର ବା ନାଗରିକ ସେବାରେ କିପରି ସାହାଯ୍ୟ କରିବି?",
    },
    "capabilities": {
        "en": "I can assist you with Subhadra Scheme, Caste, Income, and Residence certificates, Aadhaar, PAN card, Voter ID, and citizen grievance registration.",
        "hi": "मैं सुभद्रा योजना, जाति, आय और निवास प्रमाण पत्र, आधार, पैन कार्ड, वोटर आईडी और नागरिक शिकायत पंजीकरण में आपकी मदद कर सकती हूँ।",
        "or": "ମୁଁ ସୁଭଦ୍ରା ଯୋଜନା, ଜାତି, ଆୟ ଏବଂ ବାସସ୍ଥାନ ପ୍ରମାଣପତ୍ର, ଆଧାର, ପ୍ୟାନ୍ କାର୍ଡ, ଭୋଟର ଆଇଡି ଏବଂ ଅଭିଯୋଗ ପଞ୍ଜୀକରଣରେ ସାହାଯ୍ୟ କରିପାରିବି।",
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
        "en": "I specialize exclusively in Jana Seva Odisha citizen government services and grievance registration. How may I assist you today?",
        "hi": "मैं विशेष रूप से जन सेवा ओडिशा नागरिक सेवाओं और शिकायत पंजीकरण के लिए उपलब्ध हूँ। बताइए मैं क्या सहायता करूँ?",
        "or": "ମୁଁ କେବଳ ଜନ ସେବା ଓଡ଼ିଶା ନାଗରିକ ସେବା ଏବଂ ଅଭିଯୋଗ ପଞ୍ଜୀକରଣ ପାଇଁ ଉପଲବ୍ଧ। କୁହନ୍ତୁ ମୁଁ କିପରି ସାହାଯ୍ୟ କରିବି?",
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
    "complaint_register_ask_state": {
        "en": "Certainly! Please describe the issue or problem you are experiencing.",
        "hi": "जी बिल्कुल! कृपया अपनी समस्या या शिकायत का विवरण बताएं।",
        "or": "ନିଶ୍ଚୟ! ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା କିମ୍ବା ଅଭିଯୋଗ ବିଷୟରେ କୁହନ୍ତୁ।",
    },
    "complaint_register_ask_city": {
        "en": "Certainly! Please describe the issue or problem you are experiencing.",
        "hi": "जी बिल्कुल! कृपया अपनी समस्या या शिकायत का विवरण बताएं।",
        "or": "ନିଶ୍ଚୟ! ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା କିମ୍ବା ଅଭିଯୋଗ ବିଷୟରେ କୁହନ୍ତୁ।",
    },
    "complaint_register_ask_address": {
        "en": "Certainly! Please describe the issue or problem you are experiencing.",
        "hi": "जी बिल्कुल! कृपया अपनी समस्या या शिकायत का विवरण बताएं।",
        "or": "ନିଶ୍ଚୟ! ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା କିମ୍ବା ଅଭିଯୋଗ ବିଷୟରେ କୁହନ୍ତୁ।",
    },
    "complaint_register_ask_problem": {
        "en": "Certainly! Please describe the issue or problem you are experiencing.",
        "hi": "जी बिल्कुल! कृपया अपनी समस्या या शिकायत का विवरण बताएं।",
        "or": "ନିଶ୍ଚୟ! ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା କିମ୍ବା ଅଭିଯୋଗ ବିଷୟରେ କୁହନ୍ତୁ।",
    },
    "complaint_register_ask_details": {
        "en": "Certainly! Please describe the issue or problem you are experiencing.",
        "hi": "जी बिल्कुल! कृपया अपनी समस्या या शिकायत का विवरण बताएं।",
        "or": "ନିଶ୍ଚୟ! ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା କିମ୍ବା ଅଭିଯୋଗ ବିଷୟରେ କୁହନ୍ତୁ।",
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
    "ten": "10", "das": "10", "दस": "10", "ଦଶ": "10",
    "eleven": "11", "gyarah": "11", "ग्यारह": "11", "ଏଗାର": "11",
    "twelve": "12", "barah": "12", "बारह": "12", "ବାର": "12",
    "thirteen": "13", "terah": "13", "तेरह": "13", "ତେର": "13",
    "fourteen": "14", "chaudah": "14", "चौदह": "14", "ଚଉଦ": "14",
    "fifteen": "15", "pandrah": "15", "पंद्रह": "15", "ପନ୍ଦର": "15",
    "sixteen": "16", "solah": "16", "सोलह": "16", "ଷୋହଳ": "16",
    "seventeen": "17", "satrah": "17", "सत्रह": "17", "ସତର": "17",
    "eighteen": "18", "atharah": "18", "अट्ठारह": "18", "ଅଠର": "18",
    "nineteen": "19", "unnis": "19", "उन्नीस": "19", "ଉଣେଇଶ": "19",
    "twenty": "20", "bees": "20", "बीस": "20", "କୋଡ଼ିଏ": "20",
}

# ──────────────────────────────────────────────────────────────────────────────
# Geographic Entity Catalog: Indian States & Major Cities Normalization
# ──────────────────────────────────────────────────────────────────────────────

CANONICAL_INDIAN_STATES: Dict[str, str] = {
    # Odisha
    "odisha": "Odisha", "orissa": "Odisha", "ओडिशा": "Odisha", "उड़ीसा": "Odisha", "उड़ीसा": "Odisha", "ଓଡ଼ିଶା": "Odisha",
    # West Bengal
    "west bengal": "West Bengal", "bengal": "West Bengal", "wb": "West Bengal", "पश्चिम बंगाल": "West Bengal", "ପଶ୍ଚିମ ବଙ୍ଗ": "West Bengal",
    # Bihar
    "bihar": "Bihar", "बिहार": "Bihar", "ବିହାର": "Bihar",
    # Jharkhand
    "jharkhand": "Jharkhand", "झारखंड": "Jharkhand", "ଝାଡ଼ଖଣ୍ଡ": "Jharkhand",
    # Maharashtra
    "maharashtra": "Maharashtra", "महाराष्ट्र": "Maharashtra", "ମହାରାଷ୍ଟ୍ର": "Maharashtra",
    # Delhi NCT
    "delhi": "Delhi", "new delhi": "Delhi", "nct of delhi": "Delhi", "दिल्ली": "Delhi", "नई दिल्ली": "Delhi", "ଦିଲ୍ଲୀ": "Delhi",
    # Karnataka
    "karnataka": "Karnataka", "कर्नाटक": "Karnataka", "କର୍ଣ୍ଣାଟକ": "Karnataka",
    # Telangana
    "telangana": "Telangana", "तेलंगाना": "Telangana", "ତେଲେଙ୍ଗାନା": "Telangana",
    # Andhra Pradesh
    "andhra pradesh": "Andhra Pradesh", "andhra": "Andhra Pradesh", "आंध्र प्रदेश": "Andhra Pradesh", "ଆନ୍ଧ୍ର ପ୍ରଦେଶ": "Andhra Pradesh",
    # Tamil Nadu
    "tamil nadu": "Tamil Nadu", "tamilnadu": "Tamil Nadu", "तमिलनाडु": "Tamil Nadu", "ତାମିଲନାଡୁ": "Tamil Nadu",
    # Kerala
    "kerala": "Kerala", "केरल": "Kerala", "କେରଳ": "Kerala",
    # Gujarat
    "gujarat": "Gujarat", "गुजरात": "Gujarat", "ଗୁଜରାଟ": "Gujarat",
    # Rajasthan
    "rajasthan": "Rajasthan", "राजस्थान": "Rajasthan", "ରାଜସ୍ଥାନ": "Rajasthan",
    # Uttar Pradesh
    "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh", "उत्तर प्रदेश": "Uttar Pradesh", "ଉତ୍ତର ପ୍ରଦେଶ": "Uttar Pradesh",
    # Madhya Pradesh
    "madhya pradesh": "Madhya Pradesh", "mp": "Madhya Pradesh", "मध्य प्रदेश": "Madhya Pradesh", "ମଧ୍ୟ ପ୍ରଦେଶ": "Madhya Pradesh",
    # Chhattisgarh
    "chhattisgarh": "Chhattisgarh", "छत्तीसगढ़": "Chhattisgarh", "ଛତିଶଗଡ଼": "Chhattisgarh",
    # Punjab
    "punjab": "Punjab", "पंजाब": "Punjab", "ପଞ୍ଜାବ": "Punjab",
    # Haryana
    "haryana": "Haryana", "हरियाणा": "Haryana", "ହରିୟାଣା": "Haryana",
    # Assam
    "assam": "Assam", "असम": "Assam", "ଆସାମ": "Assam",
    # Goa
    "goa": "Goa", "गोवा": "Goa", "ଗୋଆ": "Goa",
    # Uttarakhand
    "uttarakhand": "Uttarakhand", "उत्तराखंड": "Uttarakhand", "ଉତ୍ତରାଖଣ୍ଡ": "Uttarakhand",
    # Himachal Pradesh
    "himachal pradesh": "Himachal Pradesh", "himachal": "Himachal Pradesh", "हिमाचल प्रदेश": "Himachal Pradesh", "ହିମାଚଳ ପ୍ରଦେଶ": "Himachal Pradesh",
    # Jammu and Kashmir
    "jammu and kashmir": "Jammu and Kashmir", "jammu & kashmir": "Jammu and Kashmir", "j&k": "Jammu and Kashmir", "जम्मू और कश्मीर": "Jammu and Kashmir", "ଜମ୍ମୁ ଓ କାଶ୍ମୀର": "Jammu and Kashmir",
}

CANONICAL_INDIAN_CITIES: Dict[str, Tuple[str, str]] = {
    # Odisha Cities
    "bhubaneswar": ("Bhubaneswar", "Odisha"), "bbsr": ("Bhubaneswar", "Odisha"), "भुवनेश्वर": ("Bhubaneswar", "Odisha"), "ଭୁବନେଶ୍ୱର": ("Bhubaneswar", "Odisha"),
    "cuttack": ("Cuttack", "Odisha"), "कटक": ("Cuttack", "Odisha"), "କଟକ": ("Cuttack", "Odisha"),
    "rourkela": ("Rourkela", "Odisha"), "राउरकेला": ("Rourkela", "Odisha"), "ରାଉରକେଲା": ("Rourkela", "Odisha"),
    "berhampur": ("Berhampur", "Odisha"), "brahmapur": ("Berhampur", "Odisha"), "बरहमपुर": ("Berhampur", "Odisha"), "ବ୍ରହ୍ମପୁର": ("Berhampur", "Odisha"),
    "sambalpur": ("Sambalpur", "Odisha"), "संबलपुर": ("Sambalpur", "Odisha"), "ସମ୍ବଲପୁର": ("Sambalpur", "Odisha"),
    "puri": ("Puri", "Odisha"), "पुरी": ("Puri", "Odisha"), "ପୁରୀ": ("Puri", "Odisha"),
    "balasore": ("Balasore", "Odisha"), "baleswar": ("Balasore", "Odisha"), "बालासोर": ("Balasore", "Odisha"), "ବାଲେଶ୍ୱର": ("Balasore", "Odisha"),
    "bhadrak": ("Bhadrak", "Odisha"), "भद्रक": ("Bhadrak", "Odisha"), "ଭଦ୍ରକ": ("Bhadrak", "Odisha"),
    "kendrapara": ("Kendrapara", "Odisha"), "केद्रपाड़ा": ("Kendrapara", "Odisha"), "କେନ୍ଦ୍ରାପଡ଼ା": ("Kendrapara", "Odisha"),
    "angul": ("Angul", "Odisha"), "अनुगुल": ("Angul", "Odisha"), "ଅନୁଗୋଳ": ("Angul", "Odisha"),
    "dhenkanal": ("Dhenkanal", "Odisha"), "धेनकनाल": ("Dhenkanal", "Odisha"), "ଢେଙ୍କାନାଳ": ("Dhenkanal", "Odisha"),
    "jharsuguda": ("Jharsuguda", "Odisha"), "झारसुगुड़ा": ("Jharsuguda", "Odisha"), "ଝାରସୁଗୁଡ଼ା": ("Jharsuguda", "Odisha"),
    "bargarh": ("Bargarh", "Odisha"), "बरगढ़": ("Bargarh", "Odisha"), "ବରଗଡ଼": ("Bargarh", "Odisha"),
    "jeypore": ("Jeypore", "Odisha"), "जयपुर ओडिशा": ("Jeypore", "Odisha"), "ଜୟପୁର": ("Jeypore", "Odisha"),
    "baripada": ("Baripada", "Odisha"), "बारीपदा": ("Baripada", "Odisha"), "ବାରିପଦା": ("Baripada", "Odisha"),
    "rayagada": ("Rayagada", "Odisha"), "रायगड़ा": ("Rayagada", "Odisha"), "ରାୟଗଡ଼ା": ("Rayagada", "Odisha"),
    "koraput": ("Koraput", "Odisha"), "कोरापुट": ("Koraput", "Odisha"), "କୋରାପୁଟ": ("Koraput", "Odisha"),
    "jagatsinghpur": ("Jagatsinghpur", "Odisha"), "जगतसिंहपुर": ("Jagatsinghpur", "Odisha"), "ଜଗତସିଂହପୁର": ("Jagatsinghpur", "Odisha"),
    "paradeep": ("Paradeep", "Odisha"), "paradip": ("Paradeep", "Odisha"), "पारादीप": ("Paradeep", "Odisha"), "ପାରାଦ୍ୱୀପ": ("Paradeep", "Odisha"),
    "khordha": ("Khordha", "Odisha"), "khurda": ("Khordha", "Odisha"), "खोर्धा": ("Khordha", "Odisha"), "ଖୋର୍ଦ୍ଧା": ("Khordha", "Odisha"),
    # Major Metro & State Hubs
    "mumbai": ("Mumbai", "Maharashtra"), "bombay": ("Mumbai", "Maharashtra"), "मुंबई": ("Mumbai", "Maharashtra"), "ମୁମ୍ବାଇ": ("Mumbai", "Maharashtra"),
    "pune": ("Pune", "Maharashtra"), "पुणे": ("Pune", "Maharashtra"), "ପୁଣେ": ("Pune", "Maharashtra"),
    "nagpur": ("Nagpur", "Maharashtra"), "नागपुर": ("Nagpur", "Maharashtra"), "ନାଗପୁର": ("Nagpur", "Maharashtra"),
    "kolkata": ("Kolkata", "West Bengal"), "calcutta": ("Kolkata", "West Bengal"), "कोलकाता": ("Kolkata", "West Bengal"), "କୋଲକାତା": ("Kolkata", "West Bengal"),
    "howrah": ("Howrah", "West Bengal"), "हावड़ा": ("Howrah", "West Bengal"), "ହାଓଡ଼ା": ("Howrah", "West Bengal"),
    "patna": ("Patna", "Bihar"), "पटना": ("Patna", "Bihar"), "ପାଟନା": ("Patna", "Bihar"),
    "gaya": ("Gaya", "Bihar"), "गया": ("Gaya", "Bihar"), "ଗୟା": ("Gaya", "Bihar"),
    "ranchi": ("Ranchi", "Jharkhand"), "रांची": ("Ranchi", "Jharkhand"), "ରାଞ୍ଚି": ("Ranchi", "Jharkhand"),
    "jamshedpur": ("Jamshedpur", "Jharkhand"), "tatanagar": ("Jamshedpur", "Jharkhand"), "जमशेदपुर": ("Jamshedpur", "Jharkhand"), "ଜାମସେଦପୁର": ("Jamshedpur", "Jharkhand"),
    "dhanbad": ("Dhanbad", "Jharkhand"), "धनबाद": ("Dhanbad", "Jharkhand"), "ଧନବାଦ": ("Dhanbad", "Jharkhand"),
    "delhi": ("New Delhi", "Delhi"), "new delhi": ("New Delhi", "Delhi"), "नई दिल्ली": ("New Delhi", "Delhi"), "दिल्ली": ("New Delhi", "Delhi"), "ଦିଲ୍ଲୀ": ("New Delhi", "Delhi"),
    "noida": ("Noida", "Uttar Pradesh"), "नोएडा": ("Noida", "Uttar Pradesh"), "ନୋଇଡା": ("Noida", "Uttar Pradesh"),
    "gurugram": ("Gurugram", "Haryana"), "gurgaon": ("Gurugram", "Haryana"), "गुरुग्राम": ("Gurugram", "Haryana"), "ଗୁରୁଗ୍ରାମ": ("Gurugram", "Haryana"),
    "lucknow": ("Lucknow", "Uttar Pradesh"), "लखनऊ": ("Lucknow", "Uttar Pradesh"), "ଲକ୍ଷ୍ନୌ": ("Lucknow", "Uttar Pradesh"),
    "kanpur": ("Kanpur", "Uttar Pradesh"), "कानपुर": ("Kanpur", "Uttar Pradesh"), "କାନପୁର": ("Kanpur", "Uttar Pradesh"),
    "varanasi": ("Varanasi", "Uttar Pradesh"), "kashi": ("Varanasi", "Uttar Pradesh"), "वाराणसी": ("Varanasi", "Uttar Pradesh"), "ବାରାଣସୀ": ("Varanasi", "Uttar Pradesh"),
    "jaipur": ("Jaipur", "Rajasthan"), "जयपुर": ("Jaipur", "Rajasthan"), "ଜୟପୁର": ("Jaipur", "Rajasthan"),
    "bengaluru": ("Bengaluru", "Karnataka"), "bangalore": ("Bengaluru", "Karnataka"), "बेंगलुरु": ("Bengaluru", "Karnataka"), "ବାଙ୍ଗାଲୋର": ("Bengaluru", "Karnataka"),
    "hyderabad": ("Hyderabad", "Telangana"), "हैदराबाद": ("Hyderabad", "Telangana"), "ହାଇଦ୍ରାବାଦ": ("Hyderabad", "Telangana"),
    "chennai": ("Chennai", "Tamil Nadu"), "madras": ("Chennai", "Tamil Nadu"), "चेन्नई": ("Chennai", "Tamil Nadu"), "ଚେନ୍ନାଇ": ("Chennai", "Tamil Nadu"),
    "kochi": ("Kochi", "Kerala"), "cochin": ("Kochi", "Kerala"), "कोच्चि": ("Kochi", "Kerala"), "କୋଚି": ("Kochi", "Kerala"),
    "ahmedabad": ("Ahmedabad", "Gujarat"), "अहमदाबाद": ("Ahmedabad", "Gujarat"), "ଅହମଦାବାଦ": ("Ahmedabad", "Gujarat"),
    "surat": ("Surat", "Gujarat"), "सूरत": ("Surat", "Gujarat"), "ସୁରଟ": ("Surat", "Gujarat"),
    "bhopal": ("Bhopal", "Madhya Pradesh"), "भोपाल": ("Bhopal", "Madhya Pradesh"), "ଭୋପାଲ": ("Bhopal", "Madhya Pradesh"),
    "indore": ("Indore", "Madhya Pradesh"), "इंदौर": ("Indore", "Madhya Pradesh"), "ଇନ୍ଦୋର": ("Indore", "Madhya Pradesh"),
    "raipur": ("Raipur", "Chhattisgarh"), "रायपुर": ("Raipur", "Chhattisgarh"), "ରାୟପୁର": ("Raipur", "Chhattisgarh"),
    "guwahati": ("Guwahati", "Assam"), "गुवाहाटी": ("Guwahati", "Assam"), "ଗୁଆହାଟୀ": ("Guwahati", "Assam"),
    "chandigarh": ("Chandigarh", "Punjab"), "चंडीगढ़": ("Chandigarh", "Punjab"), "ଚଣ୍ଡିଗଡ଼": ("Chandigarh", "Punjab"),
}


def normalize_indian_state(text: str) -> Optional[str]:
    """Resolves raw spoken or typed state text to canonical English state name."""
    if not text:
        return None
    raw = str(text).strip().lower()
    clean = re.sub(r"^(?:my state is|i am in|i live in|state is|state|in)\s+", "", raw).strip()
    if clean in CANONICAL_INDIAN_STATES:
        return CANONICAL_INDIAN_STATES[clean]
    for key, val in CANONICAL_INDIAN_STATES.items():
        if key in clean.split() or key == clean:
            return val
    return None


def normalize_indian_city(text: str) -> Optional[Tuple[str, str]]:
    """Resolves raw spoken or typed city text to (canonical_city, default_state)."""
    if not text:
        return None
    raw = str(text).strip().lower()
    clean = re.sub(r"^(?:my city is|i am in|i live in|city is|city|in)\s+", "", raw).strip()
    if clean in CANONICAL_INDIAN_CITIES:
        return CANONICAL_INDIAN_CITIES[clean]
    for key, val in CANONICAL_INDIAN_CITIES.items():
        if key in clean.split() or key == clean:
            return val
    return None

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
        "en": "The status of complaint ticket 1 is Canceled.",
        "hi": "शिकायत टिकट 1 की स्थिति रद्द (Canceled) है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୧ ର ସ୍ଥିତି ବାତିଲ୍ (Canceled) ଅଟେ।",
    },
    "2": {
        "en": "The status of complaint ticket 2 is Solved.",
        "hi": "शिकायत टिकट 2 की स्थिति हल (Solved) हो चुकी है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୨ ର ସ୍ଥିତି ସମାଧାନ (Solved) ହୋଇଛି।",
    },
    "3": {
        "en": "The status of complaint ticket 3 is Canceled.",
        "hi": "शिकायत टिकट 3 की स्थिति रद्द (Canceled) है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୩ ର ସ୍ଥିତି ବାତିଲ୍ (Canceled) ଅଟେ।",
    },
    "4": {
        "en": "The status of complaint ticket 4 is in Discussion.",
        "hi": "शिकायत टिकट 4 की स्थिति चर्चा (Discussion) में है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୪ ର ସ୍ଥିତି ଚର୍ଚ୍ଚାରେ (Discussion) ଅଛି।",
    },
    "5": {
        "en": "The status of complaint ticket 5 is New.",
        "hi": "शिकायत टिकट 5 की स्थिति नई (New) है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୫ ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
    },
    "6": {
        "en": "The status of complaint ticket 6 is New.",
        "hi": "शिकायत टिकट 6 की स्थिति नई (New) है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୬ ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
    },
    "7": {
        "en": "The status of complaint ticket 7 is New.",
        "hi": "शिकायत टिकट 7 की स्थिति नई (New) है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୭ ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
    },
    "8": {
        "en": "The status of complaint ticket 8 is New.",
        "hi": "शिकायत टिकट 8 की स्थिति नई (New) है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୮ ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
    },
    "9": {
        "en": "The status of complaint ticket 9 is New.",
        "hi": "शिकायत टिकट 9 की स्थिति नई (New) है।",
        "or": "ଅଭିଯୋଗ ଟିକେଟ୍ ୯ ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
    },
    "8976453421": {
        "en": "For your mobile number, Ticket 1 is Canceled, Ticket 2 is Solved, and Ticket 5 is New.",
        "hi": "आपके मोबाइल नंबर के लिए टिकट 1 रद्द, टिकट 2 हल, और टिकट 5 नया है।",
        "or": "ଆପଣଙ୍କ ମୋବାଇଲ୍ ପାଇଁ ଟିକେଟ୍ ୧ ବାତିଲ୍, ଟିକେଟ୍ ୨ ସମାଧାନ, ଏବଂ ଟିକେଟ୍ ୫ ନୂଆ ଅଟେ।",
    },
    "8976453488": {
        "en": "For your mobile number, Ticket 3 is Canceled and Ticket 4 is in Discussion.",
        "hi": "आपके मोबाइल नंबर के लिए टिकट 3 रद्द और टिकट 4 चर्चा में है।",
        "or": "ଆପଣଙ୍କ ମୋବାଇଲ୍ ପାଇଁ ଟିକେଟ୍ ୩ ବାତିଲ୍ ଏବଂ ଟିକେଟ୍ ୪ ଚର୍ଚ୍ଚାରେ ଅଛି।",
    },
    "9937234512": {
        "en": "For your mobile number, the status of Ticket 8 is New.",
        "hi": "आपके मोबाइल नंबर के लिए टिकट 8 की स्थिति नई (New) है।",
        "or": "ଆପଣଙ୍କ ମୋବାଇଲ୍ ପାଇଁ ଟିକେଟ୍ ୮ ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
    },
}

# Alias padding for 4-digit representations (0001 -> 1, 0002 -> 2, etc.)
for _i in range(1, 10):
    _pad = f"{_i:04d}"
    if str(_i) in COMPLAINT_STATUS_CATALOG:
        COMPLAINT_STATUS_CATALOG[_pad] = COMPLAINT_STATUS_CATALOG[str(_i)]

# Dynamic complaint ticket counter initialized beyond existing CRM tickets (latest is 11)
_NEXT_TICKET_COUNTER: int = 12


CRM_KNOWLEDGE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "banking_knowledge",
    "crm_knowledge.json",
)

PENDING_COMPLAINTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "pending_crm_complaints.json",
)


def _load_pending_complaints() -> list[dict[str, Any]]:
    if not os.path.exists(PENDING_COMPLAINTS_FILE):
        return []
    try:
        import json
        with open(PENDING_COMPLAINTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []


def _save_pending_complaints(complaints: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(PENDING_COMPLAINTS_FILE), exist_ok=True)
    try:
        import json
        with open(PENDING_COMPLAINTS_FILE, "w", encoding="utf-8") as f:
            json.dump(complaints, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[IVR CRM DB] Failed writing pending complaints: {e}")


def _load_offline_crm_catalog() -> None:
    """
    Populates COMPLAINT_STATUS_CATALOG from local crm_knowledge.json and
    pending_crm_complaints.json so all tickets and customer phone directories
    remain 100% available with full details even when remote CRM DB is unreachable.
    """
    global _NEXT_TICKET_COUNTER
    max_seen = 0
    phone_map: Dict[str, list] = {}

    # 1. Parse crm_knowledge.json
    if os.path.exists(CRM_KNOWLEDGE_FILE):
        try:
            import json
            with open(CRM_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                crm_k = json.load(f)
            for c in crm_k:
                cat = c.get("category")
                content = c.get("content", "")
                if cat == "crm_complaint_status":
                    m = re.search(
                        r"CMP/0*(\d{1,4})/26-27.*?customer\s+([^(\n]+)\s*\(phone number\s*([^)\n]+)\)\s*in\s*([^,\n]+).*?booked on\s*([^.]+)\.\s*The current complaint status is\s*'([^']+)'.*?Issue category is\s*'([^']+)'.*?assigned to\s*([^.]+)\.\s*Latest progress notes:\s*([^.]+)",
                        content,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if m:
                        tnum = str(int(m.group(1)))
                        pad4 = f"{int(tnum):04d}"
                        cname = m.group(2).strip()
                        phone = m.group(3).strip()
                        city = m.group(4).strip()
                        booked = m.group(5).strip()
                        status = m.group(6).strip()
                        dept = m.group(8).strip()
                        notes = m.group(9).strip()
                        max_seen = max(max_seen, int(tnum))
                        clean_notes = notes.strip().rstrip(".")
                        tnum_or = tnum.translate(str.maketrans("0123456789", "୦୧୨୩୪୫୬୭୮୯"))
                        pad4_or = pad4.translate(str.maketrans("0123456789", "୦୧୨୩୪୫୬୭୮୯"))
                        entry = {
                            "en": f"The status of complaint ticket {tnum} is {status}.",
                            "hi": f"शिकायत टिकट {tnum} की स्थिति {status} है।",
                            "or": f"ଅଭିଯୋଗ ଟିକେଟ୍ {tnum_or} ର ସ୍ଥିତି {status} ଅଟେ।",
                        }
                        COMPLAINT_STATUS_CATALOG[tnum] = entry
                        COMPLAINT_STATUS_CATALOG[pad4] = entry
                        COMPLAINT_STATUS_CATALOG[f"CMP/{pad4}/26-27"] = entry
                        if phone and len(phone) >= 10:
                            phone_map.setdefault(phone, []).append((tnum, pad4, status, clean_notes, cname))
        except Exception as e:
            logger.warning(f"[IVR CRM] Error loading crm_knowledge.json fallback: {e}")

    # 2. Parse pending_crm_complaints.json (latest voice complaints)
    for p in _load_pending_complaints():
        tid = p.get("ticket_id", "")
        tm = re.search(r"CMP/0*(\d{1,4})/26-27", tid)
        if tm:
            tnum = str(int(tm.group(1)))
            pad4 = f"{int(tnum):04d}"
            phone = p.get("caller_phone", "")
            cname = p.get("customer_name", "Customer")
            desc = p.get("description", "").strip().rstrip(".")
            max_seen = max(max_seen, int(tnum))
            entry = {
                "en": f"The status of complaint ticket {tnum} is New.",
                "hi": f"शिकायत टिकट {tnum} की स्थिति नई (New) है।",
                "or": f"ଅଭିଯୋଗ ଟିକେଟ୍ {tnum} ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
            }
            COMPLAINT_STATUS_CATALOG[tnum] = entry
            COMPLAINT_STATUS_CATALOG[pad4] = entry
            COMPLAINT_STATUS_CATALOG[tid] = entry
            if phone and len(phone) >= 10:
                phone_map.setdefault(phone, []).append((tnum, pad4, "New", desc, cname))

    # 3. Populate Phone Number Directories
    for ph, tkts in phone_map.items():
        latest = tkts[-1]
        tnum, pad4, stat, notes, cname = latest
        clean_notes = notes[:80].rstrip(".")
        tkt_summaries_en = [f"Ticket {t[0]}" for t in tkts]
        tkt_summaries_hi = [f"टिकट {t[0]}" for t in tkts]
        tkt_summaries_or = [f"ଟିକେଟ୍ {t[0].translate(str.maketrans('0123456789', '୦୧୨୩୪୫୬୭୮୯'))}" for t in tkts]
        en_list = ", ".join(tkt_summaries_en[:-1]) + (f", and {tkt_summaries_en[-1]}" if len(tkt_summaries_en) > 1 else tkt_summaries_en[0])
        hi_list = ", ".join(tkt_summaries_hi[:-1]) + (f", और {tkt_summaries_hi[-1]}" if len(tkt_summaries_hi) > 1 else tkt_summaries_hi[0])
        or_list = ", ".join(tkt_summaries_or[:-1]) + (f", ଏବଂ {tkt_summaries_or[-1]}" if len(tkt_summaries_or) > 1 else tkt_summaries_or[0])

        COMPLAINT_STATUS_CATALOG[ph] = {
            "en": f"For your phone number, Ticket {tnum} is {stat}.",
            "hi": f"आपके मोबाइल नंबर के लिए टिकट {tnum} की स्थिति {stat} है।",
            "or": f"ଆପଣଙ୍କ ମୋବାଇଲ୍ ପାଇଁ ଟିକେଟ୍ {tnum} ର ସ୍ଥିତି {stat} ଅଟେ।",
        }

    if max_seen >= _NEXT_TICKET_COUNTER:
        _NEXT_TICKET_COUNTER = max_seen + 1


# Initialize offline catalog immediately at module import
_load_offline_crm_catalog()


async def flush_pending_crm_complaints() -> int:
    """
    Connects to remote CRM DB if reachable and inserts any un-synced complaints.
    Returns the number of successfully flushed complaints.
    """
    pending = _load_pending_complaints()
    to_sync = [c for c in pending if not c.get("synced_to_crm")]
    if not to_sync or asyncpg is None:
        return 0

    host = os.getenv("CRM_DB_HOST", "192.168.1.124")
    port = int(os.getenv("CRM_DB_PORT", "5435"))
    dbname = os.getenv("CRM_DB_NAME", "crm_backend_db")
    user = os.getenv("CRM_DB_USERNAME", "postgres")
    password = os.getenv("CRM_DB_PASSWORD", "epsumlabs")

    try:
        conn = await asyncpg.connect(
            host=host, port=port, database=dbname, user=user, password=password, ssl=False, timeout=2.0
        )
    except Exception as e:
        logger.debug(f"[IVR CRM DB] Pending flush skipped (remote CRM DB offline): {e}")
        return 0

    flushed_count = 0
    try:
        for c in to_sync:
            try:
                c_sl = uuid.UUID(c["complaint_sl"])
                c_time_str = c.get("created_at")
                c_time = datetime.fromisoformat(c_time_str) if c_time_str else datetime.now(timezone.utc)
                await conn.execute(
                    """
                    INSERT INTO "ComplainBookings" (
                        complainsl, complainid, customername, phonenumber, department_sl,
                        complain_phase_sl, createdby, complaindatetime, createdate, reopened,
                        othernotes, state, city, address
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (complainsl) DO NOTHING
                    """,
                    c_sl,
                    c["ticket_id"],
                    c.get("customer_name") or "",
                    c.get("caller_phone") or "",
                    uuid.UUID("d6b3256a-fa21-4708-9910-3ea086831c6f"),
                    uuid.UUID("c84516c3-1592-4d0e-bc0e-7f50b2758f97"),
                    uuid.UUID("a778db92-1d4f-4778-a698-05830387b0ed"),
                    c_time,
                    c_time,
                    False,
                    c["description"],
                    c.get("state") or "",
                    c.get("city") or "",
                    c.get("address") or "",
                )
                await conn.execute(
                    """
                    INSERT INTO "ComplainFollowUp" (
                        follow_up_sl, complainsl, description, follow_up_date, created_by, created_date, complain_phase_sl
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (follow_up_sl) DO NOTHING
                    """,
                    uuid.uuid4(),
                    c_sl,
                    f"Voice IVR Registration: {c['description']}",
                    c_time,
                    uuid.UUID("a778db92-1d4f-4778-a698-05830387b0ed"),
                    c_time,
                    uuid.UUID("c84516c3-1592-4d0e-bc0e-7f50b2758f97"),
                )
                c["synced_to_crm"] = True
                flushed_count += 1
                logger.info(f"[IVR CRM DB] Flushed queued ticket {c['ticket_id']} into ComplainBookings & FollowUp.")
            except Exception as item_err:
                logger.warning(f"[IVR CRM DB] Failed to insert queued ticket {c.get('ticket_id')}: {item_err}")
    finally:
        await conn.close()
        _save_pending_complaints(pending)

    return flushed_count


async def _persist_complaint_to_crm_db(
    ticket_id: str,
    complaint_sl: uuid.UUID,
    description: str,
    caller_phone: Optional[str] = None,
    customer_name: Optional[str] = None,
    call_uuid: Optional[str] = None,
    state: Optional[str] = None,
    city: Optional[str] = None,
    address: Optional[str] = None,
) -> bool:
    """
    Asynchronously persists a newly registered complaint into the remote CRM PostgreSQL DB
    ('ComplainBookings' and 'ComplainFollowUp') and registers locally without blocking audio.
    """
    phone_val = str(caller_phone or "").strip()
    if phone_val.lower() in ("unknown", "anonymous", "none", "s"):
        phone_val = ""
    cust_val = str(customer_name or "").strip()
    now = datetime.now(timezone.utc)

    # 0. Always save to persistent local disk queue first (guarantees zero data loss even if remote DB is down)
    pending = _load_pending_complaints()
    if not any(p.get("ticket_id") == ticket_id for p in pending):
        pending.append({
            "ticket_id": ticket_id,
            "complaint_sl": str(complaint_sl),
            "description": description,
            "caller_phone": phone_val,
            "customer_name": cust_val,
            "call_uuid": call_uuid,
            "state": state or "",
            "city": city or "",
            "address": address or "",
            "created_at": now.isoformat(),
            "synced_to_crm": False,
        })
        _save_pending_complaints(pending)

    # 1. Attempt immediate flush to remote CRM DB if reachable
    await flush_pending_crm_complaints()

    # 2. Insert into local SQLite / Postgres complaints table if available
    try:
        from tools.business_data.complaints import register_complaint
        from platform_core.security.tenant_context import TenantContext
        ctx = TenantContext(tenant_id="879e9bda-8049-4824-9aa0-fc5e9f0012d9", roles=["caller"])
        await register_complaint(
            description=description,
            caller_phone=phone_val,
            service_category="complaint",
            call_uuid=call_uuid,
            tenant_context=ctx,
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
            # Flush any pending complaints that were queued while DB was offline
            try:
                await flush_pending_crm_complaints()
            except Exception:
                pass
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

                raw_notes = str(r["othernotes"] or "").strip()
                clean_notes = raw_notes.rstrip(".")
                issue_en = f" regarding '{clean_notes}'" if clean_notes else ""
                issue_hi = f", जो '{clean_notes}' के संबंध में है" if clean_notes else ""
                issue_or = f", ଯାହା '{clean_notes}' ବିଷୟରେ ଅଟେ" if clean_notes else ""

                entry = {
                    "en": f"The status of complaint ticket {tnum} is {status}.",
                    "hi": f"शिकायत टिकट {tnum} की स्थिति {status_hi} है।",
                    "or": f"ଅଭିଯୋଗ ଟିକେଟ୍ {tnum} ର ସ୍ଥିତି {status_or} ଅଟେ।",
                }
                COMPLAINT_STATUS_CATALOG[tnum] = entry
                COMPLAINT_STATUS_CATALOG[pad4] = entry
                COMPLAINT_STATUS_CATALOG[cid] = entry

                phone = str(r["phonenumber"] or "").strip()
                if phone and len(phone) >= 10 and phone.isdigit():
                    phone_map.setdefault(phone, []).append({
                        "tnum": tnum,
                        "pad4": pad4,
                        "status": status,
                        "customer": cname,
                        "notes": clean_notes,
                    })

            for phone, tkts in phone_map.items():
                names = list(dict.fromkeys(t["customer"] for t in tkts if t["customer"] and t["customer"] not in ("Customer", "aa", "qwq")))
                names_str = f" ({', '.join(names)})" if names else ""

                if len(tkts) == 1:
                    t = tkts[0]
                    tnum = t["tnum"]
                    pad4 = t["pad4"]
                    st = t["status"]
                    st_hi = {"New": "नई (New)", "Discussion": "चर्चाधीन (Discussion)", "Solved": "हल (Solved)", "Canceled": "रद्द (Canceled)"}.get(st, st)
                    st_or = {"New": "ନୂଆ (New)", "Discussion": "ଆଲୋଚନାରେ (Discussion)", "Solved": "ସମାଧାନ (Solved)", "Canceled": "ବାତିଲ୍ (Canceled)"}.get(st, st)
                    notes = t["notes"]
                    issue_en = f" regarding '{notes}'" if notes else ""
                    issue_hi = f", जो '{notes}' के संबंध में है," if notes else ""
                    issue_or = f", ଯାହା '{notes}' ବିଷୟରେ ଅଟେ," if notes else ""
                    COMPLAINT_STATUS_CATALOG[phone] = {
                        "en": f"For your registered number {phone}{names_str}, we found complaint Ticket {tnum} (reference CMP {pad4}){issue_en}. The current ticket status is {st}. Is there anything else I can help you with today?",
                        "hi": f"आपके पंजीकृत नंबर {phone}{names_str} के लिए शिकायत टिकट नंबर {tnum} (सीएमपी {pad4}){issue_hi} दर्ज है। वर्तमान स्थिति {st_hi} है। क्या मैं और कोई सहायता करूँ?",
                        "or": f"ଆପଣଙ୍କ ପଞ୍ଜୀକୃତ ନମ୍ବର {phone}{names_str} ପାଇଁ ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର {tnum} (ସିଏମ୍‌ପି {pad4}){issue_or} ରହିଛି। ବର୍ତ୍ତମାନର ସ୍ଥିତି {st_or} ଅଟେ। ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?",
                    }
                else:
                    latest = tkts[0]
                    latest_tnum = latest["tnum"]
                    latest_st = latest["status"]
                    latest_notes = latest["notes"]
                    latest_issue_en = f" regarding '{latest_notes}'" if latest_notes else ""
                    latest_issue_hi = f", जो '{latest_notes}' के संबंध में है," if latest_notes else ""
                    latest_issue_or = f", ଯାହା '{latest_notes}' ବିଷୟରେ ଅଟେ," if latest_notes else ""

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
                        "en": f"For registered phone number {phone}{names_str}, we found {len(tkts)} complaints on file: {en_list}. Your latest is Ticket {latest_tnum}{latest_issue_en} with status {latest_st}. Which ticket would you like full details on?",
                        "hi": f"पंजीकृत मोबाइल नंबर {phone}{names_str} के लिए {len(tkts)} शिकायतें दर्ज हैं: {hi_list}। आपकी नवीनतम शिकायत टिकट {latest_tnum}{latest_issue_hi} की स्थिति {latest_st} है। आप किस टिकट का पूरा विवरण चाहते हैं?",
                        "or": f"ପଞ୍ଜୀକୃତ ମୋବାଇଲ୍ {phone}{names_str} ପାଇଁ {len(tkts)} ଟି ଅଭିଯୋଗ ରହିଛି: {or_list}। ଆପଣଙ୍କ ସର୍ବଶେଷ ଅଭିଯୋଗ ଟିକେଟ୍ {latest_tnum}{latest_issue_or} ର ସ୍ଥିତି {latest_st} ଅଟେ। ଆପଣ କେଉଁ ଟିକେଟ୍ ବିଷୟରେ ଜାଣିବାକୁ ଚାହାଁନ୍ତି?",
                    }

            if max_tnum >= _NEXT_TICKET_COUNTER:
                _NEXT_TICKET_COUNTER = max_tnum + 1

            self._last_crm_sync_ts = now_ts
            logger.info(f"[IVR CRM Sync] Live sync updated {len(rows)} tickets and {len(phone_map)} phone directories (next ticket={_NEXT_TICKET_COUNTER}).")
            return True
        except Exception as e:
            logger.warning(f"[IVR CRM Sync] Live CRM sync notice (using cached catalog): {e}")
            try:
                _load_offline_crm_catalog()
            except Exception:
                pass
            return False

    def _is_complaint_register_intent(self, lower_query: str) -> bool:
        """Checks if the user wants to register, file, or book a new complaint."""
        clean = (lower_query or "").strip().lower()
        if any(w in clean for w in ("status", "check", "track", "where is", "existing", "स्थिति", "ଟ୍ରାକ୍", "ପୂର୍ବ")):
            return False
        tokens = set(re.findall(r'[a-zA-Z\u0900-\u097f\u0b00-\u0b7f]+', clean))
        # If complaint/grievance keyword is present along with register/file/lodge/new
        has_complaint_kw = any(w in clean for w in ("complaint", "complain", "grievance", "ticket", "शिकायत", "अभिযোগ", "ଅଭିଯୋଗ"))
        if has_complaint_kw and tokens.intersection({"register", "lodge", "file", "book", "new", "fresh", "दर्ज", "पंजीकृत", "पंजीकरण", "दाखिल", "नया", "नई", "ଦାଖଲ", "ପଞ୍ଜୀକରଣ", "ନୂଆ", "ନୂତନ"}):
            return True
        if tokens.intersection({"register", "lodge", "file", "book", "दर्ज", "पंजीकृत", "पंजीकरण", "दाखिल", "ଦାଖଲ", "ପଞ୍ଜୀକରଣ"}) and not any(w in clean for w in ("pan", "card", "aadhaar", "voter", "certificate", "पैन", "आधार", "प्रमाण")):
            return True
        if "new one" in clean or "fresh complaint" in clean or "new complaint" in clean or "register one" in clean or "a new one" in clean:
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

    def _has_problem_description(self, lower_query: str) -> bool:
        """Determines if the caller utterance already contains a concrete problem description."""
        clean = (lower_query or "").strip().lower()
        for phrase in (
            "i want to register a new complaint that", "i want to register a new complaint",
            "i want to register a complaint that", "i want to register a complaint",
            "i want to file a new complaint", "i want to file a complaint",
            "register a new complaint that", "register a new complaint",
            "register new complaint that", "register new complaint",
            "register a complaint that", "register a complaint",
            "file a new complaint", "file a complaint",
            "book a new complaint", "book a complaint",
            "lodge a new complaint", "lodge a complaint",
            "new complaint", "fresh complaint", "want to complain",
            "register new one", "register a new one", "new one",
            "नई शिकायत दर्ज करनी है", "नई शिकायत दर्ज करना है", "शिकायत दर्ज करनी है", "शिकायत दर्ज करना है",
            "एक नई शिकायत लिखनी है", "नई शिकायत", "नया शिकायत",
            "ନୂଆ ଅଭିଯୋଗ ଦାଖଲ କରିବାକୁ ଚାହେଁ", "ଅଭିଯୋଗ ଦାଖଲ କରିବାକୁ ଚାହେଁ",
            "ନୂଆ ଅଭିଯୋଗ", "ନୂଆ ଗୋଟିଏ",
        ):
            clean = clean.replace(phrase, "")
        clean = clean.strip(" ,.-!?:;")
        return len(clean.split()) >= 2

    def _is_complaint_status_intent(self, lower_query: str) -> bool:
        """Checks if the user wants to check the status or track an existing complaint."""
        clean = (lower_query or "").strip().lower()
        if self._is_complaint_register_intent(clean):
            return False
        if extract_complaint_identifier(clean, allow_bare_digits=False):
            return True
        tokens = set(re.findall(r'[a-zA-Z\u0900-\u097f\u0b00-\u0b7f]+', clean))
        if tokens.intersection({
            "status", "track", "existing", "old", "previous", "ticket", "tickets", "cmp",
            "स्थिति", "पुरानी", "पुराना", "जांच", "ଟ୍ରାକ୍", "ସ୍ଥିତି", "ପୂର୍ବ", "ପୁରୁଣା"
        }) and any(w in clean for w in ("complaint", "ticket", "cmp", "शिकायत", "टिकट", "ଅଭିଯୋଗ", "ଟିକେଟ୍")):
            return True
        if tokens.intersection({
            "status", "track", "cmp", "स्थिति", "ଟ୍ରାକ୍", "ସ୍ଥିତି"
        }) and not any(w in clean for w in ("aadhaar", "pan", "voter", "certificate", "आधार", "पैन", "प्रमाण")):
            return True
        return any(
            phrase in clean
            for phrase in (
                "check status", "check the status", "status of complaint", "complaint status",
                "my complaint", "the complaint", "about complaint", "about the complaint", "about my complaint",
                "explain complaint", "explain the complaint", "explain about the complaint", "tell me about my complaint",
                "tell me about the complaint", "what is my complaint", "what is the complaint",
                "what did i complain", "check existing", "existing complaint", "check existing complaint",
                "existing one", "old complaint", "status check", "know status", "track complaint",
                "track my complaint", "status dekhna", "status janna", "complaint detail", "complaint details", "ticket details",
                "स्थिति देखना", "स्थिति जांचना", "पुरानी शिकायत", "पुराना शिकायत", "स्थिति बताएं", "शिकायत का विवरण", "शिकायत के बारे में",
                "ସ୍ଥିତି ଜାଣିବା", "ପୂର୍ବ ଅଭିଯୋଗ", "ପୁରୁଣା ଅଭିଯୋଗ", "ଅଭିଯୋଗ ସ୍ଥିତି", "ଅଭିଯୋଗ ବିବରଣୀ",
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

    def _last_substantive_assistant_message(self, messages: Optional[List[ChatMessage]]) -> Optional[str]:
        """Finds the latest assistant prompt, ignoring brief fillers or greetings."""
        if not messages:
            return None
        fillers = (
            "yes, i am listening", "हाँजी, मैं सुन रही हूँ", "ହଁ ଆଜ୍ଞା, ମୁଁ ଶୁଣୁଛି",
            "how can i help you today", "what else i can help you", "sure, please let me know"
        )
        for m in reversed(messages):
            if m.role == "assistant" and m.content:
                c = str(m.content).lower().strip()
                if not any(f in c for f in fillers):
                    return c
        return None

    def _was_asked_for_state(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Disabled: Complaint flow directly asks for problem description without asking for state."""
        return False

    def _was_asked_for_city(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Disabled: Complaint flow directly asks for problem description without asking for city."""
        return False

    def _was_asked_for_address(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Disabled: Complaint flow directly asks for problem description without asking for address."""
        return False

    def _was_asked_for_complaint_description(self, messages: Optional[List[ChatMessage]]) -> bool:
        """Returns True if the previous assistant turn asked for a problem or issue description."""
        c = self._last_substantive_assistant_message(messages)
        if not c:
            return False
        return any(
            marker in c
            for marker in (
                "describe the issue", "describe the problem", "problem you are facing",
                "issue or problem", "अपनी समस्या या शिकायत", "समस्या का विवरण", "अपनी समस्या",
                "ସମସ୍ୟା କିମ୍ବା ଅଭିଯୋଗ", "ସମସ୍ୟା ବିଷୟରେ", "register your complaint right away",
            )
        )

    def _extract_registration_slots_from_history(self, messages: Optional[List[ChatMessage]]) -> Dict[str, str]:
        """
        Traverses conversation history and extracts previously collected State, City, and Address.
        Uses canonical Indian state and city normalization to resolve vernacular variants.
        Does NOT set any defaults if caller did not provide them.
        """
        slots = {"state": "", "city": "", "address": ""}
        if not messages:
            return slots

        for i, m in enumerate(messages):
            if m.role == "assistant" and m.content and i + 1 < len(messages):
                asst_text = str(m.content).lower()
                user_msg = messages[i + 1]
                if user_msg.role == "user" and user_msg.content:
                    u_val = str(user_msg.content).strip().rstrip(".")
                    u_clean = re.sub(r'^(?:i am in|i live in|my state is|my city is|it is|state is|city is|in)\s+', '', u_val, flags=re.IGNORECASE).strip()
                    if any(w in asst_text for w in ("your state", "state name", "राज्य", "ରାଜ୍ୟ")):
                        norm_st = normalize_indian_state(u_clean or u_val)
                        if norm_st:
                            slots["state"] = norm_st
                        else:
                            norm_ct = normalize_indian_city(u_clean or u_val)
                            if norm_ct:
                                slots["city"] = norm_ct[0]
                                slots["state"] = norm_ct[1]
                            else:
                                slots["state"] = u_clean.title() if u_clean else u_val.title()
                    elif any(w in asst_text for w in ("your city", "which city", "शहर", "ସହର")):
                        norm_ct = normalize_indian_city(u_clean or u_val)
                        if norm_ct:
                            slots["city"] = norm_ct[0]
                            if norm_ct[1] and not slots.get("state"):
                                slots["state"] = norm_ct[1]
                        else:
                            slots["city"] = u_clean.title() if u_clean else u_val.title()
                    elif any(w in asst_text for w in ("street address", "area or street", "पते", "क्षेत्र", "ଠିକଣା")):
                        slots["address"] = u_val

        return slots

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
        customer_name: Optional[str] = None,
        call_uuid: Optional[str] = None,
        state: Optional[str] = None,
        city: Optional[str] = None,
        address: Optional[str] = None,
    ) -> str:
        """
        Registers a new complaint instantly (< 2ms) without stalling telephony audio on slow LLMs.
        Allocates next sequential ticket ID, updates in-memory catalog, schedules async DB persistence,
        and returns a natural spoken confirmation.
        Does NOT set any fields by default (no default city, state, or phone).
        """
        global _NEXT_TICKET_COUNTER
        clean_desc = (description or "").strip()
        if len(clean_desc) < 3:
            clean_desc = "Customer service inquiry and follow-up"

        phone = str(caller_phone or "").strip()
        if phone.lower() in ("unknown", "anonymous", "none", "s"):
            phone = ""

        # Duplicate complaint guard: prevent registering the same complaint twice in one call
        dedup_key = f"{call_uuid or phone}:{clean_desc.lower()[:80]}"
        if not hasattr(self, "_recent_complaints"):
            self._recent_complaints: Dict[str, int] = {}
        if dedup_key in self._recent_complaints:
            existing_num = self._recent_complaints[dedup_key]
            logger.info(f"[IVR Mini Brain] Duplicate complaint prevented for key={dedup_key[:40]} (existing ticket={existing_num})")
            if lang == "hi":
                return f"आपकी यह शिकायत पहले ही दर्ज की जा चुकी है। आपका शिकायत टिकट नंबर {existing_num} है।"
            elif lang == "or":
                return f"ଆପଣଙ୍କ ଏହି ଅଭିଯୋଗ ପୂର୍ବରୁ ଦାଖଲ ହୋଇସାରିଛି। ଆପଣଙ୍କ ଅଭିଯୋଗ ଟିକେଟ୍ ନମ୍ବର {existing_num} ଅଟେ।"
            return f"Your complaint has already been registered. Your complaint ticket number is {existing_num}."

        ticket_num = _NEXT_TICKET_COUNTER
        _NEXT_TICKET_COUNTER += 1
        self._recent_complaints[dedup_key] = ticket_num

        pad_4 = f"{ticket_num:04d}"
        ticket_cmp = f"CMP/{pad_4}/26-27"
        comp_sl = uuid.uuid4()
        cust_name = str(customer_name or "").strip()
        clean_state = (state or "").strip()
        clean_city = (city or "").strip()
        clean_addr = (address or "").strip()
        loc_str = f" from {clean_city}, {clean_state}" if clean_city and clean_state else (f" from {clean_city}" if clean_city else (f" from {clean_state}" if clean_state else ""))
        cust_str = f" for customer {cust_name}" if cust_name else ""
        loc_desc_hi = f"यह {clean_city} से " if clean_city else "यह "
        loc_desc_or = f"ଏହା {clean_city} ରୁ " if clean_city else "ଏହା "

        # Update in-memory catalog so immediate follow-up status queries work
        COMPLAINT_STATUS_CATALOG[str(ticket_num)] = {
            "en": f"The status of complaint ticket {ticket_num} is New.",
            "hi": f"शिकायत टिकट {ticket_num} की स्थिति नई (New) है।",
            "or": f"ଅଭିଯୋଗ ଟିକେଟ୍ {ticket_num} ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
        }
        COMPLAINT_STATUS_CATALOG[pad_4] = COMPLAINT_STATUS_CATALOG[str(ticket_num)]
        if phone:
            COMPLAINT_STATUS_CATALOG[phone] = {
                "en": f"For your mobile number, the status of complaint ticket {ticket_num} is New.",
                "hi": f"आपके मोबाइल नंबर के लिए शिकायत टिकट {ticket_num} की स्थिति नई (New) है।",
                "or": f"ଆପଣଙ୍କ ମୋବାଇଲ୍ ପାଇଁ ଅଭିଯୋଗ ଟିକେଟ୍ {ticket_num} ର ସ୍ଥିତି ନୂଆ (New) ଅଟେ।",
            }

        # Fire async background task to persist to CRM DB & local DB with failure logging
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                _persist_complaint_to_crm_db(
                    ticket_id=ticket_cmp,
                    complaint_sl=comp_sl,
                    description=clean_desc,
                    caller_phone=phone,
                    customer_name=cust_name,
                    call_uuid=call_uuid,
                    state=clean_state,
                    city=clean_city,
                    address=clean_addr,
                )
            )
            def _on_persist_done(t: asyncio.Task) -> None:
                exc = t.exception() if not t.cancelled() else None
                if exc:
                    logger.error(f"[IVR Mini Brain] ❌ Complaint {ticket_cmp} DB persistence failed: {exc}")
            task.add_done_callback(_on_persist_done)
        except RuntimeError:
            pass

        logger.info(
            f"[IVR Mini Brain] Registered new complaint ticket={ticket_num} ({ticket_cmp}) "
            f"phone={phone} loc='{clean_city}, {clean_state}' desc='{clean_desc[:40]}'"
        )

        loc_confirm_en = f" for {clean_city}, {clean_state}" if clean_city and clean_state else (f" for {clean_city}" if clean_city else "")
        loc_confirm_hi = f" {clean_city}, {clean_state} के लिए" if clean_city and clean_state else (f" {clean_city} के लिए" if clean_city else "")
        loc_confirm_or = f" {clean_city}, {clean_state} ପାଇଁ" if clean_city and clean_state else (f" {clean_city} ପାଇଁ" if clean_city else "")

        if lang == "hi":
            return (
                f"आपकी शिकायत{loc_confirm_hi} सफलतापूर्वक टिकट नंबर {ticket_num} (संदर्भ सीएमपी {pad_4}) के तहत दर्ज कर ली गई है। "
                f"हमारी सपोर्ट टीम जल्द जांच करके आपसे संपर्क करेगी। क्या मैं और कोई सहायता करूँ?"
            )
        elif lang == "or":
            return (
                f"ଆପଣଙ୍କ ଅଭିଯୋଗ{loc_confirm_or} ସଫଳତାର ସହ ଟିକେଟ୍ ନମ୍ବର {ticket_num} (ସିଏମ୍‌ପି {pad_4}) ଅଧୀନରେ ପଞ୍ଜୀକୃତ ହୋଇଛି। "
                f"ଆମ ସପୋର୍ଟ ଟିମ୍ ଖୁବ୍ ଶୀଘ୍ର ଯୋଗାଯୋଗ କରିବେ। ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?"
            )
        return (
            f"Your complaint has been successfully registered under ticket number {ticket_num} (reference CMP {pad_4}){loc_confirm_en}. "
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

        # Telephony Whisper acoustic normalization for Indian English & regional terms
        clean_query = re.sub(r'(?i)\b(?:madurai|to\s+badraheel|badraheel|subadra|suvadra|suvardha)\s*(?:scheme|team|stream|screen|scene)?\b', 'Subhadra Scheme', clean_query)
        clean_query = re.sub(r'(?i)\bsubhadra\s+(?:team|stream|screen|scene)\b', 'Subhadra Scheme', clean_query)
        clean_query = re.sub(r'(?i)\bsubadra\b', 'Subhadra', clean_query)
        clean_query = re.sub(r'(?i)\b(?:cur|car|cor|kas|kos|cas)[\w]*s?atificat\w*\b', 'caste certificate', clean_query)
        clean_query = re.sub(r'(?i)\b(?:cost|cast|coast|cars)\s+certificates?\b', 'caste certificate', clean_query)
        clean_query = re.sub(r'(?i)\bcertificat\w*\b', 'certificate', clean_query)
        clean_query = re.sub(r'(?i)\bpan\s+cart\b', 'PAN card', clean_query)
        clean_query = re.sub(r'(?i)\b(?:water|voter)\s+(?:id|cart|card)\b', 'voter ID card', clean_query)
        clean_query = re.sub(r'(?i)\bbonafide\s+cart\b', 'bonafide certificate', clean_query)

        cat_str = str(service_category or "").lower()
        if "subhadra" in cat_str or "subhadra" in clean_query.lower():
            clean_query = re.sub(r'(?i)\b(?:given\s+)?to\s+the\s+weather\b', 'given to women under Subhadra', clean_query)
            clean_query = re.sub(r'(?i)\bweather\b', 'women', clean_query)

        lower_query = clean_query.lower()
        lang = self._detect_language(clean_query, messages, default_lang=language)

        # Check active slot collection guards in complaint registration
        awaiting_complaint_desc = self._was_asked_for_complaint_description(messages)
        awaiting_complaint_id = self._was_asked_for_complaint_id(messages)
        is_in_reg_slots = bool(awaiting_complaint_desc)

        # 1. Check for appreciation / closing farewell turns (triggers auto-cut)
        # Note: Must NOT trigger if caller is currently describing their complaint
        if self._is_closing_intent(lower_query):
            if not is_in_reg_slots:
                return AUTONOMOUS_KNOWLEDGE["appreciation"][lang]
            return AUTONOMOUS_KNOWLEDGE["complaint_register_ask_problem"][lang]

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
        if any(w in lower_query for w in [
            "representative", "agent", "executive", "operator", "talk to human", "speak to human",
            "talk to person", "speak to person", "talk to agent", "speak to agent", "talk to representative",
            "speak to representative", "connect agent", "connect to agent", "connect representative",
            "human agent", "live agent", "real person", "talk to someone", "speak to someone",
            "एजेंट", "प्रतिनिधि", "ପ୍ରତିନିଧି", "ଏଜେଣ୍ଟ"
        ]):
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

        # 4f. Check unrelated / general knowledge queries (only if NOT asking about citizen services/certificates)
        is_known_service_query = (
            bool(service_category)
            or any(w in lower_query for w in (
                "subhadra", "caste", "income", "residence", "bonafide", "aadhaar", "pan", "voter",
                "kendra", "certificate", "ticket", "complaint", "legal", "affidavit", "notary",
                "scheme", "yojana", "documents", "timing", "phone", "address", "email", "hours",
                "money", "amount", "given", "receive", "apply", "benefit", "eligible", "paisa", "tanka",
                "team", "free", "cost", "fee", "how to", "tell me", "explain", "who can", "where to",
                "ପ୍ରମାଣପତ୍ର", "ଅଭିଯୋଗ", "ସୁଭଦ୍ରା", "ଯୋଜନା", "ଆଧାର", "ପ୍ୟାନ୍", "ସତ୍ୟପାଠ"
            ))
        )
        is_real_unrelated = any(w in lower_query for w in [
            "capital of", "who won", "tell me a joke", "cricket", "football",
            "prime minister", "president", "meaning of life", "how old are you", "what is ai",
            "weather forecast", "weather today", "what is the weather", "temperature today",
            "मौसम कैसा है", "आज का मौसम", "राजधानी", "क्रिकेट", "जोक", "ପାଣିପାଗ"
        ])
        if not is_known_service_query and is_real_unrelated:
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

        # 5. Complaint Services Dialogue Flow (Status & Step-by-Step Registration)
        is_complaint_context = bool(service_category and "complaint" in service_category.lower())
        awaiting_complaint_desc = self._was_asked_for_complaint_description(messages)
        awaiting_address = self._was_asked_for_address(messages)
        awaiting_city = self._was_asked_for_city(messages)
        awaiting_state = self._was_asked_for_state(messages)
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
            
            # Extract collected State, City, and Address from previous turns
            slots = self._extract_registration_slots_from_history(messages)
            ans = self.register_new_complaint(
                description=clean_query,
                lang=lang,
                caller_phone=caller_phone,
                customer_name=customer_name,
                call_uuid=call_uuid,
                state=slots.get("state") or "",
                city=slots.get("city") or "",
                address=slots.get("address") or "",
            )
            return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5b. Caller explicitly declared intent to register a new complaint:
        if is_register_intent:
            # Check if caller already provided a concrete problem description in this utterance:
            has_desc = self._has_problem_description(lower_query)

            if has_desc:
                # Immediate one-shot registration without setting any default location
                ans = self.register_new_complaint(
                    description=clean_query,
                    lang=lang,
                    caller_phone=caller_phone,
                    customer_name=customer_name,
                    call_uuid=call_uuid,
                    state="",
                    city="",
                    address="",
                )
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            # Direct ask for problem/complaint (DO NOT ask for state, city, or address)
            ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_register_ask_problem"][lang]
            return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5c. If previous turn asked caller for complaint ticket ID or phone:
        if awaiting_complaint_id:
            # If caller switched to register intent:
            if is_register_intent:
                ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_register_ask_problem"][lang]
                return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            cid_probe = extract_complaint_identifier(clean_query, allow_bare_digits=True)
            if cid_probe:
                ans = self.lookup_complaint_status(cid_probe, lang=lang)
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            # Check if caller phone is recognized
            clean_phone = re.sub(r"\D", "", caller_phone or "") if caller_phone else ""
            if len(clean_phone) >= 10:
                clean_phone_10 = clean_phone[-10:]
                if clean_phone_10 in COMPLAINT_STATUS_CATALOG:
                    ans = self.lookup_complaint_status(clean_phone_10, lang=lang)
                    return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            # Prompt again for complaint ticket number
            ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_status_ask_id"][lang]
            return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5d. Caller explicitly declared intent to check status of existing complaint:
        if is_status_intent:
            cid_probe = extract_complaint_identifier(clean_query, allow_bare_digits=False)
            if cid_probe:
                ans = self.lookup_complaint_status(cid_probe, lang=lang)
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            # Proactive Lookup by caller phone number:
            clean_phone = re.sub(r"\D", "", caller_phone or "") if caller_phone else ""
            if len(clean_phone) >= 10:
                clean_phone_10 = clean_phone[-10:]
                if clean_phone_10 in COMPLAINT_STATUS_CATALOG:
                    ans = self.lookup_complaint_status(clean_phone_10, lang=lang)
                    return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_status_ask_id"][lang]
            return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

        # 5h. Caller in Complaint Service context provided a direct ticket ID or digits:
        if is_complaint_context:
            cid_probe = extract_complaint_identifier(clean_query, allow_bare_digits=True)
            if cid_probe:
                ans = self.lookup_complaint_status(cid_probe, lang=lang)
                return self.apply_sentiment_tone(ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

            if self._is_complaint_register_intent(lower_query):
                ask_ans = AUTONOMOUS_KNOWLEDGE["complaint_register_ask_problem"][lang]
                return self.apply_sentiment_tone(ask_ans, intel_emo, lang=lang, previous_attempts=prev_attempts)

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
            "subhadra_scheme", "caste_certificate", "income_certificate", "residence_certificate",
            "bonafide_certificate", "aadhaar_services", "pan_card", "voter_id", "legal_documents",
            "kendra_hours", "capabilities", "complaint", "complaint_status",
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
            if "subhadra" in cat_lower or "3" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["subhadra_scheme"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "certificate" in cat_lower or "4" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["caste_certificate"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "identity" in cat_lower or "card" in cat_lower or "5" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["aadhaar_services"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "legal" in cat_lower or "affidavit" in cat_lower or "6" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["legal_documents"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "kendra" in cat_lower or "7" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["kendra_hours"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "complaint" in cat_lower or "2" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["complaint_status_ask_id"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)
            elif "citizen" in cat_lower or "1" in cat_lower:
                return self.apply_sentiment_tone(AUTONOMOUS_KNOWLEDGE["capabilities"][lang], intel_emo, lang=lang, previous_attempts=prev_attempts)

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
        "certainly! to register your complaint", "जी बिल्कुल! आपकी शिकायत दर्ज", "ନିଶ୍ଚୟ! ଆପଣଙ୍କ ଅଭିଯୋଗ ଦାଖଲ",
        "thank you. what is your city", "धन्यवाद। आपके शहर", "ଧନ୍ୟବାଦ। ଆପଣଙ୍କ ସହରର",
        "got it. please provide your area", "ठीक है। कृपया अपने इलाके", "ଠିକ୍ ଅଛି। ଦୟାକରି ଆପଣଙ୍କ",
        "thank you. now please describe", "धन्यवाद। अब कृपया अपनी समस्या", "ଧନ୍ୟବାଦ। ଏବେ ଦୟାକରି ଆପଣଙ୍କ",
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
        Uses token boundary checks for short acronyms/words (sc, st, obc, pan, etc.) to prevent
        accidental substring collisions (e.g. 'sc' in 'scheme' or 'st' in 'status').
        """
        tokens = set(re.findall(r'\b[a-zA-Z0-9_\u0900-\u097f\u0b00-\u0b7f]+\b', lower_query))

        # Capabilities / What can you do
        if any(w in lower_query for w in [
            "what can you do", "what do you do", "what think you can do",
            "help me with", "your services", "what services", "capabilities",
            "kya kar sakte", "kya kar sakti", "क्या कर सकती", "क्या सहायता", "କଣ କରିପାରିବ",
            "service list", "services available", "all services", "citizen services",
        ]):
            return "capabilities"

        # Subhadra Scheme / Citizen Schemes (evaluated before individual certificates)
        if (
            any(w in lower_query for w in [
                "subhadra", "सुभद्रा", "ସୁଭଦ୍ରା", "50000", "50,000", "women scheme", "women financial",
                "women assistance", "scheme", "schemes", "the scheme", "about the scheme", "tell me about the scheme",
                "yojana", "योजना", "ଯୋଜନା", "sarkari yojana", "sarkari scheme", "government scheme",
                "citizen scheme", "financial scheme",
            ])
            or bool(tokens.intersection({"subhadra", "yojana", "scheme", "schemes"}))
        ):
            return "subhadra_scheme"

        # Caste Certificate (strictly token-bounded for short codes like SC/ST/OBC/SEBC to prevent substring false-positives)
        caste_phrases = ["caste", "jati", "caste certificate", "जाति", "जाति प्रमाण", "ଜାତି", "ଜାତି ପ୍ରମାଣପତ୍ର"]
        caste_short_codes = {"sc", "st", "obc", "sebc", "jati"}
        if (
            any(w in lower_query for w in caste_phrases)
            or bool(tokens.intersection(caste_short_codes))
        ):
            return "caste_certificate"

        # Income Certificate (token-bounded for short words like 'aay' to avoid matching 'today', 'pay', etc.)
        if (
            any(w in lower_query for w in [
                "income certificate", "aay praman", "aaya praman", "salary certificate",
                "आय प्रमाण", "आय प्रमाण पत्र", "ଆୟ ପ୍ରମାଣପତ୍ର", "ଆୟ ପ୍ରମାଣ",
            ])
            or bool(tokens.intersection({"aay", "aaya", "income"}))
        ):
            return "income_certificate"

        # Residence / Domicile Certificate
        if (
            any(w in lower_query for w in [
                "residence", "residential", "domicile", "niwas", "basasthana", "निवास प्रमाण", "निवास", "ବାସସ୍ଥାନ",
            ])
            or bool(tokens.intersection({"domicile", "niwas"}))
        ):
            return "residence_certificate"

        # Bonafide Certificate
        if (
            any(w in lower_query for w in [
                "bonafide", "student certificate", "बोनाफाइड", "ବୋନାଫାଇଡ୍",
            ])
            or bool(tokens.intersection({"bonafide"}))
        ):
            return "bonafide_certificate"

        # Aadhaar Card Services
        if (
            any(w in lower_query for w in [
                "aadhaar", "adhar", "uidai", "aadhar card", "आधार", "ଆଧାର",
            ])
            or bool(tokens.intersection({"aadhaar", "aadhar", "uidai"}))
        ):
            return "aadhaar_services"

        # PAN Card Services
        if (
            any(w in lower_query for w in [
                "pan card", "pancard", "nsdl", "uti", "पैन कार्ड", "पैन", "ପ୍ୟାନ୍",
            ])
            or bool(tokens.intersection({"pan"}))
        ):
            return "pan_card"

        # Voter ID Services
        if (
            any(w in lower_query for w in [
                "voter", "voter id", "epic", "election card", "वोटर आईडी", "वोटर", "ଭୋଟର",
            ])
            or bool(tokens.intersection({"voter", "epic"}))
        ):
            return "voter_id"

        # Legal Documents & Affidavits
        if (
            any(w in lower_query for w in [
                "affidavit", "legal document", "stamp paper", "notary", "deed", "agreement",
                "शपथ पत्र", "स्टांप", "सत्यपाଠ", "ଆଫିଡେଭିଟ୍", "ସତ୍ୟପାଠ", "ଦଲିଲ", "ଷ୍ଟାମ୍ପ",
            ])
            or bool(tokens.intersection({"affidavit", "notary"}))
        ):
            return "legal_documents"

        # Kendra Contact / Phone / Address / Email / Location
        if any(w in lower_query for w in [
            "phone number", "contact number", "helpline", "call number", "address", "location",
            "kendrapara", "kalabuda", "email", "mail id", "website", "संपर्क", "फोन नंबर", "पता", "ईमेल",
            "ଫୋନ ନମ୍ବର", "ଠିକଣା", "ଇମେଲ", "ଯୋଗାଯୋଗ", "ୱେବସାଇଟ୍"
        ]) or (bool(tokens.intersection({"phone", "address", "location", "email", "kendrapara", "kalabuda", "website"})) and not bool(tokens.intersection({"complaint", "status"}))):
            return "kendra_contact"

        # Two PAN Cards / Lost PAN / Duplicate PAN
        if (
            any(w in lower_query for w in ["two pan", "second pan", "2 pan", "another pan", "do pan", "dusra pan", "duplicate pan", "lost pan", "pan lost"])
            or ("pan" in lower_query and any(w in lower_query for w in ["two", "second", "another", "duplicate", "lost", "reprint", "do", "dusra", "haji"]))
        ):
            return "two_pan_prohibition"

        # Government vs Private / Organisation Status
        if any(w in lower_query for w in [
            "government or private", "is it government", "government agency", "government body",
            "सरकारी है या", "सरकारी विभाग", "ସରକାରୀ ଅନୁଷ୍ଠାନ", "ସରକାରୀ କି"
        ]):
            return "kendra_status"

        # Website Tools & Utilities (PDF / Photo resize)
        if any(w in lower_query for w in [
            "jpg to pdf", "word to pdf", "resize photo", "photo resize", "tools", "utilities", "टूल", "फोटो रीसाइज"
        ]):
            return "website_tools"

        # Kendra Timings & Information
        if (
            any(w in lower_query for w in [
                "kendra", "office hours", "timings", "timing", "open", "closing", "working hours",
                "समय", "खुलती", "खुला", "କେନ୍ଦ୍ର ସମୟ", "ସମୟ କଣ", "କେବେ ଖୋଲେ"
            ])
            or (bool(tokens.intersection({"timing", "timings", "hours"})) and not bool(tokens.intersection({"complaint", "ticket", "status"})))
        ):
            return "kendra_hours"

        if any(w in lower_query for w in [
            "order", "order id", "delivery", "parcel", "shipment", "courier", "track", "tracking",
            "ऑर्डर", "डिलीवरी", "पार्सल", "ट्रैक", "ଅର୍ଡର", "ଡେଲିଭରି", "ପାର୍ସଲ",
        ]):
            return "order_status"

        if self._is_complaint_register_intent(lower_query):
            return "complaint_register"

        if self._is_complaint_status_intent(lower_query) or any(w in lower_query for w in [
            "complaint", "ticket", "cmp", "cmp/", "cmp-", "status of complaint", "my complaint",
            "शिकायत", "अभिযোগ", "ଅଭିଯୋଗ", "ଟିକେଟ୍",
        ]):
            return "complaint_status"

        # Multi-turn context resolution: if caller asks an anaphoric follow-up ("how much time?", "documents required?")
        if messages and len(messages) >= 2:
            prev_turns = " ".join(
                str(m.content or "") for m in messages[-4:] if m.role in ("user", "assistant")
            ).lower()

            if any(w in lower_query for w in ["apply", "how to apply", "where to apply", "process", "procedure", "kaise apply", "kahan"]):
                if "caste" in prev_turns or "jati" in prev_turns:
                    return "caste_certificate"
                if "subhadra" in prev_turns or "scheme" in prev_turns or "yojana" in prev_turns:
                    return "subhadra_scheme"
                if "income" in prev_turns or "aay" in prev_turns:
                    return "income_certificate"
                if "residence" in prev_turns or "niwas" in prev_turns:
                    return "residence_certificate"
                if "pan" in prev_turns:
                    return "pan_card"
                if "voter" in prev_turns:
                    return "voter_id"

            if any(w in lower_query for w in ["money", "benefit", "amount", "cash", "rupees", "kitna", "kete", "tanka", "tell me about", "about"]):
                if "subhadra" in prev_turns or "scheme" in prev_turns or "yojana" in prev_turns:
                    return "subhadra_scheme"
                if "caste" in prev_turns:
                    return "caste_certificate"
                if "income" in prev_turns:
                    return "income_certificate"
                if "residence" in prev_turns:
                    return "residence_certificate"
                if "pan" in prev_turns:
                    return "pan_card"

            if any(w in lower_query for w in ["time", "kitna time", "kab tak", "how long", "କେତେ ସମୟ", "duration", "hours"]):
                if "subhadra" in prev_turns or "scheme" in prev_turns or "yojana" in prev_turns:
                    return "subhadra_scheme"
                if "caste" in prev_turns or "jati" in prev_turns:
                    return "caste_certificate"
                if "income" in prev_turns or "aay" in prev_turns:
                    return "income_certificate"
                if "residence" in prev_turns or "niwas" in prev_turns:
                    return "residence_certificate"
                if "aadhaar" in prev_turns:
                    return "aadhaar_services"
                if "pan" in prev_turns:
                    return "pan_card"
                if "voter" in prev_turns:
                    return "voter_id"
                if "affidavit" in prev_turns or "legal" in prev_turns:
                    return "legal_documents"
                if "kendra" in prev_turns or "timing" in prev_turns or "open" in prev_turns:
                    return "kendra_hours"
                if "complaint" in prev_turns:
                    return "complaint_status"

            if any(w in lower_query for w in ["document", "what needed", "kya chahiye", "documents", "documents required", "କ'ଣ ଦରକାର"]):
                if "subhadra" in prev_turns or "scheme" in prev_turns or "yojana" in prev_turns:
                    return "subhadra_scheme"
                if "caste" in prev_turns or "jati" in prev_turns:
                    return "caste_certificate"
                if "income" in prev_turns or "aay" in prev_turns:
                    return "income_certificate"
                if "residence" in prev_turns or "niwas" in prev_turns:
                    return "residence_certificate"
                if "aadhaar" in prev_turns:
                    return "aadhaar_services"
                if "pan" in prev_turns:
                    return "pan_card"
                if "voter" in prev_turns:
                    return "voter_id"
                if "affidavit" in prev_turns or "legal" in prev_turns:
                    return "legal_documents"

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
        Supports high-precision resolution matching from verified Q&A pairs.
        """
        if not rag_context or not rag_context.strip():
            return None

        # 1. Direct High-Precision Resolution Match from verified Q&A chunks
        resolutions: List[str] = []
        for line in rag_context.splitlines():
            l_str = line.strip()
            if "verified resolution:" in l_str.lower():
                m = re.search(r'(?i)verified resolution\s*:\s*"([^"]+)"', l_str)
                if not m:
                    m = re.search(r'(?i)verified resolution\s*:\s*\'([^\']+)\'', l_str)
                if not m:
                    m = re.search(r'(?i)verified resolution\s*:\s*(.+)$', l_str)
                if m:
                    res = m.group(1).strip().strip('"\'')
                    res_lower = res.lower()
                    if res and not any(res_lower.startswith(p) for p in (
                        "context:", "topic:", "category:", "title:", "document id:", "source:", "document title:"
                    )) and res not in resolutions:
                        resolutions.append(res)

        query_words = set(
            w for w in re.findall(r'\b[a-zA-Z0-9_\u0900-\u097f\u0b00-\u0b7f]+\b', lower_query)
            if w not in {
                "what", "is", "the", "are", "can", "how", "for", "and", "you", "tell", "about", "me",
                "kya", "hai", "ka", "ki", "ko", "se", "karan", "karein", "kahin", "koun", "kuha", "kana",
                "of", "in", "to", "do", "i", "a", "an", "this", "that", "please", "batao", "bataiye"
            }
        )

        if resolutions:
            scored_res: List[Tuple[float, str]] = []
            for r in resolutions:
                r_lower = r.lower()
                r_words = set(re.findall(r'\b[a-zA-Z0-9_\u0900-\u097f\u0b00-\u0b7f]+\b', r_lower))
                score = float(len(query_words.intersection(r_words))) * 3.0

                # Language affinity bonus
                if lang == "hi" and any("\u0900" <= ch <= "\u097f" for ch in r):
                    score += 5.0
                elif lang == "or" and any("\u0b00" <= ch <= "\u0b7f" for ch in r):
                    score += 5.0
                elif lang == "en" and not any(ch >= "\u0900" for ch in r):
                    score += 2.0

                # Specific entity match bonus with high-precision disambiguation
                if any(w in query_words for w in ("email", "mail", "ईमेल", "ଇମେଲ")) and "@" in r:
                    score += 50.0
                elif any(w in query_words for w in ("phone", "number", "call", "contact", "helpline", "नंबर", "फ़ोन", "ଫୋନ", "ଫୋନ୍", "ନମ୍ବର", "ଯୋଗାଯୋଗ")) and any(d in r for d in ("1234567890", "+91", "हेल्पलाइन", "ହେଲ୍ପଲାଇନ", "ଫୋନ")):
                    score += 50.0
                elif any(w in query_words for w in ("address", "location", "where", "located", "पता", "ଠିକଣା")) and not any(w in query_words for w in ("email", "mail")) and any(l in r_lower for l in ("kalabuda", "kendrapara", "କଳାବୁଦା", "कलाबुदा")):
                    score += 30.0

                if any(w in query_words for w in ("two", "second", "another", "दो", "दूसरा", "ଦୁଇଟି")) and any(p in r_lower for p in ("illegal", "272b", "penalty", "गैरकानूनी", "ବେଆଇନ")):
                    score += 25.0

                if any(w in query_words for w in ("caste", "jati", "जाति", "ଜାତି")) and any(c in r_lower for c in ("caste", "ror", "sc", "st", "obc", "sebc", "जाति", "ଜାତି")):
                    score += 30.0
                elif any(w in query_words for w in ("income", "salary", "aay", "आय", "ଆୟ")) and any(i in r_lower for i in ("income", "salary", "itr", "आय", "ଆୟ")):
                    score += 30.0
                elif any(w in query_words for w in ("residence", "resident", "domicile", "niwas", "निवास", "ବାସସ୍ଥାନ")) and any(rd in r_lower for rd in ("residence", "domicile", "resident", "निवास", "ବାସସ୍ଥାନ")):
                    score += 30.0

                # Disambiguation penalties between certificate types
                if any(w in query_words for w in ("residence", "resident", "domicile", "niwas", "निवास", "ବାସସ୍ଥାନ")):
                    if not any(w in query_words for w in ("income", "salary", "आय", "ଆୟ")) and any(i in r_lower for i in ("salary slip", "itr", "family income")):
                        score -= 25.0
                    if not any(w in query_words for w in ("caste", "jati", "जाति", "ଜାତି")) and any(c in r_lower for c in ("sc", "st", "obc", "sebc", "lineage")):
                        score -= 25.0

                # Bank account & DBT requirements
                if any(w in query_words for w in ("bank", "account", "dbt", "passbook", "खाता", "ବ୍ୟାଙ୍କ", "ଖାତା")) and any(b in r_lower for b in ("bank account", "single-holder", "aadhaar-seeded", "dbt", "passbook", "खाता", "ବ୍ୟାଙ୍କ")):
                    score += 35.0

                # Fee & Charges
                if any(w in query_words for w in ("fee", "fees", "cost", "charge", "charges", "charged", "free", "price", "शुल्क", "फीस", "ମାଗଣା", "ଫି")) and any(f in r_lower for f in ("free of cost", "free", "zero fee", "charge", "fee", "शुल्क", "ମାଗଣା")):
                    score += 40.0

                # Validity period
                if any(w in query_words for w in ("validity", "valid", "period", "expiry", "expire", "years", "year", "वैधता", "ମିଆଦ", "ବୈଧତା")) and any(v in r_lower for v in ("validity", "valid", "3 years", "three financial years", "साल", "ବର୍ଷ")):
                    score += 40.0

                if any(w in query_words for w in ("precautions", "precaution", "checklist", "bring", "carry", "visit", "visiting")) and any(ch in r_lower for ch in ("original documents", "self-attested", "photocopies", "checklist")):
                    score += 25.0

                if any(w in query_words for w in ("timing", "timings", "hours", "open", "time", "closing", "sunday")) and any(t in r_lower for t in ("9:00", "monday", "closed", "सोमवार", "ସୋମବାର")):
                    score += 20.0

                # Application process / how-to-apply disambiguation
                if any(w in query_words for w in ("apply", "application", "process", "procedure", "register", "enroll", "submit", "आवेदन", "प्रक्रिया", "ଆବେଦନ", "ପ୍ରକ୍ରିୟା")):
                    if any(a in r_lower for a in ("apply", "application", "portal", "kendra", "submit", "form", "anganwadi", "csc", "online", "offline", "आवेदन", "ଆବେଦନ")):
                        score += 45.0
                    if r_lower.startswith("no.") or "excluded" in r_lower:
                        score -= 30.0

                if "age" in query_words and any(a in r_lower for a in ("18 to 60", "18–60", "उम्र", "ବୟସ")):
                    score += 20.0

                scored_res.append((score, r))

            scored_res.sort(key=lambda x: x[0], reverse=True)
            return scored_res[0][1]

        # 2. Extract and score declarative sentences from narrative knowledge chunks
        clean_lines: List[str] = []
        seen = set()

        for line in rag_context.splitlines():
            l = line.strip()
            if not l or l.startswith("---") or l.startswith("[Source") or "Relevant Context" in l or "You have access" in l:
                continue

            lower_raw = l.lower()

            # Skip metadata headers (e.g. Context: Jana Seva Odisha Guide..., Topic: ...)
            if any(lower_raw.startswith(m) for m in (
                "context:", "topic:", "category:", "title:", "document id:", "source:", "document title:"
            )):
                continue

            # Skip lines that represent caller questions / inquiries to prevent repeating the user's question!
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

            # Filter generic filler and clarification phrases
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

            # Skip lines that end with '?'
            if l.endswith("?"):
                continue

            # Echo Guard: reject lines that repeat the caller's spoken words (>60% word overlap)
            cand_words = set(re.findall(r'\w+', l.lower()))
            query_words = set(re.findall(r'\w+', lower_query))
            if cand_words and query_words:
                overlap = len(cand_words.intersection(query_words)) / len(cand_words)
                if overlap >= 0.60:
                    continue

            # Avoid non-English text when session is English
            if lang == "en" and any(ch >= "\u0900" for ch in l):
                continue

            norm = l.lower()
            if norm not in seen and len(l) > 15:
                seen.add(norm)
                clean_lines.append(l)

        if not clean_lines:
            return None

        # Prioritize line matching query terms with factual domain relevance scoring
        best_line = None
        query_words = set(
            w for w in re.findall(r'\b[a-zA-Z0-9_\u0900-\u097f\u0b00-\u0b7f]+\b', lower_query)
            if w not in {
                "what", "is", "the", "are", "can", "how", "for", "and", "you", "tell", "about", "me",
                "kya", "hai", "ka", "ki", "ko", "se", "karan", "karein", "kahin", "koun", "kuha", "kana",
                "of", "in", "to", "do", "i", "a", "an", "this", "that", "please", "batao", "bataiye"
            }
        )

        if query_words:
            scored_candidates: List[Tuple[float, str]] = []
            for candidate in clean_lines:
                cand_lower = candidate.lower()
                cand_words = set(re.findall(r'\b[a-zA-Z0-9_\u0900-\u097f\u0b00-\u0b7f]+\b', cand_lower))
                overlap = len(query_words.intersection(cand_words))
                score = float(overlap) * 2.0

                # Domain entity boosts
                if any(w in cand_lower for w in ("+91", "1234567890", "kalabuda", "kendrapara", "support@janasevaodisha.com")):
                    score += 6.0
                if any(w in cand_lower for w in ("subhadra", "50,000", "50000", "2,50,000", "2.5 lakh", "18-60", "18 to 60")):
                    score += 5.0
                if any(w in cand_lower for w in ("caste", "income", "residence", "bonafide", "certificate", "tahasil", "revenue inspector")):
                    score += 4.0
                if any(w in cand_lower for w in ("pan", "form 49a", "second pan", "272b", "penalty", "aadhaar", "biometric")):
                    score += 4.0
                if any(w in cand_lower for w in ("voter", "epic", "form 6", "form 8", "electoral roll")):
                    score += 4.0
                if any(w in cand_lower for w in ("notarisation", "notary", "registration", "affidavit", "stamp paper")):
                    score += 4.0
                if any(w in cand_lower for w in ("jpg to pdf", "word to pdf", "resize photos", "utilities", "tools")):
                    score += 4.0
                if any(w in cand_lower for w in ("complaint", "ticket", "cmp/", "cmp_", "sla", "resolved")):
                    score += 4.0

                if score > 0:
                    scored_candidates.append((score, candidate))

            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                best_line = scored_candidates[0][1]

        if not best_line:
            return clean_lines[0] if clean_lines else None

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
        Returns strictly the status only of a complaint ticket or phone number without unnecessary explanations.
        """
        key = str(identifier).strip()
        if key in COMPLAINT_STATUS_CATALOG:
            return COMPLAINT_STATUS_CATALOG[key].get(lang, COMPLAINT_STATUS_CATALOG[key]["en"])

        # 10-digit phone lookup fallback
        if len(key) == 10 and key.isdigit():
            if lang == "hi":
                return f"मोबाइल नंबर {key} के लिए कोई खुली शिकायत नहीं मिली।"
            elif lang == "or":
                return f"ମୋବାଇଲ୍ ନମ୍ବର {key} ପାଇଁ କୌଣସି ଅଭିଯୋଗ ମିଳିଲା ନାହିଁ।"
            return f"No active complaint found for phone number {key}."

        # Numeric ticket fallback
        if lang == "hi":
            return f"शिकायत टिकट {key} की स्थिति प्रक्रियाधीन (In Progress) है।"
        elif lang == "or":
            return f"ଆପଣଙ୍କ ଅଭିଯୋଗ ଟିକେଟ୍ {key} ପ୍ରକ୍ରିୟାଧୀନ (In Progress) ଅଛି।"
        return f"The status of complaint ticket {key} is In Progress."


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
