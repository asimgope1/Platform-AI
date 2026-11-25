#!/usr/bin/env python3
"""
Generate Com-One Platform Feature Matrix (XLSX & CSV)
Direct ZIP/XML generator using only Python Standard Library.
"""
import os
import csv
import zipfile
import xml.etree.ElementTree as ET

DATA = [
    # Telephony & PBX Foundation (v1)
    ("Telephony & PBX", "SIP IP Telephony Bridge", "v1 Slide 3", "DONE", 
     "Live bi-directional WebSocket bridge at ws://192.168.1.133:9093 (Asterisk PBX, Ext 8000).", "None (Production active)"),
    ("Telephony & PBX", "Multi-Extension / Department Routing", "v1 Slide 3", "DONE",
     "Routes to extension 8000 with Asterisk dialplan routing support.", "Expand additional ring groups"),
    ("Telephony & PBX", "Post-Call Quality Survey (1-5)", "v1 Slide 4", "DONE",
     "Captured in IVRSession on call completion and stored in metadata.", "Hook into database survey table"),
    ("Telephony & PBX", "Call Transfer to Human Agent (Blind/Attended)", "v1 Slide 3", "NOT DONE",
     "Recognizes transfer intent but lacks Asterisk SIP REFER / AMI bridge.", "Implement Asterisk AMI Transfer command"),
    ("Telephony & PBX", "Call Hold & Resume with Audio Tones", "v1 Slide 3", "NOT DONE",
     "Basic pause handling in code, but no MOH (Music on Hold) or channel park.", "Integrate Asterisk MOH channel bridge"),
    ("Telephony & PBX", "Call Detail Records (CDR) Logging", "v1 Slide 4", "DONE",
     "Call start/end timestamps, caller ID, duration recorded in session.", "Persist into PostgreSQL CDR schema"),
    ("Telephony & PBX", "Dual-Channel Call Recording Archive", "v1 Slide 3", "NOT DONE",
     "Audio is processed in memory but full dual-channel WAV is not archived to disk.", "Configure Asterisk MixMonitor path"),

    # Cognitive Stage 01: Perceive (v2)
    ("01 · Perceive (STT)", "Real-Time Streaming ASR (Whisper)", "v2 Slide 3", "DONE",
     "Whisper-base pre-warmed on CPU in memory with 8kHz PCM telephony input.", "Test faster-whisper CTranslate2 for ~100ms latency"),
    ("01 · Perceive (STT)", "Noise & Telephony Audio Filtering", "v2 Slide 3", "DONE",
     "VAD energy thresholding and silence cutoff implemented in IVRSession.", "Add WebRTC AEC / NS pre-processing"),
    ("01 · Perceive (STT)", "Anti-Hallucination & Short Utterance Guard", "v2 Slide 3", "DONE",
     "Cleaned conditioning prompts to eliminate false 'Thank you' on short city/state utterances.", "None (Fixed & verified)"),
    ("01 · Perceive (STT)", "Multilingual Speech Recognition", "v2 Slide 3", "DONE",
     "Whisper multilingual model loaded for English, Hindi, and Indian dialects.", "Add fine-tuned Odia acoustic model"),

    # Cognitive Stage 02: Understand (v2)
    ("02 · Understand (NLU)", "Multi-Turn Complaint Slot-Filling", "v2 Slide 3", "DONE",
     "Step-by-step state machine: State -> City -> Address -> Complaint description.", "None (Tested & working)"),
    ("02 · Understand (NLU)", "Indian Geographic Entity Normalization", "v2 Slide 3", "DONE",
     "Built-in dictionary for 28 States, 8 UTs, and 50+ major cities (e.g. Mumbai, Delhi, Bhubaneswar).", "Expand tier-2/3 talukas & towns"),
    ("02 · Understand (NLU)", "Anti-Loop State Machine Guard", "v2 Slide 3", "DONE",
     "Guarded _is_closing_intent to prevent premature 'have a wonderful day' during active slot-filling.", "None (Fixed & verified)"),
    ("02 · Understand (NLU)", "High-Capacity Cloud Reasoning (Gemma 4)", "v2 Slide 8", "DONE",
     "Integrated Google Colab T4 GPU tunnel (Ollama gemma4:latest pinned with keep_alive=-1).", "Colab session restart automation"),
    ("02 · Understand (NLU)", "Ultra-Fast Local RAG Brain", "v2 Slide 8", "DONE",
     "IVRRAGMiniModel provides sub-100ms instant query resolution for banking inquiries.", "Add more domain question sets"),
    ("02 · Understand (NLU)", "Live Sentiment & Frustration Scoring", "v2 Slide 4", "DONE",
     "Calculates sentiment polarity (-1.0 to +1.0) and detects frustration cues.", "Hook into automatic escalation trigger"),
    ("02 · Understand (NLU)", "Mid-Call Language Switch (Eng/Hindi/Odia)", "v2 Slide 7", "PARTIAL",
     "Detects language preference; prompts switch language, but mid-sentence dynamic flipping needs refinement.", "Add mid-sentence bilingual code-switching"),

    # Cognitive Stage 03: Respond (v2)
    ("03 · Respond (TTS)", "Neural Voice Synthesis (Kokoro)", "v2 Slide 3", "DONE",
     "Kokoro ONNX engine pre-loaded for high naturalness English & Hindi speech.", "Optimize sample-rate downsampling"),
    ("03 · Respond (TTS)", "Native Regional Persona (Odia VITS)", "v2 Slide 7", "DONE",
     "Integrated speech_od_vits.onnx for authentic Odia voice generation.", "Add Bengali/Tamil regional models"),
    ("03 · Respond (TTS)", "Voice Profile Consistency", "v2 Slide 3", "DONE",
     "Unified speaker persona across status replies, greetings, and complaint acknowledgments.", "None (Fixed & verified)"),
    ("03 · Respond (TTS)", "Sub-100ms Pre-Rendered Audio Cache", "v2 Slide 8", "DONE",
     "52+ core IVR prompts pre-generated and cached in memory/disk.", "Pre-cache top 100 FAQ audio chunks"),

    # Cognitive Stage 04: Act (CRM & Data) (v1 & v2)
    ("04 · Act (CRM)", "Autonomous Complaint Registration", "v2 Slide 3", "DONE",
     "Generates CMP-XXXXXX tracking ID and writes directly to PostgreSQL crm_backend_db.", "None (Live in production)"),
    ("04 · Act (CRM)", "Real-Time Complaint Status Lookup", "v2 Slide 5", "DONE",
     "Lookup by CMP ID or order last 4 digits directly from live database.", "Support phone number lookup fallback"),
    ("04 · Act (CRM)", "Offline Resilience Queue & Auto-Flush", "v2 Slide 3", "DONE",
     "Buffered in pending_crm_complaints.json and flushed by daemon on DB reconnection.", "Add SMS notification on successful flush"),
    ("04 · Act (CRM)", "Lead Management Pipeline (5 Stages)", "v1 Slide 5", "NOT DONE",
     "Database tables need schema for New -> Contacted -> Qualified -> Proposal -> Converted.", "Implement Lead CRM models & API"),
    ("04 · Act (CRM)", "Follow-Up Scheduling & Calendar Booking", "v1 Slide 5", "NOT DONE",
     "Follow-up scheduling logic not connected to agent calendar/queue.", "Add appointment booking slot engine"),

    # Human-in-the-Loop & Escalation (v2)
    ("Human-in-the-Loop", "Explicit Human Agent Handover", "v2 Slide 4", "PARTIAL",
     "Detects 'talk to agent', but needs live SIP channel bridge to agent softphone queue.", "Connect to Asterisk Queue 100"),
    ("Human-in-the-Loop", "Negative Sentiment Spike Auto-Escalation", "v2 Slide 4", "NOT DONE",
     "Sentiment score calculated, but threshold-based auto-transfer not yet hooked.", "Trigger transfer when sentiment < -0.6"),
    ("Human-in-the-Loop", "Conversation Repeat Loop Escalation", "v2 Slide 4", "NOT DONE",
     "Loop counter not yet triggering automatic handover after 2 repeated queries.", "Add repeat query counter trigger"),
    ("Human-in-the-Loop", "High-Value / VIP Caller Priority Flag", "v2 Slide 4", "NOT DONE",
     "CRM schema has customer profiles, but VIP tag doesn't bypass IVR queue yet.", "Query VIP flag on incoming caller ID"),
    ("Human-in-the-Loop", "Agent Screen Pop at Handover", "v2 Slide 4", "NOT DONE",
     "No frontend modal displaying live transcript, sentiment, and CRM record to agent.", "Build WebSocket agent pop-up UI"),

    # Advanced Intelligence & Compliance (v2)
    ("Advanced AI", "Persistent Cross-Session Caller Memory", "v2 Slide 7", "NOT DONE",
     "Session memory resets on call hangup; does not recall previous tickets on new call.", "Add caller ID history lookup at call start"),
    ("Advanced AI", "PII Redaction & Compliance Layer", "v2 Slide 7", "NOT DONE",
     "Aadhaar, PAN, and card numbers are stored in raw text in transcripts.", "Add regex & NER redaction filter (e.g. XXXX-1234)"),
    ("Advanced AI", "Continuous Learning & Feedback Loop", "v2 Slide 7", "NOT DONE",
     "Unresolved/escalated calls are logged but not exported to training datasets.", "Create daily fine-tuning export pipeline"),

    # Analytics, Monitoring & HA (v1)
    ("Analytics & Monitoring", "Real-Time Supervisor Live Dashboard", "v1 Slide 4", "NOT DONE",
     "Backend metrics logged to JSON; no real-time web dashboard for active calls.", "Build FastAPI SSE/WebSocket live dashboard"),
    ("Analytics & Monitoring", "Agent Performance & CSAT Reporting", "v1 Slide 4", "NOT DONE",
     "Ratings captured in session, but no aggregated reporting interface.", "Create CSAT & SLA summary reports"),
    ("Video & HA", "Call-to-WebRTC Video Escalation", "v1 Slide 7", "NOT DONE",
     "No browser video room generation or SMS invite link dispatch.", "Implement Jitsi / WebRTC room link generator"),
    ("Video & HA", "Active-Passive PBX High Availability Cluster", "v1 Slide 8", "NOT DONE",
     "Single Asterisk instance currently running; no Corosync/Pacemaker VIP.", "Deploy secondary Asterisk standby node"),
]

def make_cell(col_idx, row_idx, value, style_idx=0):
    col_letter = chr(ord('A') + col_idx)
    ref = f"{col_letter}{row_idx}"
    val_clean = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', '&quot;')
    return f'<c r="{ref}" t="inlineStr" s="{style_idx}"><is><t>{val_clean}</t></is></c>'

def generate_xlsx(filename):
    # XML Components
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Platform Feature Status" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""

    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="4">
    <font><name val="Calibri"/><sz val="11"/></font>
    <font><b/><name val="Calibri"/><sz val="11"/><color rgb="FFFFFFFF"/></font>
    <font><b/><name val="Calibri"/><sz val="11"/><color rgb="FF1B5E20"/></font>
    <font><b/><name val="Calibri"/><sz val="11"/><color rgb="FFB71C1C"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F2937"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE8F5E9"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFEBEE"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/></border>
    <border>
      <left style="thin"><color rgb="FFD1D5DB"/></left>
      <right style="thin"><color rgb="FFD1D5DB"/></right>
      <top style="thin"><color rgb="FFD1D5DB"/></top>
      <bottom style="thin"><color rgb="FFD1D5DB"/></bottom>
    </border>
  </borders>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="3" fillId="4" borderId="1" applyFont="1" applyFill="1" applyBorder="1"/>
  </cellXfs>
</styleSheet>"""

    headers = ["Category", "Feature Name", "PDF Reference", "Status", "Current Implementation Details", "Remaining Work / Next Step"]
    
    rows_xml = []
    # Header row
    h_cells = "".join([make_cell(i, 1, h, 1) for i, h in enumerate(headers)])
    rows_xml.append(f'<row r="1" ht="26" customHeight="1">{h_cells}</row>')

    for r_idx, row_data in enumerate(DATA, start=2):
        cells = []
        status = row_data[3]
        for c_idx, val in enumerate(row_data):
            style_idx = 0
            if c_idx == 3:
                if status == "DONE":
                    style_idx = 2
                elif status in ("NOT DONE", "PARTIAL"):
                    style_idx = 3
            cells.append(make_cell(c_idx, r_idx, val, style_idx))
        rows_xml.append(f'<row r="{r_idx}" ht="20" customHeight="1">{"".join(cells)}</row>')

    sheet1 = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <cols>
    <col min="1" max="1" width="22" customWidth="1"/>
    <col min="2" max="2" width="34" customWidth="1"/>
    <col min="3" max="3" width="16" customWidth="1"/>
    <col min="4" max="4" width="14" customWidth="1"/>
    <col min="5" max="5" width="55" customWidth="1"/>
    <col min="6" max="6" width="45" customWidth="1"/>
  </cols>
  <sheetData>
    {"".join(rows_xml)}
  </sheetData>
</worksheet>"""

    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles)
        zf.writestr("xl/worksheets/sheet1.xml", sheet1)

    print(f"✅ Generated {filename}")

def generate_csv(filename):
    headers = ["Category", "Feature Name", "PDF Reference", "Status", "Current Implementation Details", "Remaining Work / Next Step"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(DATA)
    print(f"✅ Generated {filename}")

if __name__ == "__main__":
    out_dir = "/Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform"
    xlsx_path = os.path.join(out_dir, "Com_One_Platform_Feature_Status.xlsx")
    csv_path = os.path.join(out_dir, "Com_One_Platform_Feature_Status.csv")
    generate_xlsx(xlsx_path)
    generate_csv(csv_path)
