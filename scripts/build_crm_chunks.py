"""
Build CRM Knowledge Dataset from PostgreSQL CRM Database
=========================================================
Extracts live CRM records (Complaints, Customers, Phases, Types, Departments, Follow-ups)
from crm_backend_db and formats them into high-precision, conversational RAG documents.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import asyncpg
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scripts.build_crm_chunks")

HOST = os.getenv("CRM_DB_HOST", "192.168.1.124")
PORT = int(os.getenv("CRM_DB_PORT", "5435"))
DBNAME = os.getenv("CRM_DB_NAME", "crm_backend_db")
USER = os.getenv("CRM_DB_USERNAME", "postgres")
PASSWORD = os.getenv("CRM_DB_PASSWORD")
TENANT_ID = os.getenv("CRM_TENANT_ID", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")


async def extract_and_generate_crm_knowledge() -> str:
    logger.info(f"Connecting to CRM Database on {HOST}:{PORT}/{DBNAME}...")
    conn = await asyncpg.connect(
        host=HOST,
        port=PORT,
        database=DBNAME,
        user=USER,
        password=PASSWORD,
        ssl=False,
        timeout=5,
    )

    phases_rows = await conn.fetch("SELECT complain_phase_sl, complain_phase_name FROM \"ComplainPhaseMaster\"")
    phases = {r["complain_phase_sl"]: r["complain_phase_name"] for r in phases_rows}

    types_rows = await conn.fetch("SELECT complain_type_sl, complain_type_name, escalation_time FROM \"ComplainTypeMasters\"")
    types = {r["complain_type_sl"]: r["complain_type_name"] for r in types_rows}
    types_escalation = {r["complain_type_sl"]: r["escalation_time"] for r in types_rows}

    depts_rows = await conn.fetch("SELECT department_sl, department_name, department_no FROM \"DepartmentMasters\"")
    depts = {r["department_sl"]: r["department_name"] for r in depts_rows}

    customers = await conn.fetch("SELECT * FROM \"Customers\"")
    complaints = await conn.fetch("SELECT * FROM \"ComplainBookings\" ORDER BY createdate ASC")
    followups = await conn.fetch("SELECT * FROM \"ComplainFollowUp\" ORDER BY follow_up_date ASC")

    followup_map: dict[any, list[str]] = {}
    for f in followups:
        c_sl = f["complainsl"]
        desc = (f["description"] or "").strip()
        if desc:
            followup_map.setdefault(c_sl, []).append(desc)

    docs: list[dict[str, any]] = []

    # 1. Individual Complaint Ticket Status Chunks
    for cp in complaints:
        cid = cp["complainid"]
        c_sl = cp["complainsl"]
        cname = cp["customername"] or "Customer"
        phone = cp["phonenumber"] or "registered phone"
        city = cp["city"] or "Odisha"
        state = cp["state"] or "Odisha"
        phase_name = phases.get(cp["complain_phase_sl"], "In Progress")
        type_name = types.get(cp["complain_type_sl"], "General Service Issue")
        dept_name = depts.get(cp["department_sl"], "Customer Support")
        bdate = cp["complaindatetime"].strftime("%d %B %Y") if cp["complaindatetime"] else "recently"

        notes_list = followup_map.get(c_sl, [])
        notes_str = "; ".join(notes_list) if notes_list else "Ticket registered in system"

        spoken_content = (
            f"Service complaint ticket {cid} for customer {cname} (phone number {phone}) in {city}, {state} "
            f"was booked on {bdate}. The current complaint status is '{phase_name}'. "
            f"Issue category is '{type_name}' assigned to {dept_name}. "
            f"Latest progress notes: {notes_str}."
        )

        slug = cid.lower().replace("/", "_").replace("-", "_")
        docs.append({
            "tenant_id": TENANT_ID,
            "category": "crm_complaint_status",
            "topic": f"complaint_{slug}",
            "title": f"Complaint Status for Ticket {cid} ({cname})",
            "content": spoken_content,
            "language": ["en", "hi", "or"],
            "source": "live_crm_database",
            "source_version": "1.0",
            "verified": True,
            "effective_date": "2026-11-18",
            "ask_count": 1,
        })

    # 2. Customer Profiles and Registered Complaints Chunks
    for cust in customers:
        name = cust["name"]
        phone = cust["phonenumber"]
        city = cust["city"] or "Bhubaneswar"
        state = cust["state"] or "Odisha"
        if not phone or len(phone) < 6 or phone in ("aa", "qwq"):
            continue

        cust_c = [cp for cp in complaints if cp["phonenumber"] == phone or cp["customers_id"] == cust["customers_id"]]
        if cust_c:
            tickets_summary = ", ".join(
                [f"Ticket {c['complainid']} (status: {phases.get(c['complain_phase_sl'], 'Pending')})" for c in cust_c]
            )
            c_desc = f"has {len(cust_c)} complaint(s) on file: {tickets_summary}."
        else:
            c_desc = "has no active complaints on file."

        cust_content = (
            f"Customer profile for {name}: Registered contact phone number is {phone}, location {city}, {state}. "
            f"Account record {c_desc}"
        )

        docs.append({
            "tenant_id": TENANT_ID,
            "category": "crm_customer_directory",
            "topic": f"customer_{phone}",
            "title": f"Customer Profile and Account Complaints for {name} ({phone})",
            "content": cust_content,
            "language": ["en", "hi", "or"],
            "source": "live_crm_database",
            "source_version": "1.0",
            "verified": True,
            "effective_date": "2026-11-18",
            "ask_count": 1,
        })

    # 3. Overall Complaint Escalation SLA & Department Routing Policy Chunk
    dept_names_list = ", ".join([f"{name}" for name in depts.values() if name])
    sla_content = (
        "Customer Service Complaint Escalation & Department Routing Protocol: "
        "Complaints in the CRM progress through four standard phases: "
        "1. New: The complaint has been registered and is pending initial review (escalation SLA: 3 hours). "
        "2. Discussion: Active investigation and customer follow-up in progress (escalation SLA: 1 hour). "
        "3. Solved: The issue has been completely resolved and closed with customer confirmation. "
        "4. Canceled: The ticket has been closed or cancelled per customer request. "
        "Specialized routing departments include: "
        f"{dept_names_list}. "
        "Urgent Network Issues are targeted for resolution within 2 hours. "
        "Callers can check the live status of any ticket by providing their Ticket ID (e.g. CMP/0001/26-27) or registered phone number."
    )
    docs.append({
        "tenant_id": TENANT_ID,
        "category": "crm_support_policy",
        "topic": "crm_complaint_phases_and_sla",
        "title": "CRM Complaint Phases, Department Routing, and SLA Resolution Rules",
        "content": sla_content,
        "language": ["en", "hi", "or"],
        "source": "live_crm_database",
        "source_version": "1.0",
        "verified": True,
        "effective_date": "2026-11-18",
        "ask_count": 1,
    })

    out_path = os.path.join(ROOT_DIR, "data", "banking_knowledge", "crm_knowledge.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    await conn.close()
    logger.info(f"Successfully generated {len(docs)} CRM knowledge documents written to {out_path}!")
    return out_path


if __name__ == "__main__":
    asyncio.run(extract_and_generate_crm_knowledge())

