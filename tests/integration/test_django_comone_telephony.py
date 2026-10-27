"""
Integration Test for Com-one Platform v2.0 Django Architecture
Tests the 4-Stage Cognitive Pipeline, Asterisk Bridge, CRM Auto-Surfacing, and Human Escalation.
"""
from django.test import TestCase, Client

class ComoneDjangoTelephonyTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_comone_django_independent_telephony_workflow(self):
        client = self.client

        # 1. Health Endpoint
        health_res = client.get('/health/')
        self.assertEqual(health_res.status_code, 200)
        self.assertEqual(health_res.json()["telephony_mode"], "STANDALONE_SIMULATOR")

        # 2. Start Call Session (Hindi Caller)
        start_payload = {
            "caller_number": "+919876543210",
            "did_number": "1800-COM-ONE",
            "language": "hi",
        }
        start_res = client.post('/api/v2/telephony/calls/start/', data=start_payload, content_type='application/json')
        self.assertEqual(start_res.status_code, 201)
        start_data = start_res.json()
        call_id = start_data["call_id"]
        self.assertTrue(call_id.startswith("CALL-"))
        self.assertEqual(start_data["crm_customer_surfaced"]["name"], "Rahul Kumar")
        self.assertEqual(start_data["crm_customer_surfaced"]["tier"], "VIP Customer")

        # 3. Turn 1: Normal Query (UPI Refund via pgvector RAG)
        turn1_payload = {
            "text": "नमस्ते, मेरा UPI से पैसा कट गया है, रिफंड कब तक आएगा?"
        }
        turn1_res = client.post(f'/api/v2/telephony/calls/{call_id}/turn/', data=turn1_payload, content_type='application/json')
        self.assertEqual(turn1_res.status_code, 200)
        turn1_data = turn1_res.json()
        self.assertGreaterEqual(turn1_data["stage_timings_ms"]["01_perceive"], 0)
        self.assertEqual(turn1_data["intent"], "UPI_REFUND_INQUIRY")
        self.assertFalse(turn1_data["escalated"])
        self.assertTrue(len(turn1_data["ai_response_text"]) > 0)

        # 4. Turn 2: Frustrated Caller -> Human Escalation
        turn2_payload = {
            "text": "बेकार सेवा है, मुझे तुरंत एजेंट से बात करवाओ!"
        }
        turn2_res = client.post(f'/api/v2/telephony/calls/{call_id}/turn/', data=turn2_payload, content_type='application/json')
        self.assertEqual(turn2_res.status_code, 200)
        turn2_data = turn2_res.json()
        self.assertTrue(turn2_data["escalated"])
        self.assertIn(turn2_data["emotion"], ["frustrated", "angry"])
        
        # Verify 5-part Agent Handover Packet
        handover = turn2_data["escalation_handover_packet"]
        self.assertIn("01_full_verbatim_transcript", handover)
        self.assertIn("02_escalation_reason_highlighted", handover)
        self.assertIn("03_intent_sentiment_emotion", handover)
        self.assertIn("04_crm_record_auto_surfaced", handover)
        self.assertIn("05_ai_suggested_resolution_path", handover)
        self.assertEqual(handover["04_crm_record_auto_surfaced"]["name"], "Rahul Kumar")

        # 5. Check Agent Handover Dedicated View
        handover_res = client.get(f'/api/v2/telephony/calls/{call_id}/handover/')
        self.assertEqual(handover_res.status_code, 200)
        self.assertEqual(handover_res.json()["04_crm_record_auto_surfaced"]["customer_id"], "CUST_984512")

        # 6. Supervisor Live Monitor View
        monitor_res = client.get('/api/v2/telephony/supervisor/live-monitor/')
        self.assertEqual(monitor_res.status_code, 200)
        monitor_data = monitor_res.json()
        self.assertGreaterEqual(monitor_data["total_active_calls"], 1)
        matching_call = next(c for c in monitor_data["calls"] if c["call_id"] == call_id)
        self.assertTrue(matching_call["is_escalated"])
        self.assertEqual(matching_call["caller_name"], "Rahul Kumar")

        # 7. End Call -> Generate CDR
        end_payload = {"csat_rating": 4}
        end_res = client.post(f'/api/v2/telephony/calls/{call_id}/end/', data=end_payload, content_type='application/json')
        self.assertEqual(end_res.status_code, 200)
        end_data = end_res.json()
        self.assertEqual(end_data["status"], "completed")
        self.assertEqual(end_data["disposition"], "ESCALATED_TO_HUMAN")
        self.assertEqual(end_data["csat_rating"], 4)
