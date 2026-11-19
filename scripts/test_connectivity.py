import asyncio
import os
import sys
import time
import socket
import urllib.request
import json

async def test_all_connectivity():
    print("=" * 60)
    print("🔍 TESTING MULTI-ENDPOINT CONNECTIVITY")
    print("=" * 60)

    # 1. Local PostgreSQL
    print("\n1. Local PostgreSQL (localhost:5432):")
    try:
        import asyncpg
        t0 = time.perf_counter()
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password=os.getenv("LOCAL_PG_PASSWORD", "postgres"),
            database="ai_platform_test",
            timeout=3.0,
        )
        t_elapsed = (time.perf_counter() - t0) * 1000
        count = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
        await conn.close()
        print(f"   ✅ CONNECTED ({t_elapsed:.1f}ms) | {count} RAG chunks in ai_platform_test")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    # 2. Remote Staging CRM DB (192.168.1.124:5435)
    print("\n2. Remote Staging CRM DB (192.168.1.124:5435):")
    try:
        import asyncpg
        t0 = time.perf_counter()
        conn = await asyncpg.connect(
            host="192.168.1.124",
            port=5435,
            user="postgres",
            password="epsumlabs",
            database="crm_backend_db",
            ssl=False,
            timeout=5.0,
        )
        t_elapsed = (time.perf_counter() - t0) * 1000
        count = await conn.fetchval('SELECT COUNT(*) FROM "ComplainBookings"')
        latest = await conn.fetchrow('SELECT complainid, customername, complain_phase_sl FROM "ComplainBookings" ORDER BY complainsl DESC LIMIT 1')
        await conn.close()
        print(f"   ✅ CONNECTED ({t_elapsed:.1f}ms) | {count} Total Complaints | Latest: {latest['complainid']} ({latest['customername']})")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    # 3. Ollama Host (192.168.1.117:11434)
    print("\n3. Ollama Host (http://192.168.1.117:11434):")
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request("http://192.168.1.117:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
        t_elapsed = (time.perf_counter() - t0) * 1000
        models = [m.get("name") for m in data.get("models", [])]
        print(f"   ✅ ONLINE ({t_elapsed:.1f}ms) | Available Models: {', '.join(models[:4])}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    # 4. Asterisk PBX Port 9093 (192.168.1.133:9093)
    print("\n4. Asterisk PBX WebSocket (192.168.1.133:9093):")
    try:
        t0 = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect(("192.168.1.133", 9093))
        sock.close()
        t_elapsed = (time.perf_counter() - t0) * 1000
        print(f"   ✅ OPEN / READY ({t_elapsed:.1f}ms) | Port 9093 accessible")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all_connectivity())
