#!/usr/bin/env python3
"""
Switch Ollama Endpoint (Local / LAN / Google Colab Tunnel)
==========================================================
Usage:
    .venv/bin/python scripts/switch_ollama_endpoint.py --url https://xxxx.trycloudflare.com
"""
import argparse
import os
import re
import sys
import httpx

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT_DIR, ".env")
PROVIDERS_FILE = os.path.join(ROOT_DIR, "config", "providers.yaml")


def clean_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def test_endpoint(base_url: str) -> list[str]:
    print(f"Testing connectivity to: {base_url}/api/tags ...")
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            print(f"✅ Connected successfully! Available models: {models}")
            return models
        else:
            print(f"⚠️ Server returned status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    return []


def update_env_file(base_url: str):
    v1_url = f"{base_url}/v1"
    if not os.path.exists(ENV_FILE):
        with open(ENV_FILE, "w") as f:
            f.write(f"OLLAMA_BASE_URL={v1_url}\n")
        print(f"Created {ENV_FILE} with OLLAMA_BASE_URL={v1_url}")
        return

    with open(ENV_FILE, "r") as f:
        content = f.read()

    if re.search(r"^OLLAMA_BASE_URL=.*$", content, flags=re.MULTILINE):
        new_content = re.sub(
            r"^OLLAMA_BASE_URL=.*$",
            f"OLLAMA_BASE_URL={v1_url}",
            content,
            flags=re.MULTILINE,
        )
    else:
        new_content = content.rstrip() + f"\nOLLAMA_BASE_URL={v1_url}\n"

    with open(ENV_FILE, "w") as f:
        f.write(new_content)
    print(f"✅ Updated {ENV_FILE} -> OLLAMA_BASE_URL={v1_url}")


def warm_up_model(base_url: str, model_name: str = "gemma4:latest"):
    print(f"Warming up '{model_name}' on {base_url} with keep_alive=-1...")
    try:
        r = httpx.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "prompt": "Hi", "keep_alive": -1, "options": {"num_predict": 1}},
            timeout=30.0,
        )
        if r.status_code == 200:
            print(f"✅ Model '{model_name}' is pinned in memory!")
        else:
            print(f"⚠️ Warmup returned status {r.status_code}")
    except Exception as e:
        print(f"⚠️ Warmup notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Switch Ollama LLM endpoint to Google Colab Tunnel or LAN host.")
    parser.add_argument("--url", required=True, help="Base URL of Ollama or Colab Tunnel (e.g. https://xxx.trycloudflare.com)")
    parser.add_argument("--model", default="gemma4:latest", help="Model name to warm up (default: gemma4:latest)")
    args = parser.parse_args()

    clean_base = clean_url(args.url)
    models = test_endpoint(clean_base)
    update_env_file(clean_base)

    target_model = args.model
    if models and target_model not in models:
        for m in models:
            if "gemma" in m:
                target_model = m
                break
        else:
            target_model = models[0]
        print(f"ℹ️ Selected model '{target_model}' from available models.")

    warm_up_model(clean_base, target_model)
    print("\n🚀 Configuration updated! To apply, restart the server.")


if __name__ == "__main__":
    main()
