import hashlib
import secrets
from typing import Tuple


API_KEY_PREFIX = "sk_live_"


def generate_api_key() -> Tuple[str, str]:
    """Generates a raw API key and its SHA-256 hash.

    Returns:
        Tuple[str, str]: (raw_api_key, key_hash)
    """
    token = secrets.token_urlsafe(32)
    raw_key = f"{API_KEY_PREFIX}{token}"
    key_hash = hash_api_key(raw_key)
    return raw_key, key_hash


def hash_api_key(raw_key: str) -> str:
    """Hashes an API key using SHA-256."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    """Securely verifies an API key against its stored hash."""
    return secrets.compare_digest(hash_api_key(raw_key), stored_hash)
