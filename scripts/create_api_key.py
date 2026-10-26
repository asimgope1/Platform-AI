#!/usr/bin/env python3
import asyncio
import sys
import argparse
from platform_core.security.api_keys import generate_api_key
from storage.database import async_session, init_db
from storage.repositories.tenant_repo import TenantRepository
from storage.repositories.api_key_repo import ApiKeyRepository


async def main():
    parser = argparse.ArgumentParser(description="Create a new API Key for a tenant.")
    parser.add_argument("--tenant", type=str, required=True, help="Name of the tenant/organization")
    parser.add_argument("--key-name", type=str, default="Production Key", help="Descriptive name for the key")
    parser.add_argument("--scopes", type=str, default="*", help="Comma-separated scopes (default: '*')")

    args = parser.parse_args()

    await init_db()

    async with async_session() as session:
        tenant_repo = TenantRepository(session)
        api_key_repo = ApiKeyRepository(session)

        tenant = await tenant_repo.get_by_name(args.tenant)
        if not tenant:
            print(f"Tenant '{args.tenant}' not found. Creating tenant...")
            tenant = await tenant_repo.create(args.tenant)

        raw_key, key_hash = generate_api_key()
        await api_key_repo.create(
            tenant_id=tenant.id,
            key_hash=key_hash,
            name=args.key_name,
            scopes=args.scopes,
        )

        print("\n" + "=" * 60)
        print("API KEY GENERATED SUCCESSFULLY")
        print("=" * 60)
        print(f"Tenant ID:   {tenant.id}")
        print(f"Tenant Name: {tenant.name}")
        print(f"Key Name:    {args.key_name}")
        print(f"Scopes:      {args.scopes}")
        print(f"API Key:     {raw_key}")
        print("=" * 60)
        print("IMPORTANT: Store this key safely. You will NOT be able to view it again.\n")


if __name__ == "__main__":
    asyncio.run(main())
