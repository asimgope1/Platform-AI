#!/usr/bin/env python3
import asyncio
from storage.database import init_db
from platform_core.observability.logging import logger


async def main():
    print("Initializing database tables...")
    await init_db()
    print("Database tables initialized successfully.")


if __name__ == "__main__":
    asyncio.run(main())
