"""API Key management and usage statistics."""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import aiosqlite

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyConfig:
    name: str
    rate_limit: str


class KeyManager:
    """Manages API keys and tracks usage statistics in SQLite."""

    def __init__(self, keys_path: Path, db_path: Path) -> None:
        self.keys_path = keys_path
        self.db_path = db_path
        self._keys: Dict[str, KeyConfig] = {}
        self.reload_keys()

    def reload_keys(self) -> None:
        """Load API keys from JSON file."""
        if not self.keys_path.exists():
            _LOGGER.warning("Keys file not found at %s", self.keys_path)
            self._keys = {}
            return

        try:
            with open(self.keys_path, "r") as f:
                data = json.load(f)
                self._keys = {
                    key: KeyConfig(name=v["name"], rate_limit=v["rate_limit"])
                    for key, v in data.items()
                }
            _LOGGER.info("Loaded %d API keys", len(self._keys))
        except Exception as err:
            _LOGGER.error("Failed to load keys: %s", err)

    def save_keys(self) -> None:
        """Save the current keys mapping back to the JSON file."""
        try:
            data = {
                k: {"name": v.name, "rate_limit": v.rate_limit}
                for k, v in self._keys.items()
            }
            with open(self.keys_path, "w") as f:
                json.dump(data, f, indent=2)
            _LOGGER.info("Saved keys to %s", self.keys_path)
        except Exception as err:
            _LOGGER.error("Failed to save keys: %s", err)

    def add_key(self, api_key: str, name: str, rate_limit: str) -> None:
        """Add or update an API key."""
        self._keys[api_key] = KeyConfig(name=name, rate_limit=rate_limit)
        self.save_keys()

    def delete_key(self, api_key: str) -> bool:
        """Remove an API key. Returns True if found and deleted."""
        if api_key in self._keys:
            del self._keys[api_key]
            self.save_keys()
            return True
        return False

    def list_keys(self) -> None:
        """Print all configured keys."""
        if not self._keys:
            print("No API keys configured.")
            return

        print("\nConfigured API Keys:")
        print(f"{'Name':<20} | {'Rate Limit':<15} | {'Key (First 8 chars)':<15}")
        print("-" * 60)
        for key, config in self._keys.items():
            masked_key = f"{key[:8]}..." if len(key) > 8 else key
            print(f"{config.name:<20} | {config.rate_limit:<15} | {masked_key:<15}")
        print("\n")

    def get_key_config(self, api_key: str) -> Optional[KeyConfig]:
        """Get the configuration for a specific API key."""
        return self._keys.get(api_key)

    async def setup_db(self) -> None:
        """Initialize the statistics database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS stats (
                    api_key_id TEXT,
                    hour_timestamp INTEGER,
                    request_count INTEGER DEFAULT 0,
                    PRIMARY KEY (api_key_id, hour_timestamp)
                )
                """
            )
            await db.commit()

    async def log_usage(self, api_key: str) -> None:
        """Increment request count for the current hour."""
        key_config = self.get_key_config(api_key)
        if not key_config:
            return

        # Hour-level resolution for stats
        hour_ts = int(time.time() // 3600) * 3600
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO stats (api_key_id, hour_timestamp, request_count)
                VALUES (?, ?, 1)
                ON CONFLICT(api_key_id, hour_timestamp) DO UPDATE SET
                request_count = request_count + 1
                """,
                (key_config.name, hour_ts),
            )
            await db.commit()

    async def get_stats(self, days: int = 7) -> list:
        """Retrieve stats for the last N days."""
        since = int(time.time() - (days * 86400))
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT api_key_id, hour_timestamp, request_count FROM stats WHERE hour_timestamp > ? ORDER BY hour_timestamp DESC",
                (since,),
            ) as cursor:
                return await cursor.fetchall()

    async def print_stats_table(self, days: int = 7) -> None:
        """Fetch and print usage statistics in a table format."""
        stats = await self.get_stats(days)
        if not stats:
            print(f"No usage stats found for the last {days} days.")
            return

        print(f"\nUsage Statistics (Last {days} days):")
        print(f"{'API Key (Name)':<25} | {'Hour (UTC)':<20} | {'Requests':<10}")
        print("-" * 60)
        
        for key_name, ts, count in stats:
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            dt_str = dt.strftime("%Y-%m-%d %H:00")
            print(f"{key_name:<25} | {dt_str:<20} | {count:<10}")
        print("\n")
