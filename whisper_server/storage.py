"""Audio storage and retention management."""

from __future__ import annotations

import asyncio
import datetime
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

import aiofiles

_LOGGER = logging.getLogger(__name__)


class StorageManager:
    """Saves audio/transcripts and enforces retention policies."""

    def __init__(
        self,
        base_dir: Path,
        retention_days: int = 30,
        max_gb: float = 10.0,
    ) -> None:
        self.base_dir = base_dir
        self.retention_days = retention_days
        self.max_gb = max_gb
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_request(
        self,
        audio_data: bytes,
        raw_text: str,
        llm_text: Optional[str],
        key_name: str,
    ) -> str:
        """Save audio and text results to disk."""
        request_id = str(uuid.uuid4())
        today = datetime.date.today().isoformat()
        day_dir = self.base_dir / today
        day_dir.mkdir(parents=True, exist_ok=True)

        # Save audio
        audio_path = day_dir / f"{request_id}.wav"
        async with aiofiles.open(audio_path, mode="wb") as f:
            await f.write(audio_data)

        # Save metadata and results
        meta_path = day_dir / f"{request_id}_result.json"
        import json
        result_data = {
            "id": request_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "key_name": key_name,
            "raw_text": raw_text,
            "llm_text": llm_text,
        }
        async with aiofiles.open(meta_path, mode="w") as f:
            await f.write(json.dumps(result_data, indent=2))

        return request_id

    async def run_retention_loop(self, interval_seconds: int = 3600) -> None:
        """Background task to periodically clean up old data."""
        while True:
            try:
                await self.enforce_retention()
            except Exception as err:
                _LOGGER.error("Retention enforcement failed: %s", err)
            await asyncio.sleep(interval_seconds)

    async def enforce_retention(self) -> None:
        """Delete files based on age and total directory size."""
        _LOGGER.info("Enforcing storage retention policy...")
        
        # 1. Time-based cleanup
        now = datetime.datetime.now()
        for day_dir in sorted(self.base_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                dir_date = datetime.date.fromisoformat(day_dir.name)
                age = (now.date() - dir_date).days
                if age > self.retention_days:
                    _LOGGER.info("Removing old directory: %s", day_dir)
                    await asyncio.to_thread(shutil.rmtree, day_dir)
            except ValueError:
                continue

        # 2. Size-based cleanup (FIFO)
        while self._get_total_size_gb() > self.max_gb:
            dirs = sorted([d for d in self.base_dir.iterdir() if d.is_dir()])
            if not dirs:
                break
            oldest_dir = dirs[0]
            _LOGGER.info("Storage limit reached. Removing oldest directory: %s", oldest_dir)
            await asyncio.to_thread(shutil.rmtree, oldest_dir)

    def _get_total_size_gb(self) -> float:
        total_size = 0
        for p in self.base_dir.rglob("*"):
            if p.is_file():
                total_size += p.stat().st_size
        return total_size / (1024**3)
