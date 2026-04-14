"""Checkpoint manager for resumable document ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── v6: Checkpoint Configuration ─────────────────────────────────────────────
CHECKPOINT_FILE = "ingest_checkpoint.json"


class CheckpointManager:
    """Track processed filenames in a JSON file for resumable ingestion."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or CHECKPOINT_FILE)
        self.processed: set[str] = set()
        self._dirty = False
        if self.path.exists():
            try:
                self.processed = set(json.loads(self.path.read_text()))
                logger.info(
                    "Loaded checkpoint: %d processed files", len(self.processed)
                )
            except Exception as e:
                logger.warning("Failed to load checkpoint file: %s", e)

    def is_processed(self, filename: str) -> bool:
        """Check if a file has already been processed."""
        return filename in self.processed

    def mark_processed(self, filename: str) -> None:
        """Mark a file as processed."""
        if filename not in self.processed:
            self.processed.add(filename)
            self._dirty = True

    def flush(self) -> None:
        """Save checkpoint to disk if there are changes."""
        if self._dirty:
            try:
                self.path.write_text(json.dumps(list(self.processed)))
                self._dirty = False
                logger.info("Checkpoint flushed: %d files", len(self.processed))
            except Exception as e:
                logger.error("Failed to save checkpoint: %s", e)
