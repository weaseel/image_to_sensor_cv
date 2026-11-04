"""Latest processed image storage for each config entry."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime


class SnapshotData:
    """Store and publish the latest processed snapshot."""

    def __init__(self) -> None:
        """Initialize an empty snapshot store."""
        self.image: bytes | None = None
        self.last_updated: datetime | None = None
        self._listeners: set[Callable[[], None]] = set()

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Register a callback for new snapshots."""
        self._listeners.add(listener)

        def remove_listener() -> None:
            self._listeners.discard(listener)

        return remove_listener

    def set_image(self, image: bytes, last_updated: datetime) -> None:
        """Store a snapshot and notify image entities."""
        self.image = image
        self.last_updated = last_updated
        for listener in tuple(self._listeners):
            listener()