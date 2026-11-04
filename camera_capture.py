"""Helpers for preparing a camera before taking a snapshot."""
from __future__ import annotations

import asyncio
from typing import Any

from homeassistant.core import HomeAssistant

from .const import CONF_PRE_CAPTURE_DELAY_MS, CONF_PRE_CAPTURE_SWITCH


async def async_prepare_camera(hass: HomeAssistant, config: dict[str, Any]) -> None:
    """Turn on an optional light or switch and wait before capturing."""
    switch_entity = config.get(CONF_PRE_CAPTURE_SWITCH)
    if switch_entity:
        domain, separator, _ = switch_entity.partition(".")
        if not separator or domain not in {"switch", "light"}:
            raise ValueError(f"Unsupported pre-capture entity: {switch_entity}")
        await hass.services.async_call(
            domain,
            "turn_on",
            {"entity_id": switch_entity},
            blocking=True,
        )

    delay_ms = config.get(CONF_PRE_CAPTURE_DELAY_MS, 0)
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000)