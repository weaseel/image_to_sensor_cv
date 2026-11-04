"""Helpers for preparing a camera before taking a snapshot."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from homeassistant.core import HomeAssistant

from .const import CONF_PRE_CAPTURE_DELAY_MS, CONF_PRE_CAPTURE_SWITCH

_LOGGER = logging.getLogger(__name__)

@asynccontextmanager
async def async_prepare_camera(
    hass: HomeAssistant, config: dict[str, Any]
) -> AsyncIterator[None]:
    """Turn on an optional light or switch for the duration of a capture."""
    switch_entity = config.get(CONF_PRE_CAPTURE_SWITCH)
    switch_domain = None
    if switch_entity:
        switch_domain, separator, _ = switch_entity.partition(".")
        if not separator or switch_domain not in {"switch", "light"}:
            raise ValueError(f"Unsupported pre-capture entity: {switch_entity}")

    switch_enabled = False
    try:
        if switch_entity:
            await hass.services.async_call(
                switch_domain,
                "turn_on",
                {"entity_id": switch_entity},
                blocking=True,
            )
            switch_enabled = True

        delay_ms = config.get(CONF_PRE_CAPTURE_DELAY_MS, 0)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

        yield
    finally:
        if switch_enabled:
            try:
                await hass.services.async_call(
                    switch_domain,
                    "turn_off",
                    {"entity_id": switch_entity},
                    blocking=True,
                )
            except Exception:
                _LOGGER.exception(
                    "Failed to turn off pre-capture entity %s", switch_entity
                )