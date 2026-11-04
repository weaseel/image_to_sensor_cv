"""Image to Sensor CV integration for Home Assistant."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .const import DATA_SNAPSHOTS, DOMAIN
from .snapshot import SnapshotData
from .services import async_setup_services, async_unload_services

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.IMAGE]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Image to Sensor CV component."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Image to Sensor CV from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data
    hass.data.setdefault(DATA_SNAPSHOTS, {})[entry.entry_id] = SnapshotData()

    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception as err:
        _LOGGER.error(
            "Error setting up platforms for %s: %s", 
            entry.entry_id, 
            err, 
            exc_info=True
        )
        raise
    
    # Set up services
    await async_setup_services(hass)
    
    # Add update listener for config entry changes
    entry.async_on_unload(entry.add_update_listener(async_update_options))
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id)
        hass.data[DATA_SNAPSHOTS].pop(entry.entry_id, None)
        
        # Unload services if this is the last entry
        if not hass.data[DOMAIN]:
            await async_unload_services(hass)

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options."""
    # Update the stored data with new configuration
    hass.data[DOMAIN][entry.entry_id] = entry.data
    
    # Reload the entry to apply changes
    await hass.config_entries.async_reload(entry.entry_id)


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)