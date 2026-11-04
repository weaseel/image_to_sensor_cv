"""Image platform for the latest gauge snapshot."""
from __future__ import annotations

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DATA_SNAPSHOTS, DOMAIN
from .snapshot import SnapshotData


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the latest gauge snapshot image entity."""
    snapshot = hass.data[DATA_SNAPSHOTS][config_entry.entry_id]
    async_add_entities([GaugeSnapshotImageEntity(hass, config_entry, snapshot)])


class GaugeSnapshotImageEntity(ImageEntity):
    """Expose the latest annotated gauge image on its device."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        snapshot: SnapshotData,
    ) -> None:
        """Initialize the image entity."""
        super().__init__(hass)
        self._snapshot = snapshot
        self._attr_unique_id = f"{config_entry.entry_id}_latest_snapshot"
        self._attr_name = f"{config_entry.title} Latest Snapshot"
        self._attr_content_type = "image/jpeg"
        self._attr_image_last_updated = snapshot.last_updated
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": config_entry.title,
            "manufacturer": "Image to Sensor CV",
            "model": "Image Processor",
            "sw_version": "1.0.0",
        }

    @property
    def available(self) -> bool:
        """Return whether a snapshot has been taken."""
        return self._snapshot.image is not None

    def image(self) -> bytes | None:
        """Return the latest annotated snapshot."""
        return self._snapshot.image

    async def async_added_to_hass(self) -> None:
        """Listen for snapshots and update the entity state."""
        await super().async_added_to_hass()
        self.async_on_remove(self._snapshot.add_listener(self._handle_snapshot_update))

    def _handle_snapshot_update(self) -> None:
        """Publish the new snapshot timestamp."""
        self._cached_image = None
        self._attr_image_last_updated = self._snapshot.last_updated
        self.async_write_ha_state()