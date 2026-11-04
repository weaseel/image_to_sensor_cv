"""Config flow for Image to Sensor CV integration."""
from __future__ import annotations

import logging
import time
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

from aiohttp import web
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components import camera
from homeassistant.components.http import KEY_AUTHENTICATED, HomeAssistantView
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import selector
from homeassistant.components.http.auth import async_sign_path

from .const import (
    DOMAIN,
    CONF_IMAGE_SOURCE,
    CONF_IMAGE_PATH,
    CONF_CAMERA_ENTITY,
    CONF_PRE_CAPTURE_SWITCH,
    CONF_PRE_CAPTURE_DELAY_MS,
    DEFAULT_PRE_CAPTURE_DELAY_MS,
    CONF_CROP_CONFIG,
    CONF_CROP_X,
    CONF_CROP_Y,
    CONF_CROP_WIDTH,
    CONF_CROP_HEIGHT,
    CONF_PROCESSORS,
    CONF_SCAN_INTERVAL,
    CONF_MIN_ANGLE_HOURS,
    CONF_MAX_ANGLE_HOURS,
    CONF_CENTER_OFFSET_X,
    CONF_CENTER_OFFSET_Y,
    CONF_MIN_VALUE,
    CONF_MAX_VALUE,
    CONF_UNITS,
    SOURCE_FILE,
    SOURCE_CAMERA,
    PROCESSOR_ANALOG_GAUGE,
    DEFAULT_CROP_X,
    DEFAULT_CROP_Y,
    DEFAULT_CROP_WIDTH,
    DEFAULT_CROP_HEIGHT,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_MIN_ANGLE_HOURS,
    DEFAULT_MAX_ANGLE_HOURS,
    DEFAULT_CENTER_OFFSET_X,
    DEFAULT_CENTER_OFFSET_Y,
    DEFAULT_MIN_VALUE,
    DEFAULT_MAX_VALUE,
    DEFAULT_UNITS,
)
from .camera_capture import async_prepare_camera

_LOGGER = logging.getLogger(__name__)
_PREVIEW_VIEW_KEY = f"{DOMAIN}_preview_view"
_PREVIEW_MAX_BYTES = 12 * 1024 * 1024
_PREVIEW_SIZE = (960, 720)
_PREVIEW_TTL = 300


class ImagePreviewView(HomeAssistantView):
    """Serve temporary image previews for config flows."""

    url = f"/api/{DOMAIN}/preview/{{flow_id}}"
    name = f"api:{DOMAIN}:preview"
    requires_auth = False

    def __init__(self) -> None:
        """Initialize the preview cache."""
        self._images: dict[str, tuple[float, bytes]] = {}

    def set_image(self, flow_id: str, image: bytes) -> None:
        """Cache a preview image for a short time."""
        now = time.monotonic()
        self._images = {
            key: value for key, value in self._images.items() if value[0] > now
        }
        while len(self._images) >= 20:
            self._images.pop(next(iter(self._images)))
        self._images[flow_id] = (now + _PREVIEW_TTL, image)

    async def get(self, request: web.Request, flow_id: str) -> web.Response:
        """Return a cached preview image."""
        if not request[KEY_AUTHENTICATED]:
            raise web.HTTPUnauthorized

        preview = self._images.get(flow_id)
        if preview is None or preview[0] <= time.monotonic():
            self._images.pop(flow_id, None)
            raise web.HTTPNotFound

        return web.Response(
            body=preview[1],
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )


def _encode_preview(image_data: bytes, crop_config: dict[str, int]) -> bytes:
    """Resize and encode an image for display in the config flow."""
    with Image.open(BytesIO(image_data)) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("Image dimensions are too large for a preview")
        image = ImageOps.exif_transpose(image)
        source_width, source_height = image.size
        image.thumbnail(_PREVIEW_SIZE)
        image = image.convert("RGB")

        scale_x = image.width / source_width
        scale_y = image.height / source_height
        left = round(crop_config.get(CONF_CROP_X, DEFAULT_CROP_X) * scale_x)
        top = round(crop_config.get(CONF_CROP_Y, DEFAULT_CROP_Y) * scale_y)
        right = round(
            (crop_config.get(CONF_CROP_X, DEFAULT_CROP_X)
             + crop_config.get(CONF_CROP_WIDTH, DEFAULT_CROP_WIDTH))
            * scale_x
        )
        bottom = round(
            (crop_config.get(CONF_CROP_Y, DEFAULT_CROP_Y)
             + crop_config.get(CONF_CROP_HEIGHT, DEFAULT_CROP_HEIGHT))
            * scale_y
        )
        if left < right and top < bottom:
            draw = ImageDraw.Draw(image)
            draw.rectangle((left, top, right, bottom), outline="black", width=4)
            draw.rectangle((left, top, right, bottom), outline="#ff3b30", width=2)

        output = BytesIO()
        image.save(output, format="JPEG", quality=80)
        return output.getvalue()


async def _async_load_preview(
    hass: HomeAssistant,
    data: dict[str, Any],
    crop_config: dict[str, int],
) -> bytes:
    """Load a preview from the selected camera or local file."""
    if data[CONF_IMAGE_SOURCE] == SOURCE_CAMERA:
        await async_prepare_camera(hass, data)
        camera_image = await camera.async_get_image(
            hass,
            data[CONF_CAMERA_ENTITY],
            width=_PREVIEW_SIZE[0],
            height=_PREVIEW_SIZE[1],
        )
        image_data = camera_image.content
    else:
        image_path = Path(data[CONF_IMAGE_PATH])

        def read_image() -> bytes:
            if image_path.stat().st_size > _PREVIEW_MAX_BYTES:
                raise ValueError("Image file is too large for a preview")
            return image_path.read_bytes()

        image_data = await hass.async_add_executor_job(read_image)

    if len(image_data) > _PREVIEW_MAX_BYTES:
        raise ValueError("Image is too large for a preview")
    return await hass.async_add_executor_job(
        _encode_preview, image_data, crop_config
    )

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required("name"): str,
        vol.Required(CONF_IMAGE_SOURCE): selector.SelectSelector(
            selector.SelectSelectorConfig(options=[
                {"value": SOURCE_FILE, "label": "Local Image File"},
                {"value": SOURCE_CAMERA, "label": "Home Assistant Camera"},
            ])
        ),
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    # TODO: Add validation for image path or camera entity
    return {"title": data["name"]}


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Image to Sensor CV."""

    VERSION = 1

    def __init__(self):
        """Initialize the config flow."""
        self.data = {}

    @staticmethod
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            info = await validate_input(self.hass, user_input)
        except CannotConnect:
            errors["base"] = "cannot_connect"
        except InvalidAuth:
            errors["base"] = "invalid_auth"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            self.data.update(user_input)
            return await self.async_step_image_config()

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_image_config(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure image source."""
        if user_input is None:
            if self.data[CONF_IMAGE_SOURCE] == SOURCE_FILE:
                schema = vol.Schema({
                    vol.Required(CONF_IMAGE_PATH): str,
                })
            else:
                schema = vol.Schema({
                    vol.Required(CONF_CAMERA_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(domain="camera")
                    ),
                    vol.Optional(CONF_PRE_CAPTURE_SWITCH): selector.EntitySelector(
                        selector.EntitySelectorConfig(domain=["switch", "light"])
                    ),
                    vol.Optional(
                        CONF_PRE_CAPTURE_DELAY_MS,
                        default=DEFAULT_PRE_CAPTURE_DELAY_MS,
                    ): vol.All(vol.Coerce(int), vol.Range(min=0, max=60000)),
                })
            
            return self.async_show_form(step_id="image_config", data_schema=schema)

        self.data.update(user_input)
        return await self.async_step_crop_config()

    async def async_step_crop_config(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure image cropping."""
        crop_config = self.data.get(CONF_CROP_CONFIG, {})
        if user_input is not None:
            self.data[CONF_CROP_CONFIG] = user_input
            return await self.async_step_crop_preview()

        preview_link = await self._async_crop_preview_link(crop_config)
        schema = vol.Schema({
            vol.Optional(
                CONF_CROP_X,
                default=crop_config.get(CONF_CROP_X, DEFAULT_CROP_X),
            ): int,
            vol.Optional(
                CONF_CROP_Y,
                default=crop_config.get(CONF_CROP_Y, DEFAULT_CROP_Y),
            ): int,
            vol.Optional(
                CONF_CROP_WIDTH,
                default=crop_config.get(CONF_CROP_WIDTH, DEFAULT_CROP_WIDTH),
            ): int,
            vol.Optional(
                CONF_CROP_HEIGHT,
                default=crop_config.get(CONF_CROP_HEIGHT, DEFAULT_CROP_HEIGHT),
            ): int,
        })

        return self.async_show_form(
            step_id="crop_config",
            data_schema=schema,
            description_placeholders={"preview": preview_link},
        )

    async def async_step_crop_preview(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Show the selected crop area before continuing."""
        if user_input is not None:
            if user_input["action"] == "adjust":
                return await self.async_step_crop_config()
            return await self.async_step_processor_config()

        preview_link = await self._async_crop_preview_link(
            self.data[CONF_CROP_CONFIG]
        )
        schema = vol.Schema({
            vol.Required("action", default="continue"): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        {"value": "adjust", "label": "Adjust crop"},
                        {"value": "continue", "label": "Continue"},
                    ]
                )
            )
        })
        return self.async_show_form(
            step_id="crop_preview",
            data_schema=schema,
            description_placeholders={"preview": preview_link},
        )

    async def _async_crop_preview_link(
        self, crop_config: dict[str, int]
    ) -> str:
        """Create a link to the current crop preview."""
        try:
            image_data = await _async_load_preview(
                self.hass, self.data, crop_config
            )
        except (HomeAssistantError, OSError, ValueError, UnidentifiedImageError) as err:
            _LOGGER.warning("Could not load image crop preview: %s", err)
            return "The selected image preview is unavailable. Check the Home Assistant log."
        except Exception:
            _LOGGER.exception("Unexpected error loading image crop preview")
            return "The selected image preview is unavailable. Check the Home Assistant log."

        preview_view = self.hass.data.get(_PREVIEW_VIEW_KEY)
        if preview_view is None:
            preview_view = ImagePreviewView()
            self.hass.http.register_view(preview_view)
            self.hass.data[_PREVIEW_VIEW_KEY] = preview_view
        preview_view.set_image(self.flow_id, image_data)
        preview_url = async_sign_path(
            self.hass,
            f"/api/{DOMAIN}/preview/{self.flow_id}",
            timedelta(minutes=5),
        )
        return (
            f'<a href="{preview_url}" target="_blank">'
            "Open image with crop area in a new tab"
            "</a>"
        )

    async def async_step_processor_config(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure processors."""
        if user_input is None:
            schema = vol.Schema({
                vol.Optional(CONF_MIN_ANGLE_HOURS, default=DEFAULT_MIN_ANGLE_HOURS): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
                vol.Optional(CONF_MAX_ANGLE_HOURS, default=DEFAULT_MAX_ANGLE_HOURS): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
                vol.Optional(CONF_CENTER_OFFSET_X, default=DEFAULT_CENTER_OFFSET_X): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
                vol.Optional(CONF_CENTER_OFFSET_Y, default=DEFAULT_CENTER_OFFSET_Y): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
                vol.Optional(CONF_MIN_VALUE, default=DEFAULT_MIN_VALUE): vol.Coerce(float),
                vol.Optional(CONF_MAX_VALUE, default=DEFAULT_MAX_VALUE): vol.Coerce(float),
                vol.Optional(CONF_UNITS, default=DEFAULT_UNITS): str,
                vol.Optional(CONF_SCAN_INTERVAL, default=DEFAULT_SCAN_INTERVAL): vol.All(vol.Coerce(int), vol.Range(min=5, max=3600)),
            })
            
            return self.async_show_form(step_id="processor_config", data_schema=schema)

        # Create processor configuration
        processor_config = {
            "type": PROCESSOR_ANALOG_GAUGE,
            "config": user_input
        }
        
        self.data[CONF_PROCESSORS] = [processor_config]
        
        # Store scan interval at the top level
        self.data[CONF_SCAN_INTERVAL] = user_input.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)

        return self.async_create_entry(title=self.data["name"], data=self.data)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for Image to Sensor CV."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        # Store config entry for compatibility with different HA versions
        self._config_entry = config_entry

    @property
    def config_entry(self) -> config_entries.ConfigEntry:
        """Return the config entry."""
        return self._config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is None:
            current_data = self.config_entry.data
            image_source = current_data.get(CONF_IMAGE_SOURCE, SOURCE_FILE)
            crop_config = current_data.get(CONF_CROP_CONFIG, {})
            processors = current_data.get(CONF_PROCESSORS, [])
            processor_config = {}
            
            # Get current processor config if it exists
            if processors and len(processors) > 0:
                processor_config = processors[0].get("config", {})
            
            # Build complete schema with all options
            schema_fields = {
                vol.Required(
                    CONF_IMAGE_SOURCE,
                    default=image_source
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"value": SOURCE_FILE, "label": "Local Image File"},
                        {"value": SOURCE_CAMERA, "label": "Home Assistant Camera"},
                    ])
                ),
            }
            
            # Only include the relevant field based on image source
            if image_source == SOURCE_FILE:
                schema_fields[vol.Required(
                    CONF_IMAGE_PATH, 
                    default=current_data.get(CONF_IMAGE_PATH, "")
                )] = str
            else:
                schema_fields[vol.Required(
                    CONF_CAMERA_ENTITY,
                    default=current_data.get(CONF_CAMERA_ENTITY, "")
                )] = selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="camera")
                )
                schema_fields[vol.Optional(
                    CONF_PRE_CAPTURE_SWITCH,
                    default=current_data.get(CONF_PRE_CAPTURE_SWITCH),
                )] = selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["switch", "light"])
                )
                schema_fields[vol.Optional(
                    CONF_PRE_CAPTURE_DELAY_MS,
                    default=current_data.get(
                        CONF_PRE_CAPTURE_DELAY_MS, DEFAULT_PRE_CAPTURE_DELAY_MS
                    ),
                )] = vol.All(vol.Coerce(int), vol.Range(min=0, max=60000))
            
            # Add crop settings
            schema_fields.update({
                vol.Optional(
                    CONF_CROP_X, 
                    default=crop_config.get(CONF_CROP_X, DEFAULT_CROP_X)
                ): int,
                vol.Optional(
                    CONF_CROP_Y, 
                    default=crop_config.get(CONF_CROP_Y, DEFAULT_CROP_Y)
                ): int,
                vol.Optional(
                    CONF_CROP_WIDTH, 
                    default=crop_config.get(CONF_CROP_WIDTH, DEFAULT_CROP_WIDTH)
                ): int,
                vol.Optional(
                    CONF_CROP_HEIGHT, 
                    default=crop_config.get(CONF_CROP_HEIGHT, DEFAULT_CROP_HEIGHT)
                ): int,
            })
            
            # Add processor settings
            schema_fields.update({
                vol.Optional(
                    CONF_MIN_ANGLE_HOURS, 
                    default=processor_config.get(CONF_MIN_ANGLE_HOURS, DEFAULT_MIN_ANGLE_HOURS)
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
                vol.Optional(
                    CONF_MAX_ANGLE_HOURS, 
                    default=processor_config.get(CONF_MAX_ANGLE_HOURS, DEFAULT_MAX_ANGLE_HOURS)
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
                vol.Optional(
                    CONF_CENTER_OFFSET_X,
                    default=processor_config.get(
                        CONF_CENTER_OFFSET_X, DEFAULT_CENTER_OFFSET_X
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
                vol.Optional(
                    CONF_CENTER_OFFSET_Y,
                    default=processor_config.get(
                        CONF_CENTER_OFFSET_Y, DEFAULT_CENTER_OFFSET_Y
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
                vol.Optional(
                    CONF_MIN_VALUE, 
                    default=processor_config.get(CONF_MIN_VALUE, DEFAULT_MIN_VALUE)
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_MAX_VALUE, 
                    default=processor_config.get(CONF_MAX_VALUE, DEFAULT_MAX_VALUE)
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_UNITS, 
                    default=processor_config.get(CONF_UNITS, DEFAULT_UNITS)
                ): str,
                vol.Optional(
                    CONF_SCAN_INTERVAL,
                    default=current_data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
                ): vol.All(vol.Coerce(int), vol.Range(min=5, max=3600)),
            })
            
            schema = vol.Schema(schema_fields)
            return self.async_show_form(step_id="init", data_schema=schema)
        
        # Process form submission with validation
        errors = {}
        image_source = user_input[CONF_IMAGE_SOURCE]
        
        # Validate required fields based on image source
        if image_source == SOURCE_FILE:
            if CONF_IMAGE_PATH not in user_input or not user_input[CONF_IMAGE_PATH].strip():
                errors[CONF_IMAGE_PATH] = "required"
        else:  # SOURCE_CAMERA
            if CONF_CAMERA_ENTITY not in user_input or not user_input[CONF_CAMERA_ENTITY].strip():
                errors[CONF_CAMERA_ENTITY] = "required"
        
        if errors:
            # Re-show form with errors
            return self.async_show_form(
                step_id="init", 
                data_schema=self._build_options_schema(user_input),
                errors=errors
            )
        
        # Build new configuration
        crop_config = {
            CONF_CROP_X: user_input[CONF_CROP_X],
            CONF_CROP_Y: user_input[CONF_CROP_Y],
            CONF_CROP_WIDTH: user_input[CONF_CROP_WIDTH],
            CONF_CROP_HEIGHT: user_input[CONF_CROP_HEIGHT],
        }
        
        processor_config = {
            "type": PROCESSOR_ANALOG_GAUGE,
            "config": {
                CONF_MIN_ANGLE_HOURS: user_input[CONF_MIN_ANGLE_HOURS],
                CONF_MAX_ANGLE_HOURS: user_input[CONF_MAX_ANGLE_HOURS],
                CONF_CENTER_OFFSET_X: user_input.get(
                    CONF_CENTER_OFFSET_X, DEFAULT_CENTER_OFFSET_X
                ),
                CONF_CENTER_OFFSET_Y: user_input.get(
                    CONF_CENTER_OFFSET_Y, DEFAULT_CENTER_OFFSET_Y
                ),
                CONF_MIN_VALUE: user_input[CONF_MIN_VALUE],
                CONF_MAX_VALUE: user_input[CONF_MAX_VALUE],
                CONF_UNITS: user_input[CONF_UNITS],
            }
        }
        
        new_data = {
            "name": self.config_entry.data["name"],
            CONF_IMAGE_SOURCE: image_source,
            CONF_CROP_CONFIG: crop_config,
            CONF_PROCESSORS: [processor_config],
            CONF_SCAN_INTERVAL: user_input.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
        }
        
        # Add the appropriate image source field
        if image_source == SOURCE_FILE:
            new_data[CONF_IMAGE_PATH] = user_input[CONF_IMAGE_PATH]
            # Preserve existing camera entity if switching sources
            if CONF_CAMERA_ENTITY in self.config_entry.data:
                new_data[CONF_CAMERA_ENTITY] = self.config_entry.data[CONF_CAMERA_ENTITY]
        else:
            new_data[CONF_CAMERA_ENTITY] = user_input[CONF_CAMERA_ENTITY]
            if user_input.get(CONF_PRE_CAPTURE_SWITCH):
                new_data[CONF_PRE_CAPTURE_SWITCH] = user_input[
                    CONF_PRE_CAPTURE_SWITCH
                ]
            new_data[CONF_PRE_CAPTURE_DELAY_MS] = user_input.get(
                CONF_PRE_CAPTURE_DELAY_MS, DEFAULT_PRE_CAPTURE_DELAY_MS
            )
            # Preserve existing image path if switching sources
            if CONF_IMAGE_PATH in self.config_entry.data:
                new_data[CONF_IMAGE_PATH] = self.config_entry.data[CONF_IMAGE_PATH]
        
        # Update config entry
        self.hass.config_entries.async_update_entry(
            self.config_entry, data=new_data
        )
        
        return self.async_create_entry(title="", data={})

    def _build_options_schema(self, user_input: dict[str, Any]) -> vol.Schema:
        """Build options schema with user input as defaults."""
        image_source = user_input[CONF_IMAGE_SOURCE]
        
        schema_fields = {
            vol.Required(
                CONF_IMAGE_SOURCE,
                default=image_source
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(options=[
                    {"value": SOURCE_FILE, "label": "Local Image File"},
                    {"value": SOURCE_CAMERA, "label": "Home Assistant Camera"},
                ])
            ),
        }
        
        # Add image source fields - only include the relevant field
        if image_source == SOURCE_FILE:
            schema_fields[vol.Required(
                CONF_IMAGE_PATH, 
                default=user_input.get(CONF_IMAGE_PATH, "")
            )] = str
        else:
            schema_fields[vol.Required(
                CONF_CAMERA_ENTITY,
                default=user_input.get(CONF_CAMERA_ENTITY, "")
            )] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain="camera")
            )
            schema_fields[vol.Optional(
                CONF_PRE_CAPTURE_SWITCH,
                default=user_input.get(CONF_PRE_CAPTURE_SWITCH),
            )] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["switch", "light"])
            )
            schema_fields[vol.Optional(
                CONF_PRE_CAPTURE_DELAY_MS,
                default=user_input.get(
                    CONF_PRE_CAPTURE_DELAY_MS, DEFAULT_PRE_CAPTURE_DELAY_MS
                ),
            )] = vol.All(vol.Coerce(int), vol.Range(min=0, max=60000))
        
        # Add other fields
        schema_fields.update({
            vol.Optional(
                CONF_CROP_X, 
                default=user_input.get(CONF_CROP_X, DEFAULT_CROP_X)
            ): int,
            vol.Optional(
                CONF_CROP_Y, 
                default=user_input.get(CONF_CROP_Y, DEFAULT_CROP_Y)
            ): int,
            vol.Optional(
                CONF_CROP_WIDTH, 
                default=user_input.get(CONF_CROP_WIDTH, DEFAULT_CROP_WIDTH)
            ): int,
            vol.Optional(
                CONF_CROP_HEIGHT, 
                default=user_input.get(CONF_CROP_HEIGHT, DEFAULT_CROP_HEIGHT)
            ): int,
            vol.Optional(
                CONF_MIN_ANGLE_HOURS, 
                default=user_input.get(CONF_MIN_ANGLE_HOURS, DEFAULT_MIN_ANGLE_HOURS)
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
            vol.Optional(
                CONF_MAX_ANGLE_HOURS, 
                default=user_input.get(CONF_MAX_ANGLE_HOURS, DEFAULT_MAX_ANGLE_HOURS)
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
            vol.Optional(
                CONF_CENTER_OFFSET_X,
                default=user_input.get(
                    CONF_CENTER_OFFSET_X, DEFAULT_CENTER_OFFSET_X
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
            vol.Optional(
                CONF_CENTER_OFFSET_Y,
                default=user_input.get(
                    CONF_CENTER_OFFSET_Y, DEFAULT_CENTER_OFFSET_Y
                ),
            ): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
            vol.Optional(
                CONF_MIN_VALUE, 
                default=user_input.get(CONF_MIN_VALUE, DEFAULT_MIN_VALUE)
            ): vol.Coerce(float),
            vol.Optional(
                CONF_MAX_VALUE, 
                default=user_input.get(CONF_MAX_VALUE, DEFAULT_MAX_VALUE)
            ): vol.Coerce(float),
            vol.Optional(
                CONF_UNITS, 
                default=user_input.get(CONF_UNITS, DEFAULT_UNITS)
            ): str,
            vol.Optional(
                CONF_SCAN_INTERVAL,
                default=user_input.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
            ): vol.All(vol.Coerce(int), vol.Range(min=5, max=3600)),
        })
        
        return vol.Schema(schema_fields)

    async def async_step_all_options(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure all options in a single form."""
        if user_input is None:
            current_data = self.config_entry.data
            image_source = current_data.get(CONF_IMAGE_SOURCE, SOURCE_FILE)
            crop_config = current_data.get(CONF_CROP_CONFIG, {})
            processors = current_data.get(CONF_PROCESSORS, [])
            processor_config = {}
            
            # Get current processor config if it exists
            if processors and len(processors) > 0:
                processor_config = processors[0].get("config", {})
            
            # Build schema with both image source fields, but make them conditionally required
            schema_fields = {
                vol.Required(
                    CONF_IMAGE_SOURCE,
                    default=image_source
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"value": SOURCE_FILE, "label": "Local Image File"},
                        {"value": SOURCE_CAMERA, "label": "Home Assistant Camera"},
                    ])
                ),
            }
            
            # Add both image source fields - they'll be validated in the submission
            schema_fields[vol.Optional(
                CONF_IMAGE_PATH, 
                default=current_data.get(CONF_IMAGE_PATH, "") if image_source == SOURCE_FILE else ""
            )] = str
            
            schema_fields[vol.Optional(
                CONF_CAMERA_ENTITY,
                default=current_data.get(CONF_CAMERA_ENTITY, "") if image_source == SOURCE_CAMERA else ""
            )] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain="camera")
            )
            schema_fields[vol.Optional(
                CONF_PRE_CAPTURE_SWITCH,
                default=current_data.get(CONF_PRE_CAPTURE_SWITCH)
                if image_source == SOURCE_CAMERA
                else None,
            )] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["switch", "light"])
            )
            schema_fields[vol.Optional(
                CONF_PRE_CAPTURE_DELAY_MS,
                default=current_data.get(
                    CONF_PRE_CAPTURE_DELAY_MS, DEFAULT_PRE_CAPTURE_DELAY_MS
                ) if image_source == SOURCE_CAMERA else DEFAULT_PRE_CAPTURE_DELAY_MS,
            )] = vol.All(vol.Coerce(int), vol.Range(min=0, max=60000))
            
            # Add crop configuration fields
            schema_fields.update({
                vol.Optional(
                    CONF_CROP_X, 
                    default=crop_config.get(CONF_CROP_X, DEFAULT_CROP_X)
                ): int,
                vol.Optional(
                    CONF_CROP_Y, 
                    default=crop_config.get(CONF_CROP_Y, DEFAULT_CROP_Y)
                ): int,
                vol.Optional(
                    CONF_CROP_WIDTH, 
                    default=crop_config.get(CONF_CROP_WIDTH, DEFAULT_CROP_WIDTH)
                ): int,
                vol.Optional(
                    CONF_CROP_HEIGHT, 
                    default=crop_config.get(CONF_CROP_HEIGHT, DEFAULT_CROP_HEIGHT)
                ): int,
            })
            
            # Add processor configuration fields
            schema_fields.update({
                vol.Optional(
                    CONF_MIN_ANGLE_HOURS, 
                    default=processor_config.get(CONF_MIN_ANGLE_HOURS, DEFAULT_MIN_ANGLE_HOURS)
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
                vol.Optional(
                    CONF_MAX_ANGLE_HOURS, 
                    default=processor_config.get(CONF_MAX_ANGLE_HOURS, DEFAULT_MAX_ANGLE_HOURS)
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
                vol.Optional(
                    CONF_CENTER_OFFSET_X,
                    default=processor_config.get(
                        CONF_CENTER_OFFSET_X, DEFAULT_CENTER_OFFSET_X
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
                vol.Optional(
                    CONF_CENTER_OFFSET_Y,
                    default=processor_config.get(
                        CONF_CENTER_OFFSET_Y, DEFAULT_CENTER_OFFSET_Y
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
                vol.Optional(
                    CONF_MIN_VALUE, 
                    default=processor_config.get(CONF_MIN_VALUE, DEFAULT_MIN_VALUE)
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_MAX_VALUE, 
                    default=processor_config.get(CONF_MAX_VALUE, DEFAULT_MAX_VALUE)
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_UNITS, 
                    default=processor_config.get(CONF_UNITS, DEFAULT_UNITS)
                ): str,
                vol.Optional(
                    CONF_SCAN_INTERVAL,
                    default=current_data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
                ): vol.All(vol.Coerce(int), vol.Range(min=5, max=3600)),
            })
            
            schema = vol.Schema(schema_fields)
            
            # Try to generate preview image with current settings
            preview_html = await self._generate_preview_html(current_data)
            
            return self.async_show_form(
                step_id="all_options", 
                data_schema=schema,
                description_placeholders={"preview": preview_html} if preview_html else None
            )

        # Process form submission
        image_source = user_input[CONF_IMAGE_SOURCE]
        
        # Validate that the correct field is provided for the selected image source
        errors = {}
        if image_source == SOURCE_FILE:
            if not user_input.get(CONF_IMAGE_PATH, "").strip():
                errors[CONF_IMAGE_PATH] = "required"
        else:
            if not user_input.get(CONF_CAMERA_ENTITY, "").strip():
                errors[CONF_CAMERA_ENTITY] = "required"
        
        if errors:
            # Re-show form with errors
            current_data = self.config_entry.data
            crop_config = current_data.get(CONF_CROP_CONFIG, {})
            processors = current_data.get(CONF_PROCESSORS, [])
            processor_config = {}
            
            if processors and len(processors) > 0:
                processor_config = processors[0].get("config", {})
            
            # Rebuild schema with submitted values as defaults
            schema_fields = {
                vol.Required(
                    CONF_IMAGE_SOURCE,
                    default=image_source
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[
                        {"value": SOURCE_FILE, "label": "Local Image File"},
                        {"value": SOURCE_CAMERA, "label": "Home Assistant Camera"},
                    ])
                ),
                vol.Optional(
                    CONF_IMAGE_PATH, 
                    default=user_input.get(CONF_IMAGE_PATH, "")
                ): str,
                vol.Optional(
                    CONF_CAMERA_ENTITY,
                    default=user_input.get(CONF_CAMERA_ENTITY, "")
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="camera")
                ),
                vol.Optional(
                    CONF_PRE_CAPTURE_SWITCH,
                    default=user_input.get(CONF_PRE_CAPTURE_SWITCH),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["switch", "light"])
                ),
                vol.Optional(
                    CONF_PRE_CAPTURE_DELAY_MS,
                    default=user_input.get(
                        CONF_PRE_CAPTURE_DELAY_MS, DEFAULT_PRE_CAPTURE_DELAY_MS
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=0, max=60000)),
                vol.Optional(
                    CONF_CROP_X, 
                    default=user_input.get(CONF_CROP_X, DEFAULT_CROP_X)
                ): int,
                vol.Optional(
                    CONF_CROP_Y, 
                    default=user_input.get(CONF_CROP_Y, DEFAULT_CROP_Y)
                ): int,
                vol.Optional(
                    CONF_CROP_WIDTH, 
                    default=user_input.get(CONF_CROP_WIDTH, DEFAULT_CROP_WIDTH)
                ): int,
                vol.Optional(
                    CONF_CROP_HEIGHT, 
                    default=user_input.get(CONF_CROP_HEIGHT, DEFAULT_CROP_HEIGHT)
                ): int,
                vol.Optional(
                    CONF_MIN_ANGLE_HOURS, 
                    default=user_input.get(CONF_MIN_ANGLE_HOURS, DEFAULT_MIN_ANGLE_HOURS)
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
                vol.Optional(
                    CONF_MAX_ANGLE_HOURS, 
                    default=user_input.get(CONF_MAX_ANGLE_HOURS, DEFAULT_MAX_ANGLE_HOURS)
                ): vol.All(vol.Coerce(float), vol.Range(min=0, max=12)),
                vol.Optional(
                    CONF_CENTER_OFFSET_X,
                    default=user_input.get(
                        CONF_CENTER_OFFSET_X, DEFAULT_CENTER_OFFSET_X
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
                vol.Optional(
                    CONF_CENTER_OFFSET_Y,
                    default=user_input.get(
                        CONF_CENTER_OFFSET_Y, DEFAULT_CENTER_OFFSET_Y
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=-1000, max=1000)),
                vol.Optional(
                    CONF_MIN_VALUE, 
                    default=user_input.get(CONF_MIN_VALUE, DEFAULT_MIN_VALUE)
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_MAX_VALUE, 
                    default=user_input.get(CONF_MAX_VALUE, DEFAULT_MAX_VALUE)
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_UNITS, 
                    default=user_input.get(CONF_UNITS, DEFAULT_UNITS)
                ): str,
                vol.Optional(
                    CONF_SCAN_INTERVAL,
                    default=user_input.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
                ): vol.All(vol.Coerce(int), vol.Range(min=5, max=3600)),
            }
            
            schema = vol.Schema(schema_fields)
            
            return self.async_show_form(
                step_id="all_options", 
                data_schema=schema,
                errors=errors
            )
        
        # Build crop config
        crop_config = {
            CONF_CROP_X: user_input[CONF_CROP_X],
            CONF_CROP_Y: user_input[CONF_CROP_Y],
            CONF_CROP_WIDTH: user_input[CONF_CROP_WIDTH],
            CONF_CROP_HEIGHT: user_input[CONF_CROP_HEIGHT],
        }
        
        # Build processor config
        processor_config = {
            "type": PROCESSOR_ANALOG_GAUGE,
            "config": {
                CONF_MIN_ANGLE_HOURS: user_input[CONF_MIN_ANGLE_HOURS],
                CONF_MAX_ANGLE_HOURS: user_input[CONF_MAX_ANGLE_HOURS],
                CONF_CENTER_OFFSET_X: user_input.get(
                    CONF_CENTER_OFFSET_X, DEFAULT_CENTER_OFFSET_X
                ),
                CONF_CENTER_OFFSET_Y: user_input.get(
                    CONF_CENTER_OFFSET_Y, DEFAULT_CENTER_OFFSET_Y
                ),
                CONF_MIN_VALUE: user_input[CONF_MIN_VALUE],
                CONF_MAX_VALUE: user_input[CONF_MAX_VALUE],
                CONF_UNITS: user_input[CONF_UNITS],
            }
        }
        
        # Build new data
        new_data = {
            "name": self.config_entry.data["name"],
            CONF_IMAGE_SOURCE: image_source,
            CONF_CROP_CONFIG: crop_config,
            CONF_PROCESSORS: [processor_config],
            CONF_SCAN_INTERVAL: user_input.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
        }
        
        # Add image source specific data - only include the active field
        if image_source == SOURCE_FILE:
            new_data[CONF_IMAGE_PATH] = user_input[CONF_IMAGE_PATH]
        else:
            new_data[CONF_CAMERA_ENTITY] = user_input[CONF_CAMERA_ENTITY]
            if user_input.get(CONF_PRE_CAPTURE_SWITCH):
                new_data[CONF_PRE_CAPTURE_SWITCH] = user_input[
                    CONF_PRE_CAPTURE_SWITCH
                ]
            new_data[CONF_PRE_CAPTURE_DELAY_MS] = user_input.get(
                CONF_PRE_CAPTURE_DELAY_MS, DEFAULT_PRE_CAPTURE_DELAY_MS
            )

        # Update the config entry with new options
        self.hass.config_entries.async_update_entry(
            self.config_entry, data=new_data
        )

        return self.async_create_entry(title="", data={})

    async def _generate_preview_html(self, config_data: dict[str, Any]) -> str | None:
        """Generate HTML preview of the image with cropping overlay."""
        try:
            # Get crop settings for display
            crop_config = config_data.get(CONF_CROP_CONFIG, {})
            crop_x = crop_config.get(CONF_CROP_X, DEFAULT_CROP_X)
            crop_y = crop_config.get(CONF_CROP_Y, DEFAULT_CROP_Y)
            crop_width = crop_config.get(CONF_CROP_WIDTH, DEFAULT_CROP_WIDTH)
            crop_height = crop_config.get(CONF_CROP_HEIGHT, DEFAULT_CROP_HEIGHT)
            
            image_source = config_data.get(CONF_IMAGE_SOURCE, SOURCE_FILE)
            source_info = ""
            
            if image_source == SOURCE_FILE:
                image_path = config_data.get(CONF_IMAGE_PATH, "")
                source_info = f"File: {image_path}"
            else:
                camera_entity = config_data.get(CONF_CAMERA_ENTITY, "")
                source_info = f"Camera: {camera_entity}"
            
            # Return informational HTML instead of actual image preview for now
            return f"""
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h4 style="margin-top: 0;">Current Configuration Preview</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 14px;">
                    <div>
                        <strong>Image Source:</strong><br>
                        {source_info}
                    </div>
                    <div>
                        <strong>Crop Settings:</strong><br>
                        Position: ({crop_x}, {crop_y})<br>
                        Size: {crop_width} × {crop_height} pixels
                    </div>
                </div>
                <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px; font-size: 12px;">
                    💡 <strong>Tip:</strong> The crop area will be applied to extract the gauge region from your image. 
                    Make sure the crop coordinates and size cover the entire gauge face.
                </div>
                <div style="margin-top: 10px; padding: 10px; background: #f3e5f5; border-radius: 4px; font-size: 12px;">
                    🖼️ <strong>Preview Service:</strong> After saving these settings, you can use the 
                    <code>image_to_sensor_cv.generate_preview</code> service with config entry ID 
                    <code>{self.config_entry.entry_id}</code> to generate a visual preview of your crop area.
                </div>
            </div>
            """
            
        except Exception as e:
            _LOGGER.warning("Failed to generate preview info: %s", e)
            return None


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuth(HomeAssistantError):
    """Error to indicate there is invalid auth."""