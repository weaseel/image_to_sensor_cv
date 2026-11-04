"""Simplified image processing without OpenCV dependency."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, Optional
import os
from datetime import timedelta

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.components.http.auth import async_sign_path
from homeassistant.helpers.network import get_url

from .const import (
    SOURCE_FILE,
    SOURCE_CAMERA,
    CONF_IMAGE_SOURCE,
    CONF_IMAGE_PATH,
    CONF_CAMERA_ENTITY,
    CONF_CROP_CONFIG,
    CONF_CROP_X,
    CONF_CROP_Y,
    CONF_CROP_WIDTH,
    CONF_CROP_HEIGHT,
)
from .camera_capture import async_prepare_camera

_LOGGER = logging.getLogger(__name__)


class SimpleImageProcessor:
    """Simplified image processor using only PIL/Pillow."""

    def __init__(self, hass: HomeAssistant, config: Dict[str, Any], sensor_name: str = "unknown"):
        """Initialize the image processor."""
        self.hass = hass
        self.config = config
        self.sensor_name = sensor_name

    async def get_image(self) -> Optional[np.ndarray]:
        """Get image from configured source."""
        _LOGGER.debug("Getting image from source: %s", self.config[CONF_IMAGE_SOURCE])
        
        if self.config[CONF_IMAGE_SOURCE] == SOURCE_FILE:
            result = await self._get_image_from_file()
        elif self.config[CONF_IMAGE_SOURCE] == SOURCE_CAMERA:
            result = await self._get_image_from_camera()
        else:
            _LOGGER.error("Unknown image source: %s", self.config[CONF_IMAGE_SOURCE])
            return None
            
        if result is not None:
            _LOGGER.debug("Successfully acquired image with shape: %s", result.shape)
        else:
            _LOGGER.warning("Failed to acquire image from source")
            
        return result

    async def _get_image_from_file(self) -> Optional[np.ndarray]:
        """Load image from file path."""
        try:
            image_path = self.config[CONF_IMAGE_PATH]
            _LOGGER.debug("Loading image from file: %s", image_path)
            
            if not os.path.exists(image_path):
                _LOGGER.error("Image file not found: %s", image_path)
                # List files in the directory for debugging
                try:
                    dir_path = os.path.dirname(image_path)
                    if os.path.exists(dir_path):
                        files = os.listdir(dir_path)
                        _LOGGER.debug("Files in directory %s: %s", dir_path, files[:10])  # Limit to first 10
                except Exception as dir_e:
                    _LOGGER.debug("Could not list directory contents: %s", dir_e)
                return None

            # Use executor to avoid blocking I/O
            loop = asyncio.get_event_loop()
            
            def _load_image():
                _LOGGER.debug("Opening image file with PIL")
                # Load image using PIL
                pil_image = Image.open(image_path)
                _LOGGER.debug("Original image mode: %s, size: %s", pil_image.mode, pil_image.size)
                # Convert to RGB
                pil_image = pil_image.convert('RGB')
                _LOGGER.debug("Converted to RGB, size: %s", pil_image.size)
                # Convert to numpy array
                image_array = np.array(pil_image)
                _LOGGER.debug("Converted to numpy array, shape: %s, dtype: %s", 
                             image_array.shape, image_array.dtype)
                return image_array
            
            result = await loop.run_in_executor(None, _load_image)
            _LOGGER.debug("Image loaded successfully from file")
            
            # Save debug image of original loaded image
            try:
                from .debug_utils import save_debug_image
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    save_debug_image(result, "original.png", "loaded", self.sensor_name)
            except Exception as debug_e:
                _LOGGER.warning("Could not save original debug image: %s", debug_e)
            
            return result
            
        except Exception as e:
            _LOGGER.error("Error loading image from file: %s", e)
            import traceback
            _LOGGER.error("Traceback: %s", traceback.format_exc())
            return None

    async def _get_image_from_camera(self) -> Optional[np.ndarray]:
        """Get image from Home Assistant camera entity."""
        try:
            camera_entity = self.config[CONF_CAMERA_ENTITY]
            await async_prepare_camera(self.hass, self.config)
            
            # Get camera image URL
            camera_image_url = f"/api/camera_proxy/{camera_entity}"
            
            # Sign the URL with authorization signature (5 minute expiration)
            _LOGGER.debug("Signing camera proxy URL: %s", camera_image_url)
            signed_url = async_sign_path(
                self.hass, camera_image_url, timedelta(minutes=5)
            )
            _LOGGER.debug("Signed URL: %s", signed_url)
            
            # Get base URL and construct full URL
            base_url = get_url(self.hass)
            full_url = f"{base_url}{signed_url}"
            _LOGGER.debug("Full camera URL: %s", full_url)
            
            session = async_get_clientsession(self.hass)
            
            async with session.get(full_url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    # Use PIL to decode image
                    from io import BytesIO
                    pil_image = Image.open(BytesIO(image_data))
                    pil_image = pil_image.convert('RGB')
                    image_array = np.array(pil_image)
                    _LOGGER.debug("Successfully retrieved camera image, shape: %s", image_array.shape)
                    return image_array
                else:
                    _LOGGER.error("Failed to get camera image, status: %s", response.status)
                    return None
                    
        except Exception as e:
            _LOGGER.error("Error getting image from camera: %s", e)
            import traceback
            _LOGGER.error("Traceback: %s", traceback.format_exc())
            return None

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Crop image based on configuration."""
        _LOGGER.debug("Checking crop configuration")
        
        if CONF_CROP_CONFIG not in self.config:
            _LOGGER.debug("No crop configuration found, using full image")
            return image
            
        crop_config = self.config[CONF_CROP_CONFIG]
        x = crop_config.get(CONF_CROP_X, 0)
        y = crop_config.get(CONF_CROP_Y, 0)
        width = crop_config.get(CONF_CROP_WIDTH, image.shape[1])
        height = crop_config.get(CONF_CROP_HEIGHT, image.shape[0])
        
        _LOGGER.debug("Original image shape: %s", image.shape)
        _LOGGER.debug("Requested crop: x=%d, y=%d, width=%d, height=%d", x, y, width, height)
        
        # Ensure crop coordinates are within image bounds
        orig_x, orig_y, orig_width, orig_height = x, y, width, height
        x = max(0, min(x, image.shape[1]))
        y = max(0, min(y, image.shape[0]))
        width = min(width, image.shape[1] - x)
        height = min(height, image.shape[0] - y)
        
        if (x, y, width, height) != (orig_x, orig_y, orig_width, orig_height):
            _LOGGER.debug("Adjusted crop to fit image bounds: x=%d, y=%d, width=%d, height=%d", 
                         x, y, width, height)
        
        cropped = image[y:y+height, x:x+width]
        _LOGGER.debug("Cropped image shape: %s", cropped.shape)
        
        # Save debug image of cropped image
        try:
            from .debug_utils import save_debug_image
            if _LOGGER.isEnabledFor(logging.DEBUG):
                save_debug_image(cropped, "cropped.png", "processed", self.sensor_name)
        except Exception as debug_e:
            _LOGGER.warning("Could not save cropped debug image: %s", debug_e)
        
        return cropped
