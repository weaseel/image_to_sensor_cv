"""Simplified image processing without OpenCV dependency."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
import logging
import math
from typing import Any, Dict, Optional
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .const import (
    SOURCE_FILE,
    SOURCE_CAMERA,
    PROCESSOR_ANALOG_GAUGE,
    CONF_IMAGE_SOURCE,
    CONF_IMAGE_PATH,
    CONF_CAMERA_ENTITY,
    CONF_CROP_CONFIG,
    CONF_CROP_X,
    CONF_CROP_Y,
    CONF_CROP_WIDTH,
    CONF_CROP_HEIGHT,
)

_LOGGER = logging.getLogger(__name__)

# if true - instead of using best single circle candidate for gauge center,
# use weighted average of top candidates
USE_WEIGHTED_CENTER = False

class SimpleAnalogGaugeProcessor:
    """Simplified analog gauge processor using PIL instead of OpenCV."""

    def __init__(self, config: Dict[str, Any], sensor_name: str = "unknown"):
        """Initialize the analog gauge processor."""
        self.config = config
        self.sensor_name = sensor_name
        self.last_snapshot: bytes | None = None

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


    def process_image(self, image: np.ndarray) -> Optional[float]:
        """Process image and extract gauge value using simplified method."""
        try:
            _LOGGER.debug("=== Starting gauge image processing ===")
            _LOGGER.debug("Processor config: %s", self.config)
            image = self.crop_image(image)
            result = self._read_analog_gauge_simple(image)
            _LOGGER.debug("=== Gauge processing complete, result: %s ===", result)
            return result
        except Exception as e:
            _LOGGER.error("Error processing analog gauge: %s", e)
            import traceback
            _LOGGER.error("Traceback: %s", traceback.format_exc())
            return None

    def _read_analog_gauge_simple(self, image_array: np.ndarray) -> Optional[float]:
        """
        Simplified analog gauge reading using PIL and basic image processing.
        This is less accurate than OpenCV but works without additional dependencies.
        """
        try:
            _LOGGER.debug("Starting analog gauge reading process")
            _LOGGER.debug("Input image shape: %s", image_array.shape)
            self.last_snapshot = self._create_detection_snapshot(
                image_array, status="Processing"
            )
            
            # Convert numpy array back to PIL Image for processing
            pil_image = Image.fromarray(image_array)
            _LOGGER.debug("Converted to PIL image, mode: %s, size: %s", pil_image.mode, pil_image.size)
            
            # Convert to grayscale
            gray_image = pil_image.convert('L')
            _LOGGER.debug("Converted to grayscale")
            
            # Apply some basic filtering to enhance edges
            if False:
                enhanced = gray_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                _LOGGER.debug("Applied edge enhancement filter")
            else:
                enhanced = gray_image

            # Convert back to numpy for analysis
            gray_array = np.array(enhanced)
            _LOGGER.debug("Gray array shape: %s, dtype: %s", gray_array.shape, gray_array.dtype)
            _LOGGER.debug("Gray array stats - min: %d, max: %d, mean: %.2f", 
                         gray_array.min(), gray_array.max(), gray_array.mean())
            
            # Find the center of the gauge using circle detection
            height, width = gray_array.shape
            center_x, center_y, gauge_radius = self._detect_gauge_center(gray_array)
            
            _LOGGER.debug("Gauge center detected at: (%d, %d), radius: %d", center_x, center_y, gauge_radius)
            
            # Use a simplified approach to find the needle
            # This assumes the darkest line from center is the needle
            needle_angle = self._find_needle_angle_simple(gray_array, center_x, center_y, gauge_radius)
            
            if needle_angle is None:
                _LOGGER.warning("Could not detect needle angle")
                self.last_snapshot = self._create_detection_snapshot(
                    image_array,
                    center_x=center_x,
                    center_y=center_y,
                    radius=gauge_radius,
                    status="Needle not detected",
                )
                return None
            
            _LOGGER.debug("Detected needle angle: %.2f degrees", needle_angle)
            
            # Convert angle to gauge value
            value = self._angle_to_value(needle_angle)
            _LOGGER.debug("Converted angle to gauge value: %.2f", value)
            self.last_snapshot = self._create_detection_snapshot(
                image_array,
                center_x=center_x,
                center_y=center_y,
                radius=gauge_radius,
                needle_angle=needle_angle,
                value=value,
                status="Detection complete",
            )
            
            # Log detection summary at info level for easier monitoring
            try:
                from .debug_utils import log_detection_summary
                log_detection_summary(needle_angle, value, self.config)
            except Exception as summary_e:
                _LOGGER.warning("Could not log detection summary: %s", summary_e)
            
            return value

        except Exception as e:
            _LOGGER.error("Error in simplified gauge reading: %s", e)
            try:
                self.last_snapshot = self._create_detection_snapshot(
                    image_array, status=f"Detection error: {e}"
                )
            except Exception:
                _LOGGER.exception("Could not create failure snapshot")
            import traceback
            _LOGGER.error("Traceback: %s", traceback.format_exc())
            return None

    def _create_detection_snapshot(
        self,
        image_array: np.ndarray,
        *,
        center_x: int | None = None,
        center_y: int | None = None,
        radius: int | None = None,
        needle_angle: float | None = None,
        value: float | None = None,
        status: str,
    ) -> bytes:
        """Encode the analyzed image with detection details overlaid."""
        image = Image.fromarray(image_array.astype("uint8")).convert("RGB")
        draw = ImageDraw.Draw(image)

        if center_x is not None and center_y is not None and radius is not None:
            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                outline=(0, 120, 255),
                width=2,
            )
            draw.ellipse(
                (center_x - 4, center_y - 4, center_x + 4, center_y + 4),
                fill=(255, 40, 40),
            )
            if needle_angle is not None:
                needle_length = int(radius * 0.75)
                end_x = int(center_x + needle_length * math.cos(math.radians(needle_angle)))
                end_y = int(center_y - needle_length * math.sin(math.radians(needle_angle)))
                draw.line((center_x, center_y, end_x, end_y), fill=(30, 230, 80), width=3)

        captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = [f"Status: {status}", f"Captured: {captured_at}"]
        if center_x is not None and center_y is not None:
            lines.append(f"Gauge center: ({center_x}, {center_y}), radius: {radius}px")
        if needle_angle is not None:
            lines.append(f"Needle angle: {needle_angle:.1f} degrees")
        if value is not None:
            lines.append(f"Reading: {value:.2f} {self.config.get('units', '')}".rstrip())
        if all(key in self.config for key in ("min_angle_hours", "max_angle_hours")):
            lines.append(
                "Configured range: "
                f"{self.config['min_angle_hours']} to {self.config['max_angle_hours']} hours, "
                f"{self.config.get('min_value')} to {self.config.get('max_value')} "
                f"{self.config.get('units', '')}"
            )

        line_height = 15
        banner_height = min(image.height, 8 + line_height * len(lines))
        draw.rectangle((0, 0, image.width, banner_height), fill=(0, 0, 0))
        for line_index, line in enumerate(lines):
            y = 4 + line_index * line_height
            if y + line_height > image.height:
                break
            draw.text((6, y), line, fill=(255, 255, 255))

        output = BytesIO()
        image.save(output, format="JPEG", quality=90)
        return output.getvalue()

    def _detect_gauge_center(self, gray_array: np.ndarray) -> tuple[int, int, int]:
        """
        Detect the center of a circular gauge by finding the circle with strongest edges.
        
        Gauges typically have a dark frame around them, creating strong edges at the border.
        We scan various circle positions and radii to find the one with the strongest
        edge gradient (largest brightness difference between inside and outside the circle).
        
        Args:
            gray_array: Grayscale image as numpy array
            
        Returns:
            Tuple of (center_x, center_y, radius) coordinates and detected gauge radius
        """
        try:
            height, width = gray_array.shape
            _LOGGER.debug("Starting gauge center detection on image size: %dx%d", width, height)
            
            # Initial center guess (image center)
            initial_cx = width // 2
            initial_cy = height // 2
            
            # Define search ranges
            # Search for center within ±25% of image dimensions
            search_range_x = int(width * 0.25)
            search_range_y = int(height * 0.25)
            
            # Search for radius from 25% to 45% of the smaller dimension
            # (assuming gauge takes at least 25% and up to 90% of the image)
            min_dim = min(width, height)
            min_radius = int(min_dim * 0.25)
            max_radius = int(min_dim * 0.45)
            
            _LOGGER.debug("Search parameters:")
            _LOGGER.debug("  Center search range: ±%d pixels (x), ±%d pixels (y)", search_range_x, search_range_y)
            _LOGGER.debug("  Radius search range: %d to %d pixels", min_radius, max_radius)
            
            best_center = (initial_cx, initial_cy)
            best_radius = min_radius
            best_score = 0
            
            # Store all candidates for weighted average
            candidates = []
            
            # Sample center positions (coarse grid search)
            center_step = max(2, min(search_range_x, search_range_y) // 10)  # ~10 steps in each direction

            _LOGGER.debug("Step size: %d pixels", center_step)
            
            for cy in range(initial_cy - search_range_y, initial_cy + search_range_y + 1, center_step):
                if cy < 0 or cy >= height:
                    continue
                    
                for cx in range(initial_cx - search_range_x, initial_cx + search_range_x + 1, center_step):
                    if cx < 0 or cx >= width:
                        continue
                    
                    # Try different radii for this center position
                    radius_step = max(2, (max_radius - min_radius) // 10)  # ~10 steps
                    
                    for radius in range(min_radius, max_radius + 1, radius_step):
                        score = self._measure_circle_edge_strength(gray_array, cx, cy, radius)
                        
                        if score > 0:  # Valid measurement
                            candidates.append({
                                'cx': cx,
                                'cy': cy,
                                'radius': radius,
                                'score': score
                            })
                            
                            if score > best_score:
                                best_score = score
                                best_center = (cx, cy)
                                best_radius = radius
            
            if not candidates:
                _LOGGER.warning("No valid circle candidates found, using image center")
                return (initial_cx, initial_cy)
            
            # Sort candidates by score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Log top candidates with detailed breakdown
            _LOGGER.debug("Top 10 circle candidates (by edge strength):")
            for i, candidate in enumerate(candidates[:10]):
                _LOGGER.debug("  %d. Center: (%3d, %3d), Radius: %3d, Edge Score: %8.2f",
                            i+1, candidate['cx'], candidate['cy'], 
                            candidate['radius'], candidate['score'])
            
            # Use weighted average of top candidates to refine center
            # Take top 20% of candidates or at least top 5
            num_top_candidates = max(5, len(candidates) // 5)
            top_candidates = candidates[:num_top_candidates]
            
            # Weight by score
            total_weight = sum(c['score'] for c in top_candidates)
            if USE_WEIGHTED_CENTER and total_weight > 0:
                _LOGGER.debug("Weighted center calculation details (top %d candidates):", num_top_candidates)
                #for i, c in enumerate(top_candidates):
                #    weight_pct = (c['score'] / total_weight) * 100
                #    _LOGGER.debug("  %d. Center: (%3d, %3d), Radius: %3d, Score: %8.2f, Weight: %5.1f%%",
                #                i+1, c['cx'], c['cy'], c['radius'], c['score'], weight_pct)
                
                weighted_cx = sum(c['cx'] * c['score'] for c in top_candidates) / total_weight
                weighted_cy = sum(c['cy'] * c['score'] for c in top_candidates) / total_weight
                
                final_cx = int(round(weighted_cx))
                final_cy = int(round(weighted_cy))
                
                _LOGGER.debug("Weighted result:")
                _LOGGER.debug("  Total weight: %.2f", total_weight)
                _LOGGER.debug("  Weighted center: (%.1f, %.1f) -> Final: (%d, %d)",
                            weighted_cx, weighted_cy, final_cx, final_cy)
            else:
                final_cx, final_cy = best_center
                _LOGGER.debug("Using best single candidate: (%d, %d)", final_cx, final_cy)
            
            # Save debug visualization if logging is at debug level
            try:
                from .debug_utils import save_debug_image
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    # Create visualization with detected center and radius
                    from PIL import Image, ImageDraw
                    debug_img = Image.fromarray(gray_array).convert('RGB')
                    draw = ImageDraw.Draw(debug_img)
                    
                    # Draw the best circle in green
                    draw.ellipse(
                        [final_cx - best_radius, final_cy - best_radius,
                         final_cx + best_radius, final_cy + best_radius],
                        outline=(0, 255, 0), width=2
                    )
                    
                    # Draw center point in red
                    marker_size = 5
                    draw.ellipse(
                        [final_cx - marker_size, final_cy - marker_size,
                         final_cx + marker_size, final_cy + marker_size],
                        fill=(255, 0, 0)
                    )
                    
                    # Draw crosshair
                    draw.line([final_cx - 20, final_cy, final_cx + 20, final_cy], 
                             fill=(255, 0, 0), width=1)
                    draw.line([final_cx, final_cy - 20, final_cx, final_cy + 20], 
                             fill=(255, 0, 0), width=1)
                    
                    # Draw top 3 alternative circles in blue (semi-transparent effect via thinner line)
                    for candidate in candidates[1:4]:
                        draw.ellipse(
                            [candidate['cx'] - candidate['radius'], 
                             candidate['cy'] - candidate['radius'],
                             candidate['cx'] + candidate['radius'], 
                             candidate['cy'] + candidate['radius']],
                            outline=(100, 100, 255), width=1
                        )
                    
                    save_debug_image(np.array(debug_img), "gauge_center_detection.png", 
                                   "center_detection", self.sensor_name)
            except Exception as debug_e:
                _LOGGER.warning("Could not save center detection debug image: %s", debug_e)
            
            return (final_cx, final_cy, best_radius)
            
        except Exception as e:
            _LOGGER.error("Error detecting gauge center: %s", e)
            import traceback
            _LOGGER.error("Traceback: %s", traceback.format_exc())
            # Fall back to image center and estimated radius
            height, width = gray_array.shape
            fallback_radius = min(width, height) // 4
            return (width // 2, height // 2, fallback_radius)

    def _measure_circle_edge_strength(self, gray_array: np.ndarray, 
                                      cx: int, cy: int, radius: int) -> float:
        """
        Measure the edge strength at a circle's perimeter.
        
        This detects the gauge border by measuring the brightness gradient across
        the circle boundary. A strong edge (dark frame around bright gauge) will
        have a large difference between inner and outer samples.
        
        Args:
            gray_array: Grayscale image
            cx, cy: Circle center coordinates
            radius: Circle radius
            
        Returns:
            Edge strength score (higher = stronger edge), or 0 if invalid
        """
        try:
            height, width = gray_array.shape
            
            # Sample points around the circle perimeter
            # Use enough samples to get good coverage
            num_samples = max(16, int(2 * math.pi * radius))
            
            # We'll measure the gradient by comparing brightness at three positions:
            # - inside the circle (radius - offset)
            # - on the circle edge (radius)
            # - outside the circle (radius + offset)
            # The gauge border should show: bright inside, dark on edge/outside
            
            inner_offset = max(3, radius // 10)  # Sample 10% inside
            outer_offset = max(3, radius // 10)  # Sample 10% outside
            
            inner_brightness_sum = 0.0
            edge_brightness_sum = 0.0
            outer_brightness_sum = 0.0
            valid_samples = 0
            
            for i in range(num_samples):
                angle = (2 * math.pi * i) / num_samples
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                
                # Calculate positions for inner, edge, and outer samples
                x_inner = int(cx + (radius - inner_offset) * cos_a)
                y_inner = int(cy + (radius - inner_offset) * sin_a)
                
                x_edge = int(cx + radius * cos_a)
                y_edge = int(cy + radius * sin_a)
                
                x_outer = int(cx + (radius + outer_offset) * cos_a)
                y_outer = int(cy + (radius + outer_offset) * sin_a)
                
                # Check if all three points are within image bounds
                if (0 <= x_inner < width and 0 <= y_inner < height and
                    0 <= x_edge < width and 0 <= y_edge < height and
                    0 <= x_outer < width and 0 <= y_outer < height):
                    
                    inner_brightness_sum += float(gray_array[y_inner, x_inner])
                    edge_brightness_sum += float(gray_array[y_edge, x_edge])
                    outer_brightness_sum += float(gray_array[y_outer, x_outer])
                    valid_samples += 1
            
            # Require at least 75% of samples to be valid
            if valid_samples < (num_samples * 0.75):
                return 0.0
            
            if valid_samples == 0:
                return 0.0
            
            # Calculate average brightness for each ring
            avg_inner = inner_brightness_sum / valid_samples
            avg_edge = edge_brightness_sum / valid_samples
            avg_outer = outer_brightness_sum / valid_samples
            
            # Edge strength is the gradient: we want inner to be bright and edge/outer to be dark
            # Score based on the drop from inner to outer (typical gauge: bright face, dark frame)
            # Also consider the edge itself being dark
            
            # Primary score: difference between inside and outside (positive = inside brighter)
            gradient_score = avg_inner - avg_outer
            
            # Secondary score: edge should be darker than inside
            edge_contrast = avg_inner - avg_edge
            
            # Combined score: favor circles where inside is brighter than outside
            # and where there's a sharp transition at the edge
            combined_score = gradient_score + (edge_contrast * 1)
            
            # Only return positive scores (we're looking for bright-to-dark transitions)
            return max(0.0, combined_score)
            
        except Exception as e:
            _LOGGER.debug("Error measuring circle edge strength: %s", e)
            return 0.0

    def _find_needle_angle_simple(self, gray_array: np.ndarray, cx: int, cy: int, gauge_radius: int) -> Optional[float]:
        """
        Simplified needle detection using radial scanning.
        Scans in radial directions from center to find the darkest path (needle).
        
        Args:
            gray_array: Grayscale image
            cx, cy: Gauge center coordinates
            gauge_radius: Detected gauge radius (used to determine needle search area)
        """
        try:
            height, width = gray_array.shape
            # Use the detected gauge radius to set needle search area
            # Needle typically extends from center to about 70-80% of gauge radius
            needle_end_radius = int(gauge_radius * 0.75)
            needle_start_radius = max(5, needle_end_radius // 8)  # Avoid center noise, start from ~12.5% of needle length
            _LOGGER.debug("Starting needle detection - center: (%d, %d), gauge_radius: %d, needle search: %d to %d", 
                         cx, cy, gauge_radius, needle_start_radius, needle_end_radius)
            
            best_angle = None
            best_score = float('inf')
            angle_scores = []  # Store scores for analysis
            
            # Scan in 5-degree increments first for performance, then refine
            coarse_step = 5
            for angle_deg in range(0, 360, coarse_step):
                angle_rad = math.radians(angle_deg)
                
                # Sample points along this angle
                score = 0.0  # Use float to prevent overflow
                valid_points = 0

                for r in range(needle_start_radius, needle_end_radius, 1):  # Skip very center, sample every 1 pixels
                    # Fix: Image coordinates have Y-axis flipped (0,0 is top-left)
                    # Standard math: angle 0° = right (3 o'clock), 90° = up (12 o'clock)
                    # For gauge: angle 0° = up (12 o'clock), 90° = right (3 o'clock)
                    # Convert: gauge_angle = 90 - math_angle, then flip Y
                    
                    x = int(cx + r * math.cos(angle_rad))
                    y = int(cy - r * math.sin(angle_rad))  # Flip Y axis for image coordinates
                    
                    if 0 <= x < width and 0 <= y < height:
                        # Lower pixel values = darker = more likely to be needle
                        score += float(gray_array[y, x])  # Convert to float to prevent overflow
                        valid_points += 1
                
                if valid_points > 0:
                    avg_score = score / valid_points
                    angle_scores.append((angle_deg, avg_score, valid_points))
                    if avg_score < best_score:
                        best_score = avg_score
                        best_angle = angle_deg
            
            # Sort by score and log top candidates
            angle_scores.sort(key=lambda x: x[1])  # Sort by score (lower = darker)
            _LOGGER.debug("Top 10 needle angle candidates:")
            for i, (angle, score, points) in enumerate(angle_scores[:10]):
                _LOGGER.debug("  %d. Angle: %3d°, Score: %6.2f, Points: %d", 
                             i+1, angle, score, points)
            
            if best_angle is not None:
                # Refine the best angle by scanning ±2 degrees in 1-degree increments
                refined_angle = self._refine_needle_angle(gray_array, cx, cy, best_angle, needle_end_radius)
                if refined_angle is not None:
                    _LOGGER.debug("Refined needle angle from %d° to %.1f°", best_angle, refined_angle)
                    best_angle = refined_angle
                
                _LOGGER.debug("Final needle angle: %.1f° (math convention), best score: %.2f", best_angle, best_score)
                
                clock_angle_display = best_angle #(90 - best_angle) % 360
                _LOGGER.debug("Needle angle in clock convention: %.1f°", clock_angle_display)
                
                # Save debug images if logging is at debug level
                try:
                    from .debug_utils import save_debug_image, create_detection_overlay
                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        # Save the processed grayscale image
                        save_debug_image(gray_array, "gauge_processed.png", "grayscale", self.sensor_name)
                        
                        # Create and save detection overlay (using math convention angle)
                        overlay = create_detection_overlay(gray_array, cx, cy, best_angle, needle_start_radius, needle_end_radius)
                        save_debug_image(overlay, "gauge_detection.png", "overlay", self.sensor_name)
                except Exception as debug_e:
                    _LOGGER.warning("Could not save debug images: %s", debug_e)
                
                # Additional validation
                if best_score > 200:  # Threshold for "too bright" (needle should be dark)
                    _LOGGER.warning("Needle appears too bright (score: %.2f), detection may be unreliable", best_score)
                
            else:
                _LOGGER.warning("No needle angle detected - all scores were invalid")
                _LOGGER.debug("Image brightness analysis - center region:")
                # Analyze center region brightness
                center_region = gray_array[max(0, cy-20):cy+20, max(0, cx-20):cx+20]
                if center_region.size > 0:
                    _LOGGER.debug("  Center region shape: %s", center_region.shape)
                    _LOGGER.debug("  Center region stats - min: %d, max: %d, mean: %.2f", 
                                 center_region.min(), center_region.max(), center_region.mean())
            
            return best_angle if best_angle is not None else None

        except Exception as e:
            _LOGGER.error("Error finding needle angle: %s", e)
            import traceback
            _LOGGER.error("Traceback: %s", traceback.format_exc())
            return None

    def _refine_needle_angle(self, gray_array: np.ndarray, cx: int, cy: int, 
                           coarse_angle: int, radius: int) -> Optional[float]:
        """Refine the needle angle by scanning around the coarse angle."""
        try:
            height, width = gray_array.shape
            best_angle = coarse_angle
            best_score = float('inf')
            
            # Scan ±4 degrees around the coarse angle in 0.5-degree increments
            for angle_offset in range(-8, 9):  # -4 to +4 degrees in 0.5 increments
                angle_deg = coarse_angle + (angle_offset * 0.5)
                angle_rad = math.radians(angle_deg)
                
                score = 0.0  # Use float to prevent overflow
                valid_points = 0
                
                for r in range(8, radius, 2):  # More precise sampling
                    # Apply same coordinate system fix as main detection
                    x = int(cx + r * math.cos(angle_rad))
                    y = int(cy - r * math.sin(angle_rad))  # Flip Y axis for image coordinates
                    
                    if 0 <= x < width and 0 <= y < height:
                        score += float(gray_array[y, x])  # Convert to float to prevent overflow
                        valid_points += 1
                
                if valid_points > 0:
                    avg_score = score / valid_points
                    if avg_score < best_score:
                        best_score = avg_score
                        best_angle = angle_deg
            
            return best_angle
            
        except Exception as e:
            _LOGGER.error("Error refining needle angle: %s", e)
            return None

    def _angle_to_value(self, angle: float) -> float:
        """Convert angle to gauge value using configuration."""
        # Convert clock hours to degrees (clock convention: 12=0°, 3=90°, 6=180°, 9=270°)
        min_angle_deg = self.config["min_angle_hours"] * 30  # 30 degrees per hour
        max_angle_deg = self.config["max_angle_hours"] * 30
        
        min_value = self.config["min_value"]
        max_value = self.config["max_value"]

        _LOGGER.debug("Angle conversion config:")
        _LOGGER.debug("  Min angle: %.1f hours = %.1f degrees (clock)", self.config["min_angle_hours"], min_angle_deg)
        _LOGGER.debug("  Max angle: %.1f hours = %.1f degrees (clock)", self.config["max_angle_hours"], max_angle_deg)
        _LOGGER.debug("  Value range: %.2f to %.2f %s", min_value, max_value, self.config.get("units", ""))
        _LOGGER.debug("  Raw detected angle: %.1f degrees (math convention)", angle)
        
        # Convert from mathematical angle convention to clock convention
        # Math: 0°=right, 90°=up, 180°=left, 270°=down
        # Clock: 0°=up(12), 90°=right(3), 180°=down(6), 270°=left(9)
        # Conversion: clock_angle = (90 - math_angle) % 360
        clock_angle = (90 - angle) % 360
        _LOGGER.debug("  Converted to clock angle: %.1f degrees", clock_angle)

        # Handle angle wrapping (e.g., from 7 o'clock to 5 o'clock)
        if min_angle_deg > max_angle_deg:
            # Gauge spans across 0 degrees (e.g., 210° to 150°)
            _LOGGER.debug("Handling angle wrapping (gauge spans across 0°)")
            if clock_angle >= min_angle_deg:
                angle_normalized = clock_angle - min_angle_deg
            else:
                angle_normalized = (360 - min_angle_deg) + clock_angle
            total_range = (360 - min_angle_deg) + max_angle_deg
            _LOGGER.debug("  Normalized angle: %.1f°, Total range: %.1f°", angle_normalized, total_range)
        else:
            # Normal case
            _LOGGER.debug("Normal angle range (no wrapping)")
            angle_normalized = clock_angle - min_angle_deg
            total_range = max_angle_deg - min_angle_deg
            _LOGGER.debug("  Normalized angle: %.1f°, Total range: %.1f°", angle_normalized, total_range)

        if total_range == 0:
            _LOGGER.warning("Total angle range is 0, returning min value")
            return min_value
            
        # Map angle to value
        value_range = max_value - min_value
        new_value = (angle_normalized / total_range) * value_range + min_value
        
        _LOGGER.debug("  Calculated raw value: %.3f", new_value)
        
        # Clamp to valid range
        clamped_value = max(min_value, min(max_value, new_value))
        if clamped_value != new_value:
            _LOGGER.debug("  Value clamped from %.3f to %.3f", new_value, clamped_value)
        
        _LOGGER.debug("  Final gauge value: %.2f %s", clamped_value, self.config.get("units", ""))
        
        return clamped_value


def create_simple_processor(processor_type: str, config: Dict[str, Any], sensor_name: str = "unknown"):
    """Create a simplified processor based on type."""
    if processor_type == PROCESSOR_ANALOG_GAUGE:
        return SimpleAnalogGaugeProcessor(config, sensor_name)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")