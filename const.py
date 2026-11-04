"""Constants for the Image to Sensor CV integration."""

DOMAIN = "image_to_sensor_cv"
DATA_SNAPSHOTS = f"{DOMAIN}_snapshots"

# Configuration keys
CONF_IMAGE_SOURCE = "image_source"
CONF_IMAGE_PATH = "image_path"
CONF_CAMERA_ENTITY = "camera_entity"
CONF_PRE_CAPTURE_SWITCH = "pre_capture_switch"
CONF_PRE_CAPTURE_DELAY_MS = "pre_capture_delay_ms"
CONF_CROP_CONFIG = "crop_config"
CONF_CROP_X = "crop_x"
CONF_CROP_Y = "crop_y"
CONF_CROP_WIDTH = "crop_width"
CONF_CROP_HEIGHT = "crop_height"
CONF_PROCESSORS = "processors"
CONF_SCAN_INTERVAL = "scan_interval"

# Processor types
PROCESSOR_ANALOG_GAUGE = "analog_gauge_reader"

# Analog gauge processor configuration
CONF_MIN_ANGLE_HOURS = "min_angle_hours"
CONF_MAX_ANGLE_HOURS = "max_angle_hours"
CONF_CENTER_OFFSET_X = "center_offset_x"
CONF_CENTER_OFFSET_Y = "center_offset_y"
CONF_MIN_VALUE = "min_value"
CONF_MAX_VALUE = "max_value"
CONF_UNITS = "units"

# Default values
DEFAULT_SCAN_INTERVAL = 30
DEFAULT_CROP_X = 0
DEFAULT_CROP_Y = 0
DEFAULT_CROP_WIDTH = 100
DEFAULT_CROP_HEIGHT = 100
DEFAULT_MIN_ANGLE_HOURS = 7
DEFAULT_MAX_ANGLE_HOURS = 5
DEFAULT_CENTER_OFFSET_X = 0
DEFAULT_CENTER_OFFSET_Y = 0
DEFAULT_MIN_VALUE = 0
DEFAULT_MAX_VALUE = 100
DEFAULT_UNITS = ""
DEFAULT_PRE_CAPTURE_DELAY_MS = 0

# Image source types
SOURCE_FILE = "file"
SOURCE_CAMERA = "camera"