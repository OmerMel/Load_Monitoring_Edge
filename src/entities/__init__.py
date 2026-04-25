# Expose entities at module level
from .image_frame import ImageFrame
from .sensor_reading import SensorReading
from .detection import DetectionBox

__all__ = ["ImageFrame", "SensorReading", "DetectionBox"]
