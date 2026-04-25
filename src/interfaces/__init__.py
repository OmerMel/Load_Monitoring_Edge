# Expose interfaces at module level
from .imaging_device import ImagingDevice
from .sensor import Sensor
from .comms_client import CommsClient

__all__ = ["ImagingDevice", "Sensor", "CommsClient"]
