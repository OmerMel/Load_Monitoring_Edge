from dataclasses import dataclass
from datetime import datetime

@dataclass
class SensorReading:
    """Represents a single reading from a hardware sensor."""
    value: int
    timestamp: datetime
    sensor_type: str
    sensor_id: str  # Internal ID to distinguish between multiple sensors
