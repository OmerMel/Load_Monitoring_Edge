from abc import ABC, abstractmethod
from src.entities.sensor_reading import SensorReading

class Sensor(ABC):
    """Abstract Base Class for any hardware sensor (e.g., IR Sensor, ToF Sensor)."""
    
    @abstractmethod
    def read(self) -> SensorReading:
        """Reads data from the sensor.
        
        Returns:
            SensorReading: The current reading and metadata.
        """
        pass
