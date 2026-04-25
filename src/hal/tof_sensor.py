import random
from datetime import datetime
from src.interfaces.sensor import Sensor
from src.entities.sensor_reading import SensorReading

class TOFSensor(Sensor):
    """
    Hardware Abstraction for a ToF sensor.
    Currently implements a dummy reading for testing purposes.
    """
    
    def __init__(self, sensor_id: str = "tof_sensor_x"):
        self.sensor_id = sensor_id

    def read(self) -> SensorReading:
        """
        Simulates reading from the ToF hardware.
        Returns:
            SensorReading: A random integer between 0 and 20 for testing.
        """
        # In the future, this will interface with the actual ToF hardware
        val = random.randint(0, 20)
        
        return SensorReading(
            value=val,
            timestamp=datetime.now(),
            sensor_type="ToF",
            sensor_id=self.sensor_id
        )
