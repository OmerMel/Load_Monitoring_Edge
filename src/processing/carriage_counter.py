import threading
from datetime import datetime
from src.interfaces.sensor import Sensor
from src.entities.sensor_reading import SensorReading

class CarriageCounter(Sensor):
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self._current_count = 0

        # Critical lock: ensures two doors don't update the count in the same instant
        self._lock = threading.Lock()

    def person_entered(self):
        with self._lock:
            self._current_count += 1
            count = self._current_count
        print(f"\n[EVENT] Person entered. Sensors counter: {count}")

    def person_exited(self):
        with self._lock:
            # Ensures the carriage count doesn't go below 0, carriage-wide (not just per door)
            self._current_count = max(0, self._current_count - 1)
            count = self._current_count
        print(f"\n[EVENT] Person exited. Sensors counter: {count}")

    def set_count(self, count: int):
        with self._lock:
            self._current_count = max(0, count)
            new_count = self._current_count
        print(f"\n[EVENT] Counter reset to {new_count} due to drift correction")

    def read(self) -> SensorReading:
        with self._lock:
            count = self._current_count

        return SensorReading(
            value=count,
            timestamp=datetime.now(),
            sensor_type="Carriage_Counter",
            sensor_id=self.sensor_id
        )