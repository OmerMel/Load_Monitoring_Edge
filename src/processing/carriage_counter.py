import threading
from datetime import datetime
from src.interfaces.sensor import Sensor
from src.entities.sensor_reading import SensorReading

class CarriageCounter(Sensor):
    def __init__(self, sensor_id="carriage_total"):
        self.sensor_id = sensor_id
        self._current_count = 0
        
        # מנעול קריטי: מוודא ששתי דלתות לא מעדכנות את המספר באותו חלקיק שנייה
        self._lock = threading.Lock()

    def person_entered(self):
        """פונקציה שנקראת על ידי כל דלת כשמישהו נכנס"""
        with self._lock:
            self._current_count += 1
            print(f"\n[EVENT] Person entered. Sensors counter: {self._current_count}")

    def person_exited(self):
        """פונקציה שנקראת על ידי כל דלת כשמישהו יוצא"""
        with self._lock:
            # כאן אנחנו מוודאים שהקרון לא יורד מ-0, ולא בתוך הדלת הבודדת!
            self._current_count = max(0, self._current_count - 1)
            print(f"\n[EVENT] Person exited. Sensors counter: {self._current_count}")

    def set_count(self, count: int):
        """מעדכן את המונה באופן ישיר (למשל במקרה של זיהוי סחיפה)"""
        with self._lock:
            self._current_count = max(0, count)
            print(f"\n[EVENT] Counter reset to {self._current_count} due to drift correction")

    def read(self) -> SensorReading:
        """
        מממש את הממשק Sensor. 
        זה מה שה-LoadMonitorService יקרא!
        """
        with self._lock:
            count = self._current_count
            
        return SensorReading(
            value=count,
            timestamp=datetime.now(),
            sensor_type="Carriage_Counter",
            sensor_id=self.sensor_id
        )