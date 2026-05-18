import random
import time
from datetime import datetime
from src.hal.tof_unit import TofUnit
from src.interfaces.sensor import Sensor
from src.entities.sensor_reading import SensorReading
from src.processing.carriage_counter import CarriageCounter

class TofPair(Sensor):
    """
    Hardware Abstraction for a ToF sensor.
    Currently implements a dummy reading for testing purposes.
    """
    
    def __init__(self, outside_unit: TofUnit, inside_unit: TofUnit, sensor_id: str, shared_counter: CarriageCounter, poll_interval_sec: float = 0.05):
            self.outside = outside_unit
            self.inside = inside_unit
            self.sensor_id = sensor_id
            self.poll_interval = poll_interval_sec
            self.shared_counter = shared_counter
            self.state = "IDLE"
            self.is_running = False


    def start_polling(self):
        """לולאה שרצה ברקע ודוגמת את החיישנים בקצב גבוה"""
        self.is_running = True
        while self.is_running:
            self._update_state()
            time.sleep(self.poll_interval)

    def stop_polling(self):
        self.is_running = False

    def _update_state(self):
        out_blocked = self.outside.is_blocked()
        in_blocked = self.inside.is_blocked()

        # ----------------------------------------------------
        # מצב התחלתי - ממתינים לתנועה
        # ----------------------------------------------------
        if self.state == "IDLE":
            if out_blocked and not in_blocked:
                self.state = "ENTER_START"   # התחיל להיכנס
            elif in_blocked and not out_blocked:
                self.state = "EXIT_START"    # התחיל לצאת
            elif out_blocked and in_blocked:
                self.state = "UNKNOWN"       # חסימה פתאומית של שניהם (נדיר)

        # ----------------------------------------------------
        # תהליך כניסה (Person is entering)
        # ----------------------------------------------------
        elif self.state == "ENTER_START":
            if out_blocked and in_blocked:
                self.state = "ENTER_CROSSED"  # הגיע לאמצע המעבר (שניהם חסומים)
            elif not out_blocked and in_blocked:
                self.state = "ENTER_FINISHING" # דילג על האמצע (עבר מהר מאוד)
            elif not out_blocked and not in_blocked:
                self.state = "IDLE"           # אזעקת שווא / התחרט וחזר החוצה

        elif self.state in ["ENTER_CROSSED", "ENTER_FINISHING"]:
            if not out_blocked and not in_blocked:
                # המעבר הושלם בהצלחה! האדם סיים לעבור ושני החיישנים פנויים
                self.shared_counter.person_entered()
                print(f">>> [ToF {self.sensor_id}] Person ENTERED.")
                self.state = "IDLE"
            elif not out_blocked and in_blocked:
                self.state = "ENTER_FINISHING" # התקדמות טבעית קדימה
            elif out_blocked and not in_blocked:
                self.state = "ENTER_START"     # האדם הלך אחורה במקום לסיים כניסה

        # ----------------------------------------------------
        # תהליך יציאה (Person is exiting)
        # ----------------------------------------------------
        elif self.state == "EXIT_START":
            if out_blocked and in_blocked:
                self.state = "EXIT_CROSSED"   # הגיע לאמצע
            elif out_blocked and not in_blocked:
                self.state = "EXIT_FINISHING" # דילג על האמצע (עבר מהר)
            elif not out_blocked and not in_blocked:
                self.state = "IDLE"           # התחרט וחזר פנימה לקרון

        elif self.state in ["EXIT_CROSSED", "EXIT_FINISHING"]:
            if not out_blocked and not in_blocked:
                # המעבר הושלם בהצלחה!
                self.shared_counter.person_exited()
                print(f"<<< [ToF {self.sensor_id}] Person EXITED.")
                self.state = "IDLE"
            elif out_blocked and not in_blocked:
                self.state = "EXIT_FINISHING" # התקדמות טבעית החוצה
            elif not out_blocked and in_blocked:
                self.state = "EXIT_START"     # האדם הלך אחורה אל תוך הקרון

        # ----------------------------------------------------
        # מצב לא ידוע (הפרעה או חסימה פתאומית)
        # ----------------------------------------------------
        elif self.state == "UNKNOWN":
            if not out_blocked and not in_blocked:
                self.state = "IDLE" # ממתינים שיתפנה ומתאפסים

    def read(self) -> SensorReading:
        return SensorReading(
            value=self.current_count,
            timestamp=datetime.now(),
            sensor_type="ToF_Pair",
            sensor_id=self.sensor_id
        )
