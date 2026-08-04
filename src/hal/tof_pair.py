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
            self.last_update = time.time()

    def start_polling(self):
        self.is_running = True
        while self.is_running:
            try:
                self._update_state()
                self.last_update = time.time() 
            except Exception as e:
                print(f"[ToF {self.sensor_id}] Error reading sensors: {e}")
            time.sleep(self.poll_interval)

    def is_alive(self, max_silence_sec: float = 5.0) -> bool:
        """Returns False if the polling loop hasn't ticked recently (thread died or stuck)."""
        return (time.time() - self.last_update) < max_silence_sec

    def stop_polling(self):
        self.is_running = False

    def _update_state(self):
        out_blocked = self.outside.is_blocked()
        in_blocked = self.inside.is_blocked()

        # Initial state - waiting for movement
        if self.state == "IDLE":
            if out_blocked and not in_blocked:
                self.state = "ENTER_START"   # Started entering
            elif in_blocked and not out_blocked:
                self.state = "EXIT_START"    # Started exiting
            elif out_blocked and in_blocked:
                self.state = "UNKNOWN"       # Sudden blockage of both (rare)

        # Entry process (person is entering)
        elif self.state == "ENTER_START":
            if out_blocked and in_blocked:
                self.state = "ENTER_CROSSED"  # Reached the middle of the passage (both blocked)
            elif not out_blocked and in_blocked:
                self.state = "ENTER_FINISHING" # Skipped the middle (moved very fast)
            elif not out_blocked and not in_blocked:
                self.state = "IDLE"           # False alarm / changed their mind and stepped back out

        elif self.state in ["ENTER_CROSSED", "ENTER_FINISHING"]:
            if not out_blocked and not in_blocked:
                # Crossing completed successfully! Person finished passing, both sensors clear
                self.shared_counter.person_entered()
                self.state = "IDLE"
            elif not out_blocked and in_blocked:
                self.state = "ENTER_FINISHING" # Natural forward progress
            elif out_blocked and not in_blocked:
                self.state = "ENTER_START"     # Person stepped back instead of finishing the entry

        # Exit process (person is exiting)
        elif self.state == "EXIT_START":
            if out_blocked and in_blocked:
                self.state = "EXIT_CROSSED"   # Reached the middle
            elif out_blocked and not in_blocked:
                self.state = "EXIT_FINISHING" # Skipped the middle (moved fast)
            elif not out_blocked and not in_blocked:
                self.state = "IDLE"           # Changed their mind and stepped back into the carriage

        elif self.state in ["EXIT_CROSSED", "EXIT_FINISHING"]:
            if not out_blocked and not in_blocked:
                # Crossing completed successfully!
                self.shared_counter.person_exited()
                self.state = "IDLE"
            elif out_blocked and not in_blocked:
                self.state = "EXIT_FINISHING" # Natural progress outward
            elif not out_blocked and in_blocked:
                self.state = "EXIT_START"     # Person stepped back into the carriage

        # Unknown state (interference or sudden blockage)
        elif self.state == "UNKNOWN":
            if not out_blocked and not in_blocked:
                self.state = "IDLE" # Wait for it to clear and reset

    def read(self) -> SensorReading:
        return SensorReading(
            value=self.current_count,
            timestamp=datetime.now(),
            sensor_type="ToF_Pair",
            sensor_id=self.sensor_id
        )