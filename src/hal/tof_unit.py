import time
import board
import digitalio
import adafruit_vl53l0x

class TofUnit:
    def __init__(self, xshut_pin: int, target_i2c_address: int, threshold_mm: int, i2c_bus):
        self.target_address = target_i2c_address
        self.threshold = threshold_mm
        self.i2c_bus = i2c_bus
        self.sensor = None

        # Convert the pin number to a CircuitPython pin object (e.g., xshut_pin=17 -> board.D17)
        pin_name = f"D{xshut_pin}"
        if not hasattr(board, pin_name):
            raise ValueError(f"Invalid GPIO pin: {xshut_pin}")

        board_pin = getattr(board, pin_name)

        self.xshut = digitalio.DigitalInOut(board_pin)
        self.xshut.direction = digitalio.Direction.OUTPUT

        # Sensor is off by default when the object is created
        self.xshut.value = False

    def turn_off(self):
        self.xshut.value = False

    def turn_on(self):
        self.xshut.value = True
        time.sleep(0.1)  # Sensor needs a few milliseconds to wake up

    def setup_sensor(self):
        """Called only while the sensor is powered on: connects at the default
        address (0x29) and burns in the target address (kept in memory until next power-down)."""
        self.sensor = adafruit_vl53l0x.VL53L0X(self.i2c_bus)
        self.sensor.set_address(self.target_address)

    def read_distance(self) -> int:
        if self.sensor is not None:
            return self.sensor.range
        return 0

    def is_blocked(self) -> bool:
        distance = self.read_distance()
        return 0 < distance < self.threshold