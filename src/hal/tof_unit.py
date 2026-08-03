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
        
        # המרת מספר הפין לאובייקט פין של CircuitPython
        # אם xshut_pin הוא 17, אנחנו משיגים את board.D17
        pin_name = f"D{xshut_pin}"
        if not hasattr(board, pin_name):
            raise ValueError(f"Invalid GPIO pin: {xshut_pin}")
            
        board_pin = getattr(board, pin_name)
        
        # הגדרת הפין כ-Output דרך digitalio (הספרייה של Adafruit)
        self.xshut = digitalio.DigitalInOut(board_pin)
        self.xshut.direction = digitalio.Direction.OUTPUT
        
        # מכבים את החיישן כברירת מחדל בעת יצירת האובייקט
        self.xshut.value = False

    def turn_off(self):
        """מוריד מתח ל-LOW ומכבה את החיישן"""
        self.xshut.value = False

    def turn_on(self):
        """מעלה מתח ל-HIGH, מדליק וממתין לאתחול"""
        self.xshut.value = True
        time.sleep(0.1)  # חובה לתת לחיישן כמה מילישניות להתעורר

    def setup_sensor(self):
        """
        פונקציה זו נקראת *רק* כשהחיישן דלוק.
        היא מתחברת אליו בכתובת ברירת המחדל ומשנה לו לכתובת היעד.
        """
        # 1. התחברות לחיישן בכתובת המקורית שלו (0x29)
        self.sensor = adafruit_vl53l0x.VL53L0X(self.i2c_bus)
        
        # 2. צריבת הכתובת החדשה (נשמר בזיכרון עד לכיבוי הבא)
        self.sensor.set_address(self.target_address)
        
    def read_distance(self) -> int:
        """קורא מרחק מהחומרה. מחזיר מרחק במילימטרים."""
        if self.sensor is not None:
            return self.sensor.range
        return 0

    def is_blocked(self) -> bool:
        """
        מחזיר True אם מישהו עובר בטווח.
        """
        distance = self.read_distance()
        return 0 < distance < self.threshold