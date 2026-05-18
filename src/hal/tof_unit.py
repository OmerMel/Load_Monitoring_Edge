# hal/tof_unit.py
import time
from gpiozero import DigitalOutputDevice
import adafruit_vl53l0x

class TofUnit:
    def __init__(self, xshut_pin: int, target_i2c_address: int, threshold_mm: int, i2c_bus):
        self.target_address = target_i2c_address
        self.threshold = threshold_mm
        self.i2c_bus = i2c_bus
        
        # הגדרת פין ה-XSHUT כפלט (Output) לשליטה בחשמל של החיישן
        self.xshut = DigitalOutputDevice(xshut_pin)
        
        # המשתנה שיחזיק את אובייקט החיישן של Adafruit אחרי שנדליק אותו
        self.sensor = None 

    def turn_off(self):
        """מוריד מתח ל-LOW ומכבה את החיישן"""
        self.xshut.off()

    def turn_on(self):
        """מעלה מתח ל-HIGH, מדליק וממתין לאתחול"""
        self.xshut.on()
        time.sleep(0.1)  # חובה לתת לחיישן כמה מילישניות להתעורר

    def setup_sensor(self):
        """
        פונקציה זו נקראת *רק* כשהחיישן דלוק.
        היא מתחברת אליו בכתובת ברירת המחדל ומשנה לו לכתובת היעד.
        """
        # 1. התחברות לחיישן בכתובת המקורית שלו (0x29)
        self.sensor = adafruit_vl53l0x.VL53L0X(self.i2c_bus)
        
        # 2. צריבת הכתובת החדשה (נשמר בזיכרון ה-RAM של החיישן עד לכיבוי הבא)
        self.sensor.set_address(self.target_address)
        
    def read_distance(self) -> int:
        """קורא מרחק מהחומרה. מחזיר מרחק במילימטרים."""
        if self.sensor is not None:
            return self.sensor.range
        return 0

    def is_blocked(self) -> bool:
        """
        מחזיר True אם מישהו עובר בטווח.
        הערה: VL53L0X מחזיר לפעמים מספרים גבוהים מאוד (כמו 8190) כשהוא לא קולט כלום.
        לכן בודקים שהמרחק גדול מ-0 אבל קטן מהסף.
        """
        distance = self.read_distance()
        return 0 < distance < self.threshold