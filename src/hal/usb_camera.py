import cv2
import numpy as np
from typing import Optional
from datetime import datetime

from src.interfaces.imaging_device import ImagingDevice
from src.entities.image_frame import ImageFrame


class UsbCamera(ImagingDevice):

    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        print(f"USB Camera {self.camera_index} configured for resolution: {self.width}x{self.height}")


    def capture(self) -> Optional[ImageFrame]:

        # V4L2 is the standard and most stable backend for USB cameras on Raspberry Pi/Linux
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not cap.isOpened():
            print("Error: Camera is not initialized.")
            return None
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Grab a few frames first to let the camera auto-adjust to the room's lighting
        for _ in range(30):
            cap.grab()
        
        ret, frame = cap.read() # pylint: disable=no-member

        cap.release()
        
        if not ret or frame is None:
            print("Error: Failed to grab frame from USB camera.")
            return None
            
        return ImageFrame(
            data=frame,
            timestamp=datetime.now(),
            source_id=f"usb_camera_{self.camera_index}"
        )


    def cleanup(self):
        """Releases the camera resource."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            print("Camera resource released.")
