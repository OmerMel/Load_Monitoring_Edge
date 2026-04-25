import cv2
import numpy as np
from typing import Optional
from datetime import datetime

from src.interfaces.imaging_device import ImagingDevice
from src.entities.image_frame import ImageFrame

# Class that handles image capture using a USB webcam (e.g., Microsoft LifeCam)


class UsbCamera(ImagingDevice):

    # ---------------------------------------------------------------------------------------------------------------#
    # Constructor - Initialize the camera settings
    # Args: Camera index (usually 0 for the first USB camera), Image width, Image height
    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        # Initialize the VideoCapture object using V4L2 (Video4Linux2) backend
        # V4L2 is the standard and most stable backend for USB cameras on Raspberry Pi/Linux
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2) # pylint: disable=no-member
        
        if not self.cap.isOpened():
            print(f"Error: Could not open USB camera at index {self.camera_index}.")
        else:
            # Try to set the requested resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Read back the actual resolution to confirm
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera initialized at resolution: {int(actual_width)}x{int(actual_height)}")

    # ---------------------------------------------------------------------------------------------------------------#
    # Function to capture a single image from the USB camera
    # Returns: The captured image as an ImageFrame, or None if capture failed
    def capture(self) -> Optional[ImageFrame]:
        if not self.cap.isOpened():
            print("Error: Camera is not initialized.")
            return None
            
        # Read a frame from the camera
        ret, frame = self.cap.read() # pylint: disable=no-member
        
        if not ret or frame is None:
            print("Error: Failed to grab frame from USB camera.")
            return None
            
        return ImageFrame(
            data=frame,
            timestamp=datetime.now(),
            source_id=f"usb_camera_{self.camera_index}"
        )

    # ---------------------------------------------------------------------------------------------------------------#
    # Function to cleanup the camera
    def cleanup(self):
        """Releases the camera resource."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            print("Camera resource released.")
    # ---------------------------------------------------------------------------------------------------------------#
