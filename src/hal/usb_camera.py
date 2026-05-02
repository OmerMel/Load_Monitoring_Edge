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
        
        print(f"USB Camera {self.camera_index} configured for resolution: {self.width}x{self.height}")
    # ---------------------------------------------------------------------------------------------------------------#
    # Function to capture a single image from the USB camera
    # Returns: The captured image as an ImageFrame, or None if capture failed
    def capture(self) -> Optional[ImageFrame]:

        # Initialize the VideoCapture object using V4L2 (Video4Linux2) backend
        # V4L2 is the standard and most stable backend for USB cameras on Raspberry Pi/Linux
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2) # pylint: disable=no-member

        if not cap.isOpened():
            print("Error: Camera is not initialized.")
            return None
            
        # Try to set the requested resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Throwing in a few frames to let the camera balance on the light in the room
        for _ in range(30):
            cap.grab()
        
        # Read a frame from the camera
        ret, frame = cap.read() # pylint: disable=no-member

        # Closing the camera
        cap.release()
        
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
