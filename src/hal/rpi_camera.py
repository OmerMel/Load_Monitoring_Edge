import subprocess
import os
import time
import cv2
import sys
import numpy as np
from typing import Optional
from datetime import datetime

from src.interfaces.imaging_device import ImagingDevice
from src.entities.image_frame import ImageFrame

# Class that handles image capture using the Raspberry Pi camera module (ribbon cable)


class RpiCamera(ImagingDevice):

    # ---------------------------------------------------------------------------------------------------------------#
    # Constructor - Initialize the camera settings
    # Args: Image width, Image height, Time to wait before capture (in milliseconds)
    def __init__(self, width: int = 1920, height: int = 1080, timeout_ms: int = 100):
        self.width = width
        self.height = height
        self.timeout_ms = timeout_ms
        self.temp_file = "temp_capture.jpg"

    # ---------------------------------------------------------------------------------------------------------------#
    # Function to capture a single image from the camera
    # Returns: The captured image as an ImageFrame, or None if capture failed
    def capture(self) -> Optional[ImageFrame]:

        # Remove previous temp file if it exists
        if os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except OSError:
                pass

        # Construct the command
        # -t: time delay before capture (ms)
        # -o: output file path
        # -n: Do not display preview window
        # --width, --height: Resolution
        command = [
            "rpicam-jpeg",
            "-o", self.temp_file,
            "-t", str(self.timeout_ms),
            "--width", str(self.width),
            "--height", str(self.height),
            "-n"
        ]

        try:
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print(f"Error executing rpicam-jpeg: {result.stderr}")
                return None

            # Check if file was created
            if not os.path.exists(self.temp_file):
                print("Error: Output file was not created.")
                return None

            # Load the image
            frame = cv2.imread(self.temp_file) # pylint: disable=no-member
            if frame is None:
                print("Error: Failed to decode captured image.")
                return None

            return ImageFrame(
                data=frame,
                timestamp=datetime.now(),
                source_id="rpi_camera"
            )

        except FileNotFoundError:
            print(
                "Error: 'rpicam-jpeg' command not found. Ensure libcamera-apps is installed.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during capture: {e}")
            return None

    # ---------------------------------------------------------------------------------------------------------------#
    # Function to cleanup the camera
    def cleanup(self):
        """Removes temporary files."""
        if os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except OSError:
                pass
    # ---------------------------------------------------------------------------------------------------------------#
