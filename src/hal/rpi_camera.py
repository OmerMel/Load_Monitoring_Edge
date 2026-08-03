import subprocess
import os
import cv2
from typing import Optional
from datetime import datetime

from src.interfaces.imaging_device import ImagingDevice
from src.entities.image_frame import ImageFrame


class RpiCamera(ImagingDevice):

    def __init__(self, width: int = 1920, height: int = 1080, timeout_ms: int = 100):
        self.width = width
        self.height = height
        self.timeout_ms = timeout_ms
        self.temp_file = "temp_capture.jpg"

    def capture(self) -> Optional[ImageFrame]:
        if os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except OSError:
                pass

        # -t: capture delay (ms), -o: output path, -n: no preview window
        command = [
            "rpicam-jpeg",
            "-o", self.temp_file,
            "-t", str(self.timeout_ms),
            "--width", str(self.width),
            "--height", str(self.height),
            "-n"
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print(f"Error executing rpicam-jpeg: {result.stderr}")
                return None

            if not os.path.exists(self.temp_file):
                print("Error: Output file was not created.")
                return None

            frame = cv2.imread(self.temp_file)
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

    def cleanup(self):
        """Removes temporary files."""
        if os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except OSError:
                pass