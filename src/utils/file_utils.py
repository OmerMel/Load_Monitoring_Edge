import os
import cv2
import datetime
from typing import Optional


class FileManager:

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.ensure_directory_exists(self.output_dir)

    def ensure_directory_exists(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def save_image(self, image: cv2.Mat, prefix: str = "image", timestamp: bool = True) -> Optional[str]:

        if image is None:
            print("Error: Cannot save None image.")
            return None

        filename = prefix
        if timestamp:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{ts}"

        filename += ".jpg"
        filepath = os.path.join(self.output_dir, filename)

        try:
            success = cv2.imwrite(filepath, image)
            if success:
                print(f"Saved image: {filepath}")
                return filepath
            else:
                print(f"Error: Failed to write image to {filepath}")
                return None
        except Exception as e:
            print(f"Error saving image: {e}")
            return None