from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from src.entities.image_frame import ImageFrame
from src.interfaces.imaging_device import ImagingDevice


class FolderImageSource(ImagingDevice):
    """Simple image source that returns files from a folder one by one."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, images_dir: str):
        self.images_dir = Path(images_dir)
        self.image_paths = self._load_image_paths()
        self.current_index = 0

    def _load_image_paths(self) -> list[Path]:
        if not self.images_dir.exists():
            print(f"Error: images folder not found: {self.images_dir}")
            return []

        return sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

    @property
    def exhausted(self) -> bool:
        return self.current_index >= len(self.image_paths)

    def capture(self) -> Optional[ImageFrame]:
        if self.exhausted:
            print("No more images to process.")
            return None

        image_path = self.image_paths[self.current_index]
        self.current_index += 1

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Failed to decode image: {image_path}")
            return None

        return ImageFrame(
            data=image,
            timestamp=datetime.now(),
            source_id=f"file:{image_path.name}",
        )

    def cleanup(self) -> None:
        pass
