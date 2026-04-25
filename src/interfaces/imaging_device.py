from abc import ABC, abstractmethod
from typing import Optional
from src.entities.image_frame import ImageFrame

class ImagingDevice(ABC):
    """Abstract Base Class for any imaging device (e.g., USB Camera, RPi Camera)."""
    
    @abstractmethod
    def capture(self) -> Optional[ImageFrame]:
        """Captures a single image frame.
        
        Returns:
            Optional[ImageFrame]: The captured frame and metadata, or None if capture failed.
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Releases any hardware resources associated with the device."""
        pass
