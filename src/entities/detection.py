from dataclasses import dataclass

@dataclass
class DetectionBox:
    """Represents a single detected bounding box for an object (e.g., person)."""
    # Coordinates are in pixel space:
    # (x1, y1) -> top-left corner
    x1: int
    y1: int
    # (x2, y2) -> bottom-right corner
    x2: int
    y2: int
    conf: float  # Detection confidence score
