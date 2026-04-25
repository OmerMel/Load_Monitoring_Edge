import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ImageFrame:
    """Represents a single captured image frame with metadata."""
    data: np.ndarray
    timestamp: datetime
    source_id: str
