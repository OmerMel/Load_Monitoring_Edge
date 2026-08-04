from dataclasses import dataclass
from datetime import datetime
from typing import Literal


# "ok"          - the source produced a valid reading this cycle
# "unavailable" - the source failed or has no data this cycle (hardware
#                 not connected, capture failed, background thread died, etc.)
SourceStatus = Literal["ok", "unavailable"]


@dataclass(frozen=True)
class SensorDataEntity:
    """Domain representation of a single monitoring-cycle reading.

    This is the internal model used by the processing layer. It is
    decoupled from any transport format (MQTT, HTTP, etc.) — conversion
    to the wire format is the responsibility of a dedicated converter.
    """
    train_id: int
    carriage_number: int
    camera_count: int
    ir_count: int
    calculated_occupancy: int
    camera_status: SourceStatus
    ir_status: SourceStatus
    timestamp: datetime
