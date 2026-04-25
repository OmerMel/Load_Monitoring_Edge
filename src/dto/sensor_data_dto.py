from dataclasses import dataclass


@dataclass(frozen=True)
class SensorDataDTO:
    """Wire-format contract for a sensor-data update.

    Field names intentionally use the external (camelCase) schema
    expected by the backend. 
    """
    trainId: int
    carriageNumber: int
    cameraCount: int
    irCount: int
    calculatedOccupancy: int
    timestamp: str