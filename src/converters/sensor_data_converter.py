from datetime import datetime

from src.entities.sensor_data_entity import SensorDataEntity
from src.dto.sensor_data_dto import SensorDataDTO


class SensorDataConverter:
    """Mapper between SensorDataEntity (domain) and
    SensorDataDTO (wire contract)."""

    @staticmethod
    def to_dto(entity: SensorDataEntity) -> SensorDataDTO:
        return SensorDataDTO(
            trainId=entity.train_id,
            carriageNumber=entity.carriage_number,
            cameraCount=entity.camera_count,
            irCount=entity.ir_count,
            calculatedOccupancy=entity.calculated_occupancy,
            timestamp=entity.timestamp.isoformat(),
        )

    @staticmethod
    def to_entity(dto: SensorDataDTO) -> SensorDataEntity:
        return SensorDataEntity(
            train_id=dto.trainId,
            carriage_number=dto.carriageNumber,
            camera_count=dto.cameraCount,
            ir_count=dto.irCount,
            calculated_occupancy=dto.calculatedOccupancy,
            timestamp=datetime.fromisoformat(dto.timestamp),
        )
