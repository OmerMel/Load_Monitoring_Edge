from datetime import datetime
from typing import Callable, List, Optional

from src.entities.sensor_data_entity import SensorDataEntity
from src.interfaces.comms_client import CommsClient
from src.interfaces.imaging_device import ImagingDevice
from src.interfaces.sensor import Sensor
from src.processing.image_processor import ImageProcessor


class LoadMonitorService:
    """
    Core orchestrator for the Pi-Edge load monitoring system.
    Ties together image input, sensors, processing, and communications.
    """

    def __init__(
        self,
        camera: ImagingDevice,
        sensors: List[Sensor],
        processor: ImageProcessor,
        comms: CommsClient,
        train_id: int,
        carriage_number: int,
        tof_alive_check: Optional[Callable[[], bool]] = None,
    ):
        self.camera = camera
        self.sensors = sensors
        self.processor = processor
        self.comms = comms
        self.train_id = train_id
        self.carriage_number = carriage_number
        self.tof_alive_check = tof_alive_check

    def run_cycle(self) -> Optional[dict]:
        """
        Executes a single monitoring cycle:
        1. Capture image
        2. Read sensors
        3. Process image
        4. Publish update

        Either source (camera or IR) may fail independently. The cycle keeps
        going with whatever data is available, and reports each source's
        status to the server instead of silently dropping the whole update.
        """
        camera_status = "ok"
        ir_status = "ok"

        frame = self.camera.capture()
        if frame is None:
            print("Warning: Failed to capture image.")
            camera_status = "unavailable"

        try:
            final_ir_count = self.sensors[0].read().value
        except Exception as e:
            print(f"Warning: Failed to read IR sensor: {e}")
            final_ir_count = 0
            ir_status = "unavailable"

        if self.tof_alive_check is not None and not self.tof_alive_check():
            print("Warning: ToF background thread appears dead/stuck.")
            ir_status = "unavailable"

        if frame is not None:
            person_count, detections = self.processor.detect(frame)
        else:
            person_count, detections = 0, []

        sensor_data = SensorDataEntity(
            train_id=self.train_id,
            carriage_number=self.carriage_number,
            camera_count=person_count,
            ir_count=final_ir_count,
            calculated_occupancy=0,
            camera_status=camera_status,
            ir_status=ir_status,
            timestamp=datetime.now(),
        )

        success = self.comms.send_update(sensor_data)
        if not success:
            print("Warning: Failed to send update to server.")

        return {
            "sensor_data": sensor_data,
            "frame": frame,
            "detections": detections,
            "person_count": person_count,
        }