from datetime import datetime
from typing import List, Optional

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

    # ---------------------------------------------------------------------------------------------------------------#
    # Function to initialize the LoadMonitorService
    def __init__(
        self,
        camera: ImagingDevice,
        sensors: List[Sensor],
        processor: ImageProcessor,
        comms: CommsClient,
        train_id: int,
        carriage_number: int,
    ):
        self.camera = camera
        self.sensors = sensors
        self.processor = processor
        self.comms = comms
        self.train_id = train_id
        self.carriage_number = carriage_number

    # ---------------------------------------------------------------------------------------------------------------#
    # Function to run a single monitoring cycle
    def run_cycle(self) -> Optional[dict]:
        """
        Executes a single monitoring cycle:
        1. Capture image
        2. Read sensors
        3. Process image
        4. Publish update
        """
        frame = self.camera.capture()
        if frame is None:
            print("Warning: Failed to capture image.")
            return None

        total_ir_count = 0
        for sensor in self.sensors:
            reading = sensor.read()
            # TODO: Replace dummy sensor values when real hardware integration is ready.
            total_ir_count += reading.value
        final_ir_count = total_ir_count / \
            len(self.sensors) if self.sensors else 0

        person_count, detections = self.processor.detect(frame)

        sensor_data = SensorDataEntity(
            train_id=self.train_id,
            carriage_number=self.carriage_number,
            camera_count=person_count,
            ir_count=final_ir_count,
            calculated_occupancy=0,
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
    # ---------------------------------------------------------------------------------------------------------------#
