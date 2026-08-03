import json
import time
import uuid
from dataclasses import asdict

import paho.mqtt.client as mqtt

from src.entities.sensor_data_entity import SensorDataEntity
from src.converters.sensor_data_converter import SensorDataConverter
from src.interfaces.comms_client import CommsClient


class MqttSensorClient(CommsClient):
    def __init__(self, broker_address: str, train_id: str, carriage_number: int, port: int = 1883):
        self.broker_address = broker_address
        self.port = port
        self.client_id = f"train_{train_id}_carriage_{carriage_number}_{uuid.uuid4().hex[:8]}"
        self.topic = "train/sensors/updates"
        self.connected = False
        self.client = mqtt.Client(
            client_id=self.client_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_log = self._on_log

        print(
            f"Initialized MQTT client for broker: {self.broker_address}:{self.port}")

    # rc: 0 = connection successful, any other value = connection refused
    def _on_connect(self, client, userdata, flags, rc):
        print(f"on_connect rc={rc}")
        if rc == 0:
            self.connected = True
            print(f"Connected to MQTT Broker at {self.broker_address}")
        else:
            self.connected = False
            print(f"Failed to connect, return code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print(f"Disconnected from MQTT Broker (rc={rc})")

    def _on_publish(self, client, userdata, message_id):
        pass

    def _on_log(self, client, userdata, level, buffer):
        pass

    def connect(self):
        print(f"Connecting to {self.broker_address}:{self.port}...")
        self.client.connect(self.broker_address, self.port, 60)
        self.client.loop_start()

        for _ in range(20):
            if self.connected:
                break
            time.sleep(0.2)

        print(f"Connected state: {self.connected}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("MQTT Client disconnected.")

    def get_status(self) -> bool:
        return self.connected

    def send_update(self, data: SensorDataEntity) -> bool:
        if not self.connected:
            print("Client is not connected. Cannot publish.")
            return False

        dto = SensorDataConverter.to_dto(data)
        json_payload = json.dumps(asdict(dto))

        info = self.client.publish(self.topic, json_payload, qos=1)
        info.wait_for_publish()

        return info.rc == mqtt.MQTT_ERR_SUCCESS