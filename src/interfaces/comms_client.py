from abc import ABC, abstractmethod

from src.entities.sensor_data_entity import SensorDataEntity


class CommsClient(ABC):
    """Abstract Base Class for any communication client (e.g., MQTT, HTTP)."""

    @abstractmethod
    def connect(self) -> None:
        """Establishes a connection to the communication broker or server."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection cleanly."""
        pass

    @abstractmethod
    def send_update(self, data: SensorDataEntity) -> bool:
        """Sends a sensor-data update to the server.

        Args:
            data (SensorDataEntity): The domain object to send.

        Returns:
            bool: True if the update was sent successfully, False otherwise.
        """
        pass
