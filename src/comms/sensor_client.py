import requests
from typing import Optional

# ------------- Client responsible for sending sensor updates to the backend server -------------#

class SensorApiClient:

    # ---------------------------------------------------------------------------------------------------------------#
    # Constructor - Initialize the API client (The base URL of the backend server (e.g., 'http://192.168.10.2:8080'))
    def __init__(self, base_url: str):

        self.base_url = base_url.rstrip('/')
        print(f"Initialized API client with base URL: {self.base_url}")

    # ---------------------------------------------------------------------------------------------------------------#
    # Send sensor update to the server (The endpoint is /api/sensors/update)
    def send_update(
        self,
        train_id: int,
        carriage_number: int,
        tof_number: int,
        camera_number: int
    ) -> Optional[requests.Response]:

        # The endpoint is /api/sensors/update
        endpoint = f"{self.base_url}/api/sensors/update"

        # The data that is sent to the server
        payload = {
            "trainId": train_id,
            "carriageNumber": carriage_number,
            "tofNumber": tof_number,
            "cameraNumber": camera_number
        }

        try:
            print(f"Sending update to {endpoint}: {payload}")
            # Send the update to the server (POST request)
            response = requests.post(endpoint, json=payload, timeout=5)
            response.raise_for_status()
            return response  # Return the response from the server
        except requests.exceptions.RequestException as e:
            print(f"Failed to send update: {e}")
            return None
    # ---------------------------------------------------------------------------------------------------------------#
    