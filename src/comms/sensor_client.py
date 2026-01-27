from __future__ import annotations

import requests


class SensorApiClient:
    def __init__(self, base_url: str):
        # example: http://192.168.1.100:8080
        self.base_url = base_url.rstrip("/")

    def send_update(self, train_id: int, carriage_number: int, tof_number: int, camera_number: int) -> requests.Response:
        url = f"{self.base_url}/api/sensors/update"
        payload = {
            "trainId": train_id,
            "carriageNumber": carriage_number,
            "tofNumber": tof_number,
            "cameraNumber": camera_number,
        }

        # JSON + proper header
        return requests.post(url, json=payload, timeout=10)
