import os
from datetime import datetime

import cv2

# Client responsible for sending sensor updates to the backend server
from src.comms.sensor_client import SensorApiClient

# YOLO-based people detection logic
from src.vision.people_detector import PeopleDetector


def main():
    """
    Main execution flow:
    1. Load an image from disk
    2. Detect people using YOLO
    3. Save an annotated output image
    4. Send the detected people count to the backend server
    """

    # Path to the input image captured by the camera
    image_path = "/home/loadmonitoring/projects/pi-edge/images/load_car_13.jpg"

    # Load the image using OpenCV (BGR format)
    frame = cv2.imread(image_path)

    # Validate image loading
    if frame is None:
        raise SystemExit(f"Failed to load image: {image_path}")

    # Initialize the people detector
    # - yolov8n.pt: lightweight YOLO model (CPU-friendly)
    # - imgsz: inference resolution
    # - conf: confidence threshold
    # - iou: IoU threshold for non-max suppression
    detector = PeopleDetector(
        model_path="yolov8n.pt",
        imgsz=640,
        conf=0.25,
        iou=0.5
    )

    # Run people detection on the image
    # count     -> number of detected people
    # boxes     -> bounding box metadata (not used here, but available)
    # annotated -> image with drawn bounding boxes and people count
    count, boxes, annotated = detector.detect(frame)

    # Generate a timestamped output filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("outputs", f"annotated_{ts}.jpg")

    # Save the annotated image to disk
    cv2.imwrite(out_path, annotated)

    # Log detection results locally
    print(f"People detected: {count}")
    print(f"Saved annotated image: {out_path}")

    # -------------------------------------------------
    # Send detection results to backend server
    # -------------------------------------------------

    # Base URL of the backend server (Spring Boot)
    #base_url = "http://192.168.1.149:8080" ##IP of the computer server
    base_url = "http://172.20.10.8:8080" #IP of the computer server

    # Initialize API client
    client = SensorApiClient(base_url)

    # Send sensor update:
    # cameraNumber is used here to represent the number of detected people
    resp = client.send_update(
        train_id=1,
        carriage_number=1,
        tof_number=0,
        camera_number=count
    )

    # Print server response for validation/debugging
    print(f"Server response: {resp.status_code} {resp.text}")


# Entry point when running this file as a script/module
if __name__ == "__main__":
    main()
