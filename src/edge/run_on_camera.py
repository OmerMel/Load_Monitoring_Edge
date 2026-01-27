import os
import cv2
import time
import sys
import subprocess
from datetime import datetime

# Client responsible for sending sensor updates to the backend server
from src.comms.sensor_client import SensorApiClient

# YOLO-based people detection logic
from src.vision.people_detector import PeopleDetector


def capture_image_with_rpicam():
    """
    Captures a still image using the system's 'rpicam-jpeg' command
    which supports the new Raspberry Pi camera stack (libcamera).
    """
    
    temp_img_path = "temp_capture.jpg"
    
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    # -------------------------------------------------
    # Countdown Logic
    # -------------------------------------------------
    print("Camera initialized. Get ready!")
    for i in range(4, 0, -1):
        print(f"\rCapturing in {i}...", end="")
        sys.stdout.flush()
        time.sleep(1)
    
    print("\nCapturing now! 📸")

    # -------------------------------------------------
    # Run the rpicam-jpeg command
    # -------------------------------------------------
    # -t 1: time delay before capture (ms)
    # -o: output file path
    # -n: Do not display preview window
    # --width 1920 --height 1080: Resolution of the captured image
    command = [
        "rpicam-jpeg",
        "-o", temp_img_path,
        "-t", "100", 
        "--width", "1920",
        "--height", "1080",
        "-n"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error executing rpicam-jpeg: {result.stderr}")
            return None
            
        # Load the captured image 
        frame = cv2.imread(temp_img_path)
        
        return frame

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main():
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------------------------
    # 1. Capture image using rpicam command
    # -------------------------------------------------
    frame = capture_image_with_rpicam()

    # Validate image capture
    if frame is None:
        raise SystemExit("Exiting due to camera failure.")

    # Generate timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------------------------------------------------
    # 2. Save the raw 'live' image
    # -------------------------------------------------
    live_image_path = os.path.join("outputs", f"live_{ts}.jpg")
    cv2.imwrite(live_image_path, frame)
    print(f"Saved raw captured image: {live_image_path}")

    # -------------------------------------------------
    # 3. Initialize Detector & Run Detection
    # -------------------------------------------------
    detector = PeopleDetector(
        model_path="yolov8n.pt",
        imgsz=640,
        conf=0.25,
        iou=0.5
    )

    count, boxes, annotated = detector.detect(frame)

    # -------------------------------------------------
    # 4. Save the annotated image
    # -------------------------------------------------
    annotated_path = os.path.join("outputs", f"annotated_{ts}.jpg")
    cv2.imwrite(annotated_path, annotated)
    
    print(f"People detected: {count}")
    print(f"Saved annotated image: {annotated_path}")

    # -------------------------------------------------
    # 5. Send detection results to backend server
    # -------------------------------------------------
    #base_url = "http://192.168.1.149:8080" # IP of the computer server
    base_url = "http://172.20.10.8:8080" # IP of the computer server

    client = SensorApiClient(base_url)

    try:
        resp = client.send_update(
            train_id=1,
            carriage_number=1,
            tof_number=0,
            camera_number=count
        )
        print(f"Server response: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Failed to send update to server: {e}")


if __name__ == "__main__":
    main()
