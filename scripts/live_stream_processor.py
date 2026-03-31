import sys
import os
import time
import signal
import argparse
import cv2
import random

# Add the project root to the python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.file_utils import FileManager
from src.camera import Camera
from src.comms.mqtt_client import MqttSensorClient
from src.models.people_detector import PeopleDetector


# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

# ---------------------------------------------------------------------------------------------------------------#
# Function to get the ToF reading
# Returns: A random integer between 0 and 20 for testing.
def get_tof_reading() -> int:
    """
    Placeholder function to simulate ToF sensor reading.
    Returns a random integer between 0 and 20 for testing.
    """
    # In the future, this will interface with the actual ToF hardware
    return random.randint(0, 20)

# ---------------------------------------------------------------------------------------------------------------#

def main():

    global running
    signal.signal(signal.SIGINT, signal_handler)

   # ------------- Parse arguments -------------#
    parser = argparse.ArgumentParser(
        description="Run live people detection on Raspberry Pi camera.")
    parser.add_argument(
        "--broker", type=str, default="test.mosquitto.org", help="MQTT Broker address.")
    parser.add_argument("--port", type=int, default=1883,
                        help="MQTT Broker port.")
    parser.add_argument("--train-id", type=str, default="1", help="Train ID.")
    parser.add_argument("--carriage-number", type=int,
                        default=1, help="Carriage Number.")
    parser.add_argument("--model", type=str,
                        default="yolov8n.pt", help="Path to YOLO model file.")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Interval between captures (seconds).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detection.")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (e.g., 640, 1280).")
    parser.add_argument("--clahe", action="store_true",
                        help="Enable CLAHE preprocessing for better contrast.")
    parser.add_argument("--min-area", type=int, default=500,
                        help="Minimum box area to consider valid.")
    parser.add_argument("--single-shot", action="store_true",
                        help="Run a single detection cycle and exit.")

    args = parser.parse_args()

    # --------------------------------------------- Initialize components -----------------------------------------------#
    print("Initializing components...")
    try:
        # ---------------------------------------------------------------------------------------------------------------#
        # Initialize the camera
        camera = Camera(width=1920, height=1080)
        # ---------------------------------------------------------------------------------------------------------------#
        # Initialize the people detector
        detector = PeopleDetector(
            model_path=args.model, # Path to the YOLO model file
            conf=args.conf, # Confidence threshold for detection
            iou=args.iou, # IoU threshold for NMS
            imgsz=args.imgsz, # Inference image size
            min_box_area=args.min_area, # Minimum box area to consider valid
            use_clahe=args.clahe # Enable CLAHE preprocessing for better contrast
        )
        # ---------------------------------------------------------------------------------------------------------------#
        # Initialize the file manager
        file_manager = FileManager(output_dir="outputs")
        # ---------------------------------------------------------------------------------------------------------------#
        # Initialize the MQTT client
        mqtt_client = MqttSensorClient(
            broker_address=args.broker,
            train_id=args.train_id,
            carriage_number=args.carriage_number,
            port=args.port
        )
        # ---------------------------------------------------------------------------------------------------------------#
        # Connect to the MQTT broker
        mqtt_client.connect()
        # ---------------------------------------------------------------------------------------------------------------#

    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print(
        f"Starting live detection loop (Interval: {args.interval}s). Press Ctrl+C to stop.")

    while running:
        start_time = time.time()

        # 1. Capture Image
        frame = camera.capture() # Capture the image from the camera
        if frame is None:
            print("Failed to capture image. Retrying...")
            time.sleep(1)
            continue

        # 2. Detect People
        count, boxes, annotated = detector.detect(frame)
        print(f"[{time.strftime('%H:%M:%S')}] People detected: {count}")

        # 3. Save Annotated Image (Optional: could save only on detection or every N frames)
        # For now, saving every frame as per original logic, but maybe we should limit it?
        # Save the annotated image
        file_manager.save_image(annotated, prefix="live_annotated")

        # 4. Get ToF Reading
        tof_count = get_tof_reading()
        print(f"ToF Sensor reading: {tof_count}")

        # 5. Send Update to Server
        mqtt_client.send_update(
            train_id=int(args.train_id),
            carriage_number=args.carriage_number,
            tof_number=tof_count,
            camera_number=count
        )  # Send the update to the server

        # Calculate processing time (before sleep)
        processing_time = time.time() - start_time
        print(f"Cycle processing time: {processing_time:.4f} seconds")

        # Calculate sleep time to maintain interval
        elapsed = time.time() - start_time  # Calculate the elapsed time
        # Calculate the sleep time
        sleep_time = max(0, args.interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)  # Sleep for the sleep time

        if args.single_shot:
            print("Single shot mode enabled. Exiting after one cycle.")
            break

    # Cleanup
    camera.cleanup()
    mqtt_client.disconnect()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
