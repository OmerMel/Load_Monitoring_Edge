import sys
import os
import time
import signal
import argparse
import cv2

# Add the project root to the python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.people_detector import PeopleDetector
from src.comms.sensor_client import SensorApiClient
from src.utils.camera import Camera
from src.utils.file_utils import FileManager

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

def main():
    
    global running
    signal.signal(signal.SIGINT, signal_handler)

   #------------- Parse arguments -------------#
    parser = argparse.ArgumentParser(description="Run live people detection on Raspberry Pi camera.")
    parser.add_argument("--server", type=str, default="http://172.20.10.2:8080", help="Backend server URL.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model file.")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between captures (seconds).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (e.g., 640, 1280).")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing for better contrast.")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum box area to consider valid.")
    parser.add_argument("--single-shot", action="store_true", help="Run a single detection cycle and exit.")
    
    args = parser.parse_args()

    #------------- Initialize components -------------#
    print("Initializing components...")
    try:
        camera = Camera(width=1920, height=1080) # Initialize the camera
        detector = PeopleDetector(
            model_path=args.model, 
            conf=args.conf, 
            iou=args.iou,
            imgsz=args.imgsz,
            min_box_area=args.min_area,
            use_clahe=args.clahe
        ) # Initialize the people detector
        client = SensorApiClient(base_url=args.server) # Initialize the sensor API client
        file_manager = FileManager(output_dir="outputs") # Initialize the file manager
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print(f"Starting live detection loop (Interval: {args.interval}s). Press Ctrl+C to stop.")

    while running:
        start_time = time.time()

        # 1. Capture Image
        frame = camera.capture()
        if frame is None:
            print("Failed to capture image. Retrying...")
            time.sleep(1)
            continue

        # 2. Detect People
        count, boxes, annotated = detector.detect(frame)
        print(f"[{time.strftime('%H:%M:%S')}] People detected: {count}")

        # 3. Save Annotated Image (Optional: could save only on detection or every N frames)
        # For now, saving every frame as per original logic, but maybe we should limit it?
        # Let's save it.
        file_manager.save_image(annotated, prefix="live_annotated") # Save the annotated image

        # 4. Send Update to Server
        client.send_update(
            train_id=1,
            carriage_number=1,
            tof_number=0,
            camera_number=count
        ) # Send the update to the server

        # Calculate processing time (before sleep)
        processing_time = time.time() - start_time # Calculate the processing time
        print(f"Cycle processing time: {processing_time:.4f} seconds") 

        # Calculate sleep time to maintain interval
        elapsed = time.time() - start_time # Calculate the elapsed time
        sleep_time = max(0, args.interval - elapsed) # Calculate the sleep time
        if sleep_time > 0:
            time.sleep(sleep_time) # Sleep for the sleep time

        if args.single_shot:
            print("Single shot mode enabled. Exiting after one cycle.")
            break

    # Cleanup
    camera.cleanup()
    print("Exited cleanly.")

if __name__ == "__main__":
    main()
