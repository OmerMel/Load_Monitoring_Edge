import sys
import os
import argparse
import time
from datetime import datetime
import signal

# Add the project root to the python path so we can import from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.file_utils import FileManager
from src.hal import UsbCamera, RpiCamera, TOFSensor
from src.comms.mqtt_client import MqttSensorClient
from src.processing.image_processor import ImageProcessor
from src.services.load_monitor_service import LoadMonitorService
from src.sources import FolderImageSource

# --------------------------------------------Configuration---------------------------------------------------#

INTERVAL_SECONDS = 20  # 120 seconds between cycles

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TRAIN_ID = 1
CARRIAGE_NUMBER = 1

# Processing Configuration
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.30 # Confidence threshold for the detections
IOU_THRESHOLD = 0.45 # Prevents duplicate boxes around the same person
IMAGE_SIZE = 1280 # The size of the image to be processed by the model (Try to change to 1920)
MIN_BOX_AREA = 500 # Minimum box area to consider valid area of the box
USE_CLAHE = True # Improve the contrast of the image using CLAHE (if the camera suffers from low light)
OUTPUT_DIR = "outputs" # The directory to save the annotated images
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images") # The directory to save the images

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down gracefully... Finishing current cycle.")
    running = False

# ---------------------------------------------------------------------------------------------------------------#
# Function to parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Start the Raspberry Pi edge load monitoring system."
    )
    parser.add_argument(
        "--mode",
        choices=("live", "images"),
        default="live",
        help="Run with a live camera or with images from the project images folder.",
    )
    parser.add_argument(
        "--camera",
        choices=("rpi", "usb"),
        default="rpi",
        help="Camera type to use in live mode.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------------------------------------------#
# Function to build the image source (live camera or images from the project images folder)
# Args: Mode (live or images), camera type (usb or rpi), images directory
# Returns: ImageSource object
def _build_image_source(mode: str, camera_type: str, images_dir: str):
    if mode == "images":
        return FolderImageSource(images_dir)

    if camera_type == "usb":
        return UsbCamera(camera_index=0, width=1920, height=1080)

    return RpiCamera(width=1920, height=1080)


# ---------------------------------------------------------------------------------------------------------------#
# Function to display a countdown timer on the same line in the terminal
# Args: Number of seconds to countdown
def run_countdown(seconds):
    try:
        # Initial newline to separate from previous output
        print()
        for remaining in range(seconds, 0, -1):
            if not running:
                break
            sys.stdout.write(f"\rNext execution in: {remaining}s...   ")
            sys.stdout.flush()
            time.sleep(1)
        if running:
            sys.stdout.write("\rExecuting now! \n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nCountdown interrupted.")
        raise

# ---------------------------------------------------------------------------------------------------------------#
# Function to main function to run the monitor runner
def main():
    global running
    args = parse_args()

    # Print the starting message
    print("Starting Monitor Runner.")
    print(f"Mode: {args.mode}")
    if args.mode == "live":
        print(f"Camera: {args.camera}")
    print(f"Interval: {INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop.")
    print("-" * 50)

    # --------------------------------------------- Initialize components -----------------------------------------------#
    print("Initializing components...")
    
    image_source = None
    mqtt_client = None
    try:
        # Initialize the selected image source (live camera or images from the project images folder)
        image_source = _build_image_source(args.mode, args.camera, IMAGES_DIR)

        # Initialize the ToF sensor(s)
        sensors = [
            TOFSensor(sensor_id="tof_1"),
            # TOFSensor(sensor_id="tof_2"),
            # TOFSensor(sensor_id="tof_3"),
            # TOFSensor(sensor_id="tof_4")
        ]

        # Initialize the image processor
        processor = ImageProcessor(
            model_path=MODEL_PATH,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            min_box_area=MIN_BOX_AREA,
            use_clahe=USE_CLAHE,
        )

        # Initialize the file manager
        file_manager = FileManager(output_dir=OUTPUT_DIR)

        # Initialize the MQTT client
        mqtt_client = MqttSensorClient(
            broker_address=MQTT_BROKER,
            train_id=str(TRAIN_ID),
            carriage_number=CARRIAGE_NUMBER,
            port=MQTT_PORT,
        )
        
        # Connect to the MQTT broker
        mqtt_client.connect()

        # Initialize the Load Monitor Service (Orchestrator)
        load_monitor_service = LoadMonitorService(
            camera=image_source,
            sensors=sensors,
            processor=processor,
            comms=mqtt_client,
            train_id=TRAIN_ID,
            carriage_number=CARRIAGE_NUMBER,
        )

    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("Starting processing loop. Press Ctrl+C to stop.")

    # Set up signal handling for graceful shutdown
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while running:
            # 1. Log start time
            start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{start_time_str}] Starting processing cycle...")
            
            start_time = time.time()

            # 2. Execute a full monitoring cycle via the LoadMonitorService
            result = load_monitor_service.run_cycle()
            
            if result:
                print(f"[{time.strftime('%H:%M:%S')}] People detected: {result['person_count']}")
                print(f"Sensor data sent: {result['sensor_data']}")

                # Save the annotated image
                annotated_frame = processor.draw_annotations(result['frame'], result['detections'], result['person_count'])
                
                source_id = result['frame'].source_id
                if args.mode == "images" and source_id.startswith("file:"):
                    # Extract 'load_car_13' from 'file:load_car_13.jpg'
                    original_name = source_id.replace("file:", "").rsplit(".", 1)[0]
                    prefix = f"images_result_{original_name}"
                    file_manager.save_image(annotated_frame, prefix=prefix, timestamp=False)
                else:
                    file_manager.save_image(annotated_frame, prefix="live", timestamp=True)
            elif args.mode == "images" and getattr(image_source, "exhausted", False):
                print("\nAll images processed.")
                break

            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"Cycle processing time: {processing_time:.4f} seconds")

            # 3. If in images mode, we process all available images without waiting for the interval
            if args.mode == "images":
                # Add a tiny sleep to not completely lock up the CPU if loading images instantly
                time.sleep(0.1)
                continue

            # 4. If in live mode, Countdown to next execution
            if running:
                run_countdown(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nStopping Monitor Runner. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint)
        
        print("\nCleaning up resources...")
        if image_source is not None:
            image_source.cleanup()
        if mqtt_client is not None:
            mqtt_client.disconnect()
        print("Exited cleanly.")


if __name__ == "__main__":
    main()
