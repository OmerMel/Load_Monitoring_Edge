import sys
import os
import argparse
import threading
import board
import busio
import time
from datetime import datetime
import signal
import subprocess

# Add the project root to the python path so we can import from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.file_utils import FileManager
from src.hal import UsbCamera, RpiCamera, TofPair, TofUnit
from src.comms.mqtt_client import MqttSensorClient
from src.processing.image_processor import ImageProcessor
from src.processing.carriage_counter import CarriageCounter
from src.services.load_monitor_service import LoadMonitorService
from src.sources import FolderImageSource

# --------------------------------------------Configuration---------------------------------------------------#

INTERVAL_SECONDS = 30  # seconds between cycles

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TRAIN_ID = 1
CARRIAGE_NUMBER = 1

# Processing Configuration
MODEL_PATH = "yolo26s.pt"
CONFIDENCE_THRESHOLD = 0.25 # Confidence threshold for the detections
IOU_THRESHOLD = 0.45 # Prevents duplicate boxes around the same person
IMAGE_SIZE = 1280 # The size of the image to be processed by the model (Only 640 for ncnn format)
MIN_BOX_AREA = 200 # Minimum box area to consider valid area of the box
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
    parser.add_argument(
        "--no-sensors",
        action="store_true",
        help="Run the script without initializing ToF sensors.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------------------------------------------#
# Function to build the image source (live camera or images from the project images folder)
def _build_image_source(mode: str, camera_type: str, images_dir: str):
    if mode == "images":
        return FolderImageSource(images_dir)

    if camera_type == "usb":
        return UsbCamera(camera_index=0, width=1920, height=1080)

    return RpiCamera(width=1920, height=1080)


# ---------------------------------------------------------------------------------------------------------------#
# Function to display a countdown timer on the same line in the terminal
def run_countdown(seconds):
    try:
        print() # Initial newline to separate from previous output
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
# Function to initialize ToF sensors safely and assign them the shared counter
def setup_tof_sensors(i2c_bus, shared_counter: CarriageCounter) -> TofPair:
    print("[Boot Sequence] Setting up ToF I2C Addresses...")
    
    # 1. יצירת האובייקטים לפי GPIO
    tof_outside = TofUnit(xshut_pin=27, target_i2c_address=0x30, threshold_mm=1000, i2c_bus=i2c_bus)
    tof_inside = TofUnit(xshut_pin=17, target_i2c_address=0x32, threshold_mm=1000, i2c_bus=i2c_bus)

    # 2. כיבוי טוטאלי ומניעת התנגשויות
    print("ToF 0x30 turning off")
    tof_outside.turn_off()
    print("ToF 0x31 turning off")
    tof_inside.turn_off()
    time.sleep(0.5)

    # 4. הדלקה והגדרת חיישן פנימי
    print("Initializing Inside (0x31) Sensor...")
    tof_inside.turn_on()
    time.sleep(0.5)
    tof_inside.setup_sensor() 
    
    # 3. הדלקה והגדרת חיישן חיצוני
    print("Initializing Outside (0x30) Sensor...")
    tof_outside.turn_on()
    time.sleep(0.5)
    tof_outside.setup_sensor() 
    


    print("[Boot Sequence] ToF sensors ready!")
    
    # מחזירים את הזוג ומעבירים לו את המונה המשותף של הקרון!
    return TofPair(
        outside_unit=tof_outside, 
        inside_unit=tof_inside, 
        sensor_id="door_1_tof_pair",
        shared_counter=shared_counter
    )

# ---------------------------------------------------------------------------------------------------------------#
# Function to initialize all system components neatly
def initialize_system_components(args):
    print("Initializing components...")
    
    # 1. Camera / Image Source
    image_source = _build_image_source(args.mode, args.camera, IMAGES_DIR)

    # 2. Carriage Counter (The Single Source of Truth for passenger count)
    carriage_counter = CarriageCounter(sensor_id=f"carriage_{CARRIAGE_NUMBER}_total")

# 3. ToF Sensors Hardware Setup (With bypass and error handling)
    if not args.no_sensors:
        try:
            i2c_bus = busio.I2C(board.SCL, board.SDA)
            door1_sensor = setup_tof_sensors(i2c_bus, shared_counter=carriage_counter)

            # 4. Start ToF Background Thread
            tof_thread = threading.Thread(target=door1_sensor.start_polling, daemon=True)
            tof_thread.start()
            print("[Main] ToF background polling thread started successfully.")
        except Exception as e:
            print(f"\n[Warning] Could not initialize ToF sensors: {e}")
            print("[Warning] Continuing with camera only...\n")
    else:
        print("[Main] Skipping ToF sensors setup (--no-sensors flag active).")
    # 5. The orchestrator only needs to listen to the central carriage counter!
    sensors = [carriage_counter]

    # 6. Computer Vision Processor
    processor = ImageProcessor(
        model_path=MODEL_PATH,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        min_box_area=MIN_BOX_AREA,
        use_clahe=USE_CLAHE,
    )

    # 7. File Manager & MQTT
    file_manager = FileManager(output_dir=OUTPUT_DIR)
    
    mqtt_client = MqttSensorClient(
        broker_address=MQTT_BROKER,
        train_id=str(TRAIN_ID),
        carriage_number=CARRIAGE_NUMBER,
        port=MQTT_PORT,
    )
    mqtt_client.connect()

    # 8. Main Orchestrator Service
    load_monitor_service = LoadMonitorService(
        camera=image_source,
        sensors=sensors,
        processor=processor,
        comms=mqtt_client,
        train_id=TRAIN_ID,
        carriage_number=CARRIAGE_NUMBER,
    )

    return image_source, mqtt_client, file_manager, processor, load_monitor_service


# ---------------------------------------------------------------------------------------------------------------#
# Main Application Loop
def main():
    global running
    args = parse_args()

    print("Starting Monitor Runner.")
    print(f"Mode: {args.mode}")
    if args.mode == "live":
        print(f"Camera: {args.camera}")
    print(f"Interval: {INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop.")
    print("-" * 50)

    image_source = None
    mqtt_client = None

    try:
        # Initialize everything cleanly using our helper function
        image_source, mqtt_client, file_manager, processor, load_monitor_service = initialize_system_components(args)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Set up signal handling for graceful shutdown
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting processing loop. Press Ctrl+C to stop.")

    try:
        while running:
            # start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # print(f"\n[{start_time_str}] Starting processing cycle...")
            
            start_time = time.time()

            # Execute a full monitoring cycle
            result = load_monitor_service.run_cycle()
            
            if result:
                camera_val = result['person_count']
                tof_val = int(result['sensor_data'].ir_count)

                print("\n" + "." * 50)
                print("|" + " SUMMARY REPORT ".center(48) + "|")
                print("." * 50)
                print("|" + f" Camera Counter (YOLO) : {camera_val}".ljust(48) + "|")
                print("|" + f" Sensors Counter (ToF) : {tof_val}".ljust(48) + "|")
                print("." * 50 + "\n")

                # Save the annotated image
                annotated_frame = processor.draw_annotations(result['frame'], result['detections'], result['person_count'])
                source_id = result['frame'].source_id
                
                if args.mode == "images" and source_id.startswith("file:"):
                    original_name = source_id.replace("file:", "").rsplit(".", 1)[0]
                    prefix = f"images_result_{original_name}"
                    file_manager.save_image(annotated_frame, prefix=prefix, timestamp=False)
                else:
                    file_manager.save_image(annotated_frame, prefix="live", timestamp=True)
                    
            elif args.mode == "images" and getattr(image_source, "exhausted", False):
                print("\nAll images processed.")
                break

            processing_time = time.time() - start_time
            # print(f"Cycle processing time: {processing_time:.4f} seconds")

            # Handle delays based on mode
            if args.mode == "images":
                time.sleep(0.1)
                continue

            if running:
                run_countdown(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nStopping Monitor Runner. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
    finally:
        # Restore original signal handler and cleanup
        signal.signal(signal.SIGINT, original_sigint)
        
        print("\nCleaning up resources...")
        if image_source is not None:
            image_source.cleanup()
        if mqtt_client is not None:
            mqtt_client.disconnect()
        print("Exited cleanly.")


if __name__ == "__main__":
    main()