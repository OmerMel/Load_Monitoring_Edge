import sys
import os
import cv2
import time
import argparse

# ------------- This allows us to import from the src directory -------------#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.people_detector import PeopleDetector
from src.comms.mqtt_client import MqttSensorClient
from src.utils.file_utils import FileManager

def main():
   
   #------------- Parse arguments -------------#
    parser = argparse.ArgumentParser(description="Run people detection on a local image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--broker", type=str, default="test.mosquitto.org", help="MQTT Broker address.")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker port.")
    parser.add_argument("--train-id", type=str, default="1", help="Train ID.")
    parser.add_argument("--carriage-number", type=int, default=1, help="Carriage Number.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (e.g., 640, 1280).")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing for better contrast.")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum box area to consider valid.")
    
    args = parser.parse_args()

    #------------- Initialize components -------------#
    file_manager = FileManager(output_dir="outputs") # Defined the output directory
    people_detector = PeopleDetector(
        model_path=args.model, # Path to the YOLO model file
        conf=args.conf, # Confidence threshold for detection
        iou=args.iou, # Prevents duplicate boxes around the same person
        imgsz=args.imgsz, # Inference image size
        min_box_area=args.min_area, # Minimum box area to consider valid
        use_clahe=args.clahe # Enable CLAHE preprocessing for better contrast
    ) 
    mqtt_client = MqttSensorClient(
        broker_address=args.broker,
        train_id=args.train_id,
        carriage_number=args.carriage_number,
        port=args.port
    )
    mqtt_client.connect()

    #------------- Load image -------------#
    if not os.path.exists(args.image_path): # Check if the image file exists
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)

    frame = cv2.imread(args.image_path) # Read the image file
    if frame is None:
        print(f"Error: Failed to load image from {args.image_path}")
        sys.exit(1)

    print(f"Processing image: {args.image_path}")
    
    # Start timer
    start_time = time.time()

    #------------- Run detection -------------#
    count, boxes, annotated = people_detector.detect(frame)
    print(f"People detected: {count}")

    #------------- Save output -------------#
    output_path = file_manager.save_image(annotated, prefix="annotated_local")
    if output_path:
        print(f"Saved annotated image to: {output_path}")

    #------------- Send to server -------------#
    print("Sending results to MQTT broker...")
    mqtt_client.send_update(
        train_id=int(args.train_id),
        carriage_number=args.carriage_number,
        tof_number=0,
        camera_number=count
    )
    mqtt_client.disconnect()
    
    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total processing time (Detection -> Server): {duration:.4f} seconds")

if __name__ == "__main__":
    main()
