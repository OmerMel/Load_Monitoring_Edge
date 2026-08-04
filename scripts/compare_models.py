import os
import sys
import time
import json
import csv
import cv2
import torch
import numpy as np
from datetime import datetime

# Add the project root to the python path so we can import from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.processing.image_processor import ImageProcessor
from src.entities.image_frame import ImageFrame

# --------------------------------------------Configuration---------------------------------------------------#

# List of models to compare (Model Name: Model File Path)
MODELS_TO_TEST = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO26": "yolo26s.pt" 
}

IMAGE_SIZE = 1280
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MIN_BOX_AREA = 200
USE_CLAHE = True

IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_FILE = os.path.join(OUTPUT_BASE_DIR, "test_log.txt")
CSV_RESULTS_FILE = os.path.join(OUTPUT_BASE_DIR, "model_comparison_results.csv")
JSON_RESULTS_FILE = os.path.join(OUTPUT_BASE_DIR, "model_comparison_results.json")

# Ensure output directory exists
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------#

def log_message(message: str, to_console: bool = True, to_file: bool = True):
    """Log a message to the console and/or the log file."""
    if to_console:
        print(message)
    if to_file:
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")

def check_device() -> str:
    """Check if CUDA is available and return the device string."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        log_message(f"[System] CUDA is available! Using GPU: {device_name}")
        return "GPU"
    else:
        log_message("[System] CUDA not available. Using CPU.")
        return "CPU"

def get_image_files() -> list:
    """Retrieve all valid image files from the images directory."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    if not os.path.exists(IMAGES_DIR):
        log_message(f"[Error] Images directory not found: {IMAGES_DIR}")
        return []
    
    files = []
    for file in os.listdir(IMAGES_DIR):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            files.append(file)
            
    files.sort()
    return files

def run_tests():
    device = check_device()
    image_files = get_image_files()
    
    if not image_files:
        log_message("[Warning] No images found to process. Exiting test.")
        return

    log_message(f"Found {len(image_files)} images for testing.")
    log_message("-" * 60)

    # Dictionary to hold the overall results
    # Format: { image_filename: { model_name: { "count": int, "time": float, "output_path": str } } }
    results_by_image = {img: {} for img in image_files}
    
    # Dictionary to hold aggregate statistics per model
    model_stats = {
        model_name: {
            "processed_images": 0,
            "total_time": 0.0,
            "min_time": float("inf"),
            "max_time": 0.0,
            "total_persons": 0
        }
        for model_name in MODELS_TO_TEST.keys()
    }

    # Process images model by model
    for model_name, model_path in MODELS_TO_TEST.items():
        log_message(f"\n[Test] Initializing model: {model_name} ({model_path})")
        
        # Create output directory for this model
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, f"T-{model_name.lower()}")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Load the model using existing ImageProcessor
        try:
            processor = ImageProcessor(
                model_path=model_path,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                min_box_area=MIN_BOX_AREA,
                use_clahe=USE_CLAHE
            )
        except Exception as e:
            log_message(f"[Error] Failed to load model {model_name}: {e}")
            continue

        # Warm-up inference
        log_message(f"[Test] Performing warm-up inference for {model_name}...")
        warmup_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        warmup_frame = ImageFrame(data=warmup_img, timestamp=datetime.now(), source_id="warmup")
        try:
            processor.detect(warmup_frame)
        except Exception as e:
            log_message(f"[Warning] Warm-up failed for {model_name}: {e}")

        # Process all images
        for img_filename in image_files:
            img_path = os.path.join(IMAGES_DIR, img_filename)
            frame_data = cv2.imread(img_path)
            
            if frame_data is None:
                log_message(f"[Error] Failed to read image: {img_path}. Skipping.")
                continue
                
            frame = ImageFrame(data=frame_data, timestamp=datetime.now(), source_id=img_filename)
            output_path = os.path.join(model_output_dir, img_filename)
            
            try:
                # Measure processing time (inference + processing)
                start_time = time.time()
                person_count, boxes = processor.detect(frame)
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Annotate and save the image
                annotated_frame = processor.draw_annotations(frame, boxes, person_count)
                cv2.imwrite(output_path, annotated_frame)
                
                # Log this specific execution
                log_message(f"Image: {img_filename}")
                log_message(f"Model: {model_name}")
                log_message(f"Detected persons: {person_count}")
                log_message(f"Processing time: {processing_time:.4f} seconds")
                rel_output_path = os.path.relpath(output_path, PROJECT_ROOT)
                log_message(f"Output: {rel_output_path}\n")
                
                # Save results
                results_by_image[img_filename][model_name] = {
                    "count": person_count,
                    "time": processing_time,
                    "output_path": output_path
                }
                
                # Update model stats
                stats = model_stats[model_name]
                stats["processed_images"] += 1
                stats["total_time"] += processing_time
                stats["min_time"] = min(stats["min_time"], processing_time)
                stats["max_time"] = max(stats["max_time"], processing_time)
                stats["total_persons"] += person_count

            except Exception as e:
                log_message(f"[Error] Failed to process {img_filename} with {model_name}: {e}")
                
    log_message("-" * 60)
    log_message("[Test] All models finished processing.")

    # 1. Print comparison line per image
    log_message("\n--- Comparison per image ---")
    for img_filename in image_files:
        comparisons = []
        for model_name in MODELS_TO_TEST.keys():
            if model_name in results_by_image[img_filename]:
                count = results_by_image[img_filename][model_name]["count"]
                comparisons.append(f"{model_name}: {count} persons")
            else:
                comparisons.append(f"{model_name}: Failed")
        
        comp_str = " | ".join(comparisons)
        log_message(f"{img_filename} | {comp_str}")

    # 2. Save raw JSON results
    json_data = {
        "device": device,
        "date": datetime.now().isoformat(),
        "config": {
            "image_size": IMAGE_SIZE,
            "confidence": CONFIDENCE_THRESHOLD,
            "iou": IOU_THRESHOLD,
            "min_box_area": MIN_BOX_AREA,
            "use_clahe": USE_CLAHE
        },
        "stats": model_stats,
        "results": results_by_image
    }
    with open(JSON_RESULTS_FILE, "w") as f:
        json.dump(json_data, f, indent=4)
    log_message(f"\n[System] Raw JSON results saved to: {JSON_RESULTS_FILE}")

    # 3. Generate CSV results table
    generate_csv_results(image_files, results_by_image, model_stats)
    log_message(f"[System] CSV results table saved to: {CSV_RESULTS_FILE}")


def generate_csv_results(image_files, results_by_image, model_stats):
    """Generate the CSV file with formulas for deviation and summary statistics."""
    
    # Define columns
    header = ["Image Name", "Actual Person Count"]
    model_names = list(MODELS_TO_TEST.keys())
    
    for model_name in model_names:
        header.append(f"{model_name} Detected")
        header.append(f"{model_name} Deviation")

    with open(CSV_RESULTS_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Start at row 2 (header is row 1)
        current_row = 2
        
        for img in image_files:
            row = [img, ""] # Image name, Actual count (empty for manual input)
            
            for i, model_name in enumerate(model_names):
                if model_name in results_by_image[img]:
                    count = results_by_image[img][model_name]["count"]
                    row.append(count)
                    
                    # Columns: B is Actual Count.
                    # Detected counts are at: 
                    # YOLOv8n (i=0): col C, deviation D
                    # YOLO11n (i=1): col E, deviation F
                    # YOLO26 (i=2): col G, deviation H
                    actual_col = "B"
                    detected_col = chr(ord('C') + i * 2) 
                    
                    # Deviation formula: =IF(ISBLANK(B2), "", B2-C2)
                    formula = f'=IF(ISBLANK({actual_col}{current_row}), "", {actual_col}{current_row}-{detected_col}{current_row})'
                    row.append(formula)
                else:
                    row.extend(["N/A", "N/A"])
            
            writer.writerow(row)
            current_row += 1
            
        last_data_row = current_row - 1
        
        # Blank row
        writer.writerow([])
        
        # Summary Section
        writer.writerow(["--- SUMMARY ---"])
        writer.writerow(["Metric", "Formula/Value"] + [f"{m} Value" for m in model_names])
        
        # Average processing time
        avg_time_row = ["Avg Processing Time (s)", ""]
        for m in model_names:
            st = model_stats.get(m, {})
            processed = st.get("processed_images", 0)
            avg = st.get("total_time", 0) / processed if processed > 0 else 0
            avg_time_row.append(f"{avg:.4f}")
        writer.writerow(avg_time_row)
        
        # Total Detected Persons
        tot_persons_row = ["Total Detected Persons", ""]
        for i, m in enumerate(model_names):
            col = chr(ord('C') + i * 2)
            tot_persons_row.append(f'=SUM({col}2:{col}{last_data_row})')
        writer.writerow(tot_persons_row)
        
        # Mean Absolute Error (MAE)
        mae_row = ["Mean Absolute Error", ""]
        for i, m in enumerate(model_names):
            dev_col = chr(ord('D') + i * 2)
            actual_col = "B"
            # formula computes average of absolute deviations only where Actual is not blank
            formula = f'=IF(COUNT({actual_col}2:{actual_col}{last_data_row})=0, "", SUMPRODUCT(ABS({dev_col}2:{dev_col}{last_data_row}))/COUNT({actual_col}2:{actual_col}{last_data_row}))'
            mae_row.append(formula)
        writer.writerow(mae_row)
        
        # Average Absolute Percentage Error (MAPE)
        mape_row = ["Avg Abs Pct Error (MAPE)", ""]
        for i, m in enumerate(model_names):
            dev_col = chr(ord('D') + i * 2)
            actual_col = "B"
            # formula computes avg percentage error. IFERROR prevents div by 0 if Actual=0
            formula = f'=IF(COUNT({actual_col}2:{actual_col}{last_data_row})=0, "", SUMPRODUCT(IFERROR(ABS({dev_col}2:{dev_col}{last_data_row})/{actual_col}2:{actual_col}{last_data_row}, 0))/COUNT({actual_col}2:{actual_col}{last_data_row}))'
            mape_row.append(formula)
        writer.writerow(mape_row)


if __name__ == "__main__":
    # Clear old log file if it exists
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    print("Starting YOLO models comparison test flow...")
    run_tests()
    print("Tests completed successfully.")
