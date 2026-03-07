from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ultralytics import YOLO

# ---------------------------------------------------------------------------------------------------------------#
# Represents a single detected bounding box for a person


@dataclass
class DetectionBox:

    # Coordinates are in pixel space:

    # (x1, y1) -> top-left corner
    x1: int
    y1: int
    # (x2, y2) -> bottom-right corner
    x2: int
    y2: int
    conf: float  # Detection confidence score
# ---------------------------------------------------------------------------------------------------------------#
# PeopleDetector class


class PeopleDetector:

    # ---------------------------------------------------------------------------------------------------------------#
    # Constructor - Initialize the YOLO model and detection parameters

    def __init__(
        self,
        model_path: str = "yolov8n.pt",  # YOLO model file
        imgsz: int = 1280,  # Inference image size
        conf: float = 0.25,  # Confidence threshold for detections
        iou: float = 0.45,   # Prevents duplicate boxes around the same person
        min_box_area: int = 500,  # Minimum box area to consider valid
        use_clahe: bool = False   # Enable CLAHE preprocessing for better contrast
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.min_box_area = min_box_area
        self.use_clahe = use_clahe

        # Load YOLO model from disk
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)  # Load the YOLO model from disk

        # Ensure that class index 0 corresponds to "person"
        name0 = self.model.names.get(0, "")
        if str(name0).lower() != "person":
            raise ValueError(
                f"Model '{model_path}' is not COCO-person compatible. "
                f"Expected class 0 == 'person', got: {name0!r}"
            )

    # ---------------------------------------------------------------------------------------------------------------#
    # Detect people in a single image
    # Args: frame_bgr: Input image in BGR format (OpenCV default).
    # Returns: Number of detected people, List of DetectionBox objects, Image with bounding boxes and count overlay (BGR).
    def detect(self, frame_bgr: np.ndarray) -> Tuple[int, List[DetectionBox], np.ndarray]:

        # Check if the input image is None
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        # -------- Improve the contrast of the image using CLAHE --------#
        # Preprocessing - Apply CLAHE if enabled (improves the contrast of the image)
        inference_frame = frame_bgr
        if self.use_clahe:
            # Convert the image to LAB color space
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L-channel (L-channel is the lightness channel)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # Merge channels and convert back to BGR
            limg = cv2.merge((cl, a, b))
            inference_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Run YOLO inference directly on the numpy array
        results = self.model(
            inference_frame, # The image to run
            imgsz=self.imgsz, # The image size
            conf=self.conf, # The confidence threshold
            iou=self.iou, # The IoU threshold
            verbose=False, # Whether to print verbose output
        )

        boxes_out: List[DetectionBox] = []
        annotated = frame_bgr.copy()
        person_count = 0

        # Iterate over YOLO results and extract person detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # Skip non-person detections (class 0 is usually person in COCO)
                if cls != 0:
                    continue

                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calculate box area
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_box_area:
                    continue  # Skip small detections (likely noise)

                person_count += 1

                # Extract confidence score
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                boxes_out.append(DetectionBox(x1, y1, x2, y2, conf))

                # Draw bounding box around detected person
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw label with confidence score
                label = f"Person {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # Draw label background for better visibility
                cv2.rectangle(annotated, (x1, y1 - 20),
                              (x1 + label_w, y1), (0, 255, 0), -1)

                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),  # Black text on green background
                    2,
                )

        # Draw total people count in the top-right corner
        self._draw_count_overlay(annotated, person_count)

        return person_count, boxes_out, annotated

    # ---------------------------------------------------------------------------------------------------------------#
    # Helper method to draw the detection count on the image
    def _draw_count_overlay(self, image: np.ndarray, count: int):
        text = f"People: {count}"  # The text to draw on the image
        h, w = image.shape[:2]  # Get the height and width of the image

        (text_w, text_h), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )

        x_text = max(10, w - text_w - 20)
        y_text = 40

        # Draw a semi-transparent background for the text
        overlay = image.copy()
        cv2.rectangle(overlay, (x_text - 10, y_text - text_h - 10),
                      (w - 10, y_text + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        cv2.putText(
            image,
            text,
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
