from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
from ultralytics import YOLO


@dataclass
class DetectionBox:
    """
    Represents a single detected bounding box for a person.

    Coordinates are in pixel space:
    (x1, y1) -> top-left corner
    (x2, y2) -> bottom-right corner
    """
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float  # Detection confidence score


class PeopleDetector:
    """
    YOLOv8-based people detector.

    - Uses a COCO-trained YOLO model (class 0 == 'person')
    - Loads the model once and allows reuse across multiple images
    - Designed for edge devices (CPU-only, Raspberry Pi friendly)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.5
    ):
        """
        Initialize the YOLO model and detection parameters.

        Args:
            model_path: Path to YOLO weights file
            imgsz: Input image resolution for inference
            conf: Confidence threshold for detections
            iou: IoU threshold for Non-Maximum Suppression (NMS)
        """

        # Load YOLO model from disk
        self.model = YOLO(model_path)

        # Optional optimization: fuse Conv + BatchNorm layers
        # Improves inference speed on CPU in some cases
        try:
            self.model.fuse()
        except Exception:
            # Fuse may not be supported by all backends/models
            pass

        # Safety check:
        # Ensure that class index 0 corresponds to "person"
        # (required for correct people counting)
        name0 = self.model.names.get(0, "")
        if str(name0).lower() != "person":
            raise ValueError(
                f"Model '{model_path}' is not COCO-person compatible. "
                f"Expected class 0 == 'person', got: {name0!r}"
            )

        # Store detection configuration
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def detect(self, frame_bgr) -> Tuple[int, List[DetectionBox], Optional[object]]:
        """
        Detect people in a single image.

        Args:
            frame_bgr: Input image in BGR format (OpenCV default)

        Returns:
            person_count: Number of detected people
            boxes: List of DetectionBox objects
            annotated_bgr: Image with bounding boxes and count overlay (BGR)
        """

        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        # Run YOLO inference directly on the numpy array
        results = self.model(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        boxes_out: List[DetectionBox] = []
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        person_count = 0

        # Iterate over YOLO results and extract person detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # Skip non-person detections
                if cls != 0:
                    continue

                person_count += 1

                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Extract confidence score (if available)
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
                cv2.putText(
                    annotated,
                    f"Person {conf:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # Draw total people count in the top-right corner
        text = f"People: {person_count}"
        (text_w, _), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )

        x_text = max(0, w - text_w - 10)
        y_text = 30

        cv2.putText(
            annotated,
            text,
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        return person_count, boxes_out, annotated
