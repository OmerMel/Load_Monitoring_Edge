from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

from src.entities.detection import DetectionBox
from src.entities.image_frame import ImageFrame


class ImageProcessor:

    def __init__(
        self,
        model_path: str = "yolov26n.pt",
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.45,
        min_box_area: int = 200,
        use_clahe: bool = False
    ):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.min_box_area = min_box_area
        self.use_clahe = use_clahe
        self.model_path = model_path

        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

        # Ensure that class index 0 corresponds to "person"
        name0 = self.model.names.get(0, "")
        if str(name0).lower() != "person":
            raise ValueError(
                f"Model '{model_path}' is not COCO-person compatible. "
                f"Expected class 0 == 'person', got: {name0!r}"
            )

    def detect(self, frame: ImageFrame) -> Tuple[int, List[DetectionBox]]:
        if frame is None or frame.data is None:
            raise ValueError("frame or frame.data is None")

        frame_bgr = frame.data

        inference_frame = frame_bgr
        if self.use_clahe:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            limg = cv2.merge((cl, a, b))
            inference_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        results = self.model(
            inference_frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        boxes_out: List[DetectionBox] = []
        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                area = (x2 - x1) * (y2 - y1)
                if area < self.min_box_area:
                    continue

                person_count += 1

                conf = float(box.conf[0]) if box.conf is not None else 0.0

                boxes_out.append(DetectionBox(x1, y1, x2, y2, conf))

        return person_count, boxes_out

    def draw_annotations(self, frame: ImageFrame, boxes: List[DetectionBox], count: int) -> np.ndarray:
        annotated = frame.data.copy()

        for box in boxes:
            cv2.rectangle(
                annotated,
                (box.x1, box.y1),
                (box.x2, box.y2),
                (0, 255, 0),
                1
            )

            label = f"{box.conf:.2f}"
            font_scale = 0.4
            thickness = 1
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            y_bg_top = box.y1 - label_h - 4
            y_bg_bottom = box.y1
            y_text = box.y1 - 2

            if y_bg_top < 0:
                y_bg_top = box.y1
                y_bg_bottom = box.y1 + label_h + 4
                y_text = box.y1 + label_h + 2

            cv2.rectangle(annotated, (box.x1, y_bg_top),
                          (box.x1 + label_w + 4, y_bg_bottom), (0, 255, 0), -1)

            cv2.putText(
                annotated,
                label,
                (box.x1 + 2, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        self._draw_count_overlay(annotated, count)

        return annotated

    def _draw_count_overlay(self, image: np.ndarray, count: int):
        text = f"People: {count}"
        h, w = image.shape[:2]

        (text_w, text_h), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )

        x_text = max(10, w - text_w - 20)
        y_text = 40

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