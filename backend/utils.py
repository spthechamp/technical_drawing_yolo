from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64

model_path = r'C:\PROJECTS\technical_drawing_yolo\yolov8_weights\best.pt'

model = YOLO(model=model_path, task='detect')  # Ensure correct path

def read_image_from_bytes(image_bytes: bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise ValueError("Invalid image format") from e

def run_inference(image_np):
    try:
        results = model(image_np)[0]
        annotated_img = results.plot()  # Annotated image as numpy array

        detection_info = []
        for box in results.boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detection_info.append({
                "class_id": cls_id,
                "bbox": coords,
                "confidence": confidence
            })

        _, buffer = cv2.imencode(".jpg", annotated_img)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return encoded_image, detection_info

    except Exception as e:
        raise RuntimeError("Error during model inference") from e
