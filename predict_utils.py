from ultralytics import YOLO
import cv2
import uuid
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

GUN_MODEL = YOLO(os.path.join(MODEL_DIR, "gun_best.pt"))
KNIFE_MODEL = YOLO(os.path.join(MODEL_DIR, "knife_best.pt"))

def run_detection(image_path):
    # Run both models
    gun_results = GUN_MODEL.predict(source=image_path, conf=0.3, save=False)
    knife_results = KNIFE_MODEL.predict(source=image_path, conf=0.25, save=False)

    img = cv2.imread(image_path)

    # Draw gun detections
    for r in gun_results:
        img = r.plot(img=img)

    # Draw knife detections
    for r in knife_results:
        img = r.plot(img=img)

    output_path = f"output_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_path, img)

    return output_path
