from ultralytics import YOLO
import cv2
import uuid

# Load both models once
GUN_MODEL = YOLO(r"D:\Project\weapon-detection\models\gun_best.pt")
KNIFE_MODEL = YOLO(r"D:\Project\weapon-detection\models\knife_best.pt")

def run_detection(image_path):
    # Run both models
    gun_results = GUN_MODEL.predict(source=image_path, conf=0.4, save=False)
    knife_results = KNIFE_MODEL.predict(source=image_path, conf=0.10, save=False)

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
