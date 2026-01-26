from ultralytics import YOLO
# import cv2
import uuid
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

GUN_MODEL = YOLO(os.path.join(MODEL_DIR, "gun_bestv2.pt"))
KNIFE_MODEL = YOLO(os.path.join(MODEL_DIR, "knife_best.pt"))

def draw_boxes(img, results, label_name, color):
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            text = f"{label_name} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
    return img

def run_detection(image_path):
    gun_results = GUN_MODEL.predict(source=image_path, conf=0.3, save=False)
    knife_results = KNIFE_MODEL.predict(source=image_path, conf=0.25, save=False)

    img = cv2.imread(image_path)

    # Draw guns as "gun" (or change to "firearm" if you want)
    img = draw_boxes(img, gun_results, label_name="gun", color=(255, 0, 0))

    # Draw knives as "knife"
    img = draw_boxes(img, knife_results, label_name="knife", color=(0, 0, 255))

    output_path = f"output_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(output_path, img)

    return output_path
