from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import uuid
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

GUN_MODEL = YOLO(os.path.join(MODEL_DIR, "gun_bestv2.pt"))
KNIFE_MODEL = YOLO(os.path.join(MODEL_DIR, "knife_best.pt"))


def draw_boxes(img, results, label, color):
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            text = f"{label} {conf:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(0, y1 - 20)), text, fill=color, font=font)

    return img


def count_boxes(results):
    total = 0
    for r in results:
        if r.boxes is None:
            continue
        total += len(r.boxes)
    return total


def run_detection(image_path, conf=0.3, gun_conf=None, knife_conf=None):
    if gun_conf is None:
        gun_conf = conf
    if knife_conf is None:
        knife_conf = conf

    gun_results = GUN_MODEL.predict(image_path, conf=gun_conf, save=False)
    knife_results = KNIFE_MODEL.predict(image_path, conf=knife_conf, save=False)

    img = Image.open(image_path).convert("RGB")

    img = draw_boxes(img, gun_results, "gun", (0, 0, 255))
    img = draw_boxes(img, knife_results, "knife", (255, 0, 0))

    output_path = f"output_{uuid.uuid4().hex}.jpg"
    img.save(output_path)

    counts = {
        "gun": count_boxes(gun_results),
        "knife": count_boxes(knife_results),
    }

    return output_path, counts
