from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import uuid
import os

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

GUN_MODEL = YOLO(os.path.join(MODEL_DIR, "gun_bestv2.pt"))
KNIFE_MODEL = YOLO(os.path.join(MODEL_DIR, "knife_best.pt"))

# ------------------ Draw Function ------------------
def draw_boxes(img, results, label_name, color):
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

            label = f"{label_name} {conf:.2f}"

            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Text background
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

            draw.rectangle(
                [x1, y1 - th - 6, x1 + tw + 6, y1],
                fill=color
            )

            # Text
            draw.text((x1 + 3, y1 - th - 3), label, fill="white", font=font)

    return img

# ------------------ Detection Runner ------------------
def run_detection(image_path):
    gun_results = GUN_MODEL.predict(source=image_path, conf=0.3, save=False)
    knife_results = KNIFE_MODEL.predict(source=image_path, conf=0.25, save=False)

    img = Image.open(image_path).convert("RGB")

    img = draw_boxes(img, gun_results, "gun", (0, 0, 255))
    img = draw_boxes(img, knife_results, "knife", (255, 0, 0))

    output_path = f"output_{uuid.uuid4().hex}.jpg"
    img.save(output_path)

    return output_path
