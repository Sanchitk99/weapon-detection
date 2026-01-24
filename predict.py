from ultralytics import YOLO

MODEL = r"D:\Project\ChatBot\runs\detect\train\weights\best.pt"
SOURCE = r"D:\Project\weapon-detection\datasets\valid\images"

model = YOLO(MODEL)

model.predict(
    source=SOURCE,
    save=True,
    conf=0.25
)

print(" Predictions saved in runs/detect/predict/")
