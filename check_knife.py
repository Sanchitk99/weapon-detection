from ultralytics import YOLO

MODEL = r"C:\Users\Sanchit\PycharmProjects\ChatBot\runs\detect\train4\weights\best.pt"
SOURCE = r"D:\Project\weapon-detection\knife_dataset\test\images"

model = YOLO(MODEL)

results = model.predict(
    source=SOURCE,
    conf=0.25,
    save=True
)

print("Knife model prediction completed")
