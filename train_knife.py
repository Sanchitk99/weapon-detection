from ultralytics import YOLO

MODEL_PATH = r"C:\Users\Sanchit\PycharmProjects\ChatBot\runs\detect\train4\weights\last.pt"

model = YOLO(MODEL_PATH)

model.train(resume=True)
