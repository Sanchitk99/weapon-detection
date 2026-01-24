from ultralytics import YOLO

# Load the last saved checkpoint
model = YOLO(r"C:\Users\Sanchit\PycharmProjects\ChatBot\runs\detect\train\weights\last.pt")

# Resume training
model.train(resume=True)
