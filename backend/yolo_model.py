import os

if os.path.exists("yolov11.pt"):
    model = YOLO("yolov11.pt")
else:
    print("Model file does not exist!")