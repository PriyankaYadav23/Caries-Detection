from ultralytics import YOLO
model= YOLO("runs/detect/train-3/weights/best.pt")
results = model.predict("yolo_dataset/images/train/train_250.png", save=True)