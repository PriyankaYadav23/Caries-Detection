from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io 
from ultralytics import YOLO

app=FastAPI()
device=torch.device("mps" if torch.backends.mps.is_available() else("cpu"))
model=models.mobilenet_v2(weights=None)
yolo_model = YOLO("models/best.pt")
model.classifier[1]=nn.Linear(in_features=1280, out_features=4)
model.load_state_dict(torch.load("caries_model.pth" ,map_location=device))
model.eval()
model=model.to(device)

class_names= ["Caries", "Deep Caries", "Impacted", "Periapical Lesion"]
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = yolo_model.predict(image)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": results[0].names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": box.xyxy[0].tolist()
        })

    return {"detections": detections}
