from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io 

app=FastAPI()
device=torch.device("mps" if torch.backends.mps.is_available() else("cpu"))
model=models.mobilenet_v2(weights=None)
model.classifier[1]=nn.Linear(in_features=1280, out_features=4)
model.load_state_dict(torch.load("caries_model.pth" ,map_location=device))
model.eval()
model=model.to(device)

class_names= ["Caries", "Deep Caries", "Impacted", "Periapical Lesion"]
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file:UploadFile =File(...)):
    image_bytes=await file.read()
    image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    img_tensor= transform(image)
    img_tensor=img_tensor.unsqueeze(0)
    img_tensor=img_tensor.to(device)
    
    with torch.no_grad():
        output=model(img_tensor)
        _,predicted=torch.max(output,1)
        
    predicted_class=class_names[predicted.item()]
    return{"predicted_disease": predicted_class}