from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import io
import numpy as np

app = FastAPI(title="Binary DR Detection API")

# Binary class mapping
idx_to_class = {0: "No_DR", 1: "DR"}

# Model Architecture (Binary)
class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),

            nn.Flatten(),
            nn.Linear(32 * 25 * 25, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 2)  # 2 Classes
        )
    
    def forward(self, x):
        return self.net(x)

device = "cpu"
model = DRModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred_class_idx = np.argmax(probs)
        pred_class = idx_to_class[pred_class_idx]

    return JSONResponse(
        content={
            "prediction": pred_class,
            "confidence_scores": {
                idx_to_class[i]: float(probs[i]) for i in range(len(probs))
            }
        }
    )
