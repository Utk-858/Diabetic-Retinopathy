import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import json
import numpy as np

# UI Title & Description
st.title("🩺 Diabetic Retinopathy Classification App")
st.write("Upload a retina fundus image to classify the DR severity level")

# Load label mapping
with open("labels.json", "r") as f:
    label_map = json.load(f)
idx_to_class = {0: "DR", 1: "NO_DR"}


# Model Architecture (must match training)
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
            nn.Linear(32, 2)
  # 5 Classes
        )
    
    def forward(self, x):
        return self.net(x)

# Load model
device = "cpu"
model = DRModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# File upload UI
uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📌 Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class_idx = np.argmax(probs)
        pred_class = idx_to_class[pred_class_idx]

    st.subheader(f"Prediction: **{pred_class}**")

    # Show confidence scores
    st.write("### Confidence Scores")
    for idx, score in enumerate(probs):
        st.write(f"{idx_to_class[idx]}: {score:.4f}")
