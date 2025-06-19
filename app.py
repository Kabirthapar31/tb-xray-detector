import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import os

# Title
st.title("🩻 Tuberculosis (TB) Chest X-ray Detector")
st.write("Upload a chest X-ray and the AI will predict if it is **Normal** or **Tuberculosis (TB)**.")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to save model
MODEL_URL = "https://huggingface.co/kabirthapar31/tb-xray-detector/resolve/main/tb_detection_model.pth"
MODEL_PATH = "tb_detection_model.pth"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded!")

# Define the model architecture (must match how you trained it!)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Class labels
classes = ["Normal", "Tuberculosis"]

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Upload file
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Prepare the image for the model
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = classes[predicted.item()]

    st.subheader(f"🩺 Prediction: **{prediction}**")
