import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import urllib.request
import os

st.title("🩻 Tuberculosis Detection from Chest X-ray")

# ✅ URL of your model on Hugging Face
MODEL_URL = "https://huggingface.co/kabirthapar31/tb-xray-detector/raw/main/tb_detection_model.pth"
MODEL_PATH = "tb_detection_model.pth"

# ✅ Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model...'):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ✅ Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ✅ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with st.spinner('Analyzing...'):
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        class_names = ['Normal', 'Tuberculosis']
        prediction = class_names[predicted.item()]
    
    st.success(f"Prediction: **{prediction}**")
