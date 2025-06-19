import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Setup device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("tb_detection_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Class names
classes = ['Normal', 'Tuberculosis']

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(img):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="TB Chest X-ray Detection", layout="centered")

st.title("🫁 TB Chest X-ray Detection App")
st.write("Upload a Chest X-ray image to check for **Tuberculosis (TB)**.")

uploaded_file = st.file_uploader("📁 Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        prediction = predict_image(img)
    
    st.success(f"### ✅ Prediction → **{prediction}**")

st.markdown("---")
st.caption("Developed with ❤️ by [Your Name or Parabox]")

