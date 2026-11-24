import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 1. Model Structure (ResNet18)
def build_model(n_classes=2):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m

@st.cache_resource
def load_model():
    model = build_model(2)
    # Choti (half-precision) file load kar rahe hain
    checkpoint = torch.load("pneumonia_resnet18_small.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    
    # YEH LINE SABSE ZAROORI HAI: Model ko wapis float32 mein convert karne ke liye
    model = model.float() 
    
    model.eval()
    return model

model = load_model()
device = torch.device('cpu')

# 2. Transforms (Bilkul Kaggle jaisa)
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. OOD Check (Irrelevant Image detect karne ke liye)
def is_xray(image):
    img_array = np.array(image.convert('L'))
    std_dev = img_array.std()
    mean_val = img_array.mean()
    if mean_val > 180 or std_dev < 40: 
        return False
    return True

# 4. Streamlit UI
st.title("🩺 AI Pneumonia Detection System")
st.write("Chest X-Ray upload karein. Model batayega ke Pneumonia hai ya nahi.")

uploaded_file = st.file_uploader("X-Ray image choose karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if not is_xray(image):
        st.error("❌ IRRELEVANT IMAGE: Yeh X-ray nahi lag rahi. Baraye meherbani chest X-ray upload karein.")
    else:
        img_tensor = eval_tf(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = F.softmax(model(img_tensor), dim=1)[0]
        
        prob_normal = output[0].item()
        prob_pneumonia = output[1].item()
        
        if prob_pneumonia > 0.5:
            st.warning(f"⚠️ PNEUMONIA DETECTED\nConfidence: {prob_pneumonia*100:.2f}%\nPlease consult a doctor.")
        else:
            st.success(f"✅ NORMAL\nConfidence: {prob_normal*100:.2f}%\nLungs look clear.")
