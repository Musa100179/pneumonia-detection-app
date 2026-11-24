import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ---------------- 1. PAGE CONFIGURATION ----------------
st.set_page_config(page_title="PneumoScan AI", page_icon="🫁", layout="centered")

# Custom CSS for better design
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h3 {
        color: #444444;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- 2. MODEL SETUP ----------------
def build_model(n_classes=2):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m

@st.cache_resource
def load_model():
    model = build_model(2)
    checkpoint = torch.load("pneumonia_resnet18_small.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.float() 
    model.eval()
    return model

model = load_model()
device = torch.device('cpu')

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def is_xray(image):
    img_array = np.array(image.convert('L'))
    std_dev = img_array.std()
    mean_val = img_array.mean()
    if mean_val > 220 or std_dev < 20: 
        return False
    return True

# ---------------- 3. UI DESIGN ----------------
st.title("🫁 PneumoScan AI")
st.markdown("<h3>A Smart Chest X-Ray Analysis System</h3>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a Chest X-Ray (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

with col2:
    st.subheader("📊 Diagnostic Report")
    
    if uploaded_file is not None:
        if not is_xray(image):
            st.error("❌ **INVALID IMAGE**\n\nThe uploaded image does not appear to be a Chest X-Ray. Please upload a valid medical X-Ray.")
        else:
            with st.spinner("Analyzing the image..."):
                img_tensor = eval_tf(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = F.softmax(model(img_tensor), dim=1)[0]
                
                prob_normal = output[0].item()
                prob_pneumonia = output[1].item()
                
                st.write("")
                if prob_pneumonia > 0.40:
                    st.error(f"⚠️ **PNEUMONIA DETECTED**")
                    st.metric(label="Confidence Score", value=f"{prob_pneumonia*100:.2f}%")
                    st.info("Recommendation: Please consult a radiologist or a healthcare professional immediately for further evaluation.")
                else:
                    st.success(f"✅ **NORMAL**")
                    st.metric(label="Confidence Score", value=f"{prob_normal*100:.2f}%")
                    st.info("Recommendation: The lungs appear clear. However, always consult a doctor for a final diagnosis.")
    else:
        st.info("👈 Please upload an X-Ray image on the left to see the results.")
        
st.write("---")
st.caption("Disclaimer: This AI tool is for educational purposes only and is not a substitute for professional medical advice.")
        # THRESHOLD KO 0.40 KAR DIYA HAI (Zyada Pneumonia pakarne ke liye)
    if prob_pneumonia > 0.40:
            st.warning(f"⚠️ PNEUMONIA DETECTED\nConfidence: {prob_pneumonia*100:.2f}%\nPlease consult a doctor.")
        else:
            st.success(f"✅ NORMAL\nConfidence: {prob_normal*100:.2f}%\nLungs look clear.")
